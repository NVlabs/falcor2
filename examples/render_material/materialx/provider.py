# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render MaterialX materials using a MaterialX-style _options.mtlx file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable
import xml.etree.ElementTree as ET

import slangpy as spy

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..pathtracer_backend import PathTracerPreviewSettings
from ..render_backend import PATH_TRACER_BACKEND, RENDER_BACKENDS, validate_render_backend
from .discovery import (
    discover_renderable_elements,
    make_materialx_entry,
)
from ..suite import exit_code_for_result, run_provider
from ..render_material_manifest import (
    RenderEntry,
    RenderSettings,
)
from ..visualizer_backend import MaterialVisualizerSettings


DEFAULT_OPTIONS = PROJECT_ROOT / "examples" / "render_material" / "materialx" / "_options.mtlx"
DEFAULT_MATERIALX_ROOT = PROJECT_ROOT / "external" / "MaterialX"
DEFAULT_SAMPLES = 32
REFERENCE_QUALITY_SAMPLES = 256
DEFAULT_RADIANCE_IBL_PATH = "resources/Lights/san_giuseppe_bridge.hdr"


@dataclass(frozen=True)
class MaterialXOptions:
    override_files: tuple[str, ...] = ()
    shader_interfaces: int = 2
    render_backend: str = PATH_TRACER_BACKEND
    render_size: tuple[int, int] = (512, 512)
    dump_generated_code: bool = False
    render_geometry: str = "sphere.obj"
    enable_direct_lighting: bool = False
    radiance_ibl_path: str = DEFAULT_RADIANCE_IBL_PATH
    extra_library_paths: tuple[str, ...] = ()
    render_test_paths: tuple[str, ...] = ()
    enable_reference_quality: bool = False
    output_directory: str = ".tmp/render_material_mtlx"


@dataclass(frozen=True)
class MaterialXProvider:
    options: MaterialXOptions
    materialx_root: Path
    device_type: str
    samples_per_pixel_override: int | None = None
    name: str = "materialx"

    def collect_entries(self) -> tuple[RenderEntry, ...]:
        return collect_materialx_entries(self.options, self.materialx_root)

    def render_settings(self) -> RenderSettings:
        settings = render_settings(self.options)
        if self.samples_per_pixel_override is None:
            return settings
        return replace(settings, samples_per_pixel=self.samples_per_pixel_override)

    def render_backend(self) -> str:
        return self.options.render_backend

    def preview_settings(self) -> PathTracerPreviewSettings:
        return preview_settings(self.options, self.materialx_root, self.device_type)

    def visualizer_settings(self) -> MaterialVisualizerSettings:
        return MaterialVisualizerSettings(
            device_type=self.device_type,
            radiance_ibl_path=radiance_ibl_path(self.options, self.materialx_root),
            dump_generated_code=self.options.dump_generated_code,
        )

    def output_directory(self) -> Path:
        return Path(self.options.output_directory)

    def warning_messages(self) -> tuple[str, ...]:
        return tuple(warning_messages(self.options))

    def progress_subject(self, entry: RenderEntry) -> str:
        value = entry.properties.get("mtlx_path")
        return str(value) if value else entry.label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MaterialX materials with Falcor2.")
    parser.add_argument(
        "--options",
        type=Path,
        default=DEFAULT_OPTIONS,
        help="MaterialX-style _options.mtlx recipe to render.",
    )
    parser.add_argument(
        "--materialx-root",
        type=Path,
        default=DEFAULT_MATERIALX_ROOT,
        help="Root directory of the MaterialX checkout used to resolve resources.",
    )
    parser.add_argument(
        "--render-backend",
        choices=RENDER_BACKENDS,
        help="Override the renderBackend value from _options.mtlx.",
    )
    parser.add_argument(
        "--samples-per-pixel",
        type=_positive_int,
        help="Override the option-derived sample count for all rendered entries.",
    )
    parser.add_argument(
        "--device",
        choices=("automatic", "d3d12", "vulkan", "metal", "cuda"),
        default="automatic",
        help="Graphics backend device to use.",
    )
    return parser


def provider_from_args(args: argparse.Namespace) -> MaterialXProvider:
    options = read_materialx_options(args.options)
    if args.render_backend is not None:
        options = replace(options, render_backend=args.render_backend)
    return MaterialXProvider(
        options=options,
        materialx_root=args.materialx_root,
        device_type=select_materialx_device_type(args.device),
        samples_per_pixel_override=args.samples_per_pixel,
    )


def read_materialx_options(path: str | Path) -> MaterialXOptions:
    root = ET.parse(path).getroot()
    nodedef = next(
        (
            element
            for element in root.iter()
            if _local_name(element.tag) == "nodedef"
            and element.attrib.get("name") == "TestSuiteOptions"
        ),
        None,
    )
    if nodedef is None:
        raise ValueError(f"{path}: missing TestSuiteOptions nodedef")

    values = {
        element.attrib["name"]: element.attrib.get("value", "")
        for element in nodedef
        if _local_name(element.tag) == "input" and "name" in element.attrib
    }
    return MaterialXOptions(
        override_files=_split_list(values.get("overrideFiles", "")),
        shader_interfaces=_parse_int(values.get("shaderInterfaces", "2")),
        render_backend=validate_render_backend(
            values.get("renderBackend", PATH_TRACER_BACKEND) or PATH_TRACER_BACKEND
        ),
        render_size=_parse_vector2i(values.get("renderSize", "512, 512")),
        dump_generated_code=_parse_bool(values.get("dumpGeneratedCode", "false")),
        render_geometry=values.get("renderGeometry", "sphere.obj") or "sphere.obj",
        enable_direct_lighting=_parse_bool(values.get("enableDirectLighting", "false")),
        radiance_ibl_path=values.get("radianceIBLPath", DEFAULT_RADIANCE_IBL_PATH),
        extra_library_paths=_split_list(values.get("extraLibraryPaths", "")),
        render_test_paths=_split_list(values.get("renderTestPaths", "")),
        enable_reference_quality=_parse_bool(values.get("enableReferenceQuality", "false")),
        output_directory=values.get("outputDirectory", ".tmp/render_material_mtlx")
        or ".tmp/render_material_mtlx",
    )


def collect_materialx_files(options: MaterialXOptions, materialx_root: Path) -> list[Path]:
    material_paths: list[Path] = []
    for item in options.render_test_paths:
        path = _resolve_materialx_data_path(item, materialx_root)
        if path.is_dir():
            for directory in _materialx_test_directories(path):
                material_paths.extend(sorted(directory.glob("*.mtlx")))
        elif path.is_file():
            material_paths.append(path)
        else:
            raise FileNotFoundError(f"MaterialX renderTestPaths entry does not exist: {path}")

    if options.override_files:
        filters = set(options.override_files)
        material_paths = [path for path in material_paths if path.name in filters]
    return list(dict.fromkeys(material_paths))


def collect_materialx_entries(
    options: MaterialXOptions, materialx_root: Path
) -> tuple[RenderEntry, ...]:
    entries: list[RenderEntry] = []
    for material_path in collect_materialx_files(options, materialx_root):
        for element in discover_renderable_elements(material_path, materialx_root):
            entries.append(
                make_materialx_entry(
                    label=f"{material_path.stem}:{element.name}",
                    material_path=material_path,
                    materialx_root=materialx_root,
                    basepath=material_path.parent,
                    element_name=element.name,
                    element_type=element.element_type,
                    layering_mode="bsdf_mix",
                )
            )
    seen: set[str] = set()
    deduped: list[RenderEntry] = []
    for entry in entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        deduped.append(entry)
    return tuple(deduped)


def render_geometry_path(options: MaterialXOptions, materialx_root: Path) -> Path:
    geometry_path = Path(options.render_geometry)
    if not geometry_path.is_absolute():
        geometry_path = materialx_root / "resources" / "Geometry" / geometry_path
    if geometry_path.suffix.lower() != ".obj":
        raise ValueError(f"Only OBJ preview geometry is supported in V1: {geometry_path}")
    return geometry_path


def radiance_ibl_path(options: MaterialXOptions, materialx_root: Path) -> Path | None:
    if not options.radiance_ibl_path:
        return None
    return _resolve_materialx_data_path(options.radiance_ibl_path, materialx_root)


def render_settings(options: MaterialXOptions) -> RenderSettings:
    samples = REFERENCE_QUALITY_SAMPLES if options.enable_reference_quality else DEFAULT_SAMPLES
    return RenderSettings(
        width=options.render_size[0],
        height=options.render_size[1],
        samples_per_pixel=samples,
        image_extension="png",
    )


def preview_settings(
    options: MaterialXOptions,
    materialx_root: Path,
    device_type: str,
) -> PathTracerPreviewSettings:
    geometry_path = render_geometry_path(options, materialx_root)
    return PathTracerPreviewSettings(
        geometry_path=geometry_path,
        geometry_flip_v=geometry_path.name in {"sphere.obj", "plane.obj"},
        device_type=device_type,
        width=options.render_size[0],
        height=options.render_size[1],
        enable_analytic_lights=options.enable_direct_lighting,
        radiance_ibl_path=radiance_ibl_path(options, materialx_root),
        dump_generated_code=options.dump_generated_code,
    )


def select_materialx_device_type(
    requested: str,
    adapter_probe: Callable[[str], bool] | None = None,
) -> str:
    if requested == "cuda":
        raise ValueError(
            "MaterialXMaterial is disabled on CUDA while NVRTC compile-time issues are investigated."
        )
    if requested != "automatic":
        return requested

    probe = adapter_probe or _has_adapter
    for candidate in ("d3d12", "vulkan", "metal"):
        if not hasattr(spy.DeviceType, candidate):
            continue
        if probe(candidate):
            return candidate
    raise RuntimeError("No non-CUDA graphics backend is available for MaterialX rendering.")


def warning_messages(options: MaterialXOptions) -> list[str]:
    messages: list[str] = []
    if options.shader_interfaces != 2:
        # TODO: Add MaterialX reduced/complete interface variants when the native MaterialXMaterial supports them.
        messages.append(
            "shaderInterfaces is not implemented by the Falcor2 render-material preview."
        )
    if options.extra_library_paths:
        messages.append(
            "extraLibraryPaths is not implemented by the Falcor2 render-material preview."
        )
    return messages


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return exit_code_for_result(run_provider(provider_from_args(args)))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _split_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0", ""}:
        return False
    raise ValueError(f"Invalid boolean value '{value}'")


def _parse_int(value: str) -> int:
    return int(value.strip())


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _parse_vector2i(value: str) -> tuple[int, int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError(f"Expected vector2 value, got '{value}'")
    return int(float(values[0])), int(float(values[1]))


def _resolve_materialx_data_path(value: str, materialx_root: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return materialx_root / path


def _materialx_test_directories(root: Path) -> list[Path]:
    directories = [root]
    for child in sorted(root.iterdir()):
        if child.is_dir():
            directories.extend(_materialx_test_directories(child))
    return directories


def _has_adapter(device_type_name: str) -> bool:
    try:
        return bool(spy.Device.enumerate_adapters(type=getattr(spy.DeviceType, device_type_name)))
    except Exception:
        return False
