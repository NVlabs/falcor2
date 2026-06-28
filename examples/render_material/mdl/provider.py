# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render MDL materials using an _options.jsonc recipe file."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

from ..pathtracer_backend import PathTracerPreviewSettings
from ..render_backend import PATH_TRACER_BACKEND, RENDER_BACKENDS, validate_render_backend
from .discovery import discover_mdl_entries
from ..suite import exit_code_for_result, run_provider
from ..render_material_manifest import RenderEntry, RenderSettings
from ..visualizer_backend import MaterialVisualizerSettings


DEFAULT_OPTIONS = PROJECT_ROOT / "examples" / "render_material" / "mdl" / "_options.jsonc"
DEFAULT_SAMPLES = 32
REFERENCE_QUALITY_SAMPLES = 256


@dataclass(frozen=True)
class MDLOptions:
    name: str
    library_path: Path
    override_files: tuple[str, ...]
    shader_interfaces: int
    render_backend: str
    render_size: tuple[int, int]
    dump_generated_code: bool
    render_geometry: Path
    enable_direct_lighting: bool
    radiance_ibl_path: Path | None
    extra_library_paths: tuple[Path, ...]
    enable_reference_quality: bool
    output_directory: Path


@dataclass(frozen=True)
class MDLProvider:
    options: MDLOptions
    device_type: str
    samples_per_pixel_override: int | None = None

    @property
    def name(self) -> str:
        return self.options.name

    def collect_entries(self) -> tuple[RenderEntry, ...]:
        entries: list[RenderEntry] = []
        include_modules = _mdl_module_filters(self.options.override_files)
        for mdl_class_compilation, output_subdirectory in _shader_interface_variants(
            self.options.shader_interfaces
        ):
            entries.extend(
                replace(
                    entry,
                    output_subdirectory=_join_output_subdirectories(
                        entry.output_subdirectory,
                        output_subdirectory,
                    ),
                )
                for entry in discover_mdl_entries(
                    self.options.library_path,
                    include_modules=include_modules,
                    mdl_class_compilation=mdl_class_compilation,
                )
            )
        return tuple(entries)

    def render_settings(self) -> RenderSettings:
        samples = (
            REFERENCE_QUALITY_SAMPLES if self.options.enable_reference_quality else DEFAULT_SAMPLES
        )
        if self.samples_per_pixel_override is not None:
            samples = self.samples_per_pixel_override
        return RenderSettings(
            width=self.options.render_size[0],
            height=self.options.render_size[1],
            samples_per_pixel=samples,
            image_extension="png",
        )

    def render_backend(self) -> str:
        return self.options.render_backend

    def preview_settings(self) -> PathTracerPreviewSettings:
        if self.options.render_geometry.suffix.lower() != ".obj":
            raise ValueError(
                f"Only OBJ preview geometry is supported in V1: {self.options.render_geometry}"
            )
        return PathTracerPreviewSettings(
            geometry_path=self.options.render_geometry,
            geometry_flip_v=self.options.render_geometry.name in {"sphere.obj", "plane.obj"},
            device_type=self.device_type,
            width=self.options.render_size[0],
            height=self.options.render_size[1],
            enable_analytic_lights=self.options.enable_direct_lighting,
            radiance_ibl_path=self.options.radiance_ibl_path,
            dump_generated_code=self.options.dump_generated_code,
        )

    def visualizer_settings(self) -> MaterialVisualizerSettings:
        return MaterialVisualizerSettings(
            device_type=self.device_type,
            radiance_ibl_path=self.options.radiance_ibl_path,
            dump_generated_code=self.options.dump_generated_code,
        )

    def output_directory(self) -> Path:
        return self.options.output_directory

    def warning_messages(self) -> tuple[str, ...]:
        messages: list[str] = []
        if self.options.extra_library_paths:
            messages.append(
                "extraLibraryPaths is not implemented by the Falcor2 render-material MDL preview."
            )
        return tuple(messages)

    def progress_subject(self, entry: RenderEntry) -> str:
        value = entry.properties.get("mdl_material_name")
        return str(value) if value else entry.label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render MDL materials with Falcor2.")
    parser.add_argument(
        "--options",
        type=Path,
        default=DEFAULT_OPTIONS,
        help="JSONC _options recipe to render.",
    )
    parser.add_argument(
        "--render-backend",
        choices=RENDER_BACKENDS,
        help="Override the renderBackend value from _options.jsonc.",
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


def provider_from_args(args: argparse.Namespace) -> MDLProvider:
    options = read_mdl_options(args.options)
    if args.render_backend is not None:
        options = replace(options, render_backend=args.render_backend)
    return MDLProvider(
        options=options,
        device_type=args.device,
        samples_per_pixel_override=args.samples_per_pixel,
    )


def read_mdl_options(path: str | Path) -> MDLOptions:
    options_path = Path(path)
    payload = _read_jsonc(options_path)
    if int(payload.get("version", 0)) != 1:
        raise ValueError(f"{options_path}: unsupported MDL options version")
    return _read_options(payload, options_path.parent)


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return exit_code_for_result(run_provider(provider_from_args(args)))


def _read_options(payload: dict[str, object], options_dir: Path) -> MDLOptions:
    name = str(payload.get("name", "mdl"))
    render_test_paths = _read_path_list(
        payload.get("renderTestPaths"), options_dir, name, "renderTestPaths"
    )
    if len(render_test_paths) != 1:
        raise ValueError(f"{name}: renderTestPaths must contain one MDL library path")
    library_path = render_test_paths[0]

    if "radianceIBLPath" not in payload:
        radiance_ibl_path = library_path / "resources" / "environment.hdr"
    elif payload["radianceIBLPath"] is None or str(payload["radianceIBLPath"]) == "":
        radiance_ibl_path = None
    else:
        radiance_ibl_path = _resolve_path(str(payload["radianceIBLPath"]), options_dir)

    render_geometry = _resolve_path(
        str(
            payload.get(
                "renderGeometry",
                "../../../external/MaterialX/resources/Geometry/sphere.obj",
            )
        ),
        options_dir,
    )
    output_directory = Path(str(payload.get("outputDirectory") or ".tmp/render_material_mdl"))
    render_size = _read_render_size(payload.get("renderSize"), name)
    override_files_value = payload.get("overrideFiles", payload.get("includeModules"))
    override_files = _read_string_list(override_files_value, name, "overrideFiles")
    shader_interfaces = int(payload.get("shaderInterfaces", 2))
    _shader_interface_variants(shader_interfaces)
    return MDLOptions(
        name=name,
        library_path=library_path,
        override_files=override_files,
        shader_interfaces=shader_interfaces,
        render_backend=validate_render_backend(
            str(payload.get("renderBackend", PATH_TRACER_BACKEND)) or PATH_TRACER_BACKEND
        ),
        render_size=(render_size[0], render_size[1]),
        dump_generated_code=bool(payload.get("dumpGeneratedCode", False)),
        render_geometry=render_geometry,
        enable_direct_lighting=bool(payload.get("enableDirectLighting", False)),
        radiance_ibl_path=radiance_ibl_path,
        extra_library_paths=_read_path_list(
            payload.get("extraLibraryPaths"), options_dir, name, "extraLibraryPaths"
        ),
        enable_reference_quality=bool(payload.get("enableReferenceQuality", False)),
        output_directory=output_directory,
    )


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve(strict=False)


def _read_render_size(value: object, command_name: str) -> tuple[int, int]:
    if value is None:
        return (512, 512)
    if not isinstance(value, list):
        raise ValueError(f"{command_name}: renderSize must contain width and height")
    if len(value) != 2:
        raise ValueError(f"{command_name}: renderSize must contain width and height")
    return int(value[0]), int(value[1])


def _positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("value must be an integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _read_string_list(value: object, command_name: str, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{command_name}: {field_name} must be a list")
    return tuple(str(item) for item in value)


def _read_path_list(
    value: object, base_dir: Path, command_name: str, field_name: str
) -> tuple[Path, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ValueError(f"{command_name}: {field_name} must be a list")
    return tuple(_resolve_path(str(item), base_dir) for item in value)


def _mdl_module_filters(override_files: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_mdl_module_filter_name(item) for item in override_files)


def _shader_interface_variants(shader_interfaces: int) -> tuple[tuple[bool, str], ...]:
    if shader_interfaces == 1:
        return ((False, "reduced"),)
    if shader_interfaces == 2:
        return ((True, ""),)
    if shader_interfaces == 3:
        return ((False, "reduced"), (True, ""))
    raise ValueError("shaderInterfaces must be 1, 2, or 3")


def _join_output_subdirectories(first: str, second: str) -> str:
    if first and second:
        return f"{first}/{second}"
    return first or second


def _mdl_module_filter_name(value: str) -> str:
    name = value.strip().strip(":")
    if name.endswith(".mdl"):
        name = Path(name).stem
    return name


def _read_jsonc(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8-sig")
    payload = json.loads(_strip_jsonc_comments_and_trailing_commas(text))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _strip_jsonc_comments_and_trailing_commas(text: str) -> str:
    without_comments = _strip_jsonc_comments(text)
    return _strip_jsonc_trailing_commas(without_comments)


def _strip_jsonc_comments(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == "/" and next_char == "/":
            index += 2
            while index < len(text) and text[index] not in "\r\n":
                index += 1
            continue
        if char == "/" and next_char == "*":
            index += 2
            while index + 1 < len(text) and not (text[index] == "*" and text[index + 1] == "/"):
                result.append("\n" if text[index] in "\r\n" else " ")
                index += 1
            index += 2
            continue
        result.append(char)
        index += 1
    return "".join(result)


def _strip_jsonc_trailing_commas(text: str) -> str:
    result: list[str] = []
    in_string = False
    escape = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            result.append(char)
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            result.append(char)
            index += 1
            continue
        if char == ",":
            scan = index + 1
            while scan < len(text) and text[scan].isspace():
                scan += 1
            if scan < len(text) and text[scan] in "}]":
                index += 1
                continue
        result.append(char)
        index += 1
    return "".join(result)
