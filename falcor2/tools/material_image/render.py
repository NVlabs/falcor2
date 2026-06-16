# SPDX-License-Identifier: Apache-2.0
"""Renderer for source-neutral material image manifests."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable

from .schema import (
    RenderEntry,
    RenderManifest,
    RenderResult,
    RenderSettings,
    entry_fingerprint,
    read_manifest,
    read_results,
    write_manifest,
)


@dataclass(frozen=True)
class RenderExecutionOptions:
    """Execution controls that do not affect pixels or render fingerprints."""

    jobs: int = 1
    resume: bool = False
    fail_fast: bool = False
    timeout_seconds: float | None = None


@dataclass(frozen=True)
class RenderRunResult:
    """Summary returned after rendering a manifest."""

    output_dir: Path
    results: tuple[RenderResult, ...]


RenderFunc = Callable[[RenderEntry, RenderSettings, Path], None]


def render_entries(
    entries: tuple[RenderEntry, ...] | list[RenderEntry],
    settings: RenderSettings,
    output_dir: str | Path,
    execution_options: RenderExecutionOptions = RenderExecutionOptions(),
    *,
    render_func: RenderFunc | None = None,
) -> RenderRunResult:
    """Render entries to `output_dir` and write an incremental manifest.

    The default renderer creates Falcor2 materials from each entry's neutral
    property map and renders them with `MaterialVisualizer`. Tests may inject
    `render_func` to exercise scheduling and resume behavior without GPU work.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    manifest = RenderManifest(entries=tuple(entries), settings=settings)
    previous = _previous_results(manifest_path) if execution_options.resume else {}
    results: list[RenderResult | None] = [None] * len(manifest.entries)
    planned: list[tuple[int, RenderEntry]] = []

    for index, entry in enumerate(manifest.entries):
        fingerprint = entry_fingerprint(entry, settings)
        image = _image_relative_path(entry, settings)
        previous_result = previous.get(entry.entry_id)
        if (
            previous_result is not None
            and previous_result.fingerprint == fingerprint
            and previous_result.image == image
            and (root / image).exists()
        ):
            results[index] = RenderResult(
                entry_id=entry.entry_id,
                status="skipped",
                image=image,
                duration_seconds=0.0,
                message="Resume skipped unchanged entry",
                fingerprint=fingerprint,
            )
        else:
            planned.append((index, entry))

    worker = render_func or _render_entry_visualizer
    if execution_options.jobs > 1 and worker is _render_entry_visualizer and len(planned) > 1:
        with ProcessPoolExecutor(max_workers=execution_options.jobs) as executor:
            futures = {
                executor.submit(_render_one_entry, entry, settings, root, None): index
                for index, entry in planned
            }
            for future in as_completed(futures, timeout=execution_options.timeout_seconds):
                index = futures[future]
                results[index] = future.result()
                _write_progress_manifest(manifest_path, manifest, results)
                if execution_options.fail_fast and results[index].status != "pass":
                    for pending in futures:
                        pending.cancel()
                    break
    else:
        for index, entry in planned:
            results[index] = _render_one_entry(entry, settings, root, worker)
            _write_progress_manifest(manifest_path, manifest, results)
            if execution_options.fail_fast and results[index].status != "pass":
                break

    final_results = tuple(result for result in results if result is not None)
    write_manifest(manifest_path, manifest, final_results)
    return RenderRunResult(output_dir=root, results=final_results)


def _render_one_entry(
    entry: RenderEntry,
    settings: RenderSettings,
    output_dir: Path,
    render_func: RenderFunc | None,
) -> RenderResult:
    """Render one entry and convert failures into a result row."""

    image_rel = _image_relative_path(entry, settings)
    image_path = output_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fingerprint = entry_fingerprint(entry, settings)
    try:
        (render_func or _render_entry_visualizer)(entry, settings, image_path)
        status = "pass"
        message = "Rendered"
    except Exception as exc:  # noqa: BLE001 - CLI reports must survive individual entry failures.
        status = "fail"
        message = str(exc)
    return RenderResult(
        entry_id=entry.entry_id,
        status=status,
        image=image_rel if status == "pass" else None,
        duration_seconds=time.perf_counter() - started,
        message=message,
        fingerprint=fingerprint,
    )


def _render_entry_visualizer(
    entry: RenderEntry, settings: RenderSettings, image_path: Path
) -> None:
    """Render one entry through Falcor2's MaterialVisualizer backend."""

    import falcor2 as f2
    import slangpy as spy
    from falcor2.editor.utils import create_device, save_image
    from falcor2.tools.materials.material_tools import (
        MaterialVisualizer,
        apply_materialx_test_geomprop_properties,
    )

    device = create_device()
    scene = f2.Scene(device)
    props = _to_falcor_properties(entry.properties)
    if entry.material_class == "MaterialXMaterial":
        apply_materialx_test_geomprop_properties(props)
    material = scene.create_material(entry.material_class, props)
    visualizer = MaterialVisualizer(device)
    result_texture = device.create_texture(
        format=spy.Format.rgba32_float,
        width=settings.width,
        height=settings.height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label=f"material_image_{_safe_name(entry.entry_id)}",
    )
    if entry.output.kind == "material":
        texture = visualizer.show_material(
            scene,
            material,
            samples_per_pixel=settings.samples_per_pixel,
            result_texture=result_texture,
            sample_batch_size=settings.sample_batch_size,
            camera_size=(settings.width, settings.height),
        )
    else:
        textures = visualizer.show_preview_properties(
            scene,
            material,
            property_textures={entry.output.name: result_texture},
        )
        texture = textures[str(entry.output.name)]
    save_image(texture, image_path)


def _to_falcor_properties(values: dict[str, object]):
    import falcor2 as f2

    props = f2.Properties()
    for key, value in values.items():
        props[key] = _coerce_materialx_enum_property(f2, key, value)
    return props


def _coerce_materialx_enum_property(f2, key: str, value: object) -> object:
    if not isinstance(value, str):
        return value

    enum_type_by_key = {
        "mtlx_layering_mode": f2.MaterialXLayeringMode,
        "mtlx_compensation": f2.MaterialXCompensationMode,
    }
    enum_type = enum_type_by_key.get(key)
    if enum_type is None:
        return value
    try:
        return getattr(enum_type, value)
    except AttributeError as exc:
        raise ValueError(f"Unsupported {key} value '{value}'.") from exc


def _image_relative_path(entry: RenderEntry, settings: RenderSettings) -> str:
    return f"images/{_safe_name(entry.entry_id)}.{settings.image_extension}"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _previous_results(manifest_path: Path) -> dict[str, RenderResult]:
    if not manifest_path.exists():
        return {}
    try:
        read_manifest(manifest_path)
        return {result.entry_id: result for result in read_results(manifest_path)}
    except Exception:
        return {}


def _write_progress_manifest(
    path: Path,
    manifest: RenderManifest,
    results: list[RenderResult | None],
) -> None:
    write_manifest(path, manifest, (result for result in results if result is not None))
