# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared render-material suite loop."""

from __future__ import annotations

from pathlib import Path

from .pathtracer_backend import PathTracerPreviewRenderer
from .provider import MaterialProvider
from .render_material_manifest import RenderEntry
from .render_backend import (
    MATERIAL_VISUALIZER_BACKEND,
    PATH_TRACER_BACKEND,
    image_suffix_for_backend,
    validate_render_backend,
)
from .manifest_runner import RenderExecutionOptions, RenderRunResult, render_entries
from .visualizer_backend import MaterialVisualizerRenderer


def exit_code_for_result(result: RenderRunResult) -> int:
    """Return a process exit code for a completed render-material run."""

    return 1 if any(row.status == "fail" for row in result.results) else 0


def progress_message_for_entry(render_backend: str, subject: str) -> str:
    """Return a MaterialXTest-style per-entry progress line."""

    return f"Validating {_progress_backend_name(render_backend)} rendering for: " f"{subject}"


def run_provider(provider: MaterialProvider) -> RenderRunResult:
    """Render all entries from a configured material provider."""

    for message in provider.warning_messages():
        print(f"warning: {message}")

    entries = provider.collect_entries()
    render_backend = validate_render_backend(provider.render_backend())
    if render_backend == PATH_TRACER_BACKEND:
        settings = provider.preview_settings()
        _require_material_ibl(entries, settings.radiance_ibl_path)
        renderer = PathTracerPreviewRenderer(settings)
    elif render_backend == MATERIAL_VISUALIZER_BACKEND:
        settings = provider.visualizer_settings()
        _require_material_ibl(entries, settings.radiance_ibl_path)
        renderer = MaterialVisualizerRenderer(settings)
    else:
        raise ValueError(f"Unsupported render backend: {render_backend}")

    result = render_entries(
        entries,
        provider.render_settings(),
        provider.output_directory(),
        RenderExecutionOptions(
            fail_fast=False,
            image_name_suffix=image_suffix_for_backend(render_backend),
        ),
        render_func=renderer.render_entry,
        progress_message_func=lambda entry: progress_message_for_entry(
            render_backend, provider.progress_subject(entry)
        ),
    )
    print(f"Wrote {result.output_dir / 'manifest.json'}")
    for row in result.results:
        if row.status == "fail":
            print(f"failed: {row.entry_id}: {row.message}")
    return result


def _progress_backend_name(render_backend: str) -> str:
    validate_render_backend(render_backend)
    if render_backend == PATH_TRACER_BACKEND:
        return "Falcor2 PathTracer"
    return "Falcor2 MaterialVisualizer"


def _require_material_ibl(entries: tuple[RenderEntry, ...], radiance_ibl_path: object) -> None:
    if not any(entry.output.kind == "material" for entry in entries):
        return
    if radiance_ibl_path is None:
        raise ValueError("Material renders require radianceIBLPath")
    path = Path(radiance_ibl_path)
    if not path.exists():
        raise FileNotFoundError(f"radianceIBLPath does not exist: {path}")
