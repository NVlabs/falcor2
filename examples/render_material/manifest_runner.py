# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-neutral manifest runner for render-material examples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .render_material_manifest import (
    RenderEntry,
    RenderManifest,
    RenderResult,
    RenderSettings,
    write_manifest,
)


@dataclass(frozen=True)
class RenderExecutionOptions:
    """Execution controls that do not affect pixels."""

    fail_fast: bool = False
    image_name_suffix: str = ""
    manifest_name: str = "manifest.json"


@dataclass(frozen=True)
class RenderRunResult:
    """Summary returned after rendering a manifest."""

    output_dir: Path
    results: tuple[RenderResult, ...]


RenderFunc = Callable[[RenderEntry, RenderSettings, Path], None]
ProgressMessageFunc = Callable[[RenderEntry], str | None]


def render_entries(
    entries: tuple[RenderEntry, ...] | list[RenderEntry],
    settings: RenderSettings,
    output_dir: str | Path,
    execution_options: RenderExecutionOptions | None = None,
    *,
    render_func: RenderFunc,
    progress_message_func: ProgressMessageFunc | None = None,
) -> RenderRunResult:
    """Render entries to `output_dir` and write an incremental manifest."""

    if execution_options is None:
        execution_options = RenderExecutionOptions()

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / execution_options.manifest_name
    manifest = RenderManifest(entries=tuple(entries), settings=settings)
    results: list[RenderResult | None] = [None] * len(manifest.entries)
    planned: list[tuple[int, RenderEntry]] = []

    for index, entry in enumerate(manifest.entries):
        planned.append((index, entry))

    for index, entry in planned:
        _print_progress(entry, progress_message_func)
        results[index] = _render_one_entry(
            entry,
            settings,
            root,
            execution_options.image_name_suffix,
            render_func,
        )
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
    image_name_suffix: str,
    render_func: RenderFunc,
) -> RenderResult:
    """Render one entry and convert failures into a result row."""

    image_rel = _image_relative_path(entry, settings, image_name_suffix)
    image_path = output_dir / image_rel
    image_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_func(entry, settings, image_path)
        status = "pass"
        message = "Rendered"
    except Exception as exc:  # noqa: BLE001 - CLI reports must survive individual entry failures.
        status = "fail"
        message = str(exc)
    return RenderResult(
        entry_id=entry.entry_id,
        status=status,
        image=image_rel if status == "pass" else None,
        message=message,
    )


def _image_relative_path(
    entry: RenderEntry, settings: RenderSettings, image_name_suffix: str = ""
) -> str:
    if entry.artifact_name:
        file_name = (
            f"{_safe_name(entry.artifact_name)}{image_name_suffix}.{settings.image_extension}"
        )
        if entry.output_subdirectory:
            subdirectory = entry.output_subdirectory.replace("\\", "/")
            return f"{subdirectory}/{file_name}"
        return file_name

    image_dir = "images"
    if entry.output_subdirectory:
        subdirectory = entry.output_subdirectory.replace("\\", "/")
        image_dir = f"{subdirectory}/{image_dir}"
    file_name = f"{_safe_name(entry.entry_id)}{image_name_suffix}.{settings.image_extension}"
    return f"{image_dir}/{file_name}"


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _print_progress(
    entry: RenderEntry,
    progress_message_func: ProgressMessageFunc | None,
) -> None:
    if progress_message_func is None:
        return
    message = progress_message_func(entry)
    if message:
        print(message, flush=True)


def _write_progress_manifest(
    path: Path,
    manifest: RenderManifest,
    results: list[RenderResult | None],
) -> None:
    write_manifest(path, manifest, (result for result in results if result is not None))
