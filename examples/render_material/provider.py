# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider interface for render-material examples."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .pathtracer_backend import PathTracerPreviewSettings
from .visualizer_backend import MaterialVisualizerSettings
from .render_material_manifest import RenderEntry, RenderSettings


class MaterialProvider(Protocol):
    """Source-specific adapter used by the shared render-material suite."""

    @property
    def name(self) -> str: ...

    def collect_entries(self) -> tuple[RenderEntry, ...]: ...

    def render_settings(self) -> RenderSettings: ...

    def render_backend(self) -> str: ...

    def preview_settings(self) -> PathTracerPreviewSettings: ...

    def visualizer_settings(self) -> MaterialVisualizerSettings: ...

    def output_directory(self) -> Path: ...

    def warning_messages(self) -> tuple[str, ...]: ...

    def progress_subject(self, entry: RenderEntry) -> str: ...
