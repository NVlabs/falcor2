# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared render backend names for render-material suites."""

from __future__ import annotations


PATH_TRACER_BACKEND = "pathtracer"
MATERIAL_VISUALIZER_BACKEND = "material_visualizer"
RENDER_BACKENDS = (PATH_TRACER_BACKEND, MATERIAL_VISUALIZER_BACKEND)


def validate_render_backend(value: str) -> str:
    if value not in RENDER_BACKENDS:
        raise ValueError(f"renderBackend must be one of: {', '.join(RENDER_BACKENDS)}")
    return value


def image_suffix_for_backend(render_backend: str) -> str:
    validate_render_backend(render_backend)
    if render_backend == PATH_TRACER_BACKEND:
        return "_falcor2"
    return "_visualizer"
