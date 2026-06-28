# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Falcor material property construction for render-material entries."""

from __future__ import annotations

from typing import Any

from .render_material_manifest import RenderEntry


def make_material_properties(entry: RenderEntry) -> Any:
    """Create Falcor2 properties from one neutral render entry."""

    import falcor2 as f2

    enum_type_by_key = {
        "mtlx_layering_mode": f2.MaterialXLayeringMode,
        "mtlx_compensation": f2.MaterialXCompensationMode,
    }
    props = f2.Properties()
    for key, value in entry.properties.items():
        enum_type = enum_type_by_key.get(key)
        if enum_type is not None and isinstance(value, str):
            try:
                value = getattr(enum_type, value)
            except AttributeError as exc:
                raise ValueError(f"Unsupported {key} value '{value}'.") from exc
        props[key] = value
    return props
