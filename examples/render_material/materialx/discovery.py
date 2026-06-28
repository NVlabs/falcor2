# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaterialX discovery provider for material image manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..render_material_manifest import (
    RenderEntry,
    RenderOutput,
    output_identity_payload,
    stable_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATERIALX_ROOT = PROJECT_ROOT / "external" / "MaterialX"
SHADING_ELEMENT_TYPES = {
    "material",
    "surfaceshader",
    "volumeshader",
    "lightshader",
    "BSDF",
    "EDF",
    "VDF",
}


@dataclass(frozen=True)
class MaterialXRenderableElement:
    name: str
    element_type: str


def discover_renderable_elements(
    material_path: Path, materialx_root: Path
) -> tuple[MaterialXRenderableElement, ...]:
    """Return renderable element names and types from one MaterialX document."""

    import falcor2 as f2

    resolver = f2.AssetResolver()
    resolver.add_search_path(materialx_root)
    return tuple(
        MaterialXRenderableElement(
            name=str(record["name"]),
            element_type=str(record["type"]),
        )
        for record in f2.discover_materialx_renderable_elements(
            material_path,
            asset_resolver=resolver,
        )
    )


def make_materialx_entry(
    *,
    label: str,
    material_path: Path,
    materialx_root: Path,
    basepath: Path,
    element_name: str,
    element_type: str,
    layering_mode: str,
) -> RenderEntry:
    """Create one neutral render entry for a MaterialX renderable element."""

    from falcor2.tools.materials.material_tools import (
        MATERIALX_TEST_GEOMPROP_IDS,
        MATERIALX_TEST_GEOMPROP_NAMES,
    )

    output = render_output_for_element_type(element_type)
    material_basepath = basepath if basepath != materialx_root else material_path.parent
    properties = {
        "mtlx_path": str(material_path),
        "mtlx_node_name": element_name,
        "mtlx_basepath": str(material_basepath),
        "mtlx_search_paths": str(materialx_root),
        "mtlx_layering_mode": layering_mode,
        "mtlx_target_color_space_override": "lin_rec709",
        "mtlx_autogamma": False,
        "mtlx_geomprop_names": list(MATERIALX_TEST_GEOMPROP_NAMES),
        "mtlx_geomprop_ids": list(MATERIALX_TEST_GEOMPROP_IDS),
    }
    digest = stable_digest(
        {
            "provider": "materialx",
            "material_class": "MaterialXMaterial",
            "properties": _materialx_identity_properties(
                properties,
                material_path=material_path,
                material_basepath=material_basepath,
                materialx_root=materialx_root,
            ),
            "output": output_identity_payload(output),
        },
        16,
    )
    return RenderEntry(
        entry_id=f"materialx:{digest}",
        label=label,
        material_class="MaterialXMaterial",
        properties=properties,
        output=output,
        provider="materialx",
        output_subdirectory=_safe_output_path(
            _relative_or_absolute(material_path.with_suffix(""), materialx_root),
            "materialx",
        ),
        artifact_name=_safe_output_part(
            _materialx_shader_name(element_name),
            f"material_{digest[:8]}",
        ),
    )


def render_output_for_element_type(element_type: str) -> RenderOutput:
    if element_requires_shading(element_type):
        return RenderOutput(
            kind="material",
            name=None,
        )
    return RenderOutput(
        kind="preview_property",
        name="albedo",
    )


def element_requires_shading(element_type: str) -> bool:
    return element_type in SHADING_ELEMENT_TYPES


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _materialx_identity_properties(
    properties: dict[str, object],
    *,
    material_path: Path,
    material_basepath: Path,
    materialx_root: Path,
) -> dict[str, object]:
    result = dict(properties)
    result["mtlx_path"] = _relative_or_absolute(material_path, materialx_root)
    result["mtlx_basepath"] = _relative_or_absolute(material_basepath, materialx_root)
    result["mtlx_search_paths"] = _relative_or_absolute(materialx_root, materialx_root)
    return result


def _materialx_shader_name(element_name: str) -> str:
    return element_name.replace("/", "_").replace(":", "_")


def _safe_output_path(value: str, fallback: str) -> str:
    parts = [
        _safe_output_part(part, fallback) for part in value.replace("\\", "/").split("/") if part
    ]
    return "/".join(parts) or fallback


def _safe_output_part(value: str, fallback: str) -> str:
    return _safe_artifact_name(value) or fallback


def _safe_artifact_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip()
    safe = safe.lstrip(".")
    return "" if safe in {"", ".", ".."} else safe
