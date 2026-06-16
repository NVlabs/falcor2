# SPDX-License-Identifier: Apache-2.0
"""MaterialX discovery provider for material image manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .schema import RenderEntry, RenderManifest, RenderOutput, stable_digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MATERIALX_ROOT = PROJECT_ROOT / "external" / "MaterialX"


@dataclass(frozen=True)
class MaterialXSeed:
    """One MaterialX file/node seed used by the representative discovery mode."""

    label: str
    relative_path: str
    node_name: str | None = None
    expand_all_nodes: bool = False


REPRESENTATIVE_SEEDS = (
    MaterialXSeed(
        "lama",
        "resources/Materials/TestSuite/pbrlib/surfaceshader/lama/lama_layer.mtlx",
        "LamaLayerBSDFTest",
    ),
    MaterialXSeed(
        "openpbr", "resources/Materials/Examples/OpenPbr/open_pbr_default.mtlx", "Default"
    ),
    MaterialXSeed(
        "standard_surface",
        "resources/Materials/Examples/StandardSurface/standard_surface_default.mtlx",
        "Default",
    ),
    MaterialXSeed(
        "glass", "resources/Materials/Examples/StandardSurface/standard_surface_glass.mtlx", "Glass"
    ),
    MaterialXSeed(
        "car_paint",
        "resources/Materials/Examples/StandardSurface/standard_surface_carpaint.mtlx",
        "Car_Paint",
    ),
    MaterialXSeed(
        "tiled_brass",
        "resources/Materials/Examples/StandardSurface/standard_surface_brass_tiled.mtlx",
        "Tiled_Brass",
    ),
    MaterialXSeed(
        "chess_set",
        "resources/Materials/Examples/StandardSurface/standard_surface_chess_set.mtlx",
        expand_all_nodes=True,
    ),
)


def discover_representative_materialx(
    materialx_root: str | Path = DEFAULT_MATERIALX_ROOT,
    *,
    layering_mode: str = "bsdf_mix",
) -> RenderManifest:
    """Discover the small MaterialX representative set used for v1 smoke runs."""

    root = Path(materialx_root)
    entries: list[RenderEntry] = []
    for seed in REPRESENTATIVE_SEEDS:
        entries.extend(
            _entries_for_seed(root, root, seed, layering_mode=layering_mode, flip_v_texcoord=False)
        )
    return RenderManifest(
        entries=tuple(_dedupe_entries(entries)),
        provenance={"provider": "materialx", "mode": "representative"},
    )


def discover_node_names(material_path: Path, materialx_root: Path) -> tuple[str, ...]:
    """Return material node names from one MaterialX document."""

    import falcor2 as f2

    resolver = f2.AssetResolver()
    resolver.add_search_path(materialx_root)
    return tuple(f2.discover_materialx_material_node_names(material_path, asset_resolver=resolver))


def _entries_for_seed(
    search_root: Path,
    basepath: Path,
    seed: MaterialXSeed,
    *,
    layering_mode: str,
    flip_v_texcoord: bool,
) -> list[RenderEntry]:
    material_path = search_root / seed.relative_path
    if not material_path.exists():
        return []
    node_names = (
        discover_node_names(material_path, search_root)
        if seed.expand_all_nodes or seed.node_name is None
        else (seed.node_name,)
    )
    entries = []
    for node_name in sorted(node_names):
        entries.append(
            make_materialx_entry(
                label=f"{seed.label}:{node_name}",
                material_path=material_path,
                materialx_root=search_root,
                basepath=basepath,
                node_name=node_name,
                layering_mode=layering_mode,
                flip_v_texcoord=flip_v_texcoord,
            )
        )
    return entries


def make_materialx_entry(
    *,
    label: str,
    material_path: Path,
    materialx_root: Path,
    basepath: Path,
    node_name: str,
    layering_mode: str,
    flip_v_texcoord: bool = False,
) -> RenderEntry:
    """Create one neutral render entry for a MaterialX material node."""

    relative_path = _relative_or_absolute(material_path, materialx_root)
    output = RenderOutput(
        kind="material", name=None, comparison_profile="material", color_policy="raw_linear"
    )
    material_basepath = basepath if basepath != materialx_root else material_path.parent
    properties = {
        "mtlx_path": str(material_path),
        "mtlx_node_name": node_name,
        "mtlx_basepath": str(material_basepath),
        "mtlx_search_paths": str(materialx_root),
        "mtlx_layering_mode": layering_mode,
        "mtlx_flip_v_texcoord": flip_v_texcoord,
        "mtlx_target_color_space_override": "lin_rec709",
        "mtlx_autogamma": False,
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
            "output": output.__dict__,
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
        provider_metadata={
            "relative_path": relative_path,
            "node_name": node_name,
            "layering": layering_mode,
        },
    )


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


def _dedupe_entries(entries: Iterable[RenderEntry]) -> list[RenderEntry]:
    seen: set[str] = set()
    result: list[RenderEntry] = []
    for entry in entries:
        if entry.entry_id in seen:
            continue
        seen.add(entry.entry_id)
        result.append(entry)
    return result
