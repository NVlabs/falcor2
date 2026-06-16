# SPDX-License-Identifier: Apache-2.0
"""MDL discovery provider for material image manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .schema import RenderEntry, RenderManifest, RenderOutput, stable_digest


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MDL_EXAMPLES_ROOT = PROJECT_ROOT / "data" / "assets" / "mdl_sdk_examples"


def discover_mdl_materials(
    mdl_library_path: str | Path = DEFAULT_MDL_EXAMPLES_ROOT,
    *,
    include_modules: Iterable[str] = (),
    mdl_class_compilation: bool = False,
    limit: int | None = None,
) -> RenderManifest:
    """Discover MDL materials using Falcor's native MDL SDK wrapper.

    The Python side stays dependency-light. Native code performs SDK discovery
    and returns dictionaries, which this module converts to neutral render
    entries for `MDLMaterial`.
    """

    root = Path(mdl_library_path)
    records = _discover_mdl_records(root, tuple(include_modules))
    entries = [
        _entry_from_record(record, root, mdl_class_compilation=mdl_class_compilation)
        for record in records
    ]
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    return RenderManifest(
        entries=tuple(entries),
        provenance={"provider": "mdl", "api": "MDL SDK discovery"},
    )


def _discover_mdl_records(
    mdl_library_path: Path,
    include_modules: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Call the thin Python binding around Falcor's native MDL discovery."""

    import falcor2 as f2

    if not hasattr(f2, "discover_mdl_materials"):
        raise RuntimeError(
            "falcor2.discover_mdl_materials is unavailable; build the Release falcor2_ext target"
        )
    return list(f2.discover_mdl_materials(mdl_library_path, list(include_modules)))


def _entry_from_record(
    record: dict[str, Any],
    mdl_library_path: Path,
    *,
    mdl_class_compilation: bool,
) -> RenderEntry:
    material_name = str(record["mdl_material_name"])
    output = RenderOutput(
        kind="material", name=None, comparison_profile="material", color_policy="raw_linear"
    )
    properties = {
        "mdl_library_path": str(mdl_library_path),
        "mdl_material_name": material_name,
        "mdl_class_compilation": bool(mdl_class_compilation),
    }
    digest = stable_digest(
        {
            "provider": "mdl",
            "material_class": "MDLMaterial",
            "properties": _mdl_identity_properties(properties),
            "output": output.__dict__,
        },
        16,
    )
    return RenderEntry(
        entry_id=f"mdl:{digest}",
        label=str(record.get("label") or material_name),
        material_class="MDLMaterial",
        properties=properties,
        output=output,
        provider="mdl",
        provider_metadata=dict(record),
    )


def _mdl_identity_properties(properties: dict[str, object]) -> dict[str, object]:
    result = dict(properties)
    result["mdl_library_path"] = "."
    return result
