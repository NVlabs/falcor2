# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MDL discovery provider for material image manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ..render_material_manifest import (
    RenderEntry,
    RenderOutput,
    output_identity_payload,
    stable_digest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MDL_EXAMPLES_ROOT = PROJECT_ROOT / "data" / "assets" / "mdl_sdk_examples"


def discover_mdl_entries(
    mdl_library_path: str | Path = DEFAULT_MDL_EXAMPLES_ROOT,
    *,
    include_modules: Iterable[str] = (),
    mdl_class_compilation: bool = False,
    limit: int | None = None,
) -> tuple[RenderEntry, ...]:
    """Discover MDL materials as neutral render entries.

    The Python side stays dependency-light. Native code performs SDK discovery
    and returns dictionaries, which this module converts to neutral render
    entries for `MDLMaterial`.
    """

    root = Path(mdl_library_path)
    records = _discover_mdl_records(root, tuple(include_modules))
    entries = [
        make_mdl_entry(record, root, mdl_class_compilation=mdl_class_compilation)
        for record in records
    ]
    if limit is not None:
        entries = entries[: max(0, int(limit))]
    return tuple(entries)


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


def make_mdl_entry(
    record: dict[str, Any],
    mdl_library_path: Path,
    *,
    mdl_class_compilation: bool,
) -> RenderEntry:
    """Create one neutral render entry from an MDL discovery record."""

    material_name = str(record["mdl_material_name"])
    output = RenderOutput(kind="material", name=None)
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
            "output": output_identity_payload(output),
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
        output_subdirectory=_safe_output_part(
            str(record.get("module_simple_name") or "mdl"), "mdl"
        ),
        artifact_name=_safe_output_part(
            str(record.get("label") or material_name),
            f"material_{digest[:8]}",
        ),
    )


def _mdl_identity_properties(properties: dict[str, object]) -> dict[str, object]:
    result = dict(properties)
    result["mdl_library_path"] = "."
    return result


def _safe_output_part(value: str, fallback: str) -> str:
    return _safe_artifact_name(value) or fallback


def _safe_artifact_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip()
    safe = safe.lstrip(".")
    return "" if safe in {"", ".", ".."} else safe
