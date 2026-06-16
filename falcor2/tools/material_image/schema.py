# SPDX-License-Identifier: Apache-2.0
"""Source-neutral manifest schema for material image tooling.

Discovery providers emit :class:`RenderEntry` objects. The renderer and
comparator consume those entries without parsing MaterialX paths, MDL names, or
other provider-specific identifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 1
OUTPUT_KINDS = {"material", "preview_property"}
COMPARISON_PROFILES = {"material", "preview"}
COLOR_POLICIES = {"raw_linear", "display_srgb_preview"}


@dataclass(frozen=True)
class RenderOutput:
    """Description of the image a render entry should produce.

    A material output renders the full material on the visualizer sphere. A
    preview-property output renders a named property such as albedo or emission.
    The color policy is part of the entry fingerprint because it affects the
    expected image values.
    """

    kind: str = "material"
    name: str | None = None
    comparison_profile: str = "material"
    color_policy: str = "raw_linear"

    def __post_init__(self) -> None:
        if self.kind not in OUTPUT_KINDS:
            raise ValueError(f"Unsupported output kind: {self.kind}")
        if self.comparison_profile not in COMPARISON_PROFILES:
            raise ValueError(f"Unsupported comparison profile: {self.comparison_profile}")
        if self.color_policy not in COLOR_POLICIES:
            raise ValueError(f"Unsupported color policy: {self.color_policy}")
        if self.kind == "material" and self.name is not None:
            raise ValueError("Material outputs must not name a preview property")
        if self.kind == "preview_property" and self.name not in {"albedo", "emission"}:
            raise ValueError("Preview outputs must name albedo or emission")


@dataclass(frozen=True)
class RenderEntry:
    """One renderable material image entry in a manifest."""

    entry_id: str
    """Opaque stable ID used for output file names and comparison matching."""

    label: str
    """Human-readable label shown in CLI output and reports."""

    material_class: str
    """Falcor2 Material class name, for example MaterialXMaterial or MDLMaterial."""

    properties: dict[str, Any]
    """JSON-serializable properties passed to `Scene.create_material`."""

    output: RenderOutput
    provider: str
    """Short provider name such as materialx or mdl."""

    provider_metadata: dict[str, Any] = field(default_factory=dict)
    """Provider-specific facts for humans; renderer/comparator must not parse it."""

    def __post_init__(self) -> None:
        if not self.entry_id:
            raise ValueError("entry_id must not be empty")
        if not self.label:
            raise ValueError("label must not be empty")
        if not self.material_class:
            raise ValueError("material_class must not be empty")
        if not self.provider:
            raise ValueError("provider must not be empty")
        _assert_json_like(self.properties, "properties")
        _assert_json_like(self.provider_metadata, "provider_metadata")


@dataclass(frozen=True)
class RenderSettings:
    """Pixel-affecting render settings shared by all entries in one run."""

    width: int = 512
    height: int = 512
    samples_per_pixel: int = 32
    sample_batch_size: int = 32
    image_extension: str = "exr"

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.samples_per_pixel <= 0:
            raise ValueError("samples_per_pixel must be positive")
        if self.sample_batch_size <= 0:
            raise ValueError("sample_batch_size must be positive")
        normalized = self.image_extension.lower().lstrip(".")
        if normalized not in {"exr", "png", "hdr"}:
            raise ValueError("image_extension must be exr, hdr, or png")
        object.__setattr__(self, "image_extension", normalized)


@dataclass(frozen=True)
class RenderManifest:
    """A collection of entries plus optional render settings and provenance."""

    entries: tuple[RenderEntry, ...]
    settings: RenderSettings | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        for entry in self.entries:
            if entry.entry_id in seen:
                raise ValueError(f"Duplicate entry_id: {entry.entry_id}")
            seen.add(entry.entry_id)
        _assert_json_like(self.provenance, "provenance")


@dataclass(frozen=True)
class RenderResult:
    """Result of rendering one manifest entry."""

    entry_id: str
    status: str
    image: str | None
    duration_seconds: float
    message: str = ""
    fingerprint: str | None = None


def stable_digest(payload: Any, length: int = 16) -> str:
    """Return a deterministic short SHA-256 digest for JSON-like data."""

    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def entry_fingerprint(entry: RenderEntry, settings: RenderSettings) -> str:
    """Return the resume fingerprint for an entry under pixel-affecting settings."""

    return stable_digest({"entry": render_entry_to_dict(entry), "settings": asdict(settings)}, 32)


def manifest_to_dict(
    manifest: RenderManifest, results: Iterable[RenderResult] = ()
) -> dict[str, Any]:
    """Serialize a manifest and optional render results to a JSON-ready mapping."""

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "entries": [render_entry_to_dict(entry) for entry in manifest.entries],
        "provenance": manifest.provenance,
    }
    if manifest.settings is not None:
        payload["settings"] = asdict(manifest.settings)
    result_list = [asdict(result) for result in results]
    if result_list:
        payload["results"] = result_list
    return payload


def manifest_from_dict(payload: dict[str, Any]) -> RenderManifest:
    """Deserialize a manifest mapping and validate its schema version."""

    if int(payload.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported material image schema version: {payload.get('schema_version')}"
        )
    settings_payload = payload.get("settings")
    settings = RenderSettings(**settings_payload) if settings_payload is not None else None
    return RenderManifest(
        entries=tuple(render_entry_from_dict(item) for item in payload.get("entries", [])),
        settings=settings,
        provenance=dict(payload.get("provenance", {})),
    )


def render_entry_to_dict(entry: RenderEntry) -> dict[str, Any]:
    """Serialize one render entry to a JSON-ready mapping."""

    payload = asdict(entry)
    payload["output"] = asdict(entry.output)
    return payload


def render_entry_from_dict(payload: dict[str, Any]) -> RenderEntry:
    """Deserialize one render entry from a JSON-ready mapping."""

    data = dict(payload)
    data["output"] = RenderOutput(**data["output"])
    data["properties"] = dict(data.get("properties", {}))
    data["provider_metadata"] = dict(data.get("provider_metadata", {}))
    return RenderEntry(**data)


def write_manifest(
    path: str | Path,
    manifest: RenderManifest,
    results: Iterable[RenderResult] = (),
) -> None:
    """Write a manifest JSON file, creating parent directories as needed."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest_to_dict(manifest, results), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_manifest(path: str | Path) -> RenderManifest:
    """Read a manifest JSON file."""

    return manifest_from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def read_results(path: str | Path) -> tuple[RenderResult, ...]:
    """Read render results embedded in a manifest JSON file."""

    payload = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    return tuple(RenderResult(**item) for item in payload.get("results", []))


def _assert_json_like(value: Any, label: str) -> None:
    try:
        json.dumps(value, sort_keys=True, ensure_ascii=True)
    except TypeError as exc:
        raise TypeError(f"{label} must contain JSON-serializable values") from exc
