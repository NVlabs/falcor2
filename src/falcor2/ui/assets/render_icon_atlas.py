# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import tomllib
import xml.etree.ElementTree as ET

import resvg_py


ASSET_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = ASSET_DIR / "icon_atlas.toml"
PNG_PATH = ASSET_DIR / "icon_atlas.png"
METADATA_PATH = ASSET_DIR.parent / "icon_data.inc"
SVG_NAMESPACE = "http://www.w3.org/2000/svg"
VALID_NAME = re.compile(r"^[a-z][a-z0-9_]*$")
VALID_EFFECTS = {"none", "drop_shadow"}


@dataclass(frozen=True)
class AtlasConfig:
    padding: int
    max_width: int


@dataclass(frozen=True)
class IconConfig:
    name: str
    width: int
    height: int
    artwork: str
    preferred_size: float
    effect: str


@dataclass(frozen=True)
class IconPlacement:
    icon: IconConfig
    x: int
    y: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the manifest-driven SVG icon atlas and C++ metadata."
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail instead of writing when a generated output is stale.",
    )
    return parser.parse_args()


def _positive_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _load_svg(source: Path, name: str) -> tuple[int, int, str]:
    root = ET.parse(source).getroot()
    if root.tag != f"{{{SVG_NAMESPACE}}}svg":
        raise ValueError(f"icon {name!r} source root must be an SVG element")

    width = _positive_int_from_string(root.attrib.get("width"), f"icon {name!r} width")
    height = _positive_int_from_string(root.attrib.get("height"), f"icon {name!r} height")
    try:
        view_box = tuple(
            float(value) for value in root.attrib.get("viewBox", "").replace(",", " ").split()
        )
    except ValueError as error:
        raise ValueError(f"icon {name!r} viewBox must contain four numbers") from error
    if view_box != (0.0, 0.0, float(width), float(height)):
        raise ValueError(f"icon {name!r} viewBox must be '0 0 {width} {height}'")

    children = list(root)
    if not children:
        raise ValueError(f"icon {name!r} source contains no artwork")
    artwork = "".join(ET.tostring(child, encoding="unicode") for child in children)
    return width, height, artwork


def _positive_int_from_string(value: object, label: str) -> int:
    if not isinstance(value, str) or not value.isdigit():
        raise ValueError(f"{label} must be a positive integer without units")
    return _positive_int(int(value), label)


def _load_manifest() -> tuple[AtlasConfig, list[IconConfig]]:
    with MANIFEST_PATH.open("rb") as stream:
        manifest = tomllib.load(stream)

    atlas_data = manifest.get("atlas")
    if not isinstance(atlas_data, dict):
        raise ValueError("manifest must contain an [atlas] table")
    atlas = AtlasConfig(
        padding=_positive_int(atlas_data.get("padding"), "atlas.padding"),
        max_width=_positive_int(atlas_data.get("max_width"), "atlas.max_width"),
    )

    icon_data = manifest.get("icon")
    if not isinstance(icon_data, list) or not icon_data:
        raise ValueError("manifest must contain at least one [[icon]] entry")

    icons: list[IconConfig] = []
    names: set[str] = set()
    asset_root = ASSET_DIR.resolve()
    for index, entry in enumerate(icon_data):
        if not isinstance(entry, dict):
            raise ValueError(f"icon entry {index} must be a table")
        name = entry.get("name")
        if not isinstance(name, str) or not VALID_NAME.fullmatch(name):
            raise ValueError(f"icon entry {index} has invalid name {name!r}")
        if name in names:
            raise ValueError(f"duplicate icon name {name!r}")
        names.add(name)

        source_value = entry.get("source")
        if not isinstance(source_value, str):
            raise ValueError(f"icon {name!r} must specify a source path")
        source = (ASSET_DIR / source_value).resolve()
        if asset_root not in source.parents or source.suffix.lower() != ".svg":
            raise ValueError(f"icon {name!r} source must be an SVG below {ASSET_DIR}")
        if not source.is_file():
            raise ValueError(f"icon {name!r} source does not exist: {source}")
        width, height, artwork = _load_svg(source, name)
        if width + 2 * atlas.padding > atlas.max_width:
            raise ValueError(
                f"icon {name!r} width plus padding exceeds atlas.max_width ({atlas.max_width})"
            )

        preferred_size = entry.get("preferred_size")
        if (
            not isinstance(preferred_size, (int, float))
            or isinstance(preferred_size, bool)
            or preferred_size <= 0
        ):
            raise ValueError(f"icon {name!r} preferred_size must be positive")
        effect = entry.get("effect", "none")
        if effect not in VALID_EFFECTS:
            raise ValueError(f"icon {name!r} has unsupported effect {effect!r}")
        icons.append(IconConfig(name, width, height, artwork, float(preferred_size), effect))

    return atlas, icons


def _pack_icons(
    atlas: AtlasConfig, icons: list[IconConfig]
) -> tuple[list[IconPlacement], int, int]:
    placements: list[IconPlacement] = []
    cursor_x = 0
    cursor_y = 0
    row_height = 0
    width = 0

    for icon in icons:
        packed_width = icon.width + 2 * atlas.padding
        packed_height = icon.height + 2 * atlas.padding
        if cursor_x > 0 and cursor_x + packed_width > atlas.max_width:
            cursor_x = 0
            cursor_y += row_height
            row_height = 0

        placements.append(
            IconPlacement(icon=icon, x=cursor_x + atlas.padding, y=cursor_y + atlas.padding)
        )
        cursor_x += packed_width
        row_height = max(row_height, packed_height)
        width = max(width, cursor_x)

    height = cursor_y + row_height
    _validate_placements(atlas, placements, width, height)
    return placements, width, height


def _validate_placements(
    atlas: AtlasConfig,
    placements: list[IconPlacement],
    width: int,
    height: int,
) -> None:
    for index, placement in enumerate(placements):
        icon = placement.icon
        left = placement.x - atlas.padding
        top = placement.y - atlas.padding
        right = placement.x + icon.width + atlas.padding
        bottom = placement.y + icon.height + atlas.padding
        if left < 0 or top < 0 or right > width or bottom > height:
            raise RuntimeError(f"icon {icon.name!r} lies outside the generated atlas")

        for other in placements[:index]:
            separated = (
                right <= other.x - atlas.padding
                or left >= other.x + other.icon.width + atlas.padding
                or bottom <= other.y - atlas.padding
                or top >= other.y + other.icon.height + atlas.padding
            )
            if not separated:
                raise RuntimeError(
                    f"padded icon rectangles overlap for {icon.name!r} and {other.icon.name!r}"
                )


def _render_svg(placements: list[IconPlacement], width: int, height: int) -> str:
    parts = [
        f'<svg xmlns="{SVG_NAMESPACE}" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<defs>",
        '<filter id="drop-shadow" x="-30%" y="-30%" width="160%" height="160%" '
        'color-interpolation-filters="sRGB">',
        '<feGaussianBlur in="SourceAlpha" stdDeviation="2" result="blur"/>',
        '<feFlood flood-color="#000" flood-opacity="0.9" result="shadow-color"/>',
        '<feComposite in="shadow-color" in2="blur" operator="in" result="shadow"/>',
        '<feMerge><feMergeNode in="shadow"/><feMergeNode in="SourceGraphic"/></feMerge>',
        "</filter>",
        "</defs>",
    ]
    for placement in placements:
        icon = placement.icon
        parts.append(f'<g transform="translate({placement.x} {placement.y})">')
        if icon.effect == "drop_shadow":
            parts.append('<g filter="url(#drop-shadow)">')
        parts.append(icon.artwork)
        if icon.effect == "drop_shadow":
            parts.append("</g>")
        parts.append("</g>")
    parts.append("</svg>")
    return "".join(parts)


def _render_metadata(
    placements: list[IconPlacement],
    width: int,
    height: int,
) -> bytes:
    lines = [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        "// Generated by ui/assets/render_icon_atlas.py. Do not edit manually.",
        f"FALCOR_UI_ICON_ATLAS({width}, {height})",
    ]
    for placement in placements:
        icon = placement.icon
        preferred_size = f"{icon.preferred_size:g}"
        if preferred_size.isdigit():
            preferred_size += "."
        lines.append(
            f"FALCOR_UI_ICON({icon.name}, {placement.x}, {placement.y}, {icon.width}, {icon.height}, "
            f"{preferred_size}f)"
        )
    lines.append("")
    return "\n".join(lines).encode("utf-8")


def _generate() -> dict[Path, bytes]:
    ET.register_namespace("", SVG_NAMESPACE)
    atlas, icons = _load_manifest()
    placements, width, height = _pack_icons(atlas, icons)
    svg = _render_svg(placements, width, height)
    return {
        PNG_PATH: resvg_py.svg_to_bytes(svg_string=svg),
        METADATA_PATH: _render_metadata(placements, width, height),
    }


def main() -> int:
    outputs = _generate()
    check = _parse_args().check
    stale: list[Path] = []
    for path, content in outputs.items():
        if check:
            if not path.is_file() or path.read_bytes() != content:
                stale.append(path)
        else:
            path.write_bytes(content)
            print(f"Generated {path.relative_to(ASSET_DIR.parent)}")

    if stale:
        for path in stale:
            print(f"{path.name} is stale; run {Path(__file__).name} to regenerate it.")
        return 1
    if check:
        print("Icon atlas and metadata are up to date.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
