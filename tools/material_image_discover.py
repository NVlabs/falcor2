# SPDX-License-Identifier: Apache-2.0
"""CLI for writing material image discovery manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from falcor2.tools.material_image.discover_materialx import (
    DEFAULT_MATERIALX_ROOT,
    discover_representative_materialx,
)
from falcor2.tools.material_image.discover_mdl import (
    DEFAULT_MDL_EXAMPLES_ROOT,
    discover_mdl_materials,
)
from falcor2.tools.material_image.schema import write_manifest


def main() -> int:
    """Run the selected discovery provider and write a v1 manifest."""

    layering_mode_choices = ["closure_tree", "bsdf_mix"]

    parser = argparse.ArgumentParser(description="Discover renderable material image entries.")
    subparsers = parser.add_subparsers(dest="provider", required=True)

    materialx = subparsers.add_parser("materialx")
    materialx.add_argument("--mode", choices=("representative",), default="representative")
    materialx.add_argument("--materialx-root", type=Path, default=DEFAULT_MATERIALX_ROOT)
    materialx.add_argument(
        "--layering-mode",
        choices=tuple(layering_mode_choices),
        default="bsdf_mix",
    )
    materialx.add_argument("--output", type=Path, required=True)

    mdl = subparsers.add_parser("mdl")
    mdl.add_argument("--mdl-library-path", type=Path, default=DEFAULT_MDL_EXAMPLES_ROOT)
    mdl.add_argument("--include-module", action="append", default=[])
    mdl.add_argument("--mdl-class-compilation", choices=("true", "false"), default="false")
    mdl.add_argument("--limit", type=int)
    mdl.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.provider == "materialx":
        manifest = discover_representative_materialx(
            args.materialx_root,
            layering_mode=args.layering_mode,
        )
        output = args.output
    else:
        manifest = discover_mdl_materials(
            args.mdl_library_path,
            include_modules=args.include_module,
            mdl_class_compilation=args.mdl_class_compilation == "true",
            limit=args.limit,
        )
        output = args.output
    write_manifest(output, manifest)
    print(f"Wrote {len(manifest.entries)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
