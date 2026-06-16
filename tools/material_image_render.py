# SPDX-License-Identifier: Apache-2.0
"""CLI for rendering material image manifests."""

from __future__ import annotations

import argparse
from pathlib import Path

from falcor2.tools.material_image.render import RenderExecutionOptions, render_entries
from falcor2.tools.material_image.schema import RenderSettings, read_manifest


def main() -> int:
    """Render manifest entries and return non-zero when any entry fails."""

    parser = argparse.ArgumentParser(description="Render material image entries.")
    parser.add_argument("--entries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--timeout-seconds", type=float)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples-per-pixel", type=int, default=32)
    parser.add_argument("--sample-batch-size", type=int, default=32)
    parser.add_argument("--image-extension", choices=("exr", "hdr", "png"), default="exr")
    args = parser.parse_args()

    manifest = read_manifest(args.entries)
    settings = RenderSettings(
        width=args.width,
        height=args.height,
        samples_per_pixel=args.samples_per_pixel,
        sample_batch_size=args.sample_batch_size,
        image_extension=args.image_extension,
    )
    result = render_entries(
        manifest.entries,
        settings,
        args.output_dir,
        RenderExecutionOptions(
            jobs=args.jobs,
            resume=args.resume,
            fail_fast=args.fail_fast,
            timeout_seconds=args.timeout_seconds,
        ),
    )
    failed = [item for item in result.results if item.status == "fail"]
    print(f"Rendered {len(result.results)} entries to {args.output_dir}")
    if failed:
        print(f"Failed entries: {len(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
