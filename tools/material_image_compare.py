# SPDX-License-Identifier: Apache-2.0
"""CLI for comparing two material image render directories."""

from __future__ import annotations

import argparse
from pathlib import Path

from falcor2.tools.material_image.compare import compare_image_sets


def main() -> int:
    """Build a comparison report for two render result directories."""

    parser = argparse.ArgumentParser(description="Compare two material image render directories.")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--tested-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / ".reference_root").write_text(str(args.reference_root), encoding="utf-8")
    (args.output_dir / ".tested_root").write_text(str(args.tested_root), encoding="utf-8")
    report = compare_image_sets(args.reference_root, args.tested_root, args.output_dir)
    print(f"Wrote {len(report.rows)} comparison rows to {args.output_dir / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
