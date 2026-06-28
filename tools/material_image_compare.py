# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI for comparing two material image render directories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.material_image_comparison import compare_image_sets


def main() -> int:
    """Build a comparison report for two render result directories."""

    parser = argparse.ArgumentParser(description="Compare two material image render directories.")
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--tested-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = compare_image_sets(args.reference_root, args.tested_root, args.output_dir)
    print(f"Wrote {len(report.rows)} comparison rows to {args.output_dir / 'comparison.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
