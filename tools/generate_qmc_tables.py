# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract public PBRT Sobol directions and notices into packaged JSON.

Pass a directory containing pbrt/sobolmatrices.cpp and pbrt/LICENSE.txt from
PBRT_REVISION below. Input hashes are checked before writing any output.
The runtime needs no network access or source checkout.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
PBRT_REVISION = "b4ce9687e6c695f5582997c61b0c66cf064bdb4a"
SOURCE_SHA256 = {
    "pbrt/sobolmatrices.cpp": "98be104d68f0a90b4043e457b4288d7fb5b3650aae4ec60d72edc07a8435fef6",
    "pbrt/LICENSE.txt": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
}


def generate(source_dir: Path) -> None:
    for name, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((source_dir / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source hash mismatch for {name}: {actual}")
    source = (source_dir / "pbrt/sobolmatrices.cpp").read_text()
    values = [
        int(value, 16)
        for value in re.findall(r"0x[0-9a-fA-F]+", source.split("= {", 1)[1].split("};", 1)[0])
    ]
    if len(values) != 1024 * 52:
        raise ValueError(f"Expected 1024 by 52 PBRT directions, found {len(values)}")
    # Columns correspond to index bits. Discarding columns 32-51 preserves all
    # values for uint indices; all 1024 dimension matrices remain unchanged.
    data = dict(
        pbrt_revision=PBRT_REVISION,
        source_sha256=SOURCE_SHA256,
        sobol_notice=source.split("#include")[0].strip(),
        pbrt_license=(source_dir / "pbrt/LICENSE.txt").read_text(),
        sobol_directions=[values[d * 52 : d * 52 + 32] for d in range(1024)],
    )
    target = ROOT / "falcor2/utils/qmc_tables.json"
    target.write_text(json.dumps(data, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir", type=Path)
    generate(parser.parse_args().source_dir)
