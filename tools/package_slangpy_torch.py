# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Package the pinned SlangPy torch bridge for compilation against users' PyTorch."""

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from packaging.version import Version


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent


def build_sdist(source: Path, outdir: Path, version_suffix: str) -> Path:
    if not version_suffix:
        raise ValueError("A release version suffix is required")
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="slangpy-torch-", dir=outdir.parent) as staging:
        project = Path(staging) / "source"
        shutil.copytree(
            source,
            project,
            ignore=shutil.ignore_patterns("build", "dist", "*.egg-info", "__pycache__"),
        )
        metadata = project / "pyproject.toml"
        contents = metadata.read_text(encoding="utf-8")
        pattern = r'(?m)^version = "([^"]+)"$'
        match = re.search(pattern, contents)
        if match is None:
            raise RuntimeError("Cannot find slangpy-torch's project version")
        version = str(Version(match[1] + version_suffix))
        contents, count = re.subn(pattern, f'version = "{version}"', contents)
        if count != 1:
            raise RuntimeError("Expected exactly one slangpy-torch project version")
        metadata.write_text(contents, encoding="utf-8")
        # The upstream sdist path does not import torch or compile the extension.
        subprocess.run(
            [
                sys.executable,
                "-m",
                "build",
                "--sdist",
                "--no-isolation",
                "--outdir",
                str(outdir),
                str(project),
            ],
            check=True,
        )
    archives = list(outdir.glob(f"slangpy_torch-{version}.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(f"Expected one slangpy-torch sdist for {version}")
    return archives[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version-suffix", required=True)
    args = parser.parse_args()
    build_sdist(
        REPOSITORY_ROOT / "external" / "slangpy" / "src" / "slangpy_torch",
        REPOSITORY_ROOT / ".tmp" / "pip-wheels" / "wheelhouse",
        args.version_suffix,
    )


if __name__ == "__main__":
    main()
