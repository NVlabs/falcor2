# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run pybind11_mkdoc without putting header paths on the Windows command line."""

import sys
from pathlib import Path

from pybind11_mkdoc import main as mkdoc_main


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: run_pydoc.py ARGUMENT_FILE", file=sys.stderr)
        return 2

    args_file = Path(sys.argv[1])
    sys.argv = ["pybind11_mkdoc", *args_file.read_text(encoding="utf-8").splitlines()]
    return mkdoc_main() or 0


if __name__ == "__main__":
    sys.exit(main())
