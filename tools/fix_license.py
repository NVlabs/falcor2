#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
A utility for adding SPDX license headers to source files.
"""

from __future__ import annotations
import sys
import os
import re
import argparse
from datetime import datetime
from typing import Any, Sequence
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent.resolve()

INCLUDE_PATHS = [
    "falcor2",
    "slang",
    "src",
    "tests",
    "tools",
]

EXCLUDE_PATHS = [
    "src/falcor2_ext/py_doc.h",
]

EXTENSIONS = ["h", "cpp", "slang", "slangh", "py"]

# Matches old single-line SPDX-License-Identifier header (with or without trailing blank line)
OLD_SPDX_SINGLE_REGEX = re.compile(
    r"^((// )|(# ))SPDX-License-Identifier: [^\n]*\n\n?", re.MULTILINE
)
# Matches the full 2-line SPDX header block (with or without trailing blank line)
OLD_SPDX_FULL_REGEX = re.compile(
    r"^((// )|(# ))SPDX-FileCopyrightText: [^\n]*\n((// )|(# ))SPDX-License-Identifier: [^\n]*\n\n?",
    re.MULTILINE,
)
SHEBANG_REGEX = re.compile(r"^#!.*\n")

CURRENT_YEAR = datetime.now().year

SPDX_COPYRIGHT = (
    f"SPDX-FileCopyrightText: Copyright (c) {CURRENT_YEAR} NVIDIA CORPORATION"
    " & AFFILIATES. All rights reserved."
)
SPDX_LICENSE = "SPDX-License-Identifier: Apache-2.0"


def make_header(is_python: bool) -> str:
    prefix = "# " if is_python else "// "
    return f"{prefix}{SPDX_COPYRIGHT}\n{prefix}{SPDX_LICENSE}\n"


def list_files(
    root: Path,
    include: Sequence[Path | str],
    exclude: Sequence[Path | str],
    extensions: Sequence[str] = [],
):
    # collect files
    files: list[str] = []
    for path in include:
        path = os.path.normpath(root / path)
        if os.path.isdir(path):
            for dirpath, _, fnames in os.walk(path):
                fpaths = [os.path.join(dirpath, fname) for fname in fnames]
                for f in fpaths:
                    ext = os.path.splitext(f)[1][1:]
                    if ext in extensions:
                        files.append(f)
        elif os.path.isfile(path):
            files.append(path)
    # filter excluded paths
    exclude = [os.path.normpath(root / p) for p in exclude]
    files = [f for f in files if not any(f.startswith(p) for p in exclude)]
    return files


def has_correct_header(text: str, header: str) -> bool:
    """Check if the file already starts with (or has after shebang) the correct header."""
    shebang_match = SHEBANG_REGEX.match(text)
    if shebang_match:
        after_shebang = text[shebang_match.end() :]
        # skip one optional blank line after shebang
        if after_shebang.startswith("\n"):
            after_shebang = after_shebang[1:]
        return after_shebang.startswith(header)
    return text.startswith(header)


def add_spdx_identifier(path: str, text: str):
    is_python = path.endswith(".py")
    header = make_header(is_python)

    if has_correct_header(text, header):
        return text

    # Remove any existing old-style SPDX headers (full 2-line or single-line)
    text = OLD_SPDX_FULL_REGEX.sub("", text)
    text = OLD_SPDX_SINGLE_REGEX.sub("", text)

    # check for shebang line
    shebang_match = SHEBANG_REGEX.match(text)
    if shebang_match:
        shebang_line = shebang_match.group()
        rest = text[shebang_match.end() :].lstrip("\n")
        parts = [shebang_line, "\n", header]
        if rest:
            parts.append("\n")
            parts.append(rest)
        return "".join(parts)
    else:
        # add extra newline if file is not empty
        if text != "":
            header += "\n"
        return header + text


def process_file(path: str, dry_run: bool = False):
    ext = os.path.splitext(path)[1][1:]
    if not ext in EXTENSIONS:
        return

    text = open(path, "r").read()
    edited = add_spdx_identifier(path, text)
    if edited != text:
        if dry_run:
            print(edited[0:100])
        else:
            print(f"Writing file '{path}'")
            open(path, "w").write(edited)


def run(args: Any):
    files = list_files(
        root=PROJECT_DIR,
        include=INCLUDE_PATHS,
        exclude=EXCLUDE_PATHS,
        extensions=EXTENSIONS,
    )

    for file in files:
        process_file(file, args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        default=False,
        help="run without writing files",
    )

    args = parser.parse_args()

    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
