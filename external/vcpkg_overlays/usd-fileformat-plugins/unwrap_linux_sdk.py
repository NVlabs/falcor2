# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Copy the SDK tarball out of the SHA512-pinned Autodesk Linux installer."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("installer", type=Path)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()

    # SDK 2020.3.11 contains three gzip streams: readme, license, and SDK tar.
    # These offsets apply only to the archive pinned in fbxsdk.cmake. Copy the
    # third stream without executing the ELF installer or its embedded scripts.
    with args.installer.open("rb") as source:
        source.seek(572730)
        payload = source.read(41560879)
    if len(payload) != 41560879 or not payload.startswith(b"\x1f\x8b\x08"):
        raise ValueError("Unexpected FBX SDK 2020.3.11 Linux installer payload")
    args.archive.write_bytes(payload)


if __name__ == "__main__":
    main()
