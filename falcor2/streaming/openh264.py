# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install, check, or remove the separately distributed Cisco OpenH264 runtime.

No download happens during import or encoder creation. Administrator-provided
libraries can instead be selected with VideoEncoderConfig.library_path.
Close active encoders before uninstalling the shared per-user codec.
"""

import argparse
import bz2
import hashlib
import os
from pathlib import Path
import platform
import tempfile
import urllib.request

import falcor2 as f2

VERSION = "2.6.0"
ATTRIBUTION = "OpenH264 Video Codec provided by Cisco Systems, Inc."
LICENSE_PATH = Path(__file__).with_name("OPENH264_BINARY_LICENSE.txt")
_ARTIFACTS = {
    ("Windows", "amd64"): (
        "openh264-2.6.0-win64.dll",
        "dab5f2a872777f9a58b69bfa9fbcf20d9f82f2d6ec91383fd70bff49bd34ac9f",
        "2076cb5675ec6c1a4c70e7a2a322552f547b6eeed649d6dfcd9e02a543b24691",
    ),
    ("Linux", "x86_64"): (
        "libopenh264-2.6.0-linux64.8.so",
        "27ab53323c110b76214c1c72222f459d17febbcd1e252136cadc292b0308d75b",
        "2f0cde7c6a6abcf5cae76942894ea42897fa677bce4ed6c91a24dd1b041d5f04",
    ),
    ("Linux", "aarch64"): (
        "libopenh264-2.6.0-linux-arm64.8.so",
        "a78aea7970150f46bcd3bb7994c9e6dd90bd7a9ea785920f5a73f6964e3fcda7",
        "12e7b33623667cdab0e575170c147b1b36eadb77d0d2aa7ceb5afd3e58902140",
    ),
}


def runtime_directory() -> Path:
    return Path(f2.platform.app_data_directory()) / "codecs" / "openh264" / VERSION


def _artifact() -> tuple[str, str, str]:
    key = (platform.system(), platform.machine().lower())
    if key not in _ARTIFACTS:
        raise RuntimeError(
            f"No managed OpenH264 runtime for {key}; set VideoEncoderConfig.library_path"
        )
    return _ARTIFACTS[key]


def _verify(data: bytes, expected: str) -> None:
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError("OpenH264 SHA-256 mismatch; the runtime was not installed")


def install() -> Path:
    """Download the pinned Cisco binary directly and verify both archive and library."""
    name, archive_hash, library_hash = _artifact()
    destination = runtime_directory() / name
    if destination.is_file():
        _verify(destination.read_bytes(), library_hash)
        return destination
    url = f"https://ciscobinary.openh264.org/{name}.bz2"
    with urllib.request.urlopen(url, timeout=60) as response:
        archive = response.read(16 * 1024 * 1024 + 1)
    _verify(archive, archive_hash)
    library = bz2.decompress(archive)
    _verify(library, library_hash)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as file:
            temporary = Path(file.name)
            file.write(library)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def uninstall() -> Path:
    """Remove the managed codec. Close active encoders first; other codec files are untouched."""
    name, _, _ = _artifact()
    destination = runtime_directory() / name
    destination.unlink(missing_ok=True)
    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["install", "check", "uninstall"])
    args = parser.parse_args(argv)
    print(ATTRIBUTION)
    print(f"Binary license: {LICENSE_PATH}")
    try:
        if args.command == "install":
            print(install())
        elif args.command == "check":
            if not f2.streaming.VideoEncoder.check_available():
                print("OpenH264 is unavailable; enable native streaming support and run install")
                return 1
            print("OpenH264 is available")
        else:
            print(f"Removed managed runtime: {uninstall()}")
        return 0
    except (OSError, RuntimeError) as error:
        print(f"OpenH264 setup failed: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
