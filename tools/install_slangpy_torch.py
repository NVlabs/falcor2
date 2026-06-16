# SPDX-License-Identifier: Apache-2.0

"""
Install the slangpy_torch PyTorch extension.

The extension lives at external/slangpy/src/slangpy_torch and must be
installed with --no-build-isolation after torch is present in the env.
See external/slangpy/src/slangpy_torch/README.md for background.

This script:
  - Skips silently if torch is not installed.
  - Runs the pip install with output tee'd to build/slangpy_torch_install.log.
  - On failure, prints a warning referencing the log and exits 0 so callers
    (pipinstall.sh / pipinstall.bat) don't abort the overall setup.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import build as bt

SOURCE_DIR = Path(__file__).parent.parent.resolve()
EXTENSION_DIR = SOURCE_DIR / "external" / "slangpy" / "src" / "slangpy_torch"
LOG_DIR = SOURCE_DIR / "build"
LOG_PATH = LOG_DIR / "slangpy_torch_install.log"


def torch_is_installed() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def run_install(editable: bool) -> int:
    cmd = [sys.executable, "-m", "pip", "install"]
    if editable:
        cmd.append("--editable")
    cmd += [str(EXTENSION_DIR), "--no-build-isolation"]

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # On Windows, torch's BuildExtension shells out to cl.exe and (when available)
    # ninja. Neither is on PATH by default in CI shells, so use the same MSVC env
    # setup that build.py uses for the main cmake build.
    env = bt.get_build_env()

    print(f"Installing slangpy_torch (editable={editable}) ...")
    print(f"Logging output to {LOG_PATH}")
    sys.stdout.flush()

    with open(LOG_PATH, "w") as log:
        log.write(f"Command: {' '.join(cmd)}\n\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
        process.wait()
        return process.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--non-editable",
        action="store_true",
        help="Install non-editable (default is editable install).",
    )
    args = parser.parse_args()

    if not EXTENSION_DIR.is_dir():
        print(f"WARNING: slangpy_torch source not found at {EXTENSION_DIR}; " "skipping install.")
        return 0

    if not torch_is_installed():
        print(
            "torch is not installed; skipping slangpy_torch install. "
            "Install torch first and rerun setup to enable the extension."
        )
        return 0

    returncode = run_install(editable=not args.non_editable)

    if returncode != 0:
        print("")
        print("=" * 72)
        print("WARNING: slangpy_torch install failed.")
        print(f"  See {LOG_PATH} for details.")
        print("  Continuing setup; slangpy_torch will not be available.")
        print("=" * 72)
        return 0

    print("slangpy_torch install complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
