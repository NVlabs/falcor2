# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clear known NVIDIA driver disk caches before performance measurements.

Run after building and before starting the benchmark process. This clears prior
runs' disk caches; it does not promise cold caches between cases within a run.
An inaccessible file or a redirected cache path stops the measurement job.
"""

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from collections.abc import Mapping


CACHE_LOCATIONS = (
    ("LOCALAPPDATA", "NVIDIA/DXCache"),
    ("LOCALAPPDATA", "NVIDIA/GLCache"),
    ("LOCALAPPDATA", "NVIDIA/VkCache"),
    ("LOCALAPPDATA", "NVIDIA/ComputeCache"),
    ("LOCALAPPDATA", "NVIDIA/OptixCache"),
    ("LOCALAPPDATA", "D3DSCache"),
    ("APPDATA", "NVIDIA/ComputeCache"),
    ("PROGRAMDATA", "NVIDIA Corporation/NV_Cache"),
)


def linux_cache_paths(environment: Mapping[str, str]) -> list[Path]:
    """Use fixed per-user driver caches, never arbitrary override directories."""
    import pwd

    home = Path(environment.get("HOME") or pwd.getpwuid(os.getuid()).pw_dir)
    cache = Path(environment.get("XDG_CACHE_HOME") or home / ".cache")
    if not home.is_absolute() or not cache.is_absolute():
        raise ValueError("HOME and XDG_CACHE_HOME must name absolute directories")
    username = pwd.getpwuid(os.getuid()).pw_name
    paths = [
        cache / "nvidia/GLCache",
        cache / "nvidia/VkCache",
        cache / ".nv/GLCache",
        home / ".nv/GLCache",
        home / ".nv/ComputeCache",
        Path("/var/tmp") / f"OptixCache_{username}",
        Path("/tmp") / f"OptixCache_{username}",
    ]
    # Refuse overrides rather than delete an arbitrary directory or silently
    # leave the actual driver cache warm. Unset them to use the standard paths.
    for variable in ("__GL_SHADER_DISK_CACHE_PATH", "CUDA_CACHE_PATH", "OPTIX_CACHE_PATH"):
        if environment.get(variable):
            raise ValueError(f"Unset {variable} before driver-cache cleanup")
    for path in paths:
        if path.resolve() != path.absolute():
            raise ValueError(f"Refusing redirected driver cache: {path}")
    return list(dict.fromkeys(paths))


def cache_paths(environment: Mapping[str, str]) -> list[Path]:
    """Resolve only the fixed cache locations, rejecting redirected paths."""
    if sys.platform == "linux":
        return linux_cache_paths(environment)
    if sys.platform != "win32":
        raise RuntimeError("Driver-cache cleanup supports Windows and Linux only")
    paths = []
    for variable, relative in CACHE_LOCATIONS:
        value = environment.get(variable)
        if not value or not Path(value).is_absolute():
            raise ValueError(f"{variable} must name an absolute Windows data directory")
        base = Path(value)
        path = base / relative
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(base.resolve()):
            raise ValueError(f"Refusing redirected driver cache: {path}")
        paths.append(path)
    return paths


def _check_entry(path: Path) -> os.stat_result:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError(f"Refusing driver cache symlink or reparse point: {path}")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ValueError(f"Refusing unexpected driver cache entry: {path}")
    return info


def clear_cache(path: Path, skipped: list[str] | None = None) -> int:
    """Remove cache contents without removing its root or following links."""
    if not path.exists() and not path.is_symlink():
        print(f"Driver cache absent: {path}", flush=True)
        return 0
    if not stat.S_ISDIR(_check_entry(path).st_mode):
        raise ValueError(f"Driver cache is not a directory: {path}")

    def remove_contents(directory: Path) -> int:
        count = 0
        for entry in directory.iterdir():
            info = _check_entry(entry)
            if stat.S_ISDIR(info.st_mode):
                count += remove_contents(entry)
                if not any(entry.iterdir()):
                    entry.rmdir()
            else:
                try:
                    entry.unlink()
                except OSError as error:
                    if skipped is None or getattr(error, "winerror", None) not in (32, 33):
                        raise
                    skipped.append(str(entry))
                    print(f"Driver cache file locked, retained locally: {entry}", flush=True)
                else:
                    count += 1
        return count

    count = remove_contents(path)
    if skipped is None and any(path.iterdir()):
        raise RuntimeError(f"Driver cache was repopulated during cleanup: {path}")
    print(f"Driver cache cleared: {path} ({count} files)", flush=True)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--best-effort", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.best_effort and (
        os.environ.get("CI", "").lower() == "true" or os.environ.get("CI_JOB_ID")
    ):
        raise RuntimeError("CI requires complete driver-cache cleanup")
    if args.best_effort and not args.report:
        parser.error("--best-effort requires --report")
    paths = cache_paths(os.environ)
    skipped = [] if args.best_effort else None
    total = sum(clear_cache(path, skipped) for path in paths)
    remaining = (
        [
            str(entry)
            for path in paths
            if path.exists()
            for entry in path.rglob("*")
            if entry.is_file()
        ]
        if args.best_effort
        else []
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(
                {
                    "mode": "best_effort" if args.best_effort else "required",
                    "complete": not remaining,
                    "removed_files": total,
                    "locked_files": skipped or [],
                    "remaining_files": remaining,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    print(f"Driver-cache cleanup finished: {total} files removed", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Driver-cache cleanup failed; do not start measurements: {error}", file=sys.stderr)
        sys.exit(1)
