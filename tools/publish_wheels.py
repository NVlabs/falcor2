# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate and publish every distribution from one protected release pipeline."""

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

from packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    canonicalize_name,
    parse_sdist_filename,
    parse_wheel_filename,
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_PYTHON_TAGS = {f"cp{minor}" for minor in range(310, 315)}
_PLATFORMS = {"windows-x64", "linux-x64", "linux-arm64"}
_PACKAGES = {"slangpy", "falcor2"}


def _platform_family(platform: str) -> str:
    if platform == "win_amd64":
        return "windows-x64"
    if platform.startswith(("linux_", "manylinux", "musllinux")):
        if platform.endswith("_x86_64"):
            return "linux-x64"
        if platform.endswith("_aarch64"):
            return "linux-arm64"
    raise ValueError(f"Unsupported release wheel platform: {platform}")


def _core_wheels(wheelhouse: Path, version_suffix: str) -> dict[str, list[Path]]:
    expected = {(python, platform) for python in _PYTHON_TAGS for platform in _PLATFORMS}
    wheels: dict[str, dict[tuple[str, str], Path]] = {name: {} for name in _PACKAGES}
    versions: dict[str, set[str]] = {name: set() for name in _PACKAGES}
    for path in sorted(wheelhouse.glob("*.whl")):
        try:
            name, version, build, tags = parse_wheel_filename(path.name)
        except InvalidWheelFilename as error:
            raise ValueError(f"Invalid release wheel filename: {path.name}") from error
        if name not in _PACKAGES or build or len(tags) != 1:
            raise ValueError(f"Unexpected release wheel: {path.name}")
        if version.local != version_suffix.lstrip("+"):
            raise ValueError(f"Release wheel has another pipeline suffix: {path.name}")
        tag = next(iter(tags))
        if tag.interpreter not in _PYTHON_TAGS or tag.abi != tag.interpreter:
            raise ValueError(f"Release wheel has an unsupported Python ABI: {path.name}")
        key = (tag.interpreter, _platform_family(tag.platform))
        if key in wheels[name]:
            raise ValueError(f"Duplicate {name} release wheel for {key}: {path.name}")
        wheels[name][key] = path
        versions[name].add(str(version))

    for name in sorted(_PACKAGES):
        missing = expected - wheels[name].keys()
        if missing or len(wheels[name]) != len(expected) or len(versions[name]) != 1:
            raise ValueError(
                f"Expected one {name} wheel for each Python/platform target; missing {sorted(missing)}"
            )
    return {name: [wheels[name][key] for key in sorted(expected)] for name in _PACKAGES}


def _torch_sdist(wheelhouse: Path, version_suffix: str) -> Path:
    archives = sorted(wheelhouse.glob("*.tar.gz"))
    if len(archives) != 1:
        raise ValueError(f"Expected one slangpy-torch source archive, found {len(archives)}")
    archive = archives[0]
    try:
        name, version = parse_sdist_filename(archive.name)
    except InvalidSdistFilename as error:
        raise ValueError(f"Invalid release source archive: {archive.name}") from error
    if canonicalize_name(name) != "slangpy-torch" or version.local != version_suffix.lstrip("+"):
        raise ValueError(f"Source archive has another project or pipeline suffix: {archive.name}")
    return archive


def qualified_wheels(directory: Path, version_suffix: str, manifest: dict) -> list[Path]:
    # Delay the import until validation reaches Hydra; the module needs the
    # pinned HydraIPC checkout, which local filename-only tests do not need.
    from tools.publish_hydra import qualified_wheels as validate

    return validate(directory, version_suffix, manifest)


def collect_artifacts(
    wheelhouse: Path, hydra_wheelhouse: Path, version_suffix: str, manifest: dict
) -> list[Path]:
    """Return a complete release in dependency order, before any registry write."""
    if not re.fullmatch(r"\+rtr\.[0-9]+", version_suffix):
        raise ValueError("Use an immutable numeric CI suffix such as +rtr.123")
    if manifest.get("build_required") is not True:
        raise ValueError("A combined release requires freshly qualified Hydra wheels")

    core = _core_wheels(wheelhouse, version_suffix)
    torch = _torch_sdist(wheelhouse, version_suffix)
    hydra = qualified_wheels(hydra_wheelhouse, version_suffix, manifest)
    if len(hydra) != 2:
        raise ValueError("Expected qualified Windows and Linux Hydra wheels")
    return [torch, *core["slangpy"], *core["falcor2"], *hydra]


def publish(
    wheelhouse: Path,
    hydra_wheelhouse: Path,
    version_suffix: str,
    manifest: dict,
    repository_url: str,
) -> None:
    """Validate the complete set, then invoke Twine once for the upload."""
    artifacts = [
        str(path)
        for path in collect_artifacts(wheelhouse, hydra_wheelhouse, version_suffix, manifest)
    ]
    subprocess.run([sys.executable, "-m", "twine", "check", "--strict", *artifacts], check=True)
    subprocess.run(
        [sys.executable, "-m", "twine", "upload", "--repository-url", repository_url, *artifacts],
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--hydra-wheelhouse", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--version-suffix", required=True)
    parser.add_argument("--repository-url", required=True)
    args = parser.parse_args()

    if not (
        os.environ.get("CI_PIPELINE_SOURCE") in ("web", "schedule")
        and os.environ.get("BUILD_WHEELS") == "true"
        and os.environ.get("CI_COMMIT_REF_PROTECTED") == "true"
        and os.environ.get("CI_COMMIT_BRANCH") == os.environ.get("CI_DEFAULT_BRANCH")
        and os.environ.get("CI_DEFAULT_BRANCH")
    ):
        parser.error("Publication requires a protected default-branch release pipeline")

    try:
        from tools.discover_blender_releases import read_manifest

        manifest = read_manifest(args.manifest)
        publish(
            args.wheelhouse,
            args.hydra_wheelhouse,
            args.version_suffix,
            manifest,
            args.repository_url,
        )
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"Release publication failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
