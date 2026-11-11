# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resource paths shared by source-tree and installed Falcor2 packages."""

from pathlib import Path


def _package_directory(package_directory: Path | None) -> Path:
    return (package_directory or Path(__file__).resolve().parent).resolve()


def resource_root(package_directory: Path | None = None) -> Path:
    """Return the directory containing Falcor2 runtime resource folders."""
    package_directory = _package_directory(package_directory)
    if (package_directory / "slang").is_dir():
        return package_directory
    return package_directory.parent


def falcor_slang_include_paths(package_directory: Path | None = None) -> list[Path]:
    """Return Falcor-owned Slang include paths for a checkout or installed wheel."""
    root = resource_root(package_directory)
    paths = [root / "slang"]
    shader_path = root / "shaders"
    if shader_path.is_dir():
        paths.append(shader_path)
    return paths


def materialx_default_light_path(package_directory: Path | None = None) -> Path:
    """Return the MaterialX preview environment map for this package layout."""
    package_directory = _package_directory(package_directory)
    if (package_directory / "slang").is_dir():
        return package_directory / "mtlx/resources/Lights/san_giuseppe_bridge.hdr"
    return package_directory.parent / "external/MaterialX/resources/Lights/san_giuseppe_bridge.hdr"
