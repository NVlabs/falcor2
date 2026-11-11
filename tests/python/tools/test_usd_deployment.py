# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.parametrize("system_name", ["Linux", "Darwin"])
@pytest.mark.parametrize("build_python", [False, True])
def test_usd_deployment_preserves_runtime_names(
    tmp_path: Path, system_name: str, build_python: bool
) -> None:
    cmake = shutil.which("cmake")
    ninja = shutil.which("ninja")
    if cmake is None or ninja is None:
        pytest.skip("CMake and Ninja are required for deployment tests")

    # No compiler or native binaries are needed to exercise the real staging
    # and installation rules for versioned shared libraries on either platform.
    package = tmp_path / "packages" / "test-triplet"
    (package / "share" / "pxr").mkdir(parents=True)
    plugins = package / "lib" / "usd"
    plugins.mkdir(parents=True)
    (plugins / "plugInfo.json").write_text('{"Includes": []}', encoding="utf-8")
    python_package = package / "python" / "falcor2" / "pxr"
    python_package.mkdir(parents=True)
    (python_package / "__init__.py").write_text("", encoding="utf-8")

    libraries = {
        "usd_m": ("falcor2_usd_ms", None),
        "TBB::tbb": ("falcor2_tbb_2021_13", 12),
        "TBB::tbbmalloc": ("falcor2_tbbmalloc_2021_13", 2),
    }
    target_rules = []
    runtime_names = []
    for target, (name, major) in libraries.items():
        if system_name == "Linux":
            runtime_name = f"lib{name}.so" + (f".{major}" if major else "")
            filename = runtime_name + (".13" if major else "")
        else:
            runtime_name = f"lib{name}" + (f".{major}" if major else "") + ".dylib"
            filename = f"lib{name}" + (f".{major}.13" if major else "") + ".dylib"
        artifact = package / "lib" / filename
        artifact.write_text(target, encoding="utf-8")
        # Model the installed SONAME alias without requiring symlink privileges.
        (package / "lib" / runtime_name).write_text(target, encoding="utf-8")
        runtime_names.append((runtime_name, target))
        target_rules.append(
            f"add_library({target} SHARED IMPORTED)\n"
            f'set_target_properties({target} PROPERTIES IMPORTED_LOCATION "{artifact.as_posix()}" '
            f'IMPORTED_SONAME "{runtime_name}")'
        )

    deployment = Path(__file__).resolve().parents[3] / "cmake" / "falcor2_usd_deployment.cmake"
    source = "\n".join(
        [
            "cmake_minimum_required(VERSION 3.21)",
            "project(usd_deployment NONE)",
            "add_library(falcor2_usd INTERFACE)",
            *target_rules,
            f'set(pxr_DIR "{package.as_posix()}/share/pxr")',
            'set(FALCOR_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/runtime")',
            "set(FALCOR_COPY_DIRECTORY copy_directory)",
            "set(CMAKE_INSTALL_BINDIR bin)",
            f"set(FALCOR_BUILD_PYTHON {'ON' if build_python else 'OFF'})",
            f'set(VCPKG_INSTALLED_DIR "{package.parent.as_posix()}")',
            "set(VCPKG_TARGET_TRIPLET test-triplet)",
            f'include("{deployment.as_posix()}")',
        ]
    )
    (tmp_path / "CMakeLists.txt").write_text(source, encoding="utf-8")
    build = tmp_path / "build"
    install = tmp_path / "install"
    for arguments in [
        [
            "-S",
            str(tmp_path),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={ninja}",
            f"-DCMAKE_SYSTEM_NAME={system_name}",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        ["--build", str(build)],
        ["--install", str(build), "--prefix", str(install)],
    ]:
        result = subprocess.run([cmake, *arguments], capture_output=True, text=True)
        assert result.returncode == 0, result.stdout + result.stderr

    for runtime_name, contents in runtime_names:
        assert (build / "runtime" / runtime_name).read_text(encoding="utf-8") == contents
        assert (install / "bin" / runtime_name).read_text(encoding="utf-8") == contents
        if build_python:
            assert (build / "runtime" / "python" / "falcor2" / runtime_name).read_text(
                encoding="utf-8"
            ) == contents
    assert (install / "bin" / "usd" / "plugInfo.json").is_file()
    assert (install / "bin" / "pxr" / "__init__.py").is_file() == build_python
