# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import re
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

try:
    from setuptools import Extension, find_namespace_packages, setup
    from setuptools.command.build_ext import build_ext
except ImportError:
    print(
        "The preferred way to invoke 'setup.py' is via pip, as in 'pip "
        "install .'. If you wish to run the setup script directly, you must "
        "first install the build dependencies listed in pyproject.toml!",
        file=sys.stderr,
    )
    raise

SOURCE_DIR = Path(__file__).parent.resolve()

if sys.platform.startswith("win"):
    PLATFORM = "windows"
elif sys.platform.startswith("linux"):
    PLATFORM = "linux"
elif sys.platform.startswith("darwin"):
    PLATFORM = "macos"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")

CMAKE_PRESET = {
    "windows": "windows-msvc",
    "linux": "linux-gcc",
    "macos": "macos-arm64-clang",
}[PLATFORM]

# Check if native extension build is disabled
NO_CMAKE_BUILD = os.environ.get("NO_CMAKE_BUILD") == "1"
BUILD_RELEASE_WHEEL = os.environ.get("BUILD_RELEASE_WHEEL") == "1"
CMAKE_CONFIG = "Release"


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Setup environment variables
        env = os.environ.copy()
        if os.name == "nt":
            sys.path.append(str(Path(__file__).parent / "external/slangpy/tools"))
            import msvc  # type: ignore

            env = {key.upper(): value for key, value in env.items()}
            env.update({key.upper(): value for key, value in msvc.msvc14_get_vc_env("x64").items()})

        build_dir_path = SOURCE_DIR / "build/pip"
        if BUILD_RELEASE_WHEEL and build_dir_path.exists():
            shutil.rmtree(build_dir_path)
        build_dir = str(build_dir_path)

        env["FALCOR_PYTHON_EXECUTABLE"] = sys.executable
        env["FALCOR_PYTHON_ABI"] = sys.implementation.cache_tag
        keep_vars = {item for item in env.get("VCPKG_KEEP_ENV_VARS", "").split(";") if item}
        keep_vars.update(("FALCOR_PYTHON_EXECUTABLE", "FALCOR_PYTHON_ABI"))
        env["VCPKG_KEEP_ENV_VARS"] = ";".join(sorted(keep_vars))

        cmake_args = [
            "--preset",
            CMAKE_PRESET,
            "-B",
            build_dir,
            f"-DCMAKE_DEFAULT_BUILD_TYPE={CMAKE_CONFIG}",
            f"-DPython_ROOT_DIR:PATH={sys.prefix}",
            f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
            f"-DFALCOR_PYTHON_ABI:STRING={sys.implementation.cache_tag}",
            f"-DPython_FIND_REGISTRY:STRING=NEVER",
            f"-DCMAKE_INSTALL_PREFIX={extdir}",
            f"-DCMAKE_INSTALL_LIBDIR=.",
            f"-DCMAKE_INSTALL_BINDIR=.",
            f"-DCMAKE_INSTALL_INCLUDEDIR=include",
            f"-DCMAKE_INSTALL_DATAROOTDIR=.",
            "-DFALCOR_BUILD_TESTS=OFF",
            "-DSGL_BUILD_EXAMPLES=OFF",
            "-DSGL_BUILD_TESTS=OFF",
        ]

        if BUILD_RELEASE_WHEEL:
            cmake_args += [
                "-DFALCOR_PROJECT_DIR=",
                "-DFALCOR_PYTHON_HOME=",
            ]

        # Adding CMake arguments set as environment variable
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        # Configure, build and install
        subprocess.run(["cmake", *cmake_args], env=env, check=True)
        subprocess.run(
            ["cmake", "--build", build_dir, "--config", CMAKE_CONFIG], env=env, check=True
        )
        subprocess.run(
            ["cmake", "--install", build_dir, "--config", CMAKE_CONFIG], env=env, check=True
        )


VERSION_REGEX = re.compile(r"^\s*#\s*define\s+FALCOR_VERSION_([A-Z]+)\s+(.*)$", re.MULTILINE)

with open("src/falcor2/falcor2.h") as f:
    matches = dict(VERSION_REGEX.findall(f.read()))
    version = "{MAJOR}.{MINOR}.{PATCH}".format(**matches)
    version += os.environ.get("FALCOR2_VERSION_SUFFIX", "")
    print(f"version={version}")

install_requires = ["typing_extensions", "numpy>=1.26.0", "tomli>=2; python_version < '3.11'"]
if BUILD_RELEASE_WHEEL:
    if not os.environ.get("FALCOR2_VERSION_SUFFIX"):
        raise RuntimeError("Release-wheel builds require FALCOR2_VERSION_SUFFIX")
    try:
        slangpy_version = package_version("slangpy")
    except PackageNotFoundError as error:
        raise RuntimeError("Release-wheel builds require the paired SlangPy wheel") from error
    install_requires.append(f"slangpy=={slangpy_version}")

setup(
    version=version,
    packages=find_namespace_packages(include=["falcor2", "falcor2.*"]) + ["falcor2.tools.skills"],
    ext_modules=[] if NO_CMAKE_BUILD else [CMakeExtension("falcor2.falcor2_ext")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=install_requires,
    zip_safe=False,
)
