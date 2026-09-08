# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal script that handles the cross platform bit and
# windows env setup to run a CMake build.

import sys, os, subprocess, shutil, stat
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent.resolve()
CUSTOM_SLANG_SOURCE_DIR = SOURCE_DIR / "external" / "slang"
VS2022_VERSION_RANGE = "[17.0,18.0)"
VC_TOOLS_COMPONENT = "Microsoft.VisualStudio.Component.VC.Tools.x86.x64"

if sys.platform.startswith("win"):
    PLATFORM = "windows"
elif sys.platform.startswith("linux"):
    PLATFORM = "linux"
elif sys.platform.startswith("darwin"):
    PLATFORM = "macos"
else:
    raise Exception(f"Unsupported platform: {sys.platform}")


def get_default_preset() -> str:
    return {
        "windows": "windows-msvc",
        "linux": "linux-gcc",
        "macos": "macos-arm64-clang",
    }[PLATFORM]


def get_build_dir(preset: str) -> str:
    return str(SOURCE_DIR / "build" / preset)


def _find_vs2022_vcvarsall(env: dict[str, str]) -> Path:
    normalized_env = {key.lower(): value for key, value in env.items()}
    program_files = normalized_env.get("programfiles(x86)") or normalized_env.get("programfiles")
    if not program_files:
        raise RuntimeError("Cannot locate Program Files to find Visual Studio 2022.")

    vswhere = Path(program_files) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.is_file():
        raise RuntimeError(f"Cannot find Visual Studio Installer discovery tool at '{vswhere}'.")

    try:
        installation_path = (
            subprocess.check_output(
                [
                    str(vswhere),
                    "-latest",
                    "-products",
                    "*",
                    "-version",
                    VS2022_VERSION_RANGE,
                    "-requires",
                    VC_TOOLS_COMPONENT,
                    "-property",
                    "installationPath",
                    "-utf8",
                ],
                env=env,
                stderr=subprocess.STDOUT,
            )
            .decode("utf-8-sig", errors="strict")
            .strip()
        )
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        raise RuntimeError("Failed to query Visual Studio 2022 with vswhere.exe.") from exc

    if not installation_path:
        raise RuntimeError(
            "Visual Studio 2022 with the C++ x64 build tools is required but was not found."
        )

    vcvarsall = Path(installation_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
    if not vcvarsall.is_file():
        raise RuntimeError(
            f"Cannot find the Visual Studio 2022 environment script at '{vcvarsall}'."
        )

    return vcvarsall


def _get_msvc_env(plat_spec: str, env: dict[str, str]) -> dict[str, str]:
    vcvarsall = _find_vs2022_vcvarsall(env)
    command = f'cmd /u /c "{vcvarsall}" {plat_spec} && set'

    try:
        output = subprocess.check_output(command, env=env, stderr=subprocess.STDOUT).decode(
            "utf-16le", errors="replace"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            f"Failed to initialize the Visual Studio 2022 environment using '{vcvarsall}'."
        ) from exc

    build_env = {
        key.lower(): value
        for key, _, value in (line.partition("=") for line in output.splitlines())
        if key and value
    }
    if "path" not in build_env or "vctoolsinstalldir" not in build_env:
        raise RuntimeError(
            f"Visual Studio 2022 environment script '{vcvarsall}' returned an incomplete environment."
        )

    return build_env


def get_build_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.name == "nt":
        env = _get_msvc_env("x64", env)
    return env


def configure(preset: str, cmake_extra_args: list[str] = []):
    env = get_build_env()
    build_dir = get_build_dir(preset)

    # Multi-config generators (e.g. Visual Studio) don't support CMAKE_DEFAULT_BUILD_TYPE
    is_multi_config = "vs2022" in preset

    cmake_args = [
        "--preset",
        preset,
        "-B",
        build_dir,
        f"-DPython_ROOT_DIR:PATH={sys.prefix}",
        f"-DPython_FIND_REGISTRY:STRING=NEVER",
    ]
    if not is_multi_config:
        cmake_args.append("-DCMAKE_DEFAULT_BUILD_TYPE=Release")
    cmake_args += cmake_extra_args

    subprocess.run(["cmake", *cmake_args], env=env, cwd=SOURCE_DIR, check=True)


def build(preset: str, config: str = "Release"):
    env = get_build_env()
    build_dir = get_build_dir(preset)

    cmake_args = [
        "--build",
        build_dir,
        "--config",
        config,
    ]

    subprocess.run(["cmake", *cmake_args], env=env, cwd=SOURCE_DIR, check=True)


if __name__ == "__main__":
    preset = get_default_preset()
    configure(preset)
    build(preset)
