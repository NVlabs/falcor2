# SPDX-License-Identifier: Apache-2.0

# Minimal script that handles the cross platform bit and
# windows env setup to run a CMake build.

import sys, os, subprocess, shutil, stat
from pathlib import Path

SOURCE_DIR = Path(__file__).parent.parent.resolve()
CUSTOM_SLANG_SOURCE_DIR = SOURCE_DIR / "external" / "slang"

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


def get_build_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.name == "nt":
        sys.path.append(str(Path(__file__).parent.parent / "external/slangpy/tools"))
        import msvc  # type: ignore

        env = msvc.msvc14_get_vc_env("x64")
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
