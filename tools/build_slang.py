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


def get_slang_build_dir(preset: str) -> str:
    return str(CUSTOM_SLANG_SOURCE_DIR / "build" / preset)


def get_build_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.name == "nt":
        sys.path.append(str(Path(__file__).parent.parent / "external/slangpy/tools"))
        import msvc  # type: ignore

        env = msvc.msvc14_get_vc_env("x64")
    return env


def clear_slang():
    if CUSTOM_SLANG_SOURCE_DIR.exists():

        def del_rw(action, name, exc):
            # Change file permissions
            os.chmod(name, stat.S_IWRITE | stat.S_IREAD)

            # Retry the operation
            if os.path.isdir(name):
                os.rmdir(name)
            else:
                os.remove(name)

        shutil.rmtree(CUSTOM_SLANG_SOURCE_DIR, onerror=del_rw)


def clone_slang(slang_repository: str, slang_branch: str):
    env = get_build_env()

    git_args = [
        "clone",
        "--recursive",
        "--branch",
        slang_branch,
        slang_repository,
        str(CUSTOM_SLANG_SOURCE_DIR),
    ]

    subprocess.run(["git", *git_args], env=env, cwd=SOURCE_DIR, check=True)


def configure_slang():
    assert CUSTOM_SLANG_SOURCE_DIR.exists()

    env = get_build_env()

    build_dir = CUSTOM_SLANG_SOURCE_DIR / "build"
    os.makedirs(build_dir, exist_ok=True)

    subprocess.run(["cmake", "--version"], env=env, cwd=CUSTOM_SLANG_SOURCE_DIR, check=True)

    cmake_args = [
        "--preset",
        "default",
        "-B",
        build_dir,
        "-DCMAKE_DEFAULT_BUILD_TYPE=Release",
        f"-DPython_ROOT_DIR:PATH={sys.prefix}",
        f"-DPython_FIND_REGISTRY:STRING=NEVER",
    ]

    subprocess.run(["cmake", *cmake_args], env=env, cwd=CUSTOM_SLANG_SOURCE_DIR, check=True)


def build_slang(config: str = "Release"):
    assert CUSTOM_SLANG_SOURCE_DIR.exists()

    env = get_build_env()

    build_dir = CUSTOM_SLANG_SOURCE_DIR / "build"
    os.makedirs(build_dir, exist_ok=True)

    cmake_args = [
        "--build",
        build_dir,
        "--config",
        config,
        "--parallel",
    ]

    subprocess.run(["cmake", *cmake_args], env=env, cwd=CUSTOM_SLANG_SOURCE_DIR, check=True)

    # Workaround for this error on the CI:
    # CMake Warning at cmake/GitHubRelease.cmake:164 (message):
    #   Failed to download release info for version 2025.24.3 from
    #   https://api.github.com/repos/shader-slang/slang/releases/tags/v2025.24.3
    #   Falling back to latest version if it differs
    #   If you think this is failing because of GitHub API rate limiting, Github
    #   allows a higher limit if you use a token.  Try the cmake option
    #   -DSLANG_GITHUB_TOKEN=your_token_here
    # Since we do not actually need the dll for falcor2, we can just create an empty file.
    slang_llvm_dll = build_dir / config / "bin" / "slang-llvm.dll"
    if PLATFORM == "windows" and not slang_llvm_dll.exists():
        slang_llvm_dll.parent.mkdir(parents=True, exist_ok=True)
        slang_llvm_dll.touch()


if __name__ == "__main__":
    configure_slang()
    build_slang()
