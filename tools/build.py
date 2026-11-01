# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Minimal script that handles the cross platform bit and
# windows env setup to run a CMake build.

import argparse
import ntpath
import os
import shutil
import stat
import subprocess
import sys
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


def _run_vcvarsall(vcvarsall: Path, arguments: str, env: dict[str, str]) -> dict[str, str]:
    command = f'cmd /u /c "{vcvarsall}" {arguments} && set'
    try:
        output = subprocess.check_output(command, env=env, stderr=subprocess.STDOUT).decode(
            "utf-16le", errors="replace"
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Failed to run '{vcvarsall}' with '{arguments}'.") from exc

    return {
        key.lower(): value
        for key, _, value in (line.partition("=") for line in output.splitlines())
        if key and value
    }


def _validate_msvc_env(build_env: dict[str, str], vcvarsall: Path) -> None:
    if (
        "path" not in build_env
        or "vctoolsinstalldir" not in build_env
        or "vsinstalldir" not in build_env
        or build_env.get("visualstudioversion") != "17.0"
    ):
        raise RuntimeError(
            f"Visual Studio 2022 environment script '{vcvarsall}' returned an incomplete environment."
        )

    installation_path = vcvarsall.parents[3]
    expected_path = ntpath.normcase(ntpath.normpath(str(installation_path)))
    vs_path = ntpath.normcase(ntpath.normpath(build_env["vsinstalldir"]))
    vc_tools_path = ntpath.normcase(ntpath.normpath(build_env["vctoolsinstalldir"]))
    compiler_path = shutil.which("cl.exe", path=build_env["path"])
    linker_path = shutil.which("link.exe", path=build_env["path"])
    expected_prefix = expected_path + "\\"
    if (
        vs_path != expected_path
        or not vc_tools_path.startswith(expected_prefix)
        or not compiler_path
        or not ntpath.normcase(ntpath.normpath(compiler_path)).startswith(expected_prefix)
        or not linker_path
        or not ntpath.normcase(ntpath.normpath(linker_path)).startswith(expected_prefix)
    ):
        raise RuntimeError(
            "Visual Studio 2022 initialization produced a mixed toolchain. "
            "Start from a clean shell or the selected Visual Studio 2022 x64 Native Tools prompt."
        )


def _get_msvc_env(plat_spec: str, env: dict[str, str]) -> dict[str, str]:
    vcvarsall = _find_vs2022_vcvarsall(env)
    initial_env = {key.lower(): value for key, value in env.items()}
    active_vs_path = initial_env.get("vsinstalldir")
    if not active_vs_path and initial_env.get("vcinstalldir"):
        active_vs_path = str(Path(initial_env["vcinstalldir"]).parent)
    if active_vs_path:
        # Switching installations in one Developer shell leaves the old compiler on PATH.
        active_vcvarsall = Path(active_vs_path) / "VC" / "Auxiliary" / "Build" / "vcvarsall.bat"
        if not active_vcvarsall.is_file():
            raise RuntimeError(
                f"Cannot reset the active Visual Studio environment: '{active_vcvarsall}' is missing."
            )
        env = _run_vcvarsall(active_vcvarsall, "/clean_env", env)
    elif initial_env.get("vscmd_ver"):
        raise RuntimeError("Cannot identify the active Visual Studio environment for cleanup.")

    build_env = _run_vcvarsall(vcvarsall, plat_spec, env)
    _validate_msvc_env(build_env, vcvarsall)

    build_env.pop("vcpkg_visual_studio_path", None)
    build_env["VCPKG_VISUAL_STUDIO_PATH"] = str(vcvarsall.parents[3])

    return build_env


def get_build_env() -> dict[str, str]:
    env = os.environ.copy()
    if os.name == "nt":
        env = _get_msvc_env("x64", env)
    env["FALCOR_PYTHON_EXECUTABLE"] = sys.executable
    env["FALCOR_PYTHON_ABI"] = sys.implementation.cache_tag
    existing_keep_env = env.get("VCPKG_KEEP_ENV_VARS") or env.get("vcpkg_keep_env_vars", "")
    keep_vars = {item for item in existing_keep_env.split(";") if item}
    keep_vars.update(("FALCOR_PYTHON_EXECUTABLE", "FALCOR_PYTHON_ABI"))
    env["VCPKG_KEEP_ENV_VARS"] = ";".join(sorted(keep_vars))
    env.pop("vcpkg_keep_env_vars", None)
    return env


def configure(
    preset: str,
    cmake_extra_args: list[str] | None = None,
    fresh: bool = False,
):
    env = get_build_env()
    build_dir = get_build_dir(preset)

    # Multi-config generators (e.g. Visual Studio) don't support CMAKE_DEFAULT_BUILD_TYPE
    is_multi_config = "vs2022" in preset

    cmake_args = [
        "--preset",
        preset,
    ]
    if fresh:
        cmake_args.append("--fresh")
    cmake_args += [
        "-B",
        build_dir,
        f"-DPython_ROOT_DIR:PATH={sys.prefix}",
        f"-DPython_EXECUTABLE:FILEPATH={sys.executable}",
        f"-DFALCOR_PYTHON_ABI:STRING={sys.implementation.cache_tag}",
        f"-DPython_FIND_REGISTRY:STRING=NEVER",
    ]
    if not is_multi_config:
        cmake_args.append("-DCMAKE_DEFAULT_BUILD_TYPE=Release")
    if cmake_extra_args:
        cmake_args += cmake_extra_args

    try:
        subprocess.run(["cmake", *cmake_args], env=env, cwd=SOURCE_DIR, check=True)
    except subprocess.CalledProcessError:
        # Keep dependency failures visible even when CI cannot upload artifacts.
        for name in ("vcpkg-bootstrap.log", "vcpkg-manifest-install.log"):
            path = Path(build_dir) / name
            try:
                contents = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            except OSError as exc:
                print(f"Could not read {path}: {exc}", file=sys.stderr, flush=True)
                continue
            print(f"\n--- {path} ---\n{contents}", file=sys.stderr, flush=True)
        raise


def build(preset: str, config: str = "Release", target: str | None = None):
    env = get_build_env()
    build_dir = get_build_dir(preset)

    cmake_args = [
        "--build",
        build_dir,
        "--config",
        config,
    ]
    if target:
        cmake_args += ["--target", target]

    subprocess.run(["cmake", *cmake_args], env=env, cwd=SOURCE_DIR, check=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Configure and build Falcor2.")
    parser.add_argument("--preset", default=get_default_preset())
    parser.add_argument("--config", default="Release")
    parser.add_argument("--fresh", action="store_true", help="Discard the existing CMake cache.")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--configure-only", action="store_true")
    actions.add_argument("--build-only", action="store_true")
    parser.add_argument("--target", help="Build only the named CMake target.")
    args = parser.parse_args(argv)

    if args.build_only and args.fresh:
        parser.error("--fresh requires a configure step")
    if args.configure_only and args.target:
        parser.error("--target requires a build step")

    if not args.build_only:
        configure(args.preset, fresh=args.fresh)
    if not args.configure_only:
        build(args.preset, args.config, args.target)


if __name__ == "__main__":
    main()
