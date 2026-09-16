# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script for running CI tasks.
"""

import os
import sys
import platform
import argparse
from pathlib import Path
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any, Optional, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tools.build as bt
import tools.build_slang as bts
from tools import crashpad
from tools.system_telemetry import DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH
from tools.system_telemetry import SystemTelemetrySampler

Command = Union[str, Sequence[str]]


class CommandError(RuntimeError):
    def __init__(self, command: str, returncode: int, output: str):
        super().__init__(f'Error running "{command}"')
        self.command = command
        self.returncode = returncode
        self.output = output


INFO_ENV_VARS = (
    "CI",
    "CI_COMMIT_BRANCH",
    "CI_COMMIT_REF_NAME",
    "CI_COMMIT_SHA",
    "CI_CONFIG",
    "CI_DEFAULT_BRANCH",
    "CI_DEVICE_CACHE_POLICY",
    "CI_FLAGS",
    "CI_JOB_ID",
    "CI_JOB_NAME",
    "CI_JOB_STAGE",
    "CI_JOB_URL",
    "CI_MODULE_AND_SHADER_CACHE_DIR",
    "CI_MODULE_CACHE",
    "CI_OS",
    "CI_PIPELINE_ID",
    "CI_PLATFORM",
    "CI_PROJECT_DIR",
    "CI_PROJECT_PATH",
    "CI_PROJECT_URL",
    "CI_PYTEST_WORKERS",
    "CI_PYTHON",
    "CI_RUNNER_DESCRIPTION",
    "CI_RUNNER_ID",
    "CI_RUNNER_TAGS",
    "CI_SHADER_CACHE",
    "CI_USE_CUSTOM_SLANG",
    "CI_FORCE_IMAGE_TEST_REPORT",
    "CONDA_DEFAULT_ENV",
    "CONDA_ENVS_DIRS",
    "CUDA_HOME",
    "CUDA_PATH",
    "KERNELVM_OS",
    "LD_LIBRARY_PATH",
    "PATH",
    "PIP_CACHE_DIR",
    "PM_PACKAGES_ROOT",
    "PYTHONPATH",
    "UPDATE_SLANGPY",
    "VCPKG_DEFAULT_BINARY_CACHE",
    "VCPKG_DOWNLOADS",
    "VULKAN_SDK",
)

SENSITIVE_ENV_FRAGMENTS = (
    "AUTH",
    "COOKIE",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


def format_env_value(key: str, value: str) -> str:
    if any(fragment in key.upper() for fragment in SENSITIVE_ENV_FRAGMENTS):
        return "<redacted>"
    return value


def parse_bool(value: Union[bool, str], env_var: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in ("1", "true", "on", "yes"):
        return True
    if normalized in ("0", "false", "off", "no"):
        return False
    raise ValueError(
        f"Invalid value for {env_var}: {value!r}. "
        "Expected one of: 1, 0, true, false, on, off, yes, no"
    )


def get_os():
    """
    Return the OS name (windows, linux, macos).
    """
    platform = sys.platform
    if platform == "win32":
        return "windows"
    elif platform == "linux" or platform == "linux2":
        return "linux"
    elif platform == "darwin":
        return "macos"
    else:
        raise NameError(f"Unsupported OS: {sys.platform}")


def get_platform():
    """
    Return the platform name (x86_64, aarch64).
    """
    machine = platform.machine()
    if machine == "x86_64" or machine == "AMD64":
        return "x86_64"
    elif machine == "aarch64" or machine == "arm64":
        return "aarch64"
    else:
        raise NameError(f"Unsupported platform: {machine}")


def get_default_compiler():
    """
    Return the default compiler name for the current OS (msvc, gcc, clang).
    """
    if get_os() == "windows":
        return "msvc"
    elif get_os() == "linux":
        return "gcc"
    elif get_os() == "macos":
        return "clang"
    else:
        raise NameError(f"Unsupported OS: {get_os()}")


def run_command(
    command: Command,
    shell: bool = True,
    env: Optional[dict[str, str]] = None,
    fix_paths: bool = True,
    system_telemetry_path: Path | None = None,
) -> str:
    if fix_paths and get_os() == "windows":
        if isinstance(command, str):
            command = command.replace("/", "\\")
        else:
            command = [argument.replace("/", "\\") for argument in command]
    display_command = (
        command if isinstance(command, str) else subprocess.list2cmdline(list(command))
    )
    if env != None:
        new_env = os.environ.copy()
        new_env.update(env)
        env = new_env
    print(f'Running "{display_command}" ...')
    sys.stdout.flush()

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        universal_newlines=True,
        shell=shell,
        env=env,
    )
    assert process.stdout is not None
    system_sampler: SystemTelemetrySampler | None = None
    if system_telemetry_path is not None:
        try:
            system_sampler = SystemTelemetrySampler(process.pid)
            system_sampler.start()
        except Exception as exc:
            print(f"WARNING: Cannot start system telemetry: {exc}")
            system_sampler = None

    out = ""
    try:
        while True:
            nextline = process.stdout.readline()
            if nextline == "" and process.poll() is not None:
                break
            sys.stdout.write(nextline)
            sys.stdout.flush()
            out += nextline

        process.communicate()
    finally:
        if system_sampler is not None:
            try:
                system_sampler.stop()
                assert system_telemetry_path is not None
                system_sampler.write(system_telemetry_path)
            except Exception as exc:
                print(f"WARNING: Cannot write system telemetry: {exc}")
    if process.returncode != 0:
        raise CommandError(display_command, process.returncode, out)

    return out


def info(args: argparse.Namespace):
    print("CI configuration:")
    for key, value in sorted(args._get_kwargs(), key=lambda x: x[0]):
        print(f"  {key}: {value}")
    print("")

    print("Selected environment variables:")
    for key in INFO_ENV_VARS:
        if key not in os.environ:
            continue
        value = format_env_value(key, os.environ[key])
        try:
            print(f"  {key}: {value}")
        except UnicodeEncodeError:
            safe_value = value.encode("ascii", errors="replace").decode("ascii")
            print(f"  {key}: {safe_value}")
    print("  (full environment omitted)")
    print("")


def setup(args: Any):
    try:
        run_command("pip uninstall falcor2 -y")
    except Exception:
        pass
    try:
        run_command("pip uninstall slangpy -y")
    except Exception:
        pass

    os.environ["NO_CMAKE_BUILD"] = "1"

    run_command("pip install -r requirements-dev.txt")
    run_command(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128",
        fix_paths=False,
    )
    run_command("pip install --editable ./external/slangpy")
    run_command("pip install --editable .")
    run_command("python tools/install_slangpy_torch.py")

    # Setup local slang, if requested.
    bts.clear_slang()
    if args.use_custom_slang:
        print("-----------------------------------------------------")
        print(
            f"Preparing custom slang from {args.slang_repository} (branch: {args.slang_branch})...\n"
        )
        bts.clone_slang(args.slang_repository, args.slang_branch)
        bts.configure_slang()
        bts.build_slang(args.slang_config)
        print(
            f"Preparing custom slang from {args.slang_repository} (branch: {args.slang_branch})... DONE\n"
        )
        print("-----------------------------------------------------")


def configure(args: Any):
    cmake_extra_args: list[str] = []

    if args.use_custom_slang:
        cmake_extra_args += [
            "-DSGL_LOCAL_SLANG=ON",
            f"-DSGL_LOCAL_SLANG_DIR={bts.CUSTOM_SLANG_SOURCE_DIR}",
            f"-DSGL_LOCAL_SLANG_BUILD_DIR=build/{args.slang_config}",
        ]

    if "crashpad" in args.flags:
        cmake_extra_args += [
            "-DFALCOR_ENABLE_CRASHPAD=ON",
        ]

    bt.configure(args.preset, cmake_extra_args)


def build(args: Any):
    bt.build(args.preset, args.config)


def cpp_test_command(args: Any) -> list[str]:
    command = [
        f"{args.bin_dir}/falcor2_tests",
        "-r=console,junit",
        "-module-cache" if args.module_cache else "-no-module-cache",
        "-shader-cache" if args.shader_cache else "-no-shader-cache",
    ]
    if args.module_and_shader_cache_dir is not None:
        command.append(f"-module-and-shader-cache-dir={args.module_and_shader_cache_dir}")
    return command


def python_test_command(args: Any) -> list[str]:
    workers = int(getattr(args, "pytest_workers", 4))
    command = [
        "pytest",
        "./tests/python",
        "-vra",
        "-n",
        str(workers),
        f"--maxprocesses={workers}",
        "--junit-xml=reports/pytest-junit.xml",
        "--telemetry",
        "--telemetry-output=reports/test-telemetry.json",
        "--slow",
        "--device-cache-policy",
        args.device_cache_policy,
        "--module-cache" if args.module_cache else "--no-module-cache",
        "--shader-cache" if args.shader_cache else "--no-shader-cache",
    ]
    if args.module_and_shader_cache_dir is not None:
        command.append(f"--module-and-shader-cache-dir={args.module_and_shader_cache_dir}")
    return command


def unit_test_cpp(args: Any):
    if "crashpad" in args.flags:
        crashpad.setup("native")

    error: Optional[CommandError] = None
    try:
        out = run_command(cpp_test_command(args), shell=False)
    except CommandError as exc:
        out = exc.output
        error = exc
    finally:
        if "crashpad" in args.flags:
            crashpad.report("native")

    # doctest outputs both regular output and junit xml report on stdout
    # filter out regular output and write remaining to junit xml file
    report = "\n".join(filter(lambda line: line.strip().startswith("<"), out.splitlines()))
    os.makedirs("reports", exist_ok=True)
    with open("reports/doctest-junit.xml", "w") as f:
        f.write(report)

    if error is not None:
        raise error


def update_slangpy(args: Any):
    print("Updating slangpy submodule to top-of-tree ...")
    run_command("git -C external/slangpy fetch origin main", fix_paths=False)
    run_command("git -C external/slangpy checkout --detach FETCH_HEAD", fix_paths=False)
    run_command("git -C external/slangpy submodule update --init --recursive", fix_paths=False)
    print("SlangPy updated to:")
    run_command("git -C external/slangpy log -1 --oneline", fix_paths=False)


def typing_check_python(args: Any):
    run_command(f"pyright")


def unit_test_python(args: Any):
    os.makedirs("reports", exist_ok=True)
    DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH.unlink(missing_ok=True)
    Path("reports/test-telemetry.json").unlink(missing_ok=True)
    if "crashpad" in args.flags:
        crashpad.setup("python")

    report_enabled = args.config.lower() == "release"
    report_root = Path("reports/image-tests")
    test_env: dict[str, str] = {}
    if "crashpad" in args.flags:
        test_env["FALCOR_CRASHPAD_DEFER_REPORT"] = "1"
    if report_enabled:
        if report_root.exists():
            shutil.rmtree(report_root)
        raw_report_dir = report_root / "raw"
        raw_report_dir.mkdir(parents=True)
        test_env["FALCOR_IMAGE_TEST_REPORT_DIR"] = str(raw_report_dir.resolve())

    error: Optional[CommandError] = None
    try:
        run_command(
            python_test_command(args),
            shell=False,
            env=test_env or None,
            system_telemetry_path=DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH,
        )
    except CommandError as exc:
        error = exc
    finally:
        if "crashpad" in args.flags:
            crashpad.report("python")

    report_error: Optional[CommandError] = None
    if report_enabled:
        report_command = [
            "python",
            "tools/image_test_report.py",
            "--input",
            str(report_root / "raw"),
            "--output",
            str(report_root / "site"),
            "--test-exit-code",
            str(error.returncode if error is not None else 0),
        ]
        if os.environ.get("CI_JOB_ID"):
            report_command.append("--upload")
            if not args.force_image_test_report:
                report_command.append("--skip-if-no-failures")
        try:
            run_command(report_command, shell=False)
        except CommandError as exc:
            report_error = exc
            print(f"Image test report generation failed: {exc}")

    if error is not None:
        raise error
    if report_error is not None:
        raise report_error


# def coverage_report(args: Any):
#    if not "coverage" in args.flags:
#        print("Coverage flag not set, skipping coverage report.")
#    os.makedirs("reports", exist_ok=True)
#    run_command(f"gcovr -r . -f src/sgl --html reports/coverage.html")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", type=str, action="store", help="OS (windows, linux, macos)")
    parser.add_argument("--platform", type=str, action="store", help="Platform (x86_64, aarch64)")
    parser.add_argument("--compiler", type=str, action="store", help="Compiler (msvc, gcc, clang)")
    parser.add_argument("--config", type=str, action="store", help="Config (Release, Debug)")
    parser.add_argument("--python", type=str, action="store", help="Python version")
    parser.add_argument("--flags", type=str, action="store", help="Additional flags")
    parser.add_argument(
        "--module-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the persistent module cache for unit tests",
    )
    parser.add_argument(
        "--shader-cache",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable or disable the persistent shader cache for unit tests",
    )
    parser.add_argument(
        "--module-and-shader-cache-dir",
        type=str,
        help="Root directory for persistent unit-test caches",
    )
    parser.add_argument(
        "--device-cache-policy",
        choices=("session", "file", "test"),
        help="Recycle Python test devices at session, file, or test boundaries",
    )
    parser.add_argument(
        "--pytest-workers",
        type=int,
        help="Number of parallel pytest workers",
    )
    parser.add_argument(
        "--use-custom-slang",
        action="store_true",
        default=None,
        help="Use custom slang. If not set, use slangpy's slang. If set, defaults to Slang's top of tree.",
    )
    parser.add_argument(
        "--slang-repository",
        type=str,
        action="store",
        help="Custom slang repository, only valid when --use-custom-slang is set.",
    )
    parser.add_argument(
        "--slang-branch",
        type=str,
        action="store",
        help="Custom slang branch, only valid when --use-custom-slang is set.",
    )
    parser.add_argument(
        "--slang-config",
        type=str,
        action="store",
        help="Custom slang config, only valid when --use-custom-slang is set.",
    )
    parser.add_argument(
        "--force-image-test-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate and upload the image test report even when all image tests pass",
    )

    commands = parser.add_subparsers(dest="command", required=True, help="sub-command help")

    parser_info = commands.add_parser("info", help="print info about the current environment")

    parser_setup = commands.add_parser("setup", help="run setup.bat or setup.sh")

    parser_configure = commands.add_parser("configure", help="run cmake configure")

    parser_build = commands.add_parser("build", help="run cmake build")

    parser_test_cpp = commands.add_parser("unit-test-cpp", help="run unit tests (c++)")

    parser_update_slangpy = commands.add_parser(
        "update-slangpy", help="update slangpy submodule to top-of-tree"
    )

    parser_typing_check_python = commands.add_parser(
        "typing-check-python", help="run pyright typing checks (python)"
    )

    parser_test_python = commands.add_parser("unit-test-python", help="run unit tests (python)")

    # parser_coverage_report = commands.add_parser("coverage-report", help="generate coverage report")

    return parser


def main(argv: Optional[Sequence[str]] = None):
    parser = create_parser()
    args = parser.parse_args(argv)

    print("-----------------------------------------------------")
    print(args.command)

    args = vars(args)

    VARS = [
        ("os", "CI_OS", get_os()),
        ("platform", "CI_PLATFORM", get_platform()),
        ("compiler", "CI_COMPILER", get_default_compiler()),
        ("config", "CI_CONFIG", "Debug"),
        ("python", "CI_PYTHON", "3.9"),
        ("flags", "CI_FLAGS", ""),
        ("module_cache", "CI_MODULE_CACHE", False),
        ("shader_cache", "CI_SHADER_CACHE", True),
        ("module_and_shader_cache_dir", "CI_MODULE_AND_SHADER_CACHE_DIR", None),
        ("device_cache_policy", "CI_DEVICE_CACHE_POLICY", "file"),
        ("pytest_workers", "CI_PYTEST_WORKERS", 4),
        ("use_custom_slang", "CI_USE_CUSTOM_SLANG", False),
        ("slang_repository", "CI_SLANG_REPOSITORY", "https://github.com/shader-slang/slang.git"),
        ("slang_branch", "CI_SLANG_BRANCH", "master"),
        ("slang_config", "CI_SLANG_CONFIG", "Release"),
        ("force_image_test_report", "CI_FORCE_IMAGE_TEST_REPORT", False),
    ]

    for var, env_var, default_value in VARS:
        if not var in args or args[var] == None:
            args[var] = os.environ[env_var] if env_var in os.environ else default_value

    # Convert boolean environment variable values to bool.
    args["use_custom_slang"] = parse_bool(args["use_custom_slang"], "CI_USE_CUSTOM_SLANG")
    args["module_cache"] = parse_bool(args["module_cache"], "CI_MODULE_CACHE")
    args["shader_cache"] = parse_bool(args["shader_cache"], "CI_SHADER_CACHE")
    args["force_image_test_report"] = parse_bool(
        args["force_image_test_report"], "CI_FORCE_IMAGE_TEST_REPORT"
    )
    cache_dir = args["module_and_shader_cache_dir"]
    if isinstance(cache_dir, str) and not cache_dir.strip():
        args["module_and_shader_cache_dir"] = None
    if args["device_cache_policy"] not in ("session", "file", "test"):
        raise ValueError(
            f"Invalid CI_DEVICE_CACHE_POLICY: {args['device_cache_policy']!r}. "
            "Expected one of: session, file, test"
        )
    args["pytest_workers"] = int(args["pytest_workers"])
    if args["pytest_workers"] < 1:
        raise ValueError("CI_PYTEST_WORKERS must be at least 1")

    # Split flags.
    args["flags"] = args["flags"].split(",") if args["flags"] != "" else []

    # Determine cmake executable path.
    args["cmake"] = {
        "windows": "cmake.exe",
        "linux": "cmake",
        "macos": "cmake",
    }[args["os"]]

    # Determine cmake preset.
    preset = args["os"] + "-" + args["compiler"]
    if args["os"] == "macos":
        if args["platform"] == "x86_64":
            preset = preset.replace("macos", "macos-x64")
        elif args["platform"] == "aarch64":
            preset = preset.replace("macos", "macos-arm64")
    args["preset"] = preset

    # Determine binary directory.
    bin_dir = f"./build/{args['preset']}/{args['config']}"
    args["bin_dir"] = bin_dir

    args = argparse.Namespace(**args)

    {
        "info": info,
        "setup": setup,
        "configure": configure,
        "build": build,
        "unit-test-cpp": unit_test_cpp,
        "update-slangpy": update_slangpy,
        "typing-check-python": typing_check_python,
        "unit-test-python": unit_test_python,
        # "coverage-report": coverage_report,
    }[args.command](args)

    print("-----------------------------------------------------\n")

    return 0


if __name__ == "__main__":
    main()
