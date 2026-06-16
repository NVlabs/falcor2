# SPDX-License-Identifier: Apache-2.0

"""
Script for running CI tasks.
"""

import os
import sys
import platform
import argparse
import subprocess
from typing import Any, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tools.build as bt
import tools.build_slang as bts
from tools import crashpad


class CommandError(RuntimeError):
    def __init__(self, command: str, returncode: int, output: str):
        super().__init__(f'Error running "{command}"')
        self.command = command
        self.returncode = returncode
        self.output = output


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
    command: str, shell: bool = True, env: Optional[dict[str, str]] = None, fix_paths: bool = True
) -> str:
    if fix_paths and get_os() == "windows":
        command = command.replace("/", "\\")
    if env != None:
        new_env = os.environ.copy()
        new_env.update(env)
        env = new_env
    print(f'Running "{command}" ...')
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

    out = ""
    while True:
        nextline = process.stdout.readline()
        if nextline == "" and process.poll() is not None:
            break
        sys.stdout.write(nextline)
        sys.stdout.flush()
        out += nextline

    process.communicate()
    if process.returncode != 0:
        raise CommandError(command, process.returncode, out)

    return out


def info(args: argparse.Namespace):
    print("CI configuration:")
    for key, value in sorted(args._get_kwargs(), key=lambda x: x[0]):
        print(f"  {key}: {value}")
    print("")

    print("Environment variables:")
    for key, value in sorted(os.environ.items(), key=lambda x: x[0]):
        try:
            print(f"  {key}: {value}")
        except UnicodeEncodeError:
            safe_value = value.encode("ascii", errors="replace").decode("ascii")
            print(f"  {key}: {safe_value}")
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


def unit_test_cpp(args: Any):
    if "crashpad" in args.flags:
        crashpad.setup("native")

    error: Optional[CommandError] = None
    try:
        out = run_command(f"{args.bin_dir}/falcor2_tests -r=console,junit")
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
    run_command("git -C external/slangpy fetch origin", fix_paths=False)
    run_command("git -C external/slangpy checkout origin/main", fix_paths=False)
    run_command("git -C external/slangpy submodule update --init --recursive", fix_paths=False)
    print("SlangPy updated to:")
    run_command("git -C external/slangpy log -1 --oneline", fix_paths=False)


def typing_check_python(args: Any):
    run_command(f"pyright")


def unit_test_python(args: Any):
    os.makedirs("reports", exist_ok=True)
    if "crashpad" in args.flags:
        crashpad.setup("python")

    extra_args = "--slow"
    if args.config.lower() == "release":
        extra_args += " --image-tests"

    error: Optional[CommandError] = None
    try:
        run_command(
            f"pytest ./tests/python -vra -n 1 --maxprocesses=1 {extra_args} --junit-xml=reports/pytest-junit.xml",
            env={"FALCOR_CRASHPAD_DEFER_REPORT": "1"} if "crashpad" in args.flags else None,
        )
    except CommandError as exc:
        error = exc
    finally:
        if "crashpad" in args.flags:
            crashpad.report("python")

    if error is not None:
        raise error


# def coverage_report(args: Any):
#    if not "coverage" in args.flags:
#        print("Coverage flag not set, skipping coverage report.")
#    os.makedirs("reports", exist_ok=True)
#    run_command(f"gcovr -r . -f src/sgl --html reports/coverage.html")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--os", type=str, action="store", help="OS (windows, linux, macos)")
    parser.add_argument("--platform", type=str, action="store", help="Platform (x86_64, aarch64)")
    parser.add_argument("--compiler", type=str, action="store", help="Compiler (msvc, gcc, clang)")
    parser.add_argument("--config", type=str, action="store", help="Config (Release, Debug)")
    parser.add_argument("--python", type=str, action="store", help="Python version")
    parser.add_argument("--flags", type=str, action="store", help="Additional flags")
    parser.add_argument(
        "--use-custom-slang",
        type=bool,
        action="store",
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

    args = parser.parse_args()

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
        ("use_custom_slang", "CI_USE_CUSTOM_SLANG", False),
        ("slang_repository", "CI_SLANG_REPOSITORY", "https://github.com/shader-slang/slang.git"),
        ("slang_branch", "CI_SLANG_BRANCH", "master"),
        ("slang_config", "CI_SLANG_CONFIG", "Release"),
    ]

    for var, env_var, default_value in VARS:
        if not var in args or args[var] == None:
            args[var] = os.environ[env_var] if env_var in os.environ else default_value

    # Convert to proper type (any value set in CI_USE_CUSTOM_SLANG must become True)
    args["use_custom_slang"] = bool(args["use_custom_slang"])

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
