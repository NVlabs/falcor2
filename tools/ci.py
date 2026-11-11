# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Script for running CI tasks.
"""

import os
import sys
import platform
import argparse
import json
import queue
import re
from pathlib import Path
import shutil
import subprocess
import signal
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Any, Optional, TextIO, Union

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import tools.build as bt
import tools.build_slang as bts
from tools import crashpad
from tools.process_tree import ProcessTree
from tools.system_telemetry import DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH
from tools.system_telemetry import SystemTelemetrySampler

Command = Union[str, Sequence[str]]
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
PIP_WHEEL_ROOT = REPOSITORY_ROOT / ".tmp" / "pip-wheels"
_OUTPUT_LOCK = threading.Lock()
_COMBINED_TELEMETRY_PATH = Path("reports/system-telemetry-combined.json")


class CommandError(RuntimeError):
    def __init__(self, command: str, returncode: int, output: str):
        super().__init__(f'Error running "{command}" (exit code {returncode})')
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
    "CI_NIGHTLY",
    "CI_OS",
    "CI_PIPELINE_ID",
    "CI_PLATFORM",
    "CI_PARALLEL_TEST_PHASES",
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
    "CI_INSTALL_OPENH264",
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


def _terminate_process_tree(process: subprocess.Popen[str]) -> None:
    try:
        import psutil
    except ImportError:
        # Setup itself can be interrupted before it has installed psutil.
        process.kill()
        process.wait()
        return

    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
    except psutil.NoSuchProcess:
        process.wait()
        return
    processes = [*reversed(descendants), root]
    for child in processes:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for child in alive:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(alive, timeout=5)
    process.wait(timeout=5)
    if alive:
        raise RuntimeError(f"Test processes did not exit: {[child.pid for child in alive]}")


class _TestProcesses:
    """Prevent new commands from starting while an interrupted test run is cleaned up."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._processes: dict[subprocess.Popen[str], ProcessTree] = {}
        self._cancelled = False

    def start(self, command: Command, **kwargs: Any) -> subprocess.Popen[str]:
        with self._lock:
            if self._cancelled:
                raise RuntimeError("Test run was cancelled")
            tree = ProcessTree(command, **kwargs)
            self._processes[tree.process] = tree
            return tree.process

    def finish(self, process: subprocess.Popen[str]) -> None:
        with self._lock:
            tree = self._processes.pop(process, None)
            if tree is not None:
                # Give Crashpad and other helpers a bounded opportunity to finish naturally.
                tree.close(grace=1)

    def cancel(self) -> None:
        with self._lock:
            self._cancelled = True
            errors: list[Exception] = []
            for tree in self._processes.values():
                try:
                    tree.close()
                except Exception as exc:
                    errors.append(exc)
            self._processes.clear()
            if errors:
                raise RuntimeError(f"Cannot clean up test processes: {errors}")


def _write_command_output(text: str, prefix: str, log: TextIO | None) -> None:
    with _OUTPUT_LOCK:
        if log is not None:
            log.write(text)
            log.flush()
        sys.stdout.write(f"[{prefix}] {text}" if prefix else text)
        sys.stdout.flush()


def _read_managed_output(
    process: subprocess.Popen[str],
    process_group: _TestProcesses,
    prefix: str,
    log: TextIO | None,
) -> str:
    """Drain output while checking parent exit independently of inherited pipe handles."""
    assert process.stdout is not None
    events: queue.Queue[str | BaseException | None] = queue.Queue()

    def read() -> None:
        try:
            assert process.stdout is not None
            for line in iter(process.stdout.readline, ""):
                events.put(line)
        except BaseException as exc:
            events.put(exc)
        finally:
            events.put(None)

    reader = threading.Thread(target=read, name="ci-test-output", daemon=True)
    reader.start()
    output: list[str] = []
    eof = False
    finished = False
    try:
        while not eof or process.poll() is None:
            if not finished and process.poll() is not None:
                process_group.finish(process)
                finished = True
            try:
                event = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if isinstance(event, BaseException):
                raise event
            if event is None:
                eof = True
            else:
                _write_command_output(event, prefix, log)
                output.append(event)
        if not finished:
            process_group.finish(process)
        return "".join(output)
    except BaseException:
        process_group.cancel()
        raise
    finally:
        reader.join(timeout=5)
        if reader.is_alive():
            raise RuntimeError("Test output pipe remained open after process-tree cleanup")
        process.stdout.close()


def run_command(
    command: Command,
    shell: bool = True,
    env: Optional[dict[str, str]] = None,
    fix_paths: bool = True,
    system_telemetry_path: Path | None = None,
    output_prefix: str = "",
    log: TextIO | None = None,
    process_group: _TestProcesses | None = None,
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
    _write_command_output(f'Running "{display_command}" ...\n', output_prefix, log)

    start = process_group.start if process_group is not None else subprocess.Popen
    try:
        process = start(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            universal_newlines=True,
            shell=shell,
            env=env,
        )
    except Exception:
        if process_group is not None:
            process_group.cancel()
        raise
    assert process.stdout is not None
    system_sampler: SystemTelemetrySampler | None = None
    if system_telemetry_path is not None:
        try:
            system_sampler = SystemTelemetrySampler(process.pid)
            system_sampler.start()
        except Exception as exc:
            _write_command_output(
                f"WARNING: Cannot start system telemetry: {exc}\n", output_prefix, log
            )
            system_sampler = None

    out = ""
    try:
        if process_group is not None:
            out = _read_managed_output(process, process_group, output_prefix, log)
        else:
            while True:
                nextline = process.stdout.readline()
                if nextline == "" and process.poll() is not None:
                    break
                _write_command_output(nextline, output_prefix, log)
                out += nextline
            process.communicate()
    except BaseException:
        if process_group is not None:
            process_group.cancel()
        else:
            _terminate_process_tree(process)
        raise
    finally:
        if process_group is not None:
            process_group.finish(process)
        if system_sampler is not None:
            try:
                system_sampler.stop()
                assert system_telemetry_path is not None
                system_sampler.write(system_telemetry_path)
            except Exception as exc:
                _write_command_output(
                    f"WARNING: Cannot write system telemetry: {exc}\n", output_prefix, log
                )
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

    elif args.slang_version:
        # Only Git metadata is needed: CMake downloads the published compiler package.
        bts.clone_slang(args.slang_repository, f"v{args.slang_version}", checkout=False)
    _write_benchmark_provenance(args)


def _write_benchmark_provenance(args: Any) -> None:
    """Record selected revisions during preparation, before measurement loses networking."""

    def git(root: Path, *arguments: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()

    path = REPOSITORY_ROOT / "reports/benchmark-provenance.json"
    path.unlink(missing_ok=True)
    harness = os.environ.get("CI_PERF_HARNESS_COMMIT")
    if not (harness or args.use_custom_slang or args.slang_version):
        return
    falcor = git(REPOSITORY_ROOT, "rev-parse", "HEAD")
    compiler = None
    if args.use_custom_slang or args.slang_version:
        compiler = {
            "id": git(bts.CUSTOM_SLANG_SOURCE_DIR, "rev-parse", "FETCH_HEAD^{commit}"),
            "time": git(
                bts.CUSTOM_SLANG_SOURCE_DIR, "show", "-s", "--format=%cI", "FETCH_HEAD^{commit}"
            ),
            "branch": (
                "master"
                if args.slang_version or args.slang_branch == "master"
                else "(detached head)"
            ),
            "falcor_commit": falcor,
        }
        if (
            os.environ.get("CI_PERF_PROJECT") == "falcor2-slang"
            and args.use_custom_slang
            and not args.slang_version
            and re.fullmatch(r"[0-9a-fA-F]{40}", args.slang_branch)
        ):
            # Refresh ancestry without replacing the measured revision in FETCH_HEAD.
            git(
                bts.CUSTOM_SLANG_SOURCE_DIR,
                "fetch",
                "--no-write-fetch-head",
                "--no-tags",
                "--no-recurse-submodules",
                "origin",
                "+refs/heads/master:refs/remotes/origin/master",
            )
            if git(
                bts.CUSTOM_SLANG_SOURCE_DIR,
                "for-each-ref",
                "--format=%(refname:strip=3)",
                f"--contains={compiler['id']}",
                "refs/remotes/origin/master",
            ):
                compiler["branch"] = "master"
        if args.slang_version:
            compiler["release"] = {"label": f"v{args.slang_version}"}
    path.parent.mkdir(exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "falcor_commit": falcor,
                "harness_commit": harness or falcor,
                "compiler": compiler,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def configure(args: Any):
    cmake_extra_args: list[str] = []

    if args.use_custom_slang:
        cmake_extra_args += [
            "-DSGL_LOCAL_SLANG=ON",
            f"-DSGL_LOCAL_SLANG_DIR={bts.CUSTOM_SLANG_SOURCE_DIR}",
            f"-DSGL_LOCAL_SLANG_BUILD_DIR=build/{args.slang_config}",
        ]
    else:
        cmake_extra_args.append("-DSGL_LOCAL_SLANG=OFF")
        if args.slang_version:
            cmake_extra_args.append(f"-DSGL_SLANG_VERSION={args.slang_version}")
        else:
            cmake_extra_args.append("-USGL_SLANG_VERSION")

    if "crashpad" in args.flags:
        cmake_extra_args += [
            "-DFALCOR_ENABLE_CRASHPAD=ON",
        ]

    bt.configure(args.preset, cmake_extra_args)


def build(args: Any):
    bt.build(args.preset, args.config)


def build_hydra_wheel(args: Any) -> None:
    from tools.package_hydra import build_hydra_wheel as package

    package(args.version_suffix, args.manifest)


def build_wheels(args: Any) -> None:
    wheel_root = PIP_WHEEL_ROOT.resolve()
    if wheel_root.exists():
        shutil.rmtree(wheel_root)

    for project_root in (REPOSITORY_ROOT, REPOSITORY_ROOT / "external" / "slangpy"):
        for pattern in ("lib.*", "bdist.*"):
            for staging_dir in (project_root / "build").glob(pattern):
                shutil.rmtree(staging_dir)

    wheelhouse = wheel_root / "wheelhouse"
    build_dependency = wheel_root / "build-dependency"
    wheelhouse.mkdir(parents=True)

    build_command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(wheelhouse),
    ]
    build_env: dict[str, str] = {}
    if os.name == "nt":
        build_env = {key.upper(): value for key, value in bt.get_build_env().items()}
        build_env["DISTUTILS_USE_SDK"] = "1"
        scripts_dir = Path(sys.executable).parent / "Scripts"
        build_env["PATH"] = os.pathsep.join((str(scripts_dir), build_env["PATH"]))
    run_command(
        [*build_command, str(REPOSITORY_ROOT / "external" / "slangpy")],
        shell=False,
        fix_paths=False,
        env={
            **build_env,
            "BUILD_RELEASE_WHEEL": "1",
            "SLANGPY_VERSION_SUFFIX": args.version_suffix,
        },
    )

    slangpy_wheels = list(wheelhouse.glob("slangpy-*.whl"))
    if len(slangpy_wheels) != 1:
        raise RuntimeError(f"Expected one SlangPy wheel, found {len(slangpy_wheels)}")
    run_command(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(build_dependency),
            str(slangpy_wheels[0]),
        ],
        shell=False,
        fix_paths=False,
    )

    python_path = str(build_dependency)
    if inherited_python_path := os.environ.get("PYTHONPATH"):
        python_path = os.pathsep.join((python_path, inherited_python_path))
    run_command(
        [*build_command, str(REPOSITORY_ROOT)],
        shell=False,
        fix_paths=False,
        env={
            **build_env,
            "BUILD_RELEASE_WHEEL": "1",
            "FALCOR2_VERSION_SUFFIX": args.version_suffix,
            "PYTHONPATH": python_path,
        },
    )

    falcor2_wheels = list(wheelhouse.glob("falcor2-*.whl"))
    if len(falcor2_wheels) != 1:
        raise RuntimeError(f"Expected one Falcor2 wheel, found {len(falcor2_wheels)}")


def cpp_test_command(args: Any) -> list[str]:
    command = [
        f"{args.bin_dir}/falcor2_tests",
        "-r=console,junit",
        "-module-cache" if args.module_cache else "-no-module-cache",
        "-shader-cache" if args.shader_cache else "-no-shader-cache",
    ]
    if args.module_and_shader_cache_dir is not None:
        command.append(f"-module-and-shader-cache-dir={args.module_and_shader_cache_dir}")
    if bool(getattr(args, "nightly", False)):
        command.append("-nightly")
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
        "--dist=worksteal",
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
    if bool(getattr(args, "nightly", False)):
        command.append("--nightly")
    return command


def benchmark_command(args: Any) -> list[str]:
    """Run the shared suite without inheriting unit-test cache or worker settings."""
    pytest_args = args.pytest_args
    if pytest_args[:1] == ["--"]:
        pytest_args = pytest_args[1:]
    return [
        sys.executable,
        "-m",
        "pytest",
        "tests/python/benchmarks",
        *pytest_args,
        "--all",
        "--benchmark-only",
        "-n0",
        "--no-module-cache",
        "--no-shader-cache",
    ]


def benchmark(args: Any) -> None:
    run_command(
        benchmark_command(args),
        shell=False,
        fix_paths=False,
        env={
            "PYTEST_ADDOPTS": "",
            "PYTHONPATH": os.pathsep.join(
                [str(REPOSITORY_ROOT), str(REPOSITORY_ROOT / "external/slangpy")]
            ),
            "CUDA_CACHE_DISABLE": "1",
            "OPTIX_CACHE_MAXSIZE": "0",
            "__GL_SHADER_DISK_CACHE": "0",
        },
    )


def unit_test_cpp(
    args: Any,
    *,
    command_runner: Callable[..., str] | None = None,
    report_crashes: bool = True,
) -> None:
    run = command_runner or run_command
    if "crashpad" in args.flags:
        crashpad.setup("native")

    error: Optional[CommandError] = None
    try:
        out = run(cpp_test_command(args), shell=False)
    except CommandError as exc:
        out = exc.output
        error = exc
    finally:
        if report_crashes and "crashpad" in args.flags:
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


def unit_test_python(
    args: Any,
    *,
    command_runner: Callable[..., str] | None = None,
    report_crashes: bool = True,
) -> None:
    run = command_runner or run_command
    os.makedirs("reports", exist_ok=True)
    DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH.unlink(missing_ok=True)
    Path("reports/test-telemetry.json").unlink(missing_ok=True)
    # Require a working codec so CI exercises encoding instead of skipping it.
    if args.install_openh264:
        for command in ("install", "check"):
            run(
                [sys.executable, "-m", "falcor2.streaming.openh264", command],
                shell=False,
                fix_paths=False,
            )
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
        run(
            python_test_command(args),
            shell=False,
            env=test_env or None,
            system_telemetry_path=DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH,
        )
    except CommandError as exc:
        error = exc
    finally:
        if report_crashes and "crashpad" in args.flags:
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
            run(report_command, shell=False)
        except CommandError as exc:
            report_error = exc
            print(f"Image test report generation failed: {exc}")

    if error is not None:
        raise error
    if report_error is not None:
        raise report_error


def _run_test_phase(
    kind: str,
    phase: Callable[..., None],
    args: Any,
    processes: _TestProcesses,
) -> None:
    with Path(f"reports/{kind}-tests.log").open("w", encoding="utf-8") as log:
        phase(
            args,
            command_runner=partial(
                run_command, output_prefix=kind, log=log, process_group=processes
            ),
            report_crashes=False,
        )


def _interrupt_tests(signum: int, frame: Any) -> None:
    raise KeyboardInterrupt(f"Test run interrupted by signal {signum}")


def unit_test(args: Any) -> None:
    """Run both test phases, retaining diagnostics even when either phase fails."""
    Path("reports").mkdir(exist_ok=True)
    _COMBINED_TELEMETRY_PATH.unlink(missing_ok=True)
    summary_path = Path("reports/test-phases.json")
    summary_path.unlink(missing_ok=True)
    for report in ("doctest-junit.xml", "pytest-junit.xml"):
        (Path("reports") / report).unlink(missing_ok=True)
    processes = _TestProcesses()
    phases = {"native": unit_test_cpp, "python": unit_test_python}
    errors: dict[str, Exception] = {}
    durations: dict[str, float] = {}
    started_at = time.monotonic()

    def run_phase(kind: str, phase: Callable[..., None]) -> None:
        started = time.monotonic()
        try:
            _run_test_phase(kind, phase, args, processes)
        finally:
            durations[kind] = time.monotonic() - started

    parallel = bool(getattr(args, "parallel_test_phases", False))
    print(f"Running native and Python test phases {'in parallel' if parallel else 'serially'}.")
    print(
        "Combined telemetry covers this driver and both phase process trees. "
        "system-telemetry.json process metrics remain scoped to pytest; "
        "host and GPU metrics are system-wide."
    )
    sampler: SystemTelemetrySampler | None = None
    try:
        sampler = SystemTelemetrySampler(os.getpid())
        sampler.start()
    except Exception as exc:
        print(f"WARNING: Cannot start combined system telemetry: {exc}")
        sampler = None

    previous_sigterm = signal.signal(signal.SIGTERM, _interrupt_tests)
    executor: ThreadPoolExecutor | None = None
    try:
        if parallel:
            executor = ThreadPoolExecutor(max_workers=2)
            futures = {
                executor.submit(run_phase, kind, phase): kind for kind, phase in phases.items()
            }
            for future in as_completed(futures):
                kind = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors[kind] = exc
        else:
            for kind, phase in phases.items():
                try:
                    run_phase(kind, phase)
                except Exception as exc:
                    errors[kind] = exc
    except BaseException:
        processes.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if sampler is not None:
            try:
                sampler.stop()
                sampler.write(_COMBINED_TELEMETRY_PATH)
            except Exception as exc:
                print(f"WARNING: Cannot write combined system telemetry: {exc}")
        # The shared rust-minidump installer must not be used by concurrent reporters.
        if "crashpad" in args.flags:
            for kind in phases:
                try:
                    crashpad.report(kind)
                except Exception as exc:
                    errors[f"{kind} crash report"] = exc
                    print(f"Crash report generation failed ({kind}): {exc}")

    elapsed = time.monotonic() - started_at
    summary = {
        "parallel": parallel,
        "duration_seconds": elapsed,
        "success": not errors,
        "phases": {
            kind: {
                "duration_seconds": durations[kind],
                "success": kind not in errors,
                "returncode": (
                    errors[kind].returncode
                    if isinstance(errors.get(kind), CommandError)
                    else (None if kind in errors else 0)
                ),
            }
            for kind in phases
        },
        "errors": {kind: str(exc) for kind, exc in errors.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for kind in phases:
        status = f"FAILED ({errors[kind]})" if kind in errors else "PASSED"
        print(f"{kind}: {status} in {durations[kind]:.2f}s; log: reports/{kind}-tests.log")
    print(f"Combined test phases: {elapsed:.2f}s; summary: {summary_path}")
    if errors:
        error_summary = "; ".join(f"{kind}: {exc}" for kind, exc in errors.items())
        raise RuntimeError(f"Unit test phases failed: {error_summary}")


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
        "--nightly",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Run the nightly test tier, including slow tests",
    )
    parser.add_argument(
        "--parallel-test-phases",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Overlap native and Python phases in unit-test (default: serial)",
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
        "--slang-version",
        help="Published Slang version (for example 2026.19); defaults to CI_SLANG_VERSION",
    )
    parser.add_argument(
        "--force-image-test-report",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate and upload the image test report even when all image tests pass",
    )
    parser.add_argument(
        "--install-openh264",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Install and check OpenH264 before Python tests (default: enabled)",
    )

    commands = parser.add_subparsers(dest="command", required=True, help="sub-command help")

    parser_info = commands.add_parser("info", help="print info about the current environment")

    parser_setup = commands.add_parser("setup", help="run setup.bat or setup.sh")

    parser_configure = commands.add_parser("configure", help="run cmake configure")

    parser_build = commands.add_parser("build", help="run cmake build")

    parser_build_wheels = commands.add_parser(
        "build-wheels", help="build one paired SlangPy and Falcor2 wheel"
    )
    parser_build_wheels.add_argument(
        "--version-suffix",
        required=True,
        help="PEP 440 suffix appended to each project's checked-in version",
    )

    parser_hydra = commands.add_parser(
        "build-hydra-wheel", help="build and qualify this platform's Hydra plugin wheel"
    )
    parser_hydra.add_argument("--version-suffix", required=True)
    parser_hydra.add_argument("--manifest", type=Path, required=True)

    parser_test_cpp = commands.add_parser("unit-test-cpp", help="run unit tests (c++)")
    commands.add_parser("unit-test", help="run native and Python unit tests")

    parser_update_slangpy = commands.add_parser(
        "update-slangpy", help="update slangpy submodule to top-of-tree"
    )

    parser_typing_check_python = commands.add_parser(
        "typing-check-python", help="run pyright typing checks (python)"
    )

    parser_test_python = commands.add_parser("unit-test-python", help="run unit tests (python)")

    parser_benchmark = commands.add_parser(
        "benchmark", help="run Python benchmarks in the current prepared checkout"
    )
    parser_benchmark.add_argument(
        "pytest_args",
        nargs=argparse.REMAINDER,
        help="pytest options after --; benchmarks always use one process and no application disk caches",
    )

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
        ("nightly", "CI_NIGHTLY", False),
        ("parallel_test_phases", "CI_PARALLEL_TEST_PHASES", False),
        ("use_custom_slang", "CI_USE_CUSTOM_SLANG", False),
        ("slang_repository", "CI_SLANG_REPOSITORY", "https://github.com/shader-slang/slang.git"),
        ("slang_branch", "CI_SLANG_BRANCH", "master"),
        ("slang_config", "CI_SLANG_CONFIG", "Release"),
        ("slang_version", "CI_SLANG_VERSION", None),
        ("force_image_test_report", "CI_FORCE_IMAGE_TEST_REPORT", False),
        ("install_openh264", "CI_INSTALL_OPENH264", True),
    ]

    for var, env_var, default_value in VARS:
        if not var in args or args[var] == None:
            args[var] = os.environ[env_var] if env_var in os.environ else default_value

    # Convert boolean environment variable values to bool.
    args["use_custom_slang"] = parse_bool(args["use_custom_slang"], "CI_USE_CUSTOM_SLANG")
    if args["slang_version"] and args["use_custom_slang"]:
        parser.error("--slang-version and --use-custom-slang are mutually exclusive")
    args["module_cache"] = parse_bool(args["module_cache"], "CI_MODULE_CACHE")
    args["shader_cache"] = parse_bool(args["shader_cache"], "CI_SHADER_CACHE")
    args["nightly"] = parse_bool(args["nightly"], "CI_NIGHTLY")
    args["parallel_test_phases"] = parse_bool(
        args["parallel_test_phases"], "CI_PARALLEL_TEST_PHASES"
    )
    args["force_image_test_report"] = parse_bool(
        args["force_image_test_report"], "CI_FORCE_IMAGE_TEST_REPORT"
    )
    args["install_openh264"] = parse_bool(args["install_openh264"], "CI_INSTALL_OPENH264")
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
        "build-wheels": build_wheels,
        "build-hydra-wheel": build_hydra_wheel,
        "unit-test-cpp": unit_test_cpp,
        "unit-test": unit_test,
        "update-slangpy": update_slangpy,
        "typing-check-python": typing_check_python,
        "unit-test-python": unit_test_python,
        "benchmark": benchmark,
        # "coverage-report": coverage_report,
    }[args.command](args)

    print("-----------------------------------------------------\n")

    return 0


if __name__ == "__main__":
    main()
