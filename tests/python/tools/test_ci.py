# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tools import ci


def _test_args(**overrides: Any) -> SimpleNamespace:
    values = {
        "bin_dir": "build/windows-msvc/Release",
        "module_cache": False,
        "shader_cache": True,
        "module_and_shader_cache_dir": None,
        "device_cache_policy": "file",
        "config": "Release",
        "pytest_workers": 2,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_device_cache_policy_is_python_only() -> None:
    args = _test_args()

    python_command = ci.python_test_command(args)
    policy_index = python_command.index("--device-cache-policy")

    assert python_command[policy_index + 1] == "file"
    assert "--device-cache-policy" not in ci.cpp_test_command(args)


def test_python_test_command_uses_configured_workers() -> None:
    command = ci.python_test_command(_test_args(pytest_workers=3))

    assert command[:4] == [sys.executable, "-m", "pytest", "./tests/python"]
    assert command[command.index("-n") + 1] == "3"
    assert "--maxprocesses=3" in command
    assert "--dist=worksteal" in command
    assert "--junit-xml=reports/pytest-junit.xml" in command
    assert "--telemetry" in command
    assert "--telemetry-output=reports/test-telemetry.json" in command


@pytest.mark.parametrize("config", ["Release", "Debug"])
@pytest.mark.parametrize("nightly", [False, True])
def test_python_test_command_preserves_nightly_and_image_flags(config: str, nightly: bool) -> None:
    command = ci.python_test_command(_test_args(config=config, nightly=nightly))

    assert ("--nightly" in command) == nightly
    expected_image_tests = False
    assert ("--image-tests" in command) == expected_image_tests


def test_run_command_continues_when_system_telemetry_cannot_start(
    monkeypatch: Any, tmp_path: Path
) -> None:
    class FakeProcess:
        pid = 42
        stdout = io.StringIO("completed\n")
        returncode = 0

        def poll(self) -> int:
            return self.returncode

        def communicate(self) -> tuple[None, None]:
            return None, None

    class FailingSampler:
        def __init__(self, root_pid: int):
            super().__init__()
            assert root_pid == 42

        def start(self) -> None:
            raise RuntimeError("cannot start sampler")

    monkeypatch.setattr(ci.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(ci, "SystemTelemetrySampler", FailingSampler)

    output = ci.run_command(
        ["test-command"],
        shell=False,
        fix_paths=False,
        system_telemetry_path=tmp_path / "system-telemetry.json",
    )

    assert output == "completed\n"


def test_benchmark_uses_active_python_and_preserves_pytest_arguments() -> None:
    args = ci.create_parser().parse_args(
        ["benchmark", "--", "-k", "cornell", "--benchmark-iterations=4"]
    )
    command = ci.benchmark_command(args)

    assert command[:4] == [sys.executable, "-m", "pytest", "tests/python/benchmarks"]
    assert command[4:7] == ["-k", "cornell", "--benchmark-iterations=4"]
    assert command[-5:] == [
        "--all",
        "--benchmark-only",
        "-n0",
        "--no-module-cache",
        "--no-shader-cache",
    ]


@pytest.mark.parametrize("version,custom", [(None, False), ("2026.19", False), (None, True)])
def test_configure_selects_default_release_or_source_slang(
    monkeypatch: pytest.MonkeyPatch, version: str | None, custom: bool
) -> None:
    calls = []
    monkeypatch.setattr(ci.bt, "configure", lambda preset, options: calls.append(options))
    ci.configure(
        SimpleNamespace(
            preset="linux-gcc",
            use_custom_slang=custom,
            slang_version=version,
            slang_config="Release",
            flags=[],
        )
    )
    options = calls[0]
    assert f"-DSGL_LOCAL_SLANG={'ON' if custom else 'OFF'}" in options
    if custom:
        assert f"-DSGL_LOCAL_SLANG_DIR={ci.bts.CUSTOM_SLANG_SOURCE_DIR}" in options
    elif version:
        assert f"-DSGL_SLANG_VERSION={version}" in options
    else:
        assert "-USGL_SLANG_VERSION" in options


def test_benchmark_propagates_failure_without_inheriting_pytest_addopts(
    monkeypatch: Any,
) -> None:
    failure = ci.CommandError("pytest", 1, "measurement failed")

    def fail(command: list[str], **kwargs: Any) -> str:
        assert kwargs["shell"] is False
        assert kwargs["fix_paths"] is False
        assert kwargs["env"]["PYTEST_ADDOPTS"] == ""
        assert kwargs["env"]["CUDA_CACHE_DISABLE"] == "1"
        assert kwargs["env"]["OPTIX_CACHE_MAXSIZE"] == "0"
        raise failure

    monkeypatch.setattr(ci, "run_command", fail)
    with pytest.raises(ci.CommandError, match="exit code 1"):
        ci.benchmark(ci.create_parser().parse_args(["benchmark"]))
