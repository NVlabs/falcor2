# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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

    assert command[1] == "./tests/python"
    assert command[command.index("-n") + 1] == "3"
    assert "--maxprocesses=3" in command
    assert "--junit-xml=reports/pytest-junit.xml" in command
    assert "--telemetry" in command
    assert "--telemetry-output=reports/test-telemetry.json" in command


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
