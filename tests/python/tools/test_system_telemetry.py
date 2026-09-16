# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tools import system_telemetry


class _FakeProcess:
    def __init__(
        self,
        pid: int,
        parent_pid: int,
        rss: int,
        user_seconds: float,
        system_seconds: float,
        thread_count: int,
        open_resource_count: int,
        children: list[_FakeProcess] | None = None,
    ):
        super().__init__()
        self.pid = pid
        self._parent_pid = parent_pid
        self.rss = rss
        self.user_seconds = user_seconds
        self.system_seconds = system_seconds
        self.thread_count = thread_count
        self.open_resource_count = open_resource_count
        self.children_list = children or []

    def children(self, recursive: bool) -> list[_FakeProcess]:
        assert recursive
        return self.children_list

    def create_time(self) -> float:
        return 1000.0 + self.pid

    def memory_info(self) -> SimpleNamespace:
        return SimpleNamespace(rss=self.rss)

    def cpu_times(self) -> SimpleNamespace:
        return SimpleNamespace(user=self.user_seconds, system=self.system_seconds)

    def num_threads(self) -> int:
        return self.thread_count

    def num_handles(self) -> int:
        return self.open_resource_count

    def num_fds(self) -> int:
        return self.open_resource_count


class _FakePsutil:
    class NoSuchProcess(Exception):
        pass

    class AccessDenied(Exception):
        pass

    def __init__(self, root: _FakeProcess):
        super().__init__()
        self._root = root
        self._cpu_percent = iter((10.0, 50.0))

    def cpu_percent(self, interval: None) -> float:
        assert interval is None
        return next(self._cpu_percent)

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(total=1000, available=300)

    def swap_memory(self) -> SimpleNamespace:
        return SimpleNamespace(used=50)

    def Process(self, pid: int) -> _FakeProcess:
        assert pid == self._root.pid
        return self._root


def test_sampler_captures_columnar_host_and_process_tree_metrics(
    monkeypatch: Any, tmp_path: Path
) -> None:
    child = _FakeProcess(11, 10, 200, 0.2, 0.03, 3, 20)
    root = _FakeProcess(10, 1, 100, 0.1, 0.02, 2, 10, [child])
    monkeypatch.setattr(system_telemetry, "psutil", _FakePsutil(root))
    sampler = system_telemetry.SystemTelemetrySampler(10)

    sampler._capture_host_sample()
    root.children_list = []
    root.user_seconds = 0.15
    root.system_seconds = 0.025
    sampler._capture_host_sample()

    output_path = tmp_path / "reports" / "system-telemetry.json"
    sampler.write(output_path)
    payload = sampler.payload()
    host = payload["host"]
    open_resource_key = "handle_count" if os.name == "nt" else "fd_count"

    assert payload["schema_version"] == 1
    assert len(host["timestamps_ns"]) == 2
    assert host["system"]["memory_total_bytes"] == 1000
    assert host["system"]["memory_used_bytes"] == [700, 700]
    assert host["system"]["cpu_utilization"] == [None, 0.5]
    assert host["test_processes"]["process_count"] == [2, 1]
    assert host["test_processes"]["rss_bytes"] == [300, 100]
    assert host["test_processes"]["thread_count"] == [5, 2]
    assert host["test_processes"][open_resource_key] == [30, 10]
    assert host["test_processes"]["cpu_user_ns"] == [300000000, 350000000]
    assert host["test_processes"]["cpu_system_ns"] == [50000000, 55000000]
    assert "processes" not in host["test_processes"]
    assert output_path.is_file()
    assert "\n  " not in output_path.read_text(encoding="utf-8")


def test_sampler_captures_aligned_gpu_columns(monkeypatch: Any) -> None:
    outputs = iter(
        (
            "0, GPU-abc, NVIDIA RTX Test, 1024, 256, 75\n",
            "0, GPU-abc, NVIDIA RTX Test, 1024, 384, 50\n",
        )
    )

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        assert command[0] == "nvidia-smi"
        assert kwargs["check"] is False
        assert kwargs["timeout"] == system_telemetry.DEFAULT_GPU_COMMAND_TIMEOUT_SECONDS
        return SimpleNamespace(returncode=0, stdout=next(outputs), stderr="")

    monkeypatch.setattr(system_telemetry.subprocess, "run", fake_run)
    sampler = system_telemetry.SystemTelemetrySampler(10)
    sampler._capture_gpu_sample()
    sampler._capture_gpu_sample()

    gpu = sampler.payload()["gpu"]
    device = gpu["devices"]["GPU-abc"]
    assert len(gpu["timestamps_ns"]) == 2
    assert device["memory_total_bytes"] == 1024 * 1024 * 1024
    assert device["memory_used_bytes"] == [256 * 1024 * 1024, 384 * 1024 * 1024]
    assert device["utilization"] == [0.75, 0.5]


def test_missing_nvidia_smi_is_reported_only_once(monkeypatch: Any) -> None:
    calls = 0

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        nonlocal calls
        calls += 1
        raise FileNotFoundError

    monkeypatch.setattr(system_telemetry.subprocess, "run", fake_run)
    sampler = system_telemetry.SystemTelemetrySampler(10)
    sampler._capture_gpu_sample()
    sampler._capture_gpu_sample()

    assert calls == 1
    assert sampler.payload()["gpu"]["timestamps_ns"] == []
    assert sampler.payload()["errors"] == ["nvidia-smi is not installed"]
