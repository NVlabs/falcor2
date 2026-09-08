# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sample system, test-process, and GPU telemetry while the Python tests run."""

from __future__ import annotations

import csv
import io
import json
import os
from pathlib import Path
import subprocess
import threading
import time
from typing import Any
import uuid

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in an incomplete CI environment
    psutil = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH = Path("reports/system-telemetry.json")
DEFAULT_HOST_SAMPLE_INTERVAL_SECONDS = 0.25
DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS = 1.0
DEFAULT_GPU_COMMAND_TIMEOUT_SECONDS = 2.0
_MEBIBYTE = 1024 * 1024


class SystemTelemetrySampler:
    """Collect best-effort system telemetry for one test process tree."""

    def __init__(
        self,
        root_pid: int,
        host_sample_interval_seconds: float = DEFAULT_HOST_SAMPLE_INTERVAL_SECONDS,
        gpu_sample_interval_seconds: float = DEFAULT_GPU_SAMPLE_INTERVAL_SECONDS,
        gpu_command_timeout_seconds: float = DEFAULT_GPU_COMMAND_TIMEOUT_SECONDS,
    ):
        super().__init__()
        self._root_pid = root_pid
        self._host_sample_interval_seconds = host_sample_interval_seconds
        self._gpu_sample_interval_seconds = gpu_sample_interval_seconds
        self._gpu_command_timeout_seconds = gpu_command_timeout_seconds
        self._started_at_unix_ns = time.time_ns()
        self._finished_at_unix_ns: int | None = None
        self._system_memory_total_bytes: int | None = None
        self._host_timestamps_ns: list[int] = []
        self._system_columns: dict[str, list[int | float | None]] = {
            "cpu_utilization": [],
            "memory_available_bytes": [],
            "memory_used_bytes": [],
            "swap_used_bytes": [],
        }
        self._test_process_columns: dict[str, list[int]] = {
            "process_count": [],
            "cpu_user_ns": [],
            "cpu_system_ns": [],
            "rss_bytes": [],
            "thread_count": [],
            "handle_count" if os.name == "nt" else "fd_count": [],
        }
        self._process_cpu_times: dict[tuple[int, int], tuple[int, int]] = {}
        self._cpu_percent_ready = False
        self._gpu_timestamps_ns: list[int] = []
        self._gpu_devices: dict[str, dict[str, Any]] = {}
        self._gpu_sampling_enabled = True
        self._errors: list[str] = []
        self._stop_event = threading.Event()
        self._host_thread = threading.Thread(
            target=self._sample_host_until_stopped,
            name="falcor2-test-host-system-sampler",
            daemon=True,
        )
        self._gpu_thread = threading.Thread(
            target=self._sample_gpu_until_stopped,
            name="falcor2-test-gpu-system-sampler",
            daemon=True,
        )
        self._started_threads: list[threading.Thread] = []

    def start(self) -> None:
        if self._started_threads:
            raise RuntimeError("system telemetry sampler is already started")
        try:
            for thread in (self._host_thread, self._gpu_thread):
                thread.start()
                self._started_threads.append(thread)
        except Exception:
            self._stop_event.set()
            for thread in self._started_threads:
                thread.join()
            raise

    def stop(self) -> None:
        self._stop_event.set()
        for thread in self._started_threads:
            thread.join()
        self._finished_at_unix_ns = time.time_ns()

    def payload(self) -> dict[str, Any]:
        system: dict[str, Any] = dict(self._system_columns)
        system["memory_total_bytes"] = self._system_memory_total_bytes
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "started_at_unix_ns": self._started_at_unix_ns,
            "finished_at_unix_ns": self._finished_at_unix_ns,
            "root_pid": self._root_pid,
            "host": {
                "sample_interval_seconds": self._host_sample_interval_seconds,
                "timestamps_ns": self._host_timestamps_ns,
                "system": system,
                "test_processes": self._test_process_columns,
            },
            "gpu": {
                "sample_interval_seconds": self._gpu_sample_interval_seconds,
                "timestamps_ns": self._gpu_timestamps_ns,
                "devices": self._gpu_devices,
            },
        }
        if self._errors:
            payload["errors"] = sorted(set(self._errors))
        return payload

    def write(self, path: Path = DEFAULT_SYSTEM_TELEMETRY_OUTPUT_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self.payload(), separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _sample_host_until_stopped(self) -> None:
        next_sample = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now >= next_sample:
                self._capture_host_sample()
                next_sample += self._host_sample_interval_seconds
                if next_sample <= now:
                    next_sample = now + self._host_sample_interval_seconds
            self._stop_event.wait(max(0.0, next_sample - time.monotonic()))

    def _sample_gpu_until_stopped(self) -> None:
        next_sample = time.monotonic()
        while not self._stop_event.is_set() and self._gpu_sampling_enabled:
            now = time.monotonic()
            if now >= next_sample:
                self._capture_gpu_sample()
                next_sample += self._gpu_sample_interval_seconds
                if next_sample <= now:
                    next_sample = now + self._gpu_sample_interval_seconds
            self._stop_event.wait(max(0.0, next_sample - time.monotonic()))

    def _relative_timestamp_ns(self) -> int:
        return max(0, time.time_ns() - self._started_at_unix_ns)

    def _capture_host_sample(self) -> None:
        if psutil is None:
            self._errors.append("psutil is not installed")
            return

        timestamp_ns = self._relative_timestamp_ns()
        cpu_utilization: float | None = None
        memory_available_bytes: int | None = None
        memory_used_bytes: int | None = None
        swap_used_bytes: int | None = None
        try:
            cpu_percent = float(psutil.cpu_percent(interval=None))
            if self._cpu_percent_ready:
                cpu_utilization = cpu_percent / 100.0
            self._cpu_percent_ready = True
        except Exception as exc:
            self._errors.append(f"Cannot sample system CPU: {exc}")
        try:
            virtual_memory = psutil.virtual_memory()
            self._system_memory_total_bytes = int(virtual_memory.total)
            memory_available_bytes = int(virtual_memory.available)
            memory_used_bytes = int(virtual_memory.total - virtual_memory.available)
        except Exception as exc:
            self._errors.append(f"Cannot sample system memory: {exc}")
        try:
            swap_used_bytes = int(psutil.swap_memory().used)
        except Exception as exc:
            self._errors.append(f"Cannot sample system swap: {exc}")

        try:
            test_processes = self._test_process_snapshot()
        except Exception as exc:
            self._errors.append(f"Cannot sample test process tree: {exc}")
            open_resource_key = "handle_count" if os.name == "nt" else "fd_count"
            test_processes = {
                "process_count": 0,
                "cpu_user_ns": sum(value[0] for value in self._process_cpu_times.values()),
                "cpu_system_ns": sum(value[1] for value in self._process_cpu_times.values()),
                "rss_bytes": 0,
                "thread_count": 0,
                open_resource_key: 0,
            }
        self._host_timestamps_ns.append(timestamp_ns)
        self._system_columns["cpu_utilization"].append(cpu_utilization)
        self._system_columns["memory_available_bytes"].append(memory_available_bytes)
        self._system_columns["memory_used_bytes"].append(memory_used_bytes)
        self._system_columns["swap_used_bytes"].append(swap_used_bytes)
        for key, values in self._test_process_columns.items():
            values.append(test_processes[key])

    def _test_process_snapshot(self) -> dict[str, int]:
        assert psutil is not None
        open_resource_key = "handle_count" if os.name == "nt" else "fd_count"
        result = {
            "process_count": 0,
            "cpu_user_ns": 0,
            "cpu_system_ns": 0,
            "rss_bytes": 0,
            "thread_count": 0,
            open_resource_key: 0,
        }
        try:
            root = psutil.Process(self._root_pid)
            processes = [root, *root.children(recursive=True)]
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            self._errors.append(f"Cannot enumerate test process tree: {exc}")
            result["cpu_user_ns"] = sum(value[0] for value in self._process_cpu_times.values())
            result["cpu_system_ns"] = sum(value[1] for value in self._process_cpu_times.values())
            return result

        seen_pids: set[int] = set()
        for process in processes:
            if process.pid in seen_pids:
                continue
            seen_pids.add(process.pid)
            try:
                create_time_ns = int(process.create_time() * 1_000_000_000)
                memory_info = process.memory_info()
                cpu_times = process.cpu_times()
                thread_count = int(process.num_threads())
                if os.name == "nt":
                    open_resource_count = int(process.num_handles())
                else:
                    open_resource_count = int(process.num_fds())
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            except Exception as exc:
                self._errors.append(f"Cannot sample test process {process.pid}: {exc}")
                continue

            identity = (int(process.pid), create_time_ns)
            self._process_cpu_times[identity] = (
                int(float(cpu_times.user) * 1_000_000_000),
                int(float(cpu_times.system) * 1_000_000_000),
            )
            result["process_count"] += 1
            result["rss_bytes"] += int(memory_info.rss)
            result["thread_count"] += thread_count
            result[open_resource_key] += open_resource_count

        result["cpu_user_ns"] = sum(value[0] for value in self._process_cpu_times.values())
        result["cpu_system_ns"] = sum(value[1] for value in self._process_cpu_times.values())
        return result

    def _capture_gpu_sample(self) -> None:
        if not self._gpu_sampling_enabled:
            return
        command = [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        run_kwargs: dict[str, Any] = {
            "capture_output": True,
            "text": True,
            "check": False,
            "timeout": self._gpu_command_timeout_seconds,
        }
        if os.name == "nt":
            run_kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        try:
            completed = subprocess.run(command, **run_kwargs)
        except FileNotFoundError:
            self._gpu_sampling_enabled = False
            self._errors.append("nvidia-smi is not installed")
            return
        except Exception as exc:
            self._errors.append(f"Cannot sample GPUs: {exc}")
            return
        if completed.returncode != 0:
            diagnostic = completed.stderr.strip() or completed.stdout.strip()
            self._errors.append(
                f"Cannot sample GPUs: nvidia-smi exited with {completed.returncode}"
                + (f": {diagnostic}" if diagnostic else "")
            )
            return

        sample_index = len(self._gpu_timestamps_ns)
        self._gpu_timestamps_ns.append(self._relative_timestamp_ns())
        for device in self._gpu_devices.values():
            device["utilization"].append(None)
            device["memory_used_bytes"].append(None)

        rows = csv.reader(io.StringIO(completed.stdout), skipinitialspace=True)
        for row in rows:
            if not row:
                continue
            if len(row) != 6:
                self._errors.append(f"Cannot parse nvidia-smi GPU row: {row!r}")
                continue
            try:
                index = int(row[0])
                gpu_uuid = row[1]
                name = row[2]
                memory_total_bytes = _parse_mebibytes(row[3])
                memory_used_bytes = _parse_mebibytes(row[4])
                utilization = _parse_percentage(row[5])
            except ValueError as exc:
                self._errors.append(f"Cannot parse nvidia-smi GPU row: {exc}")
                continue

            device = self._gpu_devices.get(gpu_uuid)
            if device is None:
                device = {
                    "index": index,
                    "name": name,
                    "memory_total_bytes": memory_total_bytes,
                    "utilization": [None] * (sample_index + 1),
                    "memory_used_bytes": [None] * (sample_index + 1),
                }
                self._gpu_devices[gpu_uuid] = device
            device["utilization"][-1] = utilization
            device["memory_used_bytes"][-1] = memory_used_bytes


def _parse_mebibytes(value: str) -> int | None:
    if value.strip().upper() == "N/A":
        return None
    return round(float(value) * _MEBIBYTE)


def _parse_percentage(value: str) -> float | None:
    if value.strip().upper() == "N/A":
        return None
    return float(value) / 100.0
