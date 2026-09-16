# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact pytest telemetry for timings, shader compilations, RSS, and leaks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterator, Mapping
import uuid

import pytest
import slangpy as spy

try:
    import psutil
except ImportError:  # pragma: no cover - exercised only in an incomplete CI environment
    psutil = None  # type: ignore[assignment]

import falcor2.testing.helpers as helpers


SCHEMA_VERSION = 1
DEFAULT_TELEMETRY_OUTPUT_PATH = Path("reports/test-telemetry.json")
TELEMETRY_SCRATCH_ROOT = Path(".tmp/test-telemetry")
DEVICE_LEAKS_STASH_KEY = pytest.StashKey[list[str]]()


def _worker_tree_rss_snapshot() -> tuple[int | None, list[str]]:
    if psutil is None:
        return None, ["psutil is not installed"]
    try:
        root = psutil.Process(os.getpid())
        processes = [root, *root.children(recursive=True)]
        rss_bytes = 0
        seen_pids: set[int] = set()
        for process in processes:
            if process.pid in seen_pids:
                continue
            seen_pids.add(process.pid)
            try:
                rss_bytes += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return rss_bytes, []
    except Exception as exc:
        return None, [f"Cannot capture worker RSS checkpoint: {exc}"]


CompilationCursor = dict[int, list[tuple[int, int]]]


def _cache_key(report: Mapping[str, Any]) -> str | None:
    value = report.get("cache_key")
    return value if isinstance(value, str) else None


def _capture_compilation_report_delta(
    cursor: CompilationCursor,
    phase: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    shader_compilations: list[dict[str, Any]] = []
    pipeline_creations: list[dict[str, Any]] = []
    errors: list[str] = []
    try:
        devices = spy.Device.get_created_devices()
    except Exception as exc:
        return [], [], [f"Cannot enumerate devices for compilation telemetry: {exc}"]

    seen_device_ids: set[int] = set()
    for device in devices:
        device_id = id(device)
        seen_device_ids.add(device_id)
        try:
            desc = device.desc
            if not bool(desc.enable_compilation_reports):
                continue
            backend = desc.type.name
            reports = device.get_compilation_reports()
            previous_counts = cursor.get(device_id, [])
            current_counts: list[tuple[int, int]] = []
            for program_index, report in enumerate(reports):
                entry_points = report["entry_point_reports"]
                pipelines = report["pipeline_reports"]
                previous_entry_points, previous_pipelines = (
                    previous_counts[program_index]
                    if program_index < len(previous_counts)
                    else (0, 0)
                )
                current_counts.append((len(entry_points), len(pipelines)))
                if phase is None:
                    continue
                for entry_point in entry_points[previous_entry_points:]:
                    shader_compilations.append(
                        {
                            "phase": phase,
                            "backend": backend,
                            "key": _cache_key(entry_point),
                            "program": str(report["label"]),
                            "entry_point": str(entry_point["name"]),
                            "cached": bool(entry_point["is_cached"]),
                            "create_seconds": float(entry_point["create_time"]),
                            "compile_seconds": float(entry_point["compile_time"]),
                            "slang_seconds": float(entry_point["compile_slang_time"]),
                            "downstream_seconds": float(entry_point["compile_downstream_time"]),
                            "cache_size_bytes": int(entry_point["cache_size"]),
                        }
                    )
                for pipeline in pipelines[previous_pipelines:]:
                    pipeline_creations.append(
                        {
                            "phase": phase,
                            "backend": backend,
                            "key": _cache_key(pipeline),
                            "program": str(report["label"]),
                            "type": str(pipeline["type"]),
                            "cached": bool(pipeline["is_cached"]),
                            "create_seconds": float(pipeline["create_time"]),
                            "cache_size_bytes": int(pipeline["cache_size"]),
                        }
                    )
            cursor[device_id] = current_counts
        except Exception as exc:
            errors.append(f"Cannot collect compilation telemetry from device {device_id}: {exc}")
    for device_id in cursor.keys() - seen_device_ids:
        del cursor[device_id]
    return shader_compilations, pipeline_creations, errors


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


class TelemetryPlugin:
    """Collect worker-local telemetry and merge it into a single JSON artifact."""

    def __init__(self, config: pytest.Config, output_path: Path):
        super().__init__()
        self._config = config
        self._output_path = output_path
        worker_input = getattr(config, "workerinput", None)
        self._is_worker = worker_input is not None
        self._worker = str(worker_input["workerid"]) if worker_input else "main"
        self._run_id = (
            str(worker_input.get("falcor2_telemetry_run_id", worker_input["testrunuid"]))
            if worker_input
            else uuid.uuid4().hex
        )
        self._run_dir = TELEMETRY_SCRATCH_ROOT / self._run_id
        self._records: list[dict[str, Any]] = []
        self._active: dict[str, Any] | None = None
        self._compilation_cursor: CompilationCursor = {}
        self._shader_compilations: list[dict[str, Any]] = []
        self._pipeline_creations: list[dict[str, Any]] = []
        self._telemetry_errors: list[str] = []
        self._artifact_output_enabled = True

    @pytest.hookimpl(optionalhook=True)
    def pytest_configure_node(self, node: Any) -> None:
        node.workerinput["falcor2_telemetry_run_id"] = self._run_id

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        try:
            self._run_dir.mkdir(parents=True, exist_ok=True)
            if not self._is_worker:
                self._output_path.unlink(missing_ok=True)
        except OSError as exc:
            self._artifact_output_enabled = False
            self._warn_artifact_io("initialize", exc)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(
        self, item: pytest.Item, nextitem: pytest.Item | None
    ) -> Iterator[None]:
        if not self._artifact_output_enabled:
            yield
            return
        self._start_test(item)
        yield
        self._finish_test(item)

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(
        self, item: pytest.Item, call: pytest.CallInfo[Any]
    ) -> Iterator[None]:
        outcome = yield
        report = outcome.get_result()
        if self._active is not None:
            self._active["durations_seconds"][report.when] = float(report.duration)
            self._capture_runtime_state(report.when)

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        if not self._artifact_output_enabled:
            return
        try:
            if self._is_worker:
                self._write_worker_file()
                return

            if self._records:
                self._write_worker_file()
            self._merge_worker_files(exitstatus)
        except OSError as exc:
            self._warn_artifact_io("finalize", exc)

    def _start_test(self, item: pytest.Item) -> None:
        _, _, compilation_errors = _capture_compilation_report_delta(self._compilation_cursor, None)
        rss_before, rss_errors = _worker_tree_rss_snapshot()
        self._shader_compilations = []
        self._pipeline_creations = []
        self._telemetry_errors = [*compilation_errors, *rss_errors]
        self._active = {
            "nodeid": item.nodeid,
            "worker": self._worker,
            "pid": os.getpid(),
            "started_at_unix_ns": time.time_ns(),
            "wall_start": time.perf_counter(),
            "durations_seconds": {},
            "rss_before": rss_before,
        }

    def _capture_runtime_state(self, phase: str) -> None:
        shaders, pipelines, errors = _capture_compilation_report_delta(
            self._compilation_cursor, phase
        )
        self._shader_compilations.extend(shaders)
        self._pipeline_creations.extend(pipelines)
        self._telemetry_errors.extend(errors)

    def _finish_test(self, item: pytest.Item) -> None:
        if self._active is None:
            return
        self._capture_runtime_state("cleanup")
        rss_after, rss_errors = _worker_tree_rss_snapshot()
        self._telemetry_errors.extend(rss_errors)
        self._active["durations_seconds"]["wall"] = time.perf_counter() - self._active.pop(
            "wall_start"
        )
        self._active["finished_at_unix_ns"] = time.time_ns()
        self._active["shader_compilations"] = self._shader_compilations
        self._active["pipeline_creations"] = self._pipeline_creations
        self._active["rss_bytes"] = {
            "before": self._active.pop("rss_before"),
            "after_cleanup": rss_after,
        }
        stash = getattr(item, "stash", None)
        leaked_device_labels = stash.get(DEVICE_LEAKS_STASH_KEY, []) if stash is not None else []
        if leaked_device_labels:
            self._active["resource_leaks"] = {
                "devices_forcibly_closed": len(leaked_device_labels),
                "device_labels": leaked_device_labels,
            }
        if self._telemetry_errors:
            self._active["telemetry_errors"] = sorted(set(self._telemetry_errors))
        self._records.append(self._active)
        self._active = None

    def _worker_file(self) -> Path:
        safe_worker = re.sub(r"[^A-Za-z0-9_.-]+", "_", self._worker)
        return self._run_dir / f"worker-{safe_worker}.json"

    def _warn_artifact_io(self, action: str, exc: OSError) -> None:
        print(f"WARNING: Cannot {action} test telemetry artifacts: {exc}")

    def _write_worker_file(self) -> None:
        _write_json_atomic(
            self._worker_file(),
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": self._run_id,
                "worker": self._worker,
                "tests": self._records,
            },
        )

    def _merge_worker_files(self, exitstatus: int) -> None:
        tests: list[dict[str, Any]] = []
        workers: list[str] = []
        errors: list[str] = []
        worker_files = sorted(self._run_dir.glob("worker-*.json"))
        for path in worker_files:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                workers.append(str(payload["worker"]))
                tests.extend(payload["tests"])
            except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
                errors.append(f"Cannot merge {path.as_posix()}: {exc}")

        tests.sort(key=lambda record: (record["started_at_unix_ns"], record["nodeid"]))
        try:
            device_cache_policy = str(self._config.getoption("--device-cache-policy"))
        except (AttributeError, ValueError):
            device_cache_policy = "session"
        payload: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self._run_id,
            "exitstatus": int(exitstatus),
            "workers": sorted(workers),
            "configuration": {
                "module_cache_enabled": helpers.MODULE_CACHE_ENABLED,
                "shader_cache_enabled": helpers.SHADER_CACHE_ENABLED,
                "compilation_reports_enabled": helpers.COMPILATION_REPORTS_ENABLED,
                "cache_root": helpers.MODULE_AND_SHADER_CACHE_ROOT.as_posix(),
                "device_cache_policy": device_cache_policy,
            },
            "ci": {
                key: os.environ[key]
                for key in ("CI_JOB_ID", "CI_PIPELINE_ID", "CI_COMMIT_SHA")
                if key in os.environ
            },
            "tests": tests,
        }
        if errors:
            payload["merge_errors"] = errors
        _write_json_atomic(self._output_path, payload)

        for path in worker_files:
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                self._warn_artifact_io(f"remove {path.as_posix()}", exc)
        try:
            self._run_dir.rmdir()
        except OSError:
            pass
