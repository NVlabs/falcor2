# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from falcor2.testing import telemetry_plugin


def _shader_compilation(
    key: str,
    *,
    backend: str = "d3d12",
    phase: str = "call",
    cached: bool = False,
) -> dict[str, Any]:
    return {
        "phase": phase,
        "backend": backend,
        "key": key,
        "program": "module:main",
        "entry_point": "main",
        "cached": cached,
        "create_seconds": 0.5,
        "compile_seconds": 0.4 if not cached else 0.0,
        "slang_seconds": 0.3 if not cached else 0.0,
        "downstream_seconds": 0.1 if not cached else 0.0,
        "cache_size_bytes": 128,
    }


def test_capture_compilation_report_delta_reads_only_appended_reports(monkeypatch: Any) -> None:
    entry_points = [
        {
            "name": "stale",
            "cache_key": "sha1:stale",
            "create_time": 0.1,
            "compile_time": 0.1,
            "compile_slang_time": 0.1,
            "compile_downstream_time": 0.0,
            "is_cached": False,
            "cache_size": 64,
        }
    ]
    pipelines: list[dict[str, Any]] = []
    device = SimpleNamespace(
        desc=SimpleNamespace(type=SimpleNamespace(name="d3d12"), enable_compilation_reports=True),
        get_compilation_reports=lambda: [
            {
                "label": "module:main",
                "entry_point_reports": entry_points,
                "pipeline_reports": pipelines,
            }
        ],
    )
    monkeypatch.setattr(telemetry_plugin.spy.Device, "get_created_devices", lambda: [device])
    cursor: telemetry_plugin.CompilationCursor = {}

    shaders, pipeline_records, errors = telemetry_plugin._capture_compilation_report_delta(
        cursor, None
    )
    assert shaders == []
    assert pipeline_records == []
    assert errors == []

    entry_points.append(
        {
            "name": "main",
            "cache_key": "sha1:abc",
            "create_time": 0.5,
            "compile_time": 0.4,
            "compile_slang_time": 0.3,
            "compile_downstream_time": 0.1,
            "is_cached": False,
            "cache_size": 128,
        }
    )
    pipelines.append(
        {
            "type": "compute",
            "cache_key": "sha1:pipeline",
            "create_time": 0.2,
            "is_cached": True,
            "cache_size": 256,
        }
    )
    shaders, pipeline_records, errors = telemetry_plugin._capture_compilation_report_delta(
        cursor, "call"
    )

    assert errors == []
    assert shaders == [_shader_compilation("sha1:abc")]
    assert pipeline_records == [
        {
            "phase": "call",
            "backend": "d3d12",
            "key": "sha1:pipeline",
            "program": "module:main",
            "type": "compute",
            "cached": True,
            "create_seconds": 0.2,
            "cache_size_bytes": 256,
        }
    ]

    shaders, pipeline_records, errors = telemetry_plugin._capture_compilation_report_delta(
        cursor, "teardown"
    )
    assert shaders == []
    assert pipeline_records == []
    assert errors == []


def test_cache_key_preserves_unavailable_keys() -> None:
    assert telemetry_plugin._cache_key({"cache_key": "sha1:abc"}) == "sha1:abc"
    assert telemetry_plugin._cache_key({"cache_key": None}) is None
    assert telemetry_plugin._cache_key({}) is None


def test_plugin_records_compilations_rss_and_sparse_leak_data(
    monkeypatch: Any, tmp_path: Path
) -> None:
    compilation_snapshots = iter(
        (
            ([], [], []),
            ([_shader_compilation("sha1:cold")], [], []),
            ([_shader_compilation("sha1:warm", phase="cleanup", cached=True)], [], []),
        )
    )
    rss_snapshots = iter(((100, []), (140, [])))
    monkeypatch.setattr(
        telemetry_plugin,
        "_capture_compilation_report_delta",
        lambda cursor, phase: next(compilation_snapshots),
    )
    monkeypatch.setattr(
        telemetry_plugin,
        "_worker_tree_rss_snapshot",
        lambda: next(rss_snapshots),
    )
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "test-telemetry.json",
    )
    item = SimpleNamespace(
        nodeid="tests/python/test_example.py::test_cache",
        stash={telemetry_plugin.DEVICE_LEAKS_STASH_KEY: ["uncached-device"]},
    )

    plugin._start_test(item)  # type: ignore[arg-type]
    plugin._capture_runtime_state("call")
    plugin._finish_test(item)  # type: ignore[arg-type]

    record = plugin._records[0]
    assert record["shader_compilations"] == [
        _shader_compilation("sha1:cold"),
        _shader_compilation("sha1:warm", phase="cleanup", cached=True),
    ]
    assert record["pipeline_creations"] == []
    assert record["rss_bytes"] == {"before": 100, "after_cleanup": 140}
    assert record["resource_leaks"] == {
        "devices_forcibly_closed": 1,
        "device_labels": ["uncached-device"],
    }
    assert "wall" in record["durations_seconds"]
    assert "devices" not in record
    assert "markers" not in record
    assert "phases" not in record
    assert "cache_accesses" not in record

    # Restore real functions before the outer telemetry plugin finishes this test.
    monkeypatch.undo()


def test_plugin_omits_resource_leaks_when_cleanup_found_none(
    monkeypatch: Any, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        telemetry_plugin,
        "_capture_compilation_report_delta",
        lambda cursor, phase: ([], [], []),
    )
    monkeypatch.setattr(telemetry_plugin, "_worker_tree_rss_snapshot", lambda: (100, []))
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "test-telemetry.json",
    )
    item = SimpleNamespace(nodeid="tests/python/test_example.py::test_clean", stash={})

    plugin._start_test(item)  # type: ignore[arg-type]
    plugin._finish_test(item)  # type: ignore[arg-type]

    assert "resource_leaks" not in plugin._records[0]
    monkeypatch.undo()


def test_controller_merges_worker_files_in_start_order(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "test-telemetry.json"
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        output_path,
    )
    plugin._run_dir = tmp_path / "scratch" / plugin._run_id
    first = {
        "nodeid": "tests/python/test_example.py::test_first",
        "started_at_unix_ns": 10,
    }
    second = {
        "nodeid": "tests/python/test_example.py::test_second",
        "started_at_unix_ns": 20,
    }
    telemetry_plugin._write_json_atomic(
        plugin._run_dir / "worker-gw1.json",
        {"worker": "gw1", "tests": [second]},
    )
    telemetry_plugin._write_json_atomic(
        plugin._run_dir / "worker-gw0.json",
        {"worker": "gw0", "tests": [first]},
    )

    plugin._merge_worker_files(exitstatus=0)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["workers"] == ["gw0", "gw1"]
    assert [record["nodeid"] for record in payload["tests"]] == [
        "tests/python/test_example.py::test_first",
        "tests/python/test_example.py::test_second",
    ]
    assert not plugin._run_dir.exists()


def test_controller_session_start_removes_stale_output(tmp_path: Path) -> None:
    output_path = tmp_path / "reports" / "test-telemetry.json"
    output_path.parent.mkdir(parents=True)
    output_path.write_text("stale", encoding="utf-8")
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        output_path,
    )
    plugin._run_dir = tmp_path / "scratch" / plugin._run_id

    plugin.pytest_sessionstart(SimpleNamespace())  # type: ignore[arg-type]

    assert not output_path.exists()


def test_session_start_io_failure_disables_artifact_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_path = tmp_path / "test-telemetry.json"
    output_path.mkdir()
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        output_path,
    )
    plugin._run_dir = tmp_path / "scratch" / plugin._run_id

    plugin.pytest_sessionstart(SimpleNamespace())  # type: ignore[arg-type]
    plugin.pytest_sessionfinish(SimpleNamespace(), exitstatus=1)  # type: ignore[arg-type]

    assert not plugin._artifact_output_enabled
    assert "WARNING: Cannot initialize test telemetry artifacts" in capsys.readouterr().out


def test_session_finish_io_failure_preserves_test_result(
    monkeypatch: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plugin = telemetry_plugin.TelemetryPlugin(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "test-telemetry.json",
    )
    plugin._run_dir = tmp_path / "scratch" / plugin._run_id
    plugin._run_dir.mkdir(parents=True)

    def fail_write(path: Path, payload: Mapping[str, Any]) -> None:
        raise PermissionError("artifact is locked")

    monkeypatch.setattr(telemetry_plugin, "_write_json_atomic", fail_write)

    plugin.pytest_sessionfinish(SimpleNamespace(), exitstatus=1)  # type: ignore[arg-type]

    assert "WARNING: Cannot finalize test telemetry artifacts" in capsys.readouterr().out
