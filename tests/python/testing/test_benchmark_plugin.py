# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
PLUGIN_PATH = REPOSITORY_ROOT / "falcor2/testing/benchmark/plugin.py"


@dataclass(frozen=True)
class _BenchViewProject:
    id: str
    name: str
    suite_id: str
    suite_name: str
    repository: str
    producer_name: str
    producer_version: str


def _unused(*_args: Any, **_kwargs: Any) -> None:
    raise AssertionError("stubbed benchmark helper should not be called")


@contextmanager
def _stub_falcor2_imports() -> Iterator[None]:
    """Provide only the modules needed to import the benchmark plugin helpers."""
    module_names = [
        "slangpy",
        "slangpy.testing",
        "slangpy.testing.benchmark",
        "falcor2",
        "falcor2.testing",
        "falcor2.testing.helpers",
    ]
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in module_names}

    slangpy_module = types.ModuleType("slangpy")
    slangpy_testing_module = types.ModuleType("slangpy.testing")
    benchmark_module = types.ModuleType("slangpy.testing.benchmark")
    benchmark_module.build_benchview_observation = _unused
    benchmark_module.build_benchview_submissions = _unused
    benchmark_module.build_metric = _unused
    benchmark_module.lock_gpu_clocks = _unused
    benchmark_module.unlock_gpu_clocks = _unused
    benchmark_module.submit_benchview_submissions = _unused
    benchmark_module.benchview_submission_url = lambda api_base_url: (
        api_base_url.rstrip("/") + "/api/v1/submissions"
    )
    benchmark_module.BenchViewMetric = dict[str, Any]
    benchmark_module.BenchViewObservation = dict[str, Any]
    benchmark_module.BenchViewProject = _BenchViewProject
    benchmark_module.BenchViewSubmission = dict[str, Any]
    benchmark_module.get_machine_info = lambda: {}
    benchmark_module.get_commit_info = lambda: {}

    falcor2_module = types.ModuleType("falcor2")
    falcor2_module.__path__ = []
    falcor2_testing_module = types.ModuleType("falcor2.testing")
    falcor2_testing_module.__path__ = []
    helpers_module = types.ModuleType("falcor2.testing.helpers")
    helpers_module.PROJECT_ROOT = REPOSITORY_ROOT

    sys.modules["slangpy"] = slangpy_module
    sys.modules["slangpy.testing"] = slangpy_testing_module
    sys.modules["slangpy.testing.benchmark"] = benchmark_module
    sys.modules["falcor2"] = falcor2_module
    sys.modules["falcor2.testing"] = falcor2_testing_module
    sys.modules["falcor2.testing.helpers"] = helpers_module

    try:
        yield
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _load_plugin() -> Any:
    with _stub_falcor2_imports():
        spec = importlib.util.spec_from_file_location(
            "_falcor2_benchmark_plugin_for_test",
            PLUGIN_PATH,
        )
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_workloads_have_distinct_names_and_backend_independent_ids(monkeypatch: pytest.MonkeyPatch):
    plugin = _load_plugin()

    def build_observation(**kwargs: Any):
        return {
            "test": {"id": "bench.py:test_scene", "name": "test_scene"},
            "case": {"dimensions": dict(kwargs["parameters"])},
        }

    monkeypatch.setattr(plugin, "build_benchview_observation", build_observation)

    def observation(parameters: dict[str, Any], workload_parameter: str | None = None):
        return plugin.build_falcor2_observation(
            test_name="test_scene",
            filename="bench.py",
            function_name="test_scene",
            source_line=None,
            observed_at=None,
            metrics=[],
            parameters=parameters,
            adapter_name=None,
            device_type=None,
            workload_parameter=workload_parameter,
        )

    cornell = observation({"scene": "cornell-box", "device_type": "cuda"}, "scene")
    vulkan = observation({"scene": "cornell-box", "device_type": "vulkan"}, "scene")
    sponza = observation({"scene": "sponza", "device_type": "cuda"}, "scene")
    assert cornell["test"]["name"] == "Cornell box"
    assert cornell["test"]["id"] == vulkan["test"]["id"]
    assert cornell["test"]["id"] != sponza["test"]["id"]
    assert cornell["case"]["dimensions"]["device_type"] == "cuda"
    assert observation({"scene": "cornell-box"})["test"]["id"] == "bench.py:test_scene"


def _sample_submission(metric_id: str = "gpu_time") -> dict[str, Any]:
    return {
        "schemaVersion": 1,
        "idempotencyKey": f"falcor2/{metric_id}",
        "project": {"id": "falcor2", "name": "Falcor2"},
        "producer": {"name": "falcor2-pytest-benchmark", "version": "1.0.0"},
        "run": {
            "key": "git:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/suite:imagetests/config:1",
            "suite": {"id": "imagetests", "name": "Image Tests"},
            "vcs": {"repository": "https://example.test/falcor2", "revision": "a" * 40},
        },
        "execution": {"id": "execution", "requestId": "request"},
        "observations": [
            {
                "test": {
                    "id": "tests/python/benchmarks/test_render_benchmarks.py:test",
                    "name": "test",
                },
                "observedAt": "2026-09-18T12:00:00Z",
                "status": "passed",
                "metrics": [
                    {
                        "id": metric_id,
                        "name": "GPU time",
                        "unit": "ms",
                        "direction": "lower",
                        "distribution": {"samples": [1.0]},
                    }
                ],
                "metadata": {
                    "adapterName": "NVIDIA Test GPU",
                    "deviceType": "vulkan",
                },
                "environment": {
                    "telemetry": {
                        "gpus": [
                            {
                                "index": 0,
                                "temperature": 45.0,
                            }
                        ]
                    }
                },
            }
        ],
    }


def _export_config(payload_path: Path, ready_path: Path) -> Any:
    options = {
        "--benchmark-request-id": "sealed-test",
        "--benchmark-export-submissions": payload_path,
        "--benchmark-export-ready": ready_path,
    }
    config = SimpleNamespace(option=SimpleNamespace(benchmark=False))
    config.addinivalue_line = lambda *_args: None

    def getoption(name: str, default: Any = False) -> Any:
        if name == "--benchmark":
            return config.option.benchmark
        return options.get(name, default)

    config.getoption = getoption
    return config


@pytest.fixture
def export_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    plugin = _load_plugin()
    monkeypatch.setattr(plugin.helpers, "PROJECT_ROOT", tmp_path)
    submission = _sample_submission()
    payload_path = tmp_path / "payload.json"
    ready_path = tmp_path / "payload.ready"
    ready_path.write_text("stale marker\n", encoding="utf-8")
    monkeypatch.delenv("BENCHVIEW_API_KEY", raising=False)
    monkeypatch.setenv("BENCHVIEW_API_URL", "http://example.invalid/benchview")
    monkeypatch.setattr(plugin, "_get_falcor2_project_info", lambda: {})
    monkeypatch.setattr(
        plugin, "build_benchview_submissions", lambda *_args, **_kwargs: [submission]
    )
    config = _export_config(payload_path, ready_path)
    session = SimpleNamespace(config=config, exitstatus=pytest.ExitCode.OK, shouldstop=False)
    benchmark = plugin.BenchmarkPlugin(config)
    benchmark.pytest_configure(config)
    benchmark.pytest_sessionstart(session)
    getattr(config, plugin.BENCHMARK_CONTEXT_KEY)["observations"].append(
        submission["observations"][0]
    )
    return SimpleNamespace(
        plugin=plugin,
        benchmark=benchmark,
        session=session,
        payload_path=payload_path,
        ready_path=ready_path,
        submission=submission,
    )


def test_sealed_export_wraps_benchview_submissions_without_filtering() -> None:
    plugin = _load_plugin()
    submissions = [_sample_submission(), _sample_submission("cpu_time")]

    payload = plugin._build_submission_export(submissions)

    assert payload == {
        "schemaVersion": 1,
        "kind": "falcor2-benchview-export",
        "submissionSchema": "benchmark-submission-v1",
        "submissions": submissions,
    }


def test_sealed_export_writes_payload_before_ready_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plugin = _load_plugin()
    submission = _sample_submission()
    payload_path = tmp_path / "perf-results/falcor2-pathtracer-scene-bench.json"
    ready_path = tmp_path / "perf-results/falcor2-pathtracer-scene-bench.ready"
    writes = []
    write_text_atomically = plugin._write_text_atomically

    def record_write(path: Path, text: str) -> None:
        if path == ready_path:
            assert json.loads(payload_path.read_text(encoding="utf-8"))["submissions"] == [
                submission
            ]
        write_text_atomically(path, text)
        writes.append(path)

    monkeypatch.setattr(plugin, "_write_text_atomically", record_write)

    plugin._write_submission_export(
        [submission],
        payload_path=payload_path,
        ready_path=ready_path,
    )

    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert payload["submissions"] == [submission]
    assert payload["submissions"][0]["observations"][0]["metadata"]["deviceType"] == "vulkan"
    assert (
        payload["submissions"][0]["observations"][0]["environment"]["telemetry"]["gpus"][0][
            "temperature"
        ]
        == 45.0
    )
    assert ready_path.read_text(encoding="utf-8") == "ready\n"
    assert writes == [payload_path, ready_path]


@pytest.mark.parametrize("alias", ["identical", "relative", "parent"])
def test_sealed_export_rejects_aliased_payload_and_ready_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias: str
) -> None:
    plugin = _load_plugin()
    payload_path = tmp_path / "payload.json"
    payload_path.write_text("existing payload", encoding="utf-8")
    if alias == "relative":
        monkeypatch.chdir(tmp_path)
        ready_path = Path("payload.json")
    elif alias == "parent":
        directory = tmp_path / "nested"
        directory.mkdir()
        ready_path = directory / ".." / "payload.json"
    else:
        ready_path = payload_path
    config = _export_config(payload_path, ready_path)
    benchmark = plugin.BenchmarkPlugin(config)

    with pytest.raises(pytest.UsageError, match="different paths"):
        benchmark.pytest_configure(config)

    assert payload_path.read_text(encoding="utf-8") == "existing payload"


def test_sealed_export_removes_stale_ready_marker_at_session_start(export_session: Any) -> None:
    assert not export_session.ready_path.exists()


@pytest.mark.parametrize(
    "release",
    [None, {"label": "v2026.19", "releasedAt": "2026-03-01T00:00:00Z"}],
    ids=["daily", "slang-release"],
)
def test_sealed_export_session_succeeds_without_benchview_credentials(
    export_session: Any, monkeypatch: pytest.MonkeyPatch, release: dict[str, str] | None
) -> None:
    if release:
        monkeypatch.setattr(export_session.plugin, "get_commit_info", lambda: {"id": "a" * 40})
        export_session.benchmark._project_id = "falcor2-slang"
        export_session.benchmark._compiler = {
            "id": "b" * 40,
            "falcor_commit": "a" * 40,
            "release": release,
        }
    assert "BENCHVIEW_API_KEY" not in os.environ
    assert export_session.session.config.option.benchmark

    export_session.benchmark.pytest_sessionfinish(export_session.session, pytest.ExitCode.OK)

    assert export_session.session.exitstatus == pytest.ExitCode.OK
    assert json.loads(export_session.payload_path.read_text(encoding="utf-8"))["submissions"] == [
        export_session.submission
    ]
    assert export_session.ready_path.read_text(encoding="utf-8") == "ready\n"
    assert export_session.submission["observations"][0]["metadata"]["gpuClocksLocked"] is False

    assert export_session.submission["run"].get("metadata", {}).get("release") == release
    assert export_session.submission["project"].get("metadata", {}) == {}
    if release:
        assert export_session.submission["observations"][0]["case"]["dimensions"] == {
            "compiler_integration": "rebuild",
        }


@pytest.mark.parametrize("skipped", [False, True], ids=["failed", "skipped"])
def test_incomplete_session_does_not_export(export_session: Any, skipped: bool) -> None:
    if skipped:
        export_session.benchmark._config = SimpleNamespace(getoption=lambda *_args, **_kwargs: True)
        export_session.benchmark.pytest_runtest_logreport(SimpleNamespace(skipped=True))
    else:
        export_session.session.exitstatus = pytest.ExitCode.TESTS_FAILED
    export_session.benchmark.pytest_sessionfinish(
        export_session.session, export_session.session.exitstatus
    )
    assert export_session.session.exitstatus != pytest.ExitCode.OK
    assert not export_session.ready_path.exists()
    assert not export_session.payload_path.exists()


def test_history_uses_the_measured_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    plugin = _load_plugin()
    monkeypatch.setattr(plugin, "_get_falcor2_project_info", lambda: {})
    monkeypatch.setattr(
        plugin, "get_commit_info", lambda: {"id": "a" * 40, "branch": "(detached head)"}
    )
    monkeypatch.setenv("CI_COMMIT_BRANCH", "main")
    monkeypatch.setenv("CI_COMMIT_SHA", "a" * 40)
    assert plugin.benchmark_identity("falcor2", None)[2]["branch"] == "main"
    monkeypatch.setenv("CI_COMMIT_SHA", "b" * 40)
    assert plugin.benchmark_identity("falcor2", None)[2]["branch"] == "(detached head)"
    compiler = {"id": "b" * 40, "falcor_commit": "a" * 40}
    project, info, revision = plugin.benchmark_identity("falcor2-slang", compiler)
    assert project.id == "falcor2-slang"
    assert info == {"version": "a" * 40}
    assert revision["id"] == compiler["id"]
    with pytest.raises(ValueError, match="requires a compiler"):
        plugin.benchmark_identity("falcor2-slang", None)
    with pytest.raises(ValueError, match="different Falcor2 checkout"):
        plugin.benchmark_identity("falcor2-slang", dict(compiler, falcor_commit="d" * 40))


@pytest.mark.parametrize("failure", ["payload", "ready"])
def test_sealed_export_write_failure_fails_session_without_ready_marker(
    export_session: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    destination = export_session.payload_path if failure == "payload" else export_session.ready_path
    write_text = Path.write_text

    def fail_partial_write(path: Path, text: str, *args: Any, **kwargs: Any) -> int:
        if path.name.startswith(f".{destination.name}."):
            write_text(path, text[:1], *args, **kwargs)
            raise OSError("injected export write failure")
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_partial_write)

    export_session.benchmark.pytest_sessionfinish(export_session.session, pytest.ExitCode.OK)

    assert export_session.session.exitstatus == pytest.ExitCode.INTERNAL_ERROR
    assert not export_session.ready_path.exists()
    assert export_session.payload_path.exists() == (failure == "ready")
    assert not list(destination.parent.glob(".*.tmp"))
