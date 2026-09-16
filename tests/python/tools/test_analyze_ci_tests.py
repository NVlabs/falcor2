# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO
import json
from pathlib import Path
import zipfile

import pytest

from tools import analyze_ci_tests


def _artifact(files: dict[str, str]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, contents in files.items():
            archive.writestr(name, contents)
    return output.getvalue()


def test_parse_pipeline_target_from_url() -> None:
    target = analyze_ci_tests.parse_pipeline_target(
        "https://gitlab.example.com/group/subgroup/project/-/pipelines/1234",
        project=None,
        gitlab_url=None,
    )

    assert target == analyze_ci_tests.PipelineTarget(
        api_url="https://gitlab.example.com/api/v4",
        project="group/subgroup/project",
        pipeline_id=1234,
    )


def test_authentication_headers_only_discovers_tokens_for_ci_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CI_SERVER_URL", "https://gitlab.example.com")
    monkeypatch.setenv("GITLAB_TOKEN", "automatic-token")

    assert analyze_ci_tests.authentication_headers(None, "https://gitlab.example.com/api/v4") == {
        "PRIVATE-TOKEN": "automatic-token"
    }
    assert analyze_ci_tests.authentication_headers(None, "https://attacker.example/api/v4") == {}


def test_authentication_headers_allows_explicit_token_for_other_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXPLICIT_TOKEN", "explicit-token")

    assert analyze_ci_tests.authentication_headers(
        "EXPLICIT_TOKEN", "https://gitlab.example.com/api/v4"
    ) == {"PRIVATE-TOKEN": "explicit-token"}


def test_select_job_requires_an_exact_unambiguous_name() -> None:
    jobs = [
        {
            "id": 10,
            "name": "windows-release",
            "status": "success",
            "duration": 123.5,
            "queued_duration": 4.0,
            "web_url": "https://gitlab.example.com/job/10",
        },
        {"id": 11, "name": "linux-gcc-release", "status": "failed"},
    ]

    job = analyze_ci_tests.select_job(jobs, "windows-release")

    assert job.id == 10
    assert job.duration == 123.5
    with pytest.raises(analyze_ci_tests.AnalysisError, match="not found"):
        analyze_ci_tests.select_job(jobs, "windows-debug")


def test_extract_reports_filters_the_job_artifact() -> None:
    artifact = _artifact(
        {
            "reports/pytest-junit.xml": "<testsuites/>",
            "reports/doctest-junit.xml": "<testsuites/>",
            ".crashpad/report.dmp": "not a test report",
        }
    )

    reports = analyze_ci_tests.extract_reports(artifact, ["reports/*junit.xml"])

    assert [name for name, _data in reports] == [
        "reports/doctest-junit.xml",
        "reports/pytest-junit.xml",
    ]


def test_parse_and_aggregate_junit_results(tmp_path: Path) -> None:
    source = tmp_path / "tests" / "python" / "render" / "test_scene.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def test_render(device):\n" "    pass\n" "\n" "def test_quick():\n" "    pass\n",
        encoding="utf-8",
    )
    junit = b"""<?xml version="1.0"?>
<testsuites>
  <testsuite name="pytest">
    <testcase classname="tests.python.render.test_scene" name="test_render[d3d12]" time="12.5" />
    <testcase classname="tests.python.render.test_scene" name="test_render[vulkan]" time="2.5" />
    <testcase classname="tests.python.render.test_scene" name="test_quick" time="0.1">
      <skipped />
    </testcase>
  </testsuite>
</testsuites>
"""

    results = analyze_ci_tests.parse_junit_report(
        "reports/pytest-junit.xml", junit, analyze_ci_tests.SourceIndex(tmp_path)
    )
    logical = analyze_ci_tests.aggregate_results(results, "logical_test")

    assert [result.status for result in results] == ["success", "success", "skipped"]
    assert results[0].source == "tests/python/render/test_scene.py"
    assert results[0].line == 1
    assert logical[0] == analyze_ci_tests.Aggregate(
        name="tests/python/render/test_scene.py::test_render",
        count=2,
        total_seconds=15.0,
        minimum_seconds=2.5,
        median_seconds=7.5,
        maximum_seconds=12.5,
    )


def test_source_index_resolves_dotted_slang_classname(tmp_path: Path) -> None:
    source = tmp_path / "tests" / "python" / "utils" / "test_shader.slang"
    source.parent.mkdir(parents=True)
    source.write_text('TEST_CASE("shader_test") {}\n', encoding="utf-8")

    resolved, line = analyze_ci_tests.SourceIndex(tmp_path).resolve(
        file_attribute=None,
        classname="tests.python.utils.test_shader.slang",
        test_name="shader_test[d3d12]",
    )

    assert resolved == "tests/python/utils/test_shader.slang"
    assert line == 1


def test_source_index_resolves_same_method_name_in_different_classes(tmp_path: Path) -> None:
    source = tmp_path / "tests" / "python" / "test_scene.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "class TestA:\n"
        "    def test_render(self):\n"
        "        pass\n"
        "\n"
        "class TestB:\n"
        "    def test_render(self):\n"
        "        pass\n",
        encoding="utf-8",
    )
    source_index = analyze_ci_tests.SourceIndex(tmp_path)

    resolved_a, line_a = source_index.resolve(None, "tests.python.test_scene.TestA", "test_render")
    resolved_b, line_b = source_index.resolve(None, "tests.python.test_scene.TestB", "test_render")

    assert resolved_a == resolved_b == "tests/python/test_scene.py"
    assert (line_a, line_b) == (2, 6)


def test_parse_junit_rejects_doctype(tmp_path: Path) -> None:
    junit = b"""<?xml version="1.0"?>
<!DOCTYPE testsuites [<!ENTITY expanded "unsafe">]>
<testsuites><testsuite><testcase name="&expanded;" /></testsuite></testsuites>
"""

    with pytest.raises(analyze_ci_tests.AnalysisError, match="DTDs are not allowed"):
        analyze_ci_tests.parse_junit_report(
            "reports/pytest-junit.xml", junit, analyze_ci_tests.SourceIndex(tmp_path)
        )


def test_markdown_calls_out_cache_attribution_limit() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows-release",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url=None,
    )
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_pathtracer[d3d12]",
        status="success",
        observed_seconds=5.0,
        source="tests/python/test_example.py",
        line=7,
    )

    report = analyze_ci_tests.render_markdown(job, [result], top=10)

    assert "persistent-cache hits or misses" in report
    assert "fast result can be cache-warmed" in report
    assert "tests/python/test_example.py:7" in report


def test_apply_telemetry_sidecar_enriches_matching_junit_result() -> None:
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.render.test_scene",
        name="test_render[d3d12]",
        status="success",
        observed_seconds=5.0,
        source="tests/python/render/test_scene.py",
        line=10,
    )
    telemetry = b"""{
      "schema_version": 1,
      "tests": [{
        "nodeid": "tests/python/render/test_scene.py::TestScene::test_render[d3d12]",
        "worker": "gw2",
        "durations_seconds": {"setup": 0.5, "call": 4.0, "teardown": 0.25},
        "shader_compilations": [
          {"cached": true, "compile_seconds": 0.1},
          {"cached": false, "compile_seconds": 0.2}
        ],
        "pipeline_creations": []
      }]
    }"""

    enriched = analyze_ci_tests.apply_telemetry_sidecar(
        [result], analyze_ci_tests.parse_telemetry_sidecar(telemetry)
    )

    assert enriched[0].worker == "gw2"
    assert enriched[0].setup_seconds == 0.5
    assert enriched[0].call_seconds == 4.0
    assert enriched[0].teardown_seconds == 0.25
    assert enriched[0].cached_shader_compilations == 1
    assert enriched[0].cold_shader_compilations == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("durations_seconds", {"call": "slow"}),
        ("shader_compilations", [{"compile_seconds": "slow"}]),
        ("rss_bytes", {"before": "large"}),
    ],
)
def test_parse_telemetry_sidecar_rejects_invalid_numeric_fields(field: str, value: object) -> None:
    payload = {"schema_version": 1, "tests": [{"nodeid": "test_example.py::test", field: value}]}

    with pytest.raises(analyze_ci_tests.AnalysisError, match="test telemetry record 0"):
        analyze_ci_tests.parse_telemetry_sidecar(json.dumps(payload).encode())


def test_apply_telemetry_sidecar_distinguishes_same_test_name_in_classes() -> None:
    results = [
        analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname=f"tests.python.render.test_scene.{class_name}",
            name="test_render[d3d12]",
            status="success",
            observed_seconds=5.0,
            source="tests/python/render/test_scene.py",
            line=10,
        )
        for class_name in ("TestA", "TestB")
    ]
    telemetry = b"""{
      "schema_version": 1,
      "tests": [
        {"nodeid": "tests/python/render/test_scene.py::TestA::test_render[d3d12]",
         "worker": "gw0", "shader_compilations": [{"cached": true}]},
        {"nodeid": "tests/python/render/test_scene.py::TestB::test_render[d3d12]",
         "worker": "gw1", "shader_compilations": [{"cached": true}, {"cached": true}]}
      ]
    }"""

    enriched = analyze_ci_tests.apply_telemetry_sidecar(
        results, analyze_ci_tests.parse_telemetry_sidecar(telemetry)
    )

    assert [result.worker for result in enriched] == ["gw0", "gw1"]
    assert [result.cached_shader_compilations for result in enriched] == [1, 2]


def test_markdown_reports_telemetry_when_available() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows-release",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url=None,
    )
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_pathtracer[d3d12]",
        status="success",
        observed_seconds=5.0,
        source="tests/python/test_example.py",
        line=7,
        telemetry={
            "worker": "gw0",
            "durations_seconds": {
                "setup": 0.1,
                "call": 4.8,
                "teardown": 0.1,
                "wall": 5.1,
            },
            "shader_compilations": [
                {
                    "phase": "call",
                    "backend": "d3d12",
                    "key": "sha1:cold",
                    "cached": False,
                    "compile_seconds": 2.0,
                },
                {
                    "phase": "call",
                    "backend": "d3d12",
                    "key": "sha1:warm",
                    "cached": True,
                    "compile_seconds": 0.0,
                },
            ],
            "resource_leaks": {"devices_forcibly_closed": 1},
        },
    )

    report = analyze_ci_tests.render_markdown(job, [result], top=10)

    assert "Tests with telemetry: 1" in report
    assert "Shader compilations: 1 cached, 1 cold, 2.00 s compiler time" in report
    assert "Top 1 cold shader compilation producers" in report
    assert "0.100s / 4.800s / 0.100s" in report
    assert "exact shader-artifact buckets" in report
    assert "forced device cleanups" in report


def test_hierarchy_accumulates_folders_files_and_relative_times() -> None:
    results = [
        analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.render.test_scene",
            name="test_slow",
            status="success",
            observed_seconds=6.0,
            source="tests/python/render/test_scene.py",
            line=10,
        ),
        analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.render.test_scene",
            name="test_fast",
            status="success",
            observed_seconds=2.0,
            source="tests/python/render/test_scene.py",
            line=20,
        ),
        analyze_ci_tests.TestResult(
            report="reports/native-junit.xml",
            suite="native",
            classname="test_math.cpp",
            name="native_math",
            status="success",
            observed_seconds=2.0,
            source="tests/native/core/test_math.cpp",
            line=30,
        ),
    ]

    root = analyze_ci_tests.build_hierarchy(results)
    tests_folder = next(child for child in root.children if child.name == "tests")
    python_folder = next(child for child in tests_folder.children if child.name == "python")
    render_folder = next(child for child in python_folder.children if child.name == "render")
    test_file = next(child for child in render_folder.children if child.kind == "file")

    assert root.accumulated_seconds == 10.0
    assert root.relative_to_parent == 1.0
    assert python_folder.accumulated_seconds == 8.0
    assert python_folder.relative_to_parent == 0.8
    assert test_file.test_count == 2
    assert test_file.children[0].name == "test_slow"
    assert test_file.children[0].relative_to_parent == 0.75


def test_cache_adjusted_cost_allocates_each_key_once_across_consumers() -> None:
    def result(
        name: str,
        seconds: float,
        compilations: list[dict[str, object]],
        pipelines: list[dict[str, object]],
    ) -> analyze_ci_tests.TestResult:
        return analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.test_cost",
            name=name,
            status="success",
            observed_seconds=seconds,
            source="tests/python/test_cost.py",
            line=1,
            telemetry={
                "shader_compilations": compilations,
                "pipeline_creations": pipelines,
            },
        )

    shared_shader_cold_a = {
        "backend": "d3d12",
        "key": "sha1:shared-shader",
        "cached": False,
        "compile_seconds": 6.0,
    }
    shared_shader_cold_b = {
        **shared_shader_cold_a,
        "compile_seconds": 8.0,
    }
    shared_shader_cached = {
        **shared_shader_cold_a,
        "cached": True,
        "compile_seconds": 0.0,
    }
    shared_pipeline_cold = {
        "backend": "d3d12",
        "type": "compute",
        "key": "sha1:shared-pipeline",
        "cached": False,
        "create_seconds": 4.0,
    }
    shared_pipeline_cached = {
        **shared_pipeline_cold,
        "cached": True,
        "create_seconds": 1.0,
    }
    unique_shader_cold = {
        "backend": "d3d12",
        "key": "sha1:unique-shader",
        "cached": False,
        "compile_seconds": 2.0,
    }
    unkeyed_pipeline = {
        "backend": "cuda",
        "type": "compute",
        "cached": False,
        "create_seconds": 3.0,
    }
    results = [
        result(
            "test_cold",
            20.0,
            [shared_shader_cold_a, shared_shader_cold_b],
            [shared_pipeline_cold],
        ),
        result(
            "test_cached",
            5.0,
            [shared_shader_cached],
            [shared_pipeline_cached],
        ),
        result("test_unique", 8.0, [unique_shader_cold], [unkeyed_pipeline]),
    ]

    analysis = analyze_ci_tests.build_cache_cost_analysis(results)

    assert analysis.shader_artifact_cost_seconds == {
        "d3d12:sha1:shared-shader": 7.0,
        "d3d12:sha1:unique-shader": 2.0,
    }
    assert analysis.pipeline_cached_baseline_seconds["d3d12:sha1:shared-pipeline"] == pytest.approx(
        1.0
    )
    assert analysis.pipeline_artifact_cost_seconds == {"d3d12:sha1:shared-pipeline": 3.0}
    assert analysis.tests[0] == analyze_ci_tests.CacheAdjustedTestCost(
        measured_seconds=20.0,
        estimated_warm_seconds=3.0,
        observed_cold_shader_seconds=14.0,
        observed_keyed_pipeline_miss_seconds=3.0,
        allocated_shader_seconds=3.5,
        allocated_pipeline_seconds=1.5,
        cache_adjusted_seconds=8.0,
    )
    assert analysis.tests[1].cache_adjusted_seconds == pytest.approx(10.0)
    assert analysis.tests[2].estimated_warm_seconds == pytest.approx(6.0)
    assert analysis.tests[2].allocated_pipeline_seconds == 0.0
    assert sum(cost.cache_adjusted_seconds for cost in analysis.tests) == pytest.approx(26.0)

    hierarchy = analyze_ci_tests.build_hierarchy(results, analysis)
    buckets = analyze_ci_tests.build_shader_buckets(results, analysis)
    assert hierarchy.cache_adjusted_seconds == pytest.approx(26.0)
    assert sum(bucket.cache_adjusted_seconds for bucket in buckets) == pytest.approx(26.0)
    shared_bucket = next(bucket for bucket in buckets if len(bucket.test_indices) == 2)
    assert shared_bucket.cache_adjusted_seconds == pytest.approx(18.0)
    assert shared_bucket.keyed_pipeline_cache_misses == 1
    assert shared_bucket.unkeyed_pipeline_creations == 0
    unique_bucket = next(bucket for bucket in buckets if len(bucket.test_indices) == 1)
    assert unique_bucket.unkeyed_pipeline_creations == 1


def test_shader_buckets_group_exact_backend_and_key_sets() -> None:
    def result(
        name: str,
        seconds: float,
        compilations: list[dict[str, object]],
        pipelines: list[dict[str, object]] | None = None,
    ) -> analyze_ci_tests.TestResult:
        return analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.test_example",
            name=name,
            status="success",
            observed_seconds=seconds,
            source="tests/python/test_example.py",
            line=1,
            telemetry={
                "shader_compilations": compilations,
                "pipeline_creations": pipelines or [],
            },
        )

    key_a = {
        "backend": "d3d12",
        "key": "sha1:a",
        "cached": True,
        "compile_seconds": 0.0,
    }
    key_b = {
        "backend": "d3d12",
        "key": "sha1:b",
        "cached": False,
        "compile_seconds": 1.5,
    }
    pipeline_a = {
        "backend": "d3d12",
        "key": "sha1:pipeline-a",
        "cached": True,
        "create_seconds": 0.1,
    }
    pipeline_b = {
        "backend": "d3d12",
        "key": "sha1:pipeline-b",
        "cached": False,
        "create_seconds": 0.8,
    }
    results = [
        result("test_one", 4.0, [key_a, key_b], [pipeline_a]),
        result("test_two", 2.0, [key_b, key_a], [pipeline_b]),
        result("test_three", 1.0, [key_a]),
        result("test_no_shaders", 10.0, []),
    ]

    buckets = analyze_ci_tests.build_shader_buckets(results)

    assert len(buckets) == 2
    assert buckets[0].shader_keys == ("d3d12:sha1:a", "d3d12:sha1:b")
    assert buckets[0].test_indices == (0, 1)
    assert buckets[0].observed_seconds == 6.0
    assert (buckets[0].cached_compilations, buckets[0].cold_compilations) == (2, 2)
    assert buckets[0].compile_seconds == 3.0
    assert buckets[0].pipeline_keys == (
        "d3d12:sha1:pipeline-a",
        "d3d12:sha1:pipeline-b",
    )
    assert (
        buckets[0].cached_pipeline_creations,
        buckets[0].cold_pipeline_creations,
    ) == (1, 1)
    assert buckets[0].pipeline_create_seconds == 0.9


def test_json_contains_hierarchy_and_compact_sidecar_information() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows-release",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url=None,
    )
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_render",
        status="success",
        observed_seconds=5.0,
        source="tests/python/test_example.py",
        line=7,
    )
    sidecar = analyze_ci_tests.parse_telemetry_sidecar(
        b"""{
          "schema_version": 1,
          "run_id": "run-1",
          "workers": ["gw0"],
          "configuration": {"shader_cache_enabled": true, "device_cache_policy": "file"},
          "tests": [{
            "nodeid": "tests/python/test_example.py::test_render",
            "worker": "gw0",
            "durations_seconds": {"call": 5.0, "wall": 5.1},
            "shader_compilations": [{
              "backend": "d3d12", "key": "sha1:abc",
              "cached": false, "compile_seconds": 1.25
            }],
            "pipeline_creations": [{
              "backend": "d3d12", "key": "sha1:pipeline",
              "cached": true, "create_seconds": 0.2
            }],
            "rss_bytes": {"before": 100, "after_cleanup": 120},
            "resource_leaks": {
              "devices_forcibly_closed": 1,
              "device_labels": ["uncached-device"]
            }
          }]
        }"""
    )
    enriched = analyze_ci_tests.apply_telemetry_sidecar([result], sidecar)

    report = json.loads(analyze_ci_tests.render_json(job, enriched, sidecar))

    assert report["schema_version"] == 1
    assert report["telemetry"]["metadata"]["run_id"] == "run-1"
    assert report["telemetry"]["matched_test_count"] == 1
    assert report["hierarchy"]["accumulated_seconds"] == 5.0
    test_telemetry = report["tests"][0]["telemetry"]
    assert test_telemetry["shader_compilations"][0]["key"] == "sha1:abc"
    assert test_telemetry["resource_leaks"]["devices_forcibly_closed"] == 1
    assert "device_events" not in test_telemetry
    assert report["summary"]["cold_shader_compilations"] == 1
    assert report["summary"]["shader_compile_seconds"] == 1.25
    assert report["summary"]["cached_pipeline_creations"] == 1
    assert report["summary"]["cold_pipeline_creations"] == 0
    assert report["summary"]["keyed_pipeline_cache_misses"] == 0
    assert report["summary"]["unkeyed_pipeline_creations"] == 0
    assert report["summary"]["pipeline_create_seconds"] == 0.2
    assert report["summary"]["cache_adjusted_seconds"] == pytest.approx(5.0)
    assert report["cache_cost_model"]["kind"] == "per_key_equal_allocation"
    assert (
        report["cache_cost_model"]["unkeyed_pipeline_work"] == "retained_in_estimated_warm_seconds"
    )
    assert report["tests"][0]["cache_cost"]["allocated_shader_seconds"] == 1.25
    assert report["shader_buckets"][0]["shader_keys"] == ["d3d12:sha1:abc"]
    assert report["shader_buckets"][0]["pipeline_keys"] == ["d3d12:sha1:pipeline"]
    markdown = analyze_ci_tests.render_markdown(job, enriched, top=10, telemetry=sidecar)
    assert "Device cache policy: file" in markdown
    html_report = analyze_ci_tests.render_html(job, enriched, telemetry=sidecar)
    assert '<div class="label">Telemetry</div>' not in html_report
    assert "1/1 matched" not in html_report


def test_json_counts_telemetry_without_worker_attribution() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows-release",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url=None,
    )
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_render",
        status="success",
        observed_seconds=5.0,
        source="tests/python/test_example.py",
        line=7,
        telemetry={"shader_cache_hits": 3, "shader_cache_misses": 1},
    )

    report = json.loads(analyze_ci_tests.render_json(job, [result]))

    assert report["summary"]["telemetry_test_count"] == 1
    assert report["interpretation"]["shader_attribution_available"] is False
    assert report["interpretation"]["xdist_worker_attribution_available"] is False


def test_html_renders_hierarchy_bars_and_escaped_telemetry() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows<release>",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url="https://gitlab.example.com/job/10?a=1&b=2",
    )
    results = [
        analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.test_example",
            name="test_<unsafe>&slow",
            status="success",
            observed_seconds=3.0,
            source="tests/python/test_example.py",
            line=7,
            telemetry={
                "nodeid": "tests/python/test_example.py::test_<unsafe>&slow",
                "worker": "gw0",
                "durations_seconds": {"call": 3.0, "wall": 3.1},
                "shader_compilations": [
                    {
                        "phase": "call",
                        "backend": "d3d12",
                        "key": "sha1:abc<unsafe>",
                        "entry_point": "main",
                        "cached": False,
                        "compile_seconds": 2.0,
                    }
                ],
                "pipeline_creations": [
                    {
                        "phase": "call",
                        "backend": "d3d12",
                        "key": "sha1:pipeline<unsafe>",
                        "type": "compute",
                        "cached": True,
                        "create_seconds": 0.25,
                    }
                ],
            },
        ),
        analyze_ci_tests.TestResult(
            report="reports/pytest-junit.xml",
            suite="pytest",
            classname="tests.python.test_example",
            name="test_fast",
            status="success",
            observed_seconds=1.0,
            source="tests/python/test_example.py",
            line=8,
        ),
    ]

    report = analyze_ci_tests.render_html(job, results)

    assert report.startswith("<!doctype html>")
    assert "test_example.py" in report
    assert "test_&lt;unsafe&gt;&amp;slow" in report
    assert "test_<unsafe>&slow" not in report
    assert 'style="width:75.000%"' in report
    assert "0/1" in report
    assert "Exact shader-artifact buckets" in report
    assert '<details class="panel collapsible-panel"' in report
    assert report.index("Folder, file, or test") < report.index("Exact shader-artifact buckets")
    assert "sha1:abc&lt;unsafe&gt;" in report
    assert "sha1:pipeline&lt;unsafe&gt;" in report
    assert "gw0" in report
    assert "Expand all" in report
    assert 'data-metric="adjusted"' in report
    assert 'data-adjusted-seconds="1"' in report
    assert 'window.location.hash === "#cache-adjusted"' in report
    assert "Estimated warm execution" in report
    assert "\x15b8" not in report


def test_system_sidecar_enriches_checkpoints_and_all_report_formats() -> None:
    job = analyze_ci_tests.Job(
        id=10,
        name="windows-release",
        status="success",
        duration=20.0,
        queued_duration=1.0,
        web_url=None,
    )
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_render",
        status="success",
        observed_seconds=5.0,
        source="tests/python/test_example.py",
        line=7,
        telemetry={
            "nodeid": "tests/python/test_example.py::test_render",
            "worker": "gw0",
            "pid": 42,
            "started_at_unix_ns": 100,
            "finished_at_unix_ns": 200,
            "rss_bytes": {"before": 90, "after_cleanup": 125},
        },
    )
    sidecar = analyze_ci_tests.parse_system_sidecar(
        b"""{
          "schema_version": 1,
          "started_at_unix_ns": 100,
          "host": {
            "sample_interval_seconds": 0.25,
            "timestamps_ns": [0, 250000000],
            "system": {
              "memory_total_bytes": 1000,
              "cpu_utilization": [null, 0.5],
              "memory_available_bytes": [400, 300],
              "memory_used_bytes": [600, 700],
              "swap_used_bytes": [10, 20]
            },
            "test_processes": {
              "process_count": [5, 6],
              "cpu_user_ns": [100000000, 300000000],
              "cpu_system_ns": [20000000, 70000000],
              "rss_bytes": [300, 400],
              "thread_count": [20, 24],
              "handle_count": [100, 130]
            }
          },
          "gpu": {
            "sample_interval_seconds": 1.0,
            "timestamps_ns": [0, 1000000000],
            "devices": {
              "GPU-abc": {
                "index": 0,
                "name": "RTX Test",
                "memory_total_bytes": 1000,
                "utilization": [0.5, 0.75],
                "memory_used_bytes": [500, 800]
              }
            }
          }
        }"""
    )

    enriched = analyze_ci_tests.apply_test_rss_checkpoints([result])

    assert enriched[0].worker_rss_retained_bytes == 35

    markdown = analyze_ci_tests.render_markdown(job, enriched, top=10, system_telemetry=sidecar)
    assert "Average system CPU utilization: 50.0%" in markdown
    assert "Peak system memory used" in markdown
    assert "Peak test-process handle count: 130" in markdown
    assert "GPU 0 (RTX Test): 62.5% average, 75.0% peak" in markdown
    assert "Top 1 post-teardown worker RSS increases" in markdown

    json_report = json.loads(analyze_ci_tests.render_json(job, enriched, system_telemetry=sidecar))
    summary = json_report["system_telemetry"]["summary"]
    assert summary["test_process_rss_peak_bytes"] == 400
    assert summary["open_resource_peak"] == 130
    assert summary["gpus"]["GPU-abc"]["memory_used_peak_bytes"] == 800
    assert json_report["tests"][0]["telemetry"]["memory"]["worker_rss_retained_bytes"] == 35

    html_report = analyze_ci_tests.render_html(job, enriched, system_telemetry=sidecar)
    assert "System telemetry timeline" in html_report
    assert "Elapsed time" in html_report
    assert html_report.count('<svg class="telemetry-chart-svg"') == 3
    assert "System, test-process, and GPU memory over elapsed time" in html_report
    assert "System CPU and GPU utilization over elapsed time" in html_report
    assert "Test-process open resources, threads, and processes over elapsed time" in html_report
    assert "System memory used" in html_report
    assert "Test-process RSS" in html_report
    assert "GPU 0 VRAM" in html_report
    assert "System CPU" in html_report
    assert "Test-process handle count" in html_report
    assert "Processes / threads" in html_report
    assert "Post-teardown retained RSS" in html_report
    assert "RTX Test" in html_report


def test_parse_system_sidecar_rejects_invalid_numeric_columns() -> None:
    payload = {
        "schema_version": 1,
        "host": {
            "timestamps_ns": [0],
            "system": {"memory_total_bytes": 1000, "memory_used_bytes": ["large"]},
            "test_processes": {},
        },
        "gpu": {"timestamps_ns": [], "devices": {}},
    }

    with pytest.raises(analyze_ci_tests.AnalysisError, match="not a finite number"):
        analyze_ci_tests.parse_system_sidecar(json.dumps(payload).encode())


def test_test_sidecar_keeps_rss_checkpoints_without_system_sidecar() -> None:
    result = analyze_ci_tests.TestResult(
        report="reports/pytest-junit.xml",
        suite="pytest",
        classname="tests.python.test_example",
        name="test_fast",
        status="success",
        observed_seconds=0.01,
        source="tests/python/test_example.py",
        line=8,
        telemetry={
            "worker": "gw0",
            "pid": 42,
            "started_at_unix_ns": 100,
            "finished_at_unix_ns": 110,
            "rss_bytes": {"before": 100, "after_cleanup": 120},
        },
    )
    enriched = analyze_ci_tests.apply_test_rss_checkpoints([result])

    assert enriched[0].worker_rss_retained_bytes == 20
