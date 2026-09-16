# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch and analyze JUnit timings from one GitLab CI job.

The report deliberately calls the measurements "observed time". JUnit does not
record shader-cache hits, cache ownership, or pytest-xdist worker assignment, so
the timings identify candidates for source inspection rather than isolated test
costs.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import hashlib
import html
import json
import math
import os
import re
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_REPORT_GLOB = "reports/*junit.xml"
DEFAULT_TELEMETRY_PATH = "reports/test-telemetry.json"
DEFAULT_SYSTEM_TELEMETRY_PATH = "reports/system-telemetry.json"
DEFAULT_GITLAB_URL = "https://gitlab.com"
REPORT_SCHEMA_VERSION = 1
MAX_REPORT_BYTES = 100 * 1024 * 1024
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class AnalysisError(RuntimeError):
    """Error that should be presented without a traceback."""


@dataclass(frozen=True)
class PipelineTarget:
    api_url: str
    project: str
    pipeline_id: int


@dataclass(frozen=True)
class Job:
    id: int
    name: str
    status: str
    duration: float | None
    queued_duration: float | None
    web_url: str | None


@dataclass(frozen=True)
class TestResult:
    report: str
    suite: str
    classname: str
    name: str
    status: str
    observed_seconds: float
    source: str | None
    line: int | None
    telemetry: Mapping[str, Any] | None = None

    @property
    def location(self) -> str:
        if self.source is None:
            return self.classname or self.suite
        if self.line is None:
            return self.source
        return f"{self.source}:{self.line}"

    @property
    def logical_name(self) -> str:
        return re.sub(r"\[[^]]*\]$", "", self.name)

    @property
    def worker(self) -> str | None:
        return _telemetry_string(self.telemetry, "worker")

    @property
    def setup_seconds(self) -> float | None:
        return _test_phase_duration(self.telemetry, "setup")

    @property
    def call_seconds(self) -> float | None:
        return _test_phase_duration(self.telemetry, "call")

    @property
    def teardown_seconds(self) -> float | None:
        return _test_phase_duration(self.telemetry, "teardown")

    @property
    def wall_seconds(self) -> float | None:
        durations = _telemetry_value(self.telemetry, "durations_seconds")
        if isinstance(durations, dict):
            return _telemetry_float(durations, "wall")
        return None

    @property
    def shader_compilations(self) -> tuple[Mapping[str, Any], ...]:
        value = _telemetry_value(self.telemetry, "shader_compilations")
        if not isinstance(value, list):
            return ()
        return tuple(compilation for compilation in value if isinstance(compilation, dict))

    @property
    def pipeline_creations(self) -> tuple[Mapping[str, Any], ...]:
        value = _telemetry_value(self.telemetry, "pipeline_creations")
        if not isinstance(value, list):
            return ()
        return tuple(pipeline for pipeline in value if isinstance(pipeline, dict))

    @property
    def cached_shader_compilations(self) -> int | None:
        if self.telemetry is None:
            return None
        return sum(bool(compilation.get("cached")) for compilation in self.shader_compilations)

    @property
    def cold_shader_compilations(self) -> int | None:
        if self.telemetry is None:
            return None
        return sum(not bool(compilation.get("cached")) for compilation in self.shader_compilations)

    @property
    def shader_compile_seconds(self) -> float:
        return sum(
            _telemetry_float(compilation, "compile_seconds") or 0.0
            for compilation in self.shader_compilations
        )

    @property
    def cached_pipeline_creations(self) -> int | None:
        if not isinstance(_telemetry_value(self.telemetry, "pipeline_creations"), list):
            return None
        return sum(bool(pipeline.get("cached")) for pipeline in self.pipeline_creations)

    @property
    def cold_pipeline_creations(self) -> int | None:
        if not isinstance(_telemetry_value(self.telemetry, "pipeline_creations"), list):
            return None
        return sum(not bool(pipeline.get("cached")) for pipeline in self.pipeline_creations)

    @property
    def keyed_pipeline_cache_misses(self) -> int | None:
        if not isinstance(_telemetry_value(self.telemetry, "pipeline_creations"), list):
            return None
        return sum(
            not bool(pipeline.get("cached")) and bool(_telemetry_string(pipeline, "key"))
            for pipeline in self.pipeline_creations
        )

    @property
    def unkeyed_pipeline_creations(self) -> int | None:
        if not isinstance(_telemetry_value(self.telemetry, "pipeline_creations"), list):
            return None
        return sum(
            not bool(pipeline.get("cached")) and _telemetry_string(pipeline, "key") is None
            for pipeline in self.pipeline_creations
        )

    @property
    def pipeline_create_seconds(self) -> float:
        return sum(
            _telemetry_float(pipeline, "create_seconds") or 0.0
            for pipeline in self.pipeline_creations
        )

    @property
    def devices_forcibly_closed(self) -> int:
        leaks = _telemetry_value(self.telemetry, "resource_leaks")
        if not isinstance(leaks, dict):
            return 0
        return _telemetry_int(leaks, "devices_forcibly_closed") or 0

    @property
    def memory(self) -> Mapping[str, Any] | None:
        value = _telemetry_value(self.telemetry, "memory")
        return value if isinstance(value, dict) else None

    @property
    def worker_rss_retained_bytes(self) -> int | None:
        return _telemetry_int(self.memory, "worker_rss_retained_bytes")


@dataclass(frozen=True)
class TelemetrySidecar:
    schema_version: int
    metadata: Mapping[str, Any]
    tests: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class SystemSidecar:
    schema_version: int
    metadata: Mapping[str, Any]
    host: Mapping[str, Any]
    gpu: Mapping[str, Any]


@dataclass(frozen=True)
class CacheAdjustedTestCost:
    measured_seconds: float
    estimated_warm_seconds: float
    observed_cold_shader_seconds: float
    observed_keyed_pipeline_miss_seconds: float
    allocated_shader_seconds: float
    allocated_pipeline_seconds: float
    cache_adjusted_seconds: float


@dataclass(frozen=True)
class CacheCostAnalysis:
    tests: tuple[CacheAdjustedTestCost, ...]
    shader_artifact_cost_seconds: Mapping[str, float]
    pipeline_artifact_cost_seconds: Mapping[str, float]
    pipeline_cached_baseline_seconds: Mapping[str, float]


@dataclass(frozen=True)
class ShaderBucket:
    bucket_id: str
    shader_keys: tuple[str, ...]
    test_indices: tuple[int, ...]
    observed_seconds: float
    cached_compilations: int
    cold_compilations: int
    compile_seconds: float
    pipeline_keys: tuple[str, ...]
    cached_pipeline_creations: int
    cold_pipeline_creations: int
    keyed_pipeline_cache_misses: int
    unkeyed_pipeline_creations: int
    pipeline_create_seconds: float
    estimated_warm_seconds: float
    allocated_shader_seconds: float
    allocated_pipeline_seconds: float
    cache_adjusted_seconds: float


@dataclass(frozen=True)
class HierarchyNode:
    kind: str
    name: str
    path: str
    accumulated_seconds: float
    relative_to_parent: float
    cache_adjusted_seconds: float
    cache_adjusted_relative_to_parent: float
    test_count: int
    telemetry_test_count: int
    cached_shader_compilations: int
    cold_shader_compilations: int
    shader_compile_seconds: float
    devices_forcibly_closed: int
    status_counts: Mapping[str, int]
    children: tuple[HierarchyNode, ...] = ()
    test_index: int | None = None


@dataclass
class _HierarchyBranch:
    kind: str
    name: str
    path: str
    branches: dict[tuple[str, str], _HierarchyBranch] = field(default_factory=dict)
    tests: list[tuple[int, TestResult]] = field(default_factory=list)


@dataclass(frozen=True)
class Aggregate:
    name: str
    count: int
    total_seconds: float
    minimum_seconds: float
    median_seconds: float
    maximum_seconds: float


class SourceIndex:
    """Resolve JUnit identities to repository test source locations."""

    def __init__(self, root: Path):
        super().__init__()
        self._root = root.resolve()
        self._by_basename: dict[str, list[Path]] = {}
        self._python_lines: dict[Path, dict[str, int]] = {}
        self._index_files()

    def _index_files(self) -> None:
        for directory, patterns in (
            (self._root / "tests" / "python", ("*.py", "*.slang")),
            (self._root / "tests" / "native", ("*.cpp", "*.h")),
        ):
            if not directory.is_dir():
                continue
            for pattern in patterns:
                for path in directory.rglob(pattern):
                    self._by_basename.setdefault(path.name, []).append(path)

    def resolve(
        self, file_attribute: str | None, classname: str, test_name: str
    ) -> tuple[str | None, int | None]:
        path = self._resolve_path(file_attribute, classname)
        if path is None:
            return None, None

        line = self._resolve_line(path, classname, test_name)
        try:
            source = path.relative_to(self._root).as_posix()
        except ValueError:
            source = path.as_posix()
        return source, line

    def _resolve_path(self, file_attribute: str | None, classname: str) -> Path | None:
        if file_attribute:
            candidate = Path(file_attribute)
            if not candidate.is_absolute():
                candidate = self._root / candidate
            if candidate.is_file():
                return candidate.resolve()

            matches = self._by_basename.get(Path(file_attribute).name, [])
            if len(matches) == 1:
                return matches[0]

        if classname.endswith(".slang"):
            parts = classname.split(".")[:-1]
            candidate = self._root.joinpath(*parts).with_suffix(".slang")
            if candidate.is_file():
                return candidate.resolve()

        if classname.endswith((".cpp", ".h", ".py", ".slang")):
            matches = self._by_basename.get(Path(classname).name, [])
            if len(matches) == 1:
                return matches[0]

        parts = classname.split(".")
        while parts:
            candidate = self._root.joinpath(*parts).with_suffix(".py")
            if candidate.is_file():
                return candidate.resolve()
            parts.pop()
        return None

    def _resolve_line(self, path: Path, classname: str, test_name: str) -> int | None:
        logical_name = re.sub(r"\[[^]]*\]$", "", test_name)
        if path.suffix == ".py":
            lines = self._python_lines.get(path)
            if lines is None:
                lines = _python_function_lines(path)
                self._python_lines[path] = lines
            classname_parts = classname.split(".")
            for index in range(len(classname_parts)):
                qualified_name = ".".join(classname_parts[index:] + [logical_name])
                if qualified_name in lines:
                    return lines[qualified_name]
            return lines.get(logical_name)

        quoted_name = f'"{logical_name}"'
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
        ):
            if quoted_name in line:
                return line_number
        return None


def _python_function_lines(path: Path) -> dict[str, int]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError, UnicodeError):
        return {}

    lines: dict[str, int] = {}

    def visit(statements: Sequence[ast.stmt], class_names: tuple[str, ...] = ()) -> None:
        for statement in statements:
            if isinstance(statement, ast.ClassDef):
                visit(statement.body, (*class_names, statement.name))
            elif isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                lines.setdefault(statement.name, statement.lineno)
                if class_names:
                    lines[".".join((*class_names, statement.name))] = statement.lineno

    visit(tree.body)
    return lines


class GitLabClient:
    def __init__(self, api_url: str, headers: Mapping[str, str]):
        super().__init__()
        self._api_url = api_url.rstrip("/")
        self._headers = dict(headers)

    def get_json_pages(self, path: str) -> list[dict[str, Any]]:
        separator = "&" if "?" in path else "?"
        page = 1
        results: list[dict[str, Any]] = []
        while True:
            payload, response_headers = self._get_json(f"{path}{separator}per_page=100&page={page}")
            if not isinstance(payload, list):
                raise AnalysisError(f"GitLab returned a non-list response for {path}")
            results.extend(item for item in payload if isinstance(item, dict))
            next_page = response_headers.get("X-Next-Page", "")
            if not next_page:
                return results
            try:
                page = int(next_page)
            except ValueError as exc:
                raise AnalysisError(
                    f"GitLab returned invalid X-Next-Page value {next_page!r}"
                ) from exc

    def get_bytes(self, path: str) -> bytes:
        data, _headers = self._request(path, accept="application/octet-stream")
        return data

    def _get_json(self, path: str) -> tuple[Any, Mapping[str, str]]:
        data, headers = self._request(path, accept="application/json")
        try:
            return json.loads(data), headers
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AnalysisError(f"GitLab returned invalid JSON for {path}") from exc

    def _request(self, path: str, accept: str) -> tuple[bytes, Mapping[str, str]]:
        request = urllib.request.Request(
            f"{self._api_url}/{path.lstrip('/')}",
            headers={"Accept": accept, **self._headers},
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return response.read(), response.headers
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            if len(detail) > 500:
                detail = detail[:500] + "..."
            suffix = f": {detail}" if detail else ""
            raise AnalysisError(f"GitLab request failed with HTTP {exc.code}{suffix}") from exc
        except urllib.error.URLError as exc:
            raise AnalysisError(f"GitLab request failed: {exc.reason}") from exc


def parse_pipeline_target(
    pipeline: str, project: str | None, gitlab_url: str | None
) -> PipelineTarget:
    if pipeline.isdigit():
        resolved_project = project or os.environ.get("CI_PROJECT_ID")
        if not resolved_project:
            raise AnalysisError("--project is required when PIPELINE is an ID")
        server_url = gitlab_url or os.environ.get("CI_SERVER_URL") or DEFAULT_GITLAB_URL
        return PipelineTarget(
            api_url=_api_url(server_url),
            project=resolved_project,
            pipeline_id=int(pipeline),
        )

    parsed = urllib.parse.urlparse(pipeline)
    match = re.fullmatch(r"/(.+)/-/pipelines/(\d+)/?", parsed.path)
    if parsed.scheme not in ("http", "https") or not parsed.netloc or match is None:
        raise AnalysisError("PIPELINE must be a numeric ID or a URL ending in /-/pipelines/<id>")
    inferred_server_url = f"{parsed.scheme}://{parsed.netloc}"
    return PipelineTarget(
        api_url=_api_url(gitlab_url or inferred_server_url),
        project=project or urllib.parse.unquote(match.group(1)),
        pipeline_id=int(match.group(2)),
    )


def _api_url(server_url: str) -> str:
    normalized = server_url.rstrip("/")
    if normalized.endswith("/api/v4"):
        return normalized
    return normalized + "/api/v4"


def authentication_headers(token_env: str | None, api_url: str) -> dict[str, str]:
    if token_env:
        token = os.environ.get(token_env)
        if not token:
            raise AnalysisError(f"environment variable {token_env!r} is not set")
        header = "JOB-TOKEN" if token_env == "CI_JOB_TOKEN" else "PRIVATE-TOKEN"
        return {header: token}

    ci_server_url = os.environ.get("CI_SERVER_URL")
    trusted_origin = _url_origin(ci_server_url) if ci_server_url else None
    if trusted_origin is None or trusted_origin != _url_origin(api_url):
        return {}

    for environment_name, header in (
        ("GITLAB_TOKEN", "PRIVATE-TOKEN"),
        ("PRIVATE_TOKEN", "PRIVATE-TOKEN"),
        ("CI_JOB_TOKEN", "JOB-TOKEN"),
    ):
        token = os.environ.get(environment_name)
        if token:
            return {header: token}
    return {}


def _url_origin(url: str) -> tuple[str, str, int] | None:
    try:
        parsed = urllib.parse.urlsplit(url)
        if parsed.scheme not in ("http", "https") or parsed.hostname is None:
            return None
        default_port = 443 if parsed.scheme == "https" else 80
        return parsed.scheme, parsed.hostname.lower(), parsed.port or default_port
    except ValueError:
        return None


def select_job(jobs: Sequence[Mapping[str, Any]], job_name: str) -> Job:
    matches = [job for job in jobs if job.get("name") == job_name]
    if not matches:
        available = sorted(str(job.get("name", "")) for job in jobs)
        listing = ", ".join(name for name in available if name)
        raise AnalysisError(f"job {job_name!r} not found; available jobs: {listing}")
    if len(matches) > 1:
        ids = ", ".join(str(job.get("id")) for job in matches)
        raise AnalysisError(f"job name {job_name!r} is ambiguous; matching job IDs: {ids}")

    match = matches[0]
    try:
        return Job(
            id=int(match["id"]),
            name=str(match["name"]),
            status=str(match.get("status", "unknown")),
            duration=_optional_float(match.get("duration")),
            queued_duration=_optional_float(match.get("queued_duration")),
            web_url=str(match["web_url"]) if match.get("web_url") else None,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AnalysisError(f"GitLab returned invalid metadata for job {job_name!r}") from exc


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def extract_reports(artifact_data: bytes, report_globs: Sequence[str]) -> list[tuple[str, bytes]]:
    reports = _extract_artifact_files(artifact_data, report_globs)
    if not reports:
        patterns = ", ".join(repr(pattern) for pattern in report_globs)
        raise AnalysisError(f"no JUnit reports matching {patterns} found in the job artifact")
    return reports


def _extract_artifact_files(
    artifact_data: bytes, patterns: Sequence[str]
) -> list[tuple[str, bytes]]:
    try:
        archive = zipfile.ZipFile(BytesIO(artifact_data))
    except zipfile.BadZipFile as exc:
        raise AnalysisError("the job artifact is not a valid ZIP archive") from exc

    reports: list[tuple[str, bytes]] = []
    with archive:
        for entry in archive.infolist():
            name = PurePosixPath(entry.filename).as_posix()
            if entry.is_dir() or not any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                continue
            if entry.file_size > MAX_REPORT_BYTES:
                raise AnalysisError(f"JUnit report {name!r} exceeds {MAX_REPORT_BYTES} bytes")
            reports.append((name, archive.read(entry)))
    return sorted(reports)


def parse_telemetry_sidecar(data: bytes) -> TelemetrySidecar:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("cannot parse test telemetry JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AnalysisError("unsupported test telemetry schema")
    telemetry_tests = payload.get("tests")
    if not isinstance(telemetry_tests, list):
        raise AnalysisError("test telemetry has no test list")

    tests: list[Mapping[str, Any]] = []
    for index, record in enumerate(telemetry_tests):
        if not isinstance(record, dict):
            raise AnalysisError(f"test telemetry record {index} is not an object")
        _validate_test_telemetry_record(record, index)
        tests.append(record)
    metadata = {
        key: value for key, value in payload.items() if key not in ("schema_version", "tests")
    }
    return TelemetrySidecar(
        schema_version=int(payload["schema_version"]),
        metadata=metadata,
        tests=tuple(tests),
    )


def _validate_test_telemetry_record(record: Mapping[str, Any], index: int) -> None:
    label = f"test telemetry record {index}"
    for key in ("pid", "started_at_unix_ns", "finished_at_unix_ns"):
        _validate_numeric_field(record, key, label, integer=True)

    durations = record.get("durations_seconds")
    if durations is not None:
        if not isinstance(durations, dict):
            raise AnalysisError(f"{label} field 'durations_seconds' is not an object")
        for key in ("setup", "call", "teardown", "wall"):
            _validate_numeric_field(durations, key, f"{label} durations")

    for collection_key, seconds_key in (
        ("shader_compilations", "compile_seconds"),
        ("pipeline_creations", "create_seconds"),
    ):
        collection = record.get(collection_key)
        if collection is None:
            continue
        if not isinstance(collection, list):
            raise AnalysisError(f"{label} field {collection_key!r} is not a list")
        for item_index, item in enumerate(collection):
            if not isinstance(item, dict):
                raise AnalysisError(
                    f"{label} field {collection_key!r} item {item_index} is not an object"
                )
            _validate_numeric_field(
                item, seconds_key, f"{label} field {collection_key!r} item {item_index}"
            )

    rss = record.get("rss_bytes")
    if rss is not None:
        if not isinstance(rss, dict):
            raise AnalysisError(f"{label} field 'rss_bytes' is not an object")
        for key in ("before", "after_cleanup"):
            _validate_numeric_field(rss, key, f"{label} RSS", integer=True)

    leaks = record.get("resource_leaks")
    if leaks is not None:
        if not isinstance(leaks, dict):
            raise AnalysisError(f"{label} field 'resource_leaks' is not an object")
        _validate_numeric_field(
            leaks,
            "devices_forcibly_closed",
            f"{label} resource leaks",
            integer=True,
        )


def _validate_numeric_field(
    values: Mapping[str, Any], key: str, label: str, *, integer: bool = False
) -> None:
    value = values.get(key)
    if value is None:
        return
    valid = type(value) is int if integer else _is_finite_number(value)
    if not valid:
        expected = "integer" if integer else "finite number"
        raise AnalysisError(f"{label} field {key!r} is not a valid {expected}")


def _is_finite_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def apply_telemetry_sidecar(
    results: Sequence[TestResult], sidecar: TelemetrySidecar
) -> list[TestResult]:
    telemetry_tests = sidecar.tests

    by_tail: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    by_leaf: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in telemetry_tests:
        nodeid = record.get("nodeid")
        if not isinstance(nodeid, str):
            continue
        source, separator, test_tail = nodeid.partition("::")
        if not separator:
            continue
        normalized_source = source.replace("\\", "/")
        test_name = test_tail.rsplit("::", maxsplit=1)[-1]
        by_tail.setdefault((normalized_source, test_tail), []).append(record)
        by_leaf.setdefault((normalized_source, test_name), []).append(record)

    enriched: list[TestResult] = []
    used_records: set[int] = set()
    for result in results:
        record: Mapping[str, Any] | None = None
        source = result.source or ""
        for tail in _telemetry_tail_candidates(result):
            matches = [
                candidate
                for candidate in by_tail.get((source, tail), [])
                if id(candidate) not in used_records
            ]
            if len(matches) == 1:
                record = matches[0]
                break
        if record is None:
            matches = [
                candidate
                for candidate in by_leaf.get((source, result.name), [])
                if id(candidate) not in used_records
            ]
            if len(matches) == 1:
                record = matches[0]
        if record is None:
            enriched.append(result)
            continue
        used_records.add(id(record))
        enriched.append(replace(result, telemetry=record))
    return apply_test_rss_checkpoints(enriched)


def _telemetry_tail_candidates(result: TestResult) -> list[str]:
    candidates: list[str] = []
    if result.source and result.source.endswith(".py"):
        module_name = result.source[:-3].replace("/", ".").replace("\\", ".")
        class_prefix = module_name + "."
        if result.classname.startswith(class_prefix):
            class_name = result.classname[len(class_prefix) :]
            if class_name:
                candidates.append(class_name.replace(".", "::") + "::" + result.name)
    candidates.append(result.name)
    return candidates


def parse_system_sidecar(data: bytes) -> SystemSidecar:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnalysisError("cannot parse system telemetry JSON") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise AnalysisError("unsupported system telemetry schema")
    host = payload.get("host")
    gpu = payload.get("gpu")
    if not isinstance(host, dict) or not isinstance(gpu, dict):
        raise AnalysisError("system telemetry has no host or GPU data")
    _validate_system_columns(host, "host")
    _validate_system_columns(gpu, "gpu")
    metadata = {
        key: value for key, value in payload.items() if key not in ("schema_version", "host", "gpu")
    }
    return SystemSidecar(schema_version=1, metadata=metadata, host=host, gpu=gpu)


def _validate_system_columns(section: Mapping[str, Any], label: str) -> None:
    timestamps = section.get("timestamps_ns")
    if not isinstance(timestamps, list):
        raise AnalysisError(f"system telemetry has no {label} timestamp list")
    for index, timestamp in enumerate(timestamps):
        if isinstance(timestamp, bool) or not isinstance(timestamp, int):
            raise AnalysisError(
                f"system telemetry {label} timestamp {index} is not a valid integer"
            )
    sample_count = len(timestamps)
    column_groups: list[Mapping[str, Any]] = []
    if label == "host":
        for key in ("system", "test_processes"):
            group = section.get(key)
            if not isinstance(group, dict):
                raise AnalysisError(f"system telemetry has no host {key} columns")
            column_groups.append(group)
    else:
        devices = section.get("devices")
        if not isinstance(devices, dict):
            raise AnalysisError("system telemetry has no GPU device columns")
        for device_name, device in devices.items():
            if not isinstance(device, dict):
                raise AnalysisError(f"system telemetry GPU device {device_name!r} is not an object")
            _validate_numeric_field(
                device,
                "index",
                f"system telemetry GPU device {device_name!r}",
                integer=True,
            )
            _validate_numeric_field(
                device,
                "memory_total_bytes",
                f"system telemetry GPU device {device_name!r}",
                integer=True,
            )
            column_groups.append(device)
    for group in column_groups:
        for key, values in group.items():
            if not isinstance(values, list):
                continue
            if len(values) != sample_count:
                raise AnalysisError(
                    f"system telemetry column {key!r} has {len(values)} values "
                    f"for {sample_count} {label} timestamps"
                )
            for index, value in enumerate(values):
                if value is not None and not _is_finite_number(value):
                    raise AnalysisError(
                        f"system telemetry column {key!r} value {index} is not a finite number"
                    )

    if label == "host":
        system = section["system"]
        assert isinstance(system, dict)
        _validate_numeric_field(system, "memory_total_bytes", "system telemetry host", integer=True)


def apply_test_rss_checkpoints(results: Sequence[TestResult]) -> list[TestResult]:
    """Derive per-test retained RSS from checkpoints in the test telemetry sidecar."""
    enriched: list[TestResult] = []
    for result in results:
        telemetry = result.telemetry
        if telemetry is None:
            enriched.append(result)
            continue
        rss_checkpoints = _telemetry_value(telemetry, "rss_bytes")
        if isinstance(rss_checkpoints, dict):
            rss_before = _telemetry_int(rss_checkpoints, "before")
            rss_after = _telemetry_int(rss_checkpoints, "after_cleanup")
        else:
            rss_before = None
            rss_after = None
        if rss_before is not None and rss_after is not None:
            memory = {
                "worker_rss_before_bytes": rss_before,
                "worker_rss_after_bytes": rss_after,
                "worker_rss_retained_bytes": rss_after - rss_before,
            }
        else:
            enriched.append(result)
            continue
        enriched_telemetry = dict(telemetry)
        enriched_telemetry["memory"] = memory
        enriched.append(replace(result, telemetry=enriched_telemetry))
    return enriched


def _telemetry_value(telemetry: Mapping[str, Any] | None, key: str) -> Any:
    return telemetry.get(key) if telemetry is not None else None


def _telemetry_string(telemetry: Mapping[str, Any] | None, key: str) -> str | None:
    value = _telemetry_value(telemetry, key)
    return str(value) if value is not None else None


def _telemetry_int(telemetry: Mapping[str, Any] | None, key: str) -> int | None:
    value = _telemetry_value(telemetry, key)
    return int(value) if value is not None else None


def _telemetry_float(telemetry: Mapping[str, Any] | None, key: str) -> float | None:
    value = _telemetry_value(telemetry, key)
    return float(value) if value is not None else None


def _test_phase_duration(telemetry: Mapping[str, Any] | None, phase: str) -> float | None:
    durations = _telemetry_value(telemetry, "durations_seconds")
    if isinstance(durations, dict):
        value = durations.get(phase)
        return float(value) if value is not None else None
    return None


class _JUnitTreeBuilder(ET.TreeBuilder):
    def __init__(self, report_name: str):
        super().__init__()
        self._report_name = report_name

    def doctype(self, name: str, pubid: str | None, system: str | None) -> None:
        raise AnalysisError(
            f"cannot parse JUnit report {self._report_name!r}: DTDs are not allowed"
        )


def parse_junit_report(name: str, data: bytes, source_index: SourceIndex) -> list[TestResult]:
    try:
        parser = ET.XMLParser(target=_JUnitTreeBuilder(name))
        root = ET.fromstring(data, parser=parser)
    except ET.ParseError as exc:
        raise AnalysisError(f"cannot parse JUnit report {name!r}: {exc}") from exc

    results: list[TestResult] = []

    def visit(element: ET.Element, suite_name: str) -> None:
        tag = _local_tag(element.tag)
        if tag == "testsuite":
            suite_name = element.get("name", suite_name)
        if tag == "testcase":
            classname = element.get("classname", "")
            test_name = element.get("name", "<unnamed>")
            source, line = source_index.resolve(element.get("file"), classname, test_name)
            results.append(
                TestResult(
                    report=name,
                    suite=suite_name,
                    classname=classname,
                    name=test_name,
                    status=_test_status(element),
                    observed_seconds=_test_time(element, name, test_name),
                    source=source,
                    line=line,
                )
            )
            return
        for child in element:
            visit(child, suite_name)

    visit(root, "")
    return results


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _test_status(element: ET.Element) -> str:
    child_tags = {_local_tag(child.tag) for child in element}
    for status in ("error", "failure", "skipped"):
        if status in child_tags:
            return status
    return "success"


def _test_time(element: ET.Element, report_name: str, test_name: str) -> float:
    value = element.get("time", "0")
    try:
        seconds = float(value)
    except ValueError as exc:
        raise AnalysisError(
            f"test {test_name!r} in {report_name!r} has invalid time {value!r}"
        ) from exc
    if not math.isfinite(seconds) or seconds < 0:
        raise AnalysisError(f"test {test_name!r} in {report_name!r} has invalid time {value!r}")
    return seconds


def aggregate_results(results: Iterable[TestResult], key: str) -> list[Aggregate]:
    grouped: dict[str, list[float]] = {}
    for result in results:
        if key == "source":
            name = result.source or result.classname or result.suite or "<unknown>"
        elif key == "logical_test":
            name = f"{result.location.rsplit(':', maxsplit=1)[0]}::{result.logical_name}"
        elif key == "report":
            name = result.report
        else:
            raise ValueError(f"unknown aggregate key {key!r}")
        grouped.setdefault(name, []).append(result.observed_seconds)

    aggregates = [
        Aggregate(
            name=name,
            count=len(values),
            total_seconds=sum(values),
            minimum_seconds=min(values),
            median_seconds=statistics.median(values),
            maximum_seconds=max(values),
        )
        for name, values in grouped.items()
    ]
    return sorted(aggregates, key=lambda item: (-item.total_seconds, item.name))


def _artifact_identity(event: Mapping[str, Any]) -> str | None:
    key = _telemetry_string(event, "key")
    if key is None:
        return None
    backend = _telemetry_string(event, "backend") or "unknown"
    return f"{backend}:{key}"


def _pipeline_fallback_identity(event: Mapping[str, Any]) -> tuple[str, str]:
    return (
        _telemetry_string(event, "backend") or "unknown",
        _telemetry_string(event, "type") or "unknown",
    )


def build_cache_cost_analysis(results: Sequence[TestResult]) -> CacheCostAnalysis:
    shader_identities_by_test = [set[str]() for _ in results]
    pipeline_identities_by_test = [set[str]() for _ in results]
    shader_consumers: dict[str, set[int]] = {}
    pipeline_consumers: dict[str, set[int]] = {}
    shader_cold_samples: dict[str, list[float]] = {}
    pipeline_cold_samples: dict[str, list[float]] = {}
    pipeline_cached_samples: dict[str, list[float]] = {}
    pipeline_cached_fallback_samples: dict[tuple[str, str], list[float]] = {}

    for test_index, result in enumerate(results):
        for compilation in result.shader_compilations:
            identity = _artifact_identity(compilation)
            if identity is None:
                continue
            shader_identities_by_test[test_index].add(identity)
            shader_consumers.setdefault(identity, set()).add(test_index)
            if not bool(compilation.get("cached")):
                shader_cold_samples.setdefault(identity, []).append(
                    _telemetry_float(compilation, "compile_seconds") or 0.0
                )

        for pipeline in result.pipeline_creations:
            identity = _artifact_identity(pipeline)
            if identity is None:
                continue
            pipeline_identities_by_test[test_index].add(identity)
            pipeline_consumers.setdefault(identity, set()).add(test_index)
            seconds = _telemetry_float(pipeline, "create_seconds") or 0.0
            if bool(pipeline.get("cached")):
                pipeline_cached_samples.setdefault(identity, []).append(seconds)
                pipeline_cached_fallback_samples.setdefault(
                    _pipeline_fallback_identity(pipeline), []
                ).append(seconds)
            else:
                pipeline_cold_samples.setdefault(identity, []).append(seconds)

    fallback_baselines = {
        identity: statistics.median(samples)
        for identity, samples in pipeline_cached_fallback_samples.items()
    }
    pipeline_baselines: dict[str, float] = {}
    for result in results:
        for pipeline in result.pipeline_creations:
            identity = _artifact_identity(pipeline)
            if identity is None or identity in pipeline_baselines:
                continue
            cached_samples = pipeline_cached_samples.get(identity)
            pipeline_baselines[identity] = (
                statistics.median(cached_samples)
                if cached_samples
                else fallback_baselines.get(_pipeline_fallback_identity(pipeline), 0.0)
            )

    shader_artifact_costs = {
        identity: statistics.median(samples) for identity, samples in shader_cold_samples.items()
    }
    pipeline_artifact_costs = {
        identity: max(0.0, statistics.median(samples) - pipeline_baselines[identity])
        for identity, samples in pipeline_cold_samples.items()
    }

    test_costs: list[CacheAdjustedTestCost] = []
    for test_index, result in enumerate(results):
        observed_cold_shader_seconds = sum(
            _telemetry_float(compilation, "compile_seconds") or 0.0
            for compilation in result.shader_compilations
            if not bool(compilation.get("cached")) and _artifact_identity(compilation) is not None
        )
        observed_keyed_pipeline_miss_seconds = 0.0
        for pipeline in result.pipeline_creations:
            identity = _artifact_identity(pipeline)
            if bool(pipeline.get("cached")) or identity is None:
                continue
            observed_keyed_pipeline_miss_seconds += max(
                0.0,
                (_telemetry_float(pipeline, "create_seconds") or 0.0)
                - pipeline_baselines[identity],
            )
        estimated_warm_seconds = max(
            0.0,
            result.observed_seconds
            - observed_cold_shader_seconds
            - observed_keyed_pipeline_miss_seconds,
        )
        allocated_shader_seconds = sum(
            shader_artifact_costs.get(identity, 0.0) / len(shader_consumers[identity])
            for identity in shader_identities_by_test[test_index]
        )
        allocated_pipeline_seconds = sum(
            pipeline_artifact_costs.get(identity, 0.0) / len(pipeline_consumers[identity])
            for identity in pipeline_identities_by_test[test_index]
        )
        test_costs.append(
            CacheAdjustedTestCost(
                measured_seconds=result.observed_seconds,
                estimated_warm_seconds=estimated_warm_seconds,
                observed_cold_shader_seconds=observed_cold_shader_seconds,
                observed_keyed_pipeline_miss_seconds=(observed_keyed_pipeline_miss_seconds),
                allocated_shader_seconds=allocated_shader_seconds,
                allocated_pipeline_seconds=allocated_pipeline_seconds,
                cache_adjusted_seconds=(
                    estimated_warm_seconds + allocated_shader_seconds + allocated_pipeline_seconds
                ),
            )
        )

    return CacheCostAnalysis(
        tests=tuple(test_costs),
        shader_artifact_cost_seconds=shader_artifact_costs,
        pipeline_artifact_cost_seconds=pipeline_artifact_costs,
        pipeline_cached_baseline_seconds=pipeline_baselines,
    )


def build_shader_buckets(
    results: Sequence[TestResult],
    cost_analysis: CacheCostAnalysis | None = None,
) -> list[ShaderBucket]:
    cost_analysis = cost_analysis or build_cache_cost_analysis(results)
    grouped: dict[tuple[str, ...], list[tuple[int, TestResult]]] = {}
    for test_index, result in enumerate(results):
        shader_keys = tuple(
            sorted(
                {
                    f"{_telemetry_string(compilation, 'backend') or 'unknown'}:{key}"
                    for compilation in result.shader_compilations
                    if (key := _telemetry_string(compilation, "key"))
                }
            )
        )
        if shader_keys:
            grouped.setdefault(shader_keys, []).append((test_index, result))

    buckets: list[ShaderBucket] = []
    for shader_keys, entries in grouped.items():
        pipeline_keys = tuple(
            sorted(
                {
                    f"{_telemetry_string(pipeline, 'backend') or 'unknown'}:{key}"
                    for _, result in entries
                    for pipeline in result.pipeline_creations
                    if (key := _telemetry_string(pipeline, "key"))
                }
            )
        )
        buckets.append(
            ShaderBucket(
                bucket_id=hashlib.sha256("\n".join(shader_keys).encode("utf-8")).hexdigest(),
                shader_keys=shader_keys,
                test_indices=tuple(test_index for test_index, _ in entries),
                observed_seconds=sum(result.observed_seconds for _, result in entries),
                cached_compilations=sum(
                    result.cached_shader_compilations or 0 for _, result in entries
                ),
                cold_compilations=sum(
                    result.cold_shader_compilations or 0 for _, result in entries
                ),
                compile_seconds=sum(result.shader_compile_seconds for _, result in entries),
                pipeline_keys=pipeline_keys,
                cached_pipeline_creations=sum(
                    result.cached_pipeline_creations or 0 for _, result in entries
                ),
                cold_pipeline_creations=sum(
                    result.cold_pipeline_creations or 0 for _, result in entries
                ),
                keyed_pipeline_cache_misses=sum(
                    result.keyed_pipeline_cache_misses or 0 for _, result in entries
                ),
                unkeyed_pipeline_creations=sum(
                    result.unkeyed_pipeline_creations or 0 for _, result in entries
                ),
                pipeline_create_seconds=sum(
                    result.pipeline_create_seconds for _, result in entries
                ),
                estimated_warm_seconds=sum(
                    cost_analysis.tests[test_index].estimated_warm_seconds
                    for test_index, _ in entries
                ),
                allocated_shader_seconds=sum(
                    cost_analysis.tests[test_index].allocated_shader_seconds
                    for test_index, _ in entries
                ),
                allocated_pipeline_seconds=sum(
                    cost_analysis.tests[test_index].allocated_pipeline_seconds
                    for test_index, _ in entries
                ),
                cache_adjusted_seconds=sum(
                    cost_analysis.tests[test_index].cache_adjusted_seconds
                    for test_index, _ in entries
                ),
            )
        )
    return sorted(buckets, key=lambda bucket: (-bucket.observed_seconds, bucket.shader_keys))


def build_hierarchy(
    results: Sequence[TestResult],
    cost_analysis: CacheCostAnalysis | None = None,
) -> HierarchyNode:
    cost_analysis = cost_analysis or build_cache_cost_analysis(results)
    root = _HierarchyBranch(kind="root", name="All tests", path="")
    for test_index, result in enumerate(results):
        source = result.source
        if source is None:
            unresolved_name = result.classname or result.suite or result.report or "unknown"
            source = f"<unresolved>/{unresolved_name}"
        parts = PurePosixPath(source.replace("\\", "/")).parts
        if not parts:
            parts = ("<unresolved>", "unknown")

        branch = root
        current_parts: list[str] = []
        for part in parts[:-1]:
            current_parts.append(part)
            key = ("folder", part)
            branch = branch.branches.setdefault(
                key,
                _HierarchyBranch(
                    kind="folder",
                    name=part,
                    path=PurePosixPath(*current_parts).as_posix(),
                ),
            )

        file_name = parts[-1]
        file_path = PurePosixPath(*parts).as_posix()
        file_branch = branch.branches.setdefault(
            ("file", file_name),
            _HierarchyBranch(kind="file", name=file_name, path=file_path),
        )
        file_branch.tests.append((test_index, result))

    return replace(
        _finalize_branch(root, cost_analysis),
        relative_to_parent=1.0,
        cache_adjusted_relative_to_parent=1.0,
    )


def _finalize_branch(branch: _HierarchyBranch, cost_analysis: CacheCostAnalysis) -> HierarchyNode:
    children = [_finalize_branch(child, cost_analysis) for child in branch.branches.values()]
    children.extend(
        _test_hierarchy_node(test_index, result, cost_analysis.tests[test_index])
        for test_index, result in branch.tests
    )
    accumulated_seconds = sum(child.accumulated_seconds for child in children)
    cache_adjusted_seconds = sum(child.cache_adjusted_seconds for child in children)
    if accumulated_seconds > 0:
        children = [
            replace(
                child,
                relative_to_parent=child.accumulated_seconds / accumulated_seconds,
            )
            for child in children
        ]
    if cache_adjusted_seconds > 0:
        children = [
            replace(
                child,
                cache_adjusted_relative_to_parent=(
                    child.cache_adjusted_seconds / cache_adjusted_seconds
                ),
            )
            for child in children
        ]
    children.sort(key=lambda child: (-child.accumulated_seconds, child.kind, child.name))

    status_counts: dict[str, int] = {}
    for child in children:
        for status, count in child.status_counts.items():
            status_counts[status] = status_counts.get(status, 0) + count

    return HierarchyNode(
        kind=branch.kind,
        name=branch.name,
        path=branch.path,
        accumulated_seconds=accumulated_seconds,
        relative_to_parent=0.0,
        cache_adjusted_seconds=cache_adjusted_seconds,
        cache_adjusted_relative_to_parent=0.0,
        test_count=sum(child.test_count for child in children),
        telemetry_test_count=sum(child.telemetry_test_count for child in children),
        cached_shader_compilations=sum(child.cached_shader_compilations for child in children),
        cold_shader_compilations=sum(child.cold_shader_compilations for child in children),
        shader_compile_seconds=sum(child.shader_compile_seconds for child in children),
        devices_forcibly_closed=sum(child.devices_forcibly_closed for child in children),
        status_counts=status_counts,
        children=tuple(children),
    )


def _test_hierarchy_node(
    test_index: int,
    result: TestResult,
    cost: CacheAdjustedTestCost,
) -> HierarchyNode:
    return HierarchyNode(
        kind="test",
        name=result.name,
        path=f"{result.source or '<unresolved>'}::{result.name}",
        accumulated_seconds=result.observed_seconds,
        relative_to_parent=0.0,
        cache_adjusted_seconds=cost.cache_adjusted_seconds,
        cache_adjusted_relative_to_parent=0.0,
        test_count=1,
        telemetry_test_count=int(result.telemetry is not None),
        cached_shader_compilations=result.cached_shader_compilations or 0,
        cold_shader_compilations=result.cold_shader_compilations or 0,
        shader_compile_seconds=result.shader_compile_seconds,
        devices_forcibly_closed=result.devices_forcibly_closed,
        status_counts={result.status: 1},
        test_index=test_index,
    )


def render_markdown(
    job: Job,
    results: Sequence[TestResult],
    top: int,
    system_telemetry: SystemSidecar | None = None,
    telemetry: TelemetrySidecar | None = None,
) -> str:
    total = sum(result.observed_seconds for result in results)
    status_counts: dict[str, int] = {}
    for result in results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
    telemetry_results = [result for result in results if result.telemetry is not None]
    total_cached_compilations = sum(
        result.cached_shader_compilations or 0 for result in telemetry_results
    )
    total_cold_compilations = sum(
        result.cold_shader_compilations or 0 for result in telemetry_results
    )
    total_compile_seconds = sum(result.shader_compile_seconds for result in telemetry_results)
    total_cached_pipelines = sum(
        result.cached_pipeline_creations or 0 for result in telemetry_results
    )
    total_keyed_pipeline_misses = sum(
        result.keyed_pipeline_cache_misses or 0 for result in telemetry_results
    )
    total_unkeyed_pipelines = sum(
        result.unkeyed_pipeline_creations or 0 for result in telemetry_results
    )
    total_pipeline_seconds = sum(result.pipeline_create_seconds for result in telemetry_results)
    system_summary = _summarize_system_telemetry(system_telemetry)
    device_cache_policy = _device_cache_policy(telemetry)

    lines = [
        f"# CI test timing analysis: `{job.name}`",
        "",
        f"- Job ID: {job.id}",
        f"- Status: {job.status}",
        f"- Job URL: {job.web_url or 'unknown'}",
        f"- Job duration: {_format_optional_seconds(job.duration)}",
        f"- Queue duration: {_format_optional_seconds(job.queued_duration)}",
        f"- Tests: {len(results)} ({_format_status_counts(status_counts)})",
        f"- Sum of observed test times: {_format_seconds(total)}",
        f"- Tests with telemetry: {len(telemetry_results)}",
        f"- Shader compilations: {total_cached_compilations} cached, {total_cold_compilations} cold, {_format_seconds(total_compile_seconds)} compiler time",
        f"- Pipeline creations: {total_cached_pipelines} cached, {total_keyed_pipeline_misses} keyed misses, {total_unkeyed_pipelines} unkeyed, {_format_seconds(total_pipeline_seconds)} creation time",
    ]
    if device_cache_policy is not None:
        lines.append(f"- Device cache policy: {device_cache_policy}")
    if system_summary is not None:
        lines.extend(
            (
                f"- Average system CPU utilization: {_format_percentage(system_summary['system_cpu_utilization_average'])}",
                f"- Peak system CPU utilization: {_format_percentage(system_summary['system_cpu_utilization_peak'])}",
                f"- Peak system memory used: {_format_bytes(system_summary['system_memory_used_peak_bytes'])}",
                f"- Minimum system memory available: {_format_bytes(system_summary['system_memory_available_min_bytes'])}",
                f"- Peak test-process RSS: {_format_bytes(system_summary['test_process_rss_peak_bytes'])}",
                f"- Peak test-process {str(system_summary['open_resource_kind']).replace('_', ' ')}: {system_summary['open_resource_peak'] if system_summary['open_resource_peak'] is not None else 'not available'}",
            )
        )
        gpus = system_summary.get("gpus")
        if isinstance(gpus, dict):
            for gpu_uuid, gpu in gpus.items():
                if not isinstance(gpu, dict):
                    continue
                lines.append(
                    f"- GPU {gpu.get('index', '?')} ({gpu.get('name') or gpu_uuid}): "
                    f"{_format_percentage(_telemetry_float(gpu, 'utilization_average'))} average, "
                    f"{_format_percentage(_telemetry_float(gpu, 'utilization_peak'))} peak, "
                    f"{_format_bytes(_telemetry_int(gpu, 'memory_used_peak_bytes'))} peak memory"
                )
    lines.extend(
        (
            "",
            "## Interpretation warning",
            "",
            _interpretation_warning(bool(telemetry_results), system_telemetry is not None),
            "",
            f"## Top {min(top, len(results))} individual observations",
            "",
            "| Observed | Setup / call / teardown | Shader cached/cold/compiler | Worker | Test | Source | Status |",
            "| ---: | --- | ---: | --- | --- | --- | --- |",
        )
    )
    for result in sorted(results, key=lambda item: -item.observed_seconds)[:top]:
        lines.append(
            "| "
            + " | ".join(
                (
                    _format_seconds(result.observed_seconds),
                    _format_phases(result),
                    _format_cache(result),
                    result.worker or "-",
                    _escape_table(result.name),
                    _escape_table(result.location),
                    result.status,
                )
            )
            + " |"
        )

    retained_results = sorted(
        (result for result in results if result.worker_rss_retained_bytes is not None),
        key=lambda item: -(item.worker_rss_retained_bytes or 0),
    )[:top]
    if retained_results:
        lines.extend(
            (
                "",
                f"## Top {len(retained_results)} post-teardown worker RSS increases",
                "",
                "| Retained RSS | Worker | Test | Source |",
                "| ---: | --- | --- | --- |",
            )
        )
        for result in retained_results:
            lines.append(
                f"| {_format_signed_bytes(result.worker_rss_retained_bytes)} | "
                f"{result.worker or '-'} | {_escape_table(result.name)} | "
                f"{_escape_table(result.location)} |"
            )

    cold_compilations = sorted(
        (
            result
            for result in telemetry_results
            if result.cold_shader_compilations is not None and result.cold_shader_compilations > 0
        ),
        key=lambda item: (-item.shader_compile_seconds, -item.observed_seconds),
    )[:top]
    if cold_compilations:
        lines.extend(
            (
                "",
                f"## Top {len(cold_compilations)} cold shader compilation producers",
                "",
                "| Compiler | Cold | Cached | Observed | Worker | Test | Source |",
                "| ---: | ---: | ---: | ---: | --- | --- | --- |",
            )
        )
        for result in cold_compilations:
            lines.append(
                f"| {_format_seconds(result.shader_compile_seconds)} | "
                f"{result.cold_shader_compilations} | {result.cached_shader_compilations or 0} | "
                f"{_format_seconds(result.observed_seconds)} | {result.worker} | "
                f"{_escape_table(result.name)} | {_escape_table(result.location)} |"
            )

    shader_buckets = build_shader_buckets(results)[:top]
    if shader_buckets:
        lines.extend(
            (
                "",
                f"## Top {len(shader_buckets)} exact shader-artifact buckets",
                "",
                "Tests in one bucket compiled or loaded the same backend-and-shader-key set. Observed times are summed once per test within the bucket.",
                "",
                "| Observed | Shader compiler | Pipeline creation | Tests | Shaders | Pipelines | Shader cached/cold | Pipeline cached/miss/unkeyed | Bucket |",
                "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            )
        )
        for bucket in shader_buckets:
            lines.append(
                f"| {_format_seconds(bucket.observed_seconds)} | {_format_seconds(bucket.compile_seconds)} | "
                f"{_format_seconds(bucket.pipeline_create_seconds)} | {len(bucket.test_indices)} | "
                f"{len(bucket.shader_keys)} | {len(bucket.pipeline_keys)} | "
                f"{bucket.cached_compilations}/{bucket.cold_compilations} | "
                f"{bucket.cached_pipeline_creations}/{bucket.keyed_pipeline_cache_misses}/"
                f"{bucket.unkeyed_pipeline_creations} | "
                f"`{bucket.bucket_id[:16]}` |"
            )

    forced_cleanups = [result for result in telemetry_results if result.devices_forcibly_closed]
    if forced_cleanups:
        lines.extend(
            (
                "",
                f"## Top {min(top, len(forced_cleanups))} forced device cleanups",
                "",
                "| Devices closed | Observed | Worker | Test |",
                "| ---: | ---: | --- | --- |",
            )
        )
        for result in sorted(
            forced_cleanups,
            key=lambda item: (-item.devices_forcibly_closed, -item.observed_seconds),
        )[:top]:
            lines.append(
                f"| {result.devices_forcibly_closed} | {_format_seconds(result.observed_seconds)} | "
                f"{result.worker} | {_escape_table(result.name)} |"
            )

    executed_results = [result for result in results if result.status != "skipped"]
    logical = aggregate_results(executed_results, "logical_test")[:top]
    lines.extend(
        (
            "",
            f"## Top {min(top, len(logical))} logical tests",
            "",
            "Parameterized variants are combined here. Large min/max spreads are candidates for "
            "backend differences, cache warming, or order-dependent setup and require source-level inspection.",
            "",
            "| Total | Variants | Median | Min | Max | Logical test |",
            "| ---: | ---: | ---: | ---: | ---: | --- |",
        )
    )
    for item in logical:
        lines.append(
            f"| {_format_seconds(item.total_seconds)} | {item.count} | "
            f"{_format_seconds(item.median_seconds)} | {_format_seconds(item.minimum_seconds)} | "
            f"{_format_seconds(item.maximum_seconds)} | {_escape_table(item.name)} |"
        )

    sources = aggregate_results(results, "source")[:top]
    lines.extend(
        (
            "",
            f"## Top {min(top, len(sources))} source files",
            "",
            "| Total | Tests | Median | Max | Source |",
            "| ---: | ---: | ---: | ---: | --- |",
        )
    )
    for item in sources:
        lines.append(
            f"| {_format_seconds(item.total_seconds)} | {item.count} | "
            f"{_format_seconds(item.median_seconds)} | {_format_seconds(item.maximum_seconds)} | "
            f"{_escape_table(item.name)} |"
        )

    reports = aggregate_results(results, "report")
    lines.extend(
        (
            "",
            "## Report totals",
            "",
            "| Observed total | Tests | Report |",
            "| ---: | ---: | --- |",
        )
    )
    for item in reports:
        lines.append(
            f"| {_format_seconds(item.total_seconds)} | {item.count} | {_escape_table(item.name)} |"
        )
    return "\n".join(lines) + "\n"


def render_json(
    job: Job,
    results: Sequence[TestResult],
    telemetry: TelemetrySidecar | None = None,
    system_telemetry: SystemSidecar | None = None,
) -> str:
    telemetry_results = [result for result in results if result.telemetry is not None]
    cost_analysis = build_cache_cost_analysis(results)
    payload = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "job": asdict(job),
        "interpretation": {
            "timing_kind": "observed_junit_duration",
            "shader_attribution_available": any(
                result.shader_compilations for result in telemetry_results
            ),
            "xdist_worker_attribution_available": any(
                result.worker is not None for result in telemetry_results
            ),
        },
        "summary": {
            "test_count": len(results),
            "observed_seconds": sum(result.observed_seconds for result in results),
            "cache_adjusted_seconds": sum(
                cost.cache_adjusted_seconds for cost in cost_analysis.tests
            ),
            "estimated_warm_seconds": sum(
                cost.estimated_warm_seconds for cost in cost_analysis.tests
            ),
            "allocated_shader_seconds": sum(
                cost.allocated_shader_seconds for cost in cost_analysis.tests
            ),
            "allocated_pipeline_seconds": sum(
                cost.allocated_pipeline_seconds for cost in cost_analysis.tests
            ),
            "telemetry_test_count": len(telemetry_results),
            "cached_shader_compilations": sum(
                result.cached_shader_compilations or 0 for result in telemetry_results
            ),
            "cold_shader_compilations": sum(
                result.cold_shader_compilations or 0 for result in telemetry_results
            ),
            "shader_compile_seconds": sum(
                result.shader_compile_seconds for result in telemetry_results
            ),
            "pipeline_creations": sum(
                len(result.pipeline_creations) for result in telemetry_results
            ),
            "cached_pipeline_creations": sum(
                result.cached_pipeline_creations or 0 for result in telemetry_results
            ),
            "cold_pipeline_creations": sum(
                result.cold_pipeline_creations or 0 for result in telemetry_results
            ),
            "keyed_pipeline_cache_misses": sum(
                result.keyed_pipeline_cache_misses or 0 for result in telemetry_results
            ),
            "unkeyed_pipeline_creations": sum(
                result.unkeyed_pipeline_creations or 0 for result in telemetry_results
            ),
            "pipeline_create_seconds": sum(
                result.pipeline_create_seconds for result in telemetry_results
            ),
            "devices_forcibly_closed": sum(
                result.devices_forcibly_closed for result in telemetry_results
            ),
            "status_counts": _status_counts(results),
        },
        "telemetry": _serialize_telemetry_summary(telemetry, telemetry_results),
        "cache_cost_model": {
            "kind": "per_key_equal_allocation",
            "shader_cost_estimator": "median_uncached_compile_seconds",
            "pipeline_cost_estimator": ("median_uncached_create_seconds_minus_cached_baseline"),
            "allocation": "equal_per_distinct_test_consumer",
            "unkeyed_pipeline_work": "retained_in_estimated_warm_seconds",
            "estimated_warm_floor_seconds": 0.0,
            "shader_artifact_count": len(cost_analysis.shader_artifact_cost_seconds),
            "pipeline_artifact_count": len(cost_analysis.pipeline_artifact_cost_seconds),
            "shader_artifact_seconds": sum(cost_analysis.shader_artifact_cost_seconds.values()),
            "pipeline_artifact_seconds": sum(cost_analysis.pipeline_artifact_cost_seconds.values()),
        },
        "system_telemetry": _serialize_system_telemetry(system_telemetry),
        "hierarchy": _serialize_hierarchy(build_hierarchy(results, cost_analysis)),
        "shader_buckets": [
            asdict(bucket) for bucket in build_shader_buckets(results, cost_analysis)
        ],
        "tests": [
            _serialize_test_result(result, cost_analysis.tests[index])
            for index, result in enumerate(results)
        ],
        "logical_tests": [
            asdict(item)
            for item in aggregate_results(
                [result for result in results if result.status != "skipped"],
                "logical_test",
            )
        ],
        "sources": [asdict(item) for item in aggregate_results(results, "source")],
        "reports": [asdict(item) for item in aggregate_results(results, "report")],
    }
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _status_counts(results: Iterable[TestResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status] = counts.get(result.status, 0) + 1
    return counts


def _serialize_telemetry_summary(
    sidecar: TelemetrySidecar | None,
    matched_results: Sequence[TestResult],
) -> dict[str, Any]:
    if sidecar is None:
        return {
            "available": bool(matched_results),
            "metadata_available": False,
            "sidecar_test_count": 0,
            "matched_test_count": len(matched_results),
        }
    return {
        "available": True,
        "metadata_available": True,
        "schema_version": sidecar.schema_version,
        "sidecar_test_count": len(sidecar.tests),
        "matched_test_count": len(matched_results),
        "unmatched_test_count": max(0, len(sidecar.tests) - len(matched_results)),
        "metadata": dict(sidecar.metadata),
    }


def _device_cache_policy(sidecar: TelemetrySidecar | None) -> str | None:
    if sidecar is None:
        return None
    configuration = sidecar.metadata.get("configuration")
    if not isinstance(configuration, dict):
        return None
    policy = configuration.get("device_cache_policy")
    return str(policy) if policy is not None else None


def _serialize_test_result(
    result: TestResult, cost: CacheAdjustedTestCost | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "report": result.report,
        "suite": result.suite,
        "classname": result.classname,
        "name": result.name,
        "logical_name": result.logical_name,
        "status": result.status,
        "observed_seconds": result.observed_seconds,
        "source": result.source,
        "line": result.line,
        "location": result.location,
    }
    if cost is not None:
        payload["cache_cost"] = asdict(cost)
    if result.telemetry is not None:
        payload["telemetry"] = _serialize_test_telemetry(result.telemetry)
    return payload


def _serialize_test_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        key: telemetry[key]
        for key in (
            "nodeid",
            "worker",
            "pid",
            "started_at_unix_ns",
            "finished_at_unix_ns",
            "durations_seconds",
            "rss_bytes",
            "resource_leaks",
            "telemetry_errors",
        )
        if key in telemetry
    }
    shader_compilations = telemetry.get("shader_compilations")
    if isinstance(shader_compilations, list):
        payload["shader_compilations"] = [
            dict(compilation)
            for compilation in shader_compilations
            if isinstance(compilation, dict)
        ]
    pipeline_creations = telemetry.get("pipeline_creations")
    if isinstance(pipeline_creations, list):
        payload["pipeline_creations"] = [
            dict(pipeline) for pipeline in pipeline_creations if isinstance(pipeline, dict)
        ]
    memory = telemetry.get("memory")
    if isinstance(memory, dict):
        payload["memory"] = dict(memory)
    return payload


def _serialize_system_telemetry(sidecar: SystemSidecar | None) -> dict[str, Any]:
    if sidecar is None:
        return {"available": False}
    return {
        "available": True,
        "schema_version": sidecar.schema_version,
        "metadata": dict(sidecar.metadata),
        "summary": _summarize_system_telemetry(sidecar),
        "host": dict(sidecar.host),
        "gpu": dict(sidecar.gpu),
    }


def _numeric_column(group: Mapping[str, Any], key: str) -> list[int | float]:
    values = group.get(key)
    if not isinstance(values, list):
        return []
    return [value for value in values if isinstance(value, (int, float))]


def _summarize_system_telemetry(sidecar: SystemSidecar | None) -> dict[str, Any] | None:
    if sidecar is None:
        return None
    host_timestamps = _numeric_column(sidecar.host, "timestamps_ns")
    system = sidecar.host.get("system")
    test_processes = sidecar.host.get("test_processes")
    if not isinstance(system, dict) or not isinstance(test_processes, dict):
        return None
    cpu_utilization = _numeric_column(system, "cpu_utilization")
    memory_used = _numeric_column(system, "memory_used_bytes")
    memory_available = _numeric_column(system, "memory_available_bytes")
    swap_used = _numeric_column(system, "swap_used_bytes")
    process_rss = _numeric_column(test_processes, "rss_bytes")
    process_count = _numeric_column(test_processes, "process_count")
    thread_count = _numeric_column(test_processes, "thread_count")
    cpu_user = _numeric_column(test_processes, "cpu_user_ns")
    cpu_system = _numeric_column(test_processes, "cpu_system_ns")
    open_resource_key = (
        "handle_count" if isinstance(test_processes.get("handle_count"), list) else "fd_count"
    )
    open_resources = _numeric_column(test_processes, open_resource_key)
    duration_seconds = (
        (host_timestamps[-1] - host_timestamps[0]) / 1e9 if len(host_timestamps) > 1 else 0.0
    )
    process_cpu_seconds = (
        ((cpu_user[-1] - cpu_user[0]) + (cpu_system[-1] - cpu_system[0])) / 1e9
        if len(cpu_user) > 1 and len(cpu_system) > 1
        else 0.0
    )
    gpu_summary: dict[str, Any] = {}
    gpu_devices = sidecar.gpu.get("devices")
    if isinstance(gpu_devices, dict):
        for gpu_uuid, device in gpu_devices.items():
            if not isinstance(device, dict):
                continue
            utilization = _numeric_column(device, "utilization")
            memory = _numeric_column(device, "memory_used_bytes")
            gpu_summary[str(gpu_uuid)] = {
                "index": _telemetry_int(device, "index"),
                "name": _telemetry_string(device, "name"),
                "memory_total_bytes": _telemetry_int(device, "memory_total_bytes"),
                "utilization_average": statistics.fmean(utilization) if utilization else None,
                "utilization_peak": max(utilization) if utilization else None,
                "memory_used_peak_bytes": max(memory) if memory else None,
            }
    return {
        "host_sample_count": len(host_timestamps),
        "gpu_sample_count": len(_numeric_column(sidecar.gpu, "timestamps_ns")),
        "system_memory_total_bytes": _telemetry_int(system, "memory_total_bytes"),
        "system_cpu_utilization_average": (
            statistics.fmean(cpu_utilization) if cpu_utilization else None
        ),
        "system_cpu_utilization_peak": max(cpu_utilization) if cpu_utilization else None,
        "system_memory_used_peak_bytes": max(memory_used) if memory_used else None,
        "system_memory_available_min_bytes": min(memory_available) if memory_available else None,
        "system_swap_used_peak_bytes": max(swap_used) if swap_used else None,
        "test_process_rss_peak_bytes": max(process_rss) if process_rss else None,
        "test_process_count_peak": max(process_count) if process_count else None,
        "test_process_thread_count_peak": max(thread_count) if thread_count else None,
        "test_process_cpu_seconds": process_cpu_seconds,
        "test_process_average_cpu_cores": (
            process_cpu_seconds / duration_seconds if duration_seconds > 0.0 else None
        ),
        "open_resource_kind": open_resource_key,
        "open_resource_start": open_resources[0] if open_resources else None,
        "open_resource_peak": max(open_resources) if open_resources else None,
        "open_resource_end": open_resources[-1] if open_resources else None,
        "open_resource_delta": (
            open_resources[-1] - open_resources[0] if len(open_resources) > 1 else None
        ),
        "gpus": gpu_summary,
    }


def _serialize_hierarchy(node: HierarchyNode) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": node.kind,
        "name": node.name,
        "path": node.path,
        "accumulated_seconds": node.accumulated_seconds,
        "relative_to_parent": node.relative_to_parent,
        "cache_adjusted_seconds": node.cache_adjusted_seconds,
        "cache_adjusted_relative_to_parent": node.cache_adjusted_relative_to_parent,
        "test_count": node.test_count,
        "telemetry_test_count": node.telemetry_test_count,
        "cached_shader_compilations": node.cached_shader_compilations,
        "cold_shader_compilations": node.cold_shader_compilations,
        "shader_compile_seconds": node.shader_compile_seconds,
        "devices_forcibly_closed": node.devices_forcibly_closed,
        "status_counts": dict(node.status_counts),
    }
    if node.test_index is not None:
        payload["test_index"] = node.test_index
    if node.children:
        payload["children"] = [_serialize_hierarchy(child) for child in node.children]
    return payload


def render_html(
    job: Job,
    results: Sequence[TestResult],
    telemetry: TelemetrySidecar | None = None,
    system_telemetry: SystemSidecar | None = None,
) -> str:
    cost_analysis = build_cache_cost_analysis(results)
    hierarchy = build_hierarchy(results, cost_analysis)
    shader_buckets = build_shader_buckets(results, cost_analysis)
    telemetry_results = [result for result in results if result.telemetry is not None]
    cached_pipeline_creations = sum(
        result.cached_pipeline_creations or 0 for result in telemetry_results
    )
    keyed_pipeline_cache_misses = sum(
        result.keyed_pipeline_cache_misses or 0 for result in telemetry_results
    )
    unkeyed_pipeline_creations = sum(
        result.unkeyed_pipeline_creations or 0 for result in telemetry_results
    )
    pipeline_create_seconds = sum(result.pipeline_create_seconds for result in telemetry_results)
    status_counts = _status_counts(results)
    system_summary = _summarize_system_telemetry(system_telemetry)
    device_cache_policy = _device_cache_policy(telemetry)
    job_url = html.escape(job.web_url or "", quote=True)
    job_link = f'<a href="{job_url}">Job {job.id}</a>' if job.web_url else f"Job {job.id}"
    if telemetry is not None:
        unmatched_count = max(0, len(telemetry.tests) - len(telemetry_results))
        telemetry_notice = (
            f"Telemetry join warning: {unmatched_count:,} of {len(telemetry.tests):,} sidecar "
            "records could not be matched to JUnit tests."
            if unmatched_count
            else ""
        )
        telemetry_metadata = _html_sidecar_metadata(telemetry)
    elif telemetry_results:
        telemetry_notice = (
            f"{len(telemetry_results):,} tests contain telemetry; sidecar metadata was not provided "
            "to the renderer."
        )
        telemetry_metadata = ""
    else:
        telemetry_notice = (
            "No telemetry sidecar was present; cache and worker columns are unavailable."
        )
        telemetry_metadata = ""
    telemetry_notice_html = (
        f'<p class="note warning">{html.escape(telemetry_notice)}</p>' if telemetry_notice else ""
    )
    system_cards = ""
    system_panel = ""
    if system_summary is not None:
        system_cards = (
            _html_card(
                "Average system CPU",
                _format_percentage(system_summary["system_cpu_utilization_average"]),
            )
            + _html_card(
                "Peak system used",
                _format_bytes(system_summary["system_memory_used_peak_bytes"]),
            )
            + _html_card(
                "Minimum available",
                _format_bytes(system_summary["system_memory_available_min_bytes"]),
            )
            + _html_card(
                "Peak test-process RSS",
                _format_bytes(system_summary["test_process_rss_peak_bytes"]),
            )
            + _html_card(
                "Peak open resources",
                (
                    str(system_summary["open_resource_peak"])
                    if system_summary["open_resource_peak"] is not None
                    else "not available"
                ),
            )
        )
        assert system_telemetry is not None
        system_panel = _render_html_system_telemetry(system_telemetry, system_summary)
    shader_bucket_panel = _render_html_shader_buckets(shader_buckets, results)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CI test analysis: {html.escape(job.name)}</title>
<style>
:root {{ color-scheme: light dark; --bg: #f5f7fa; --panel: #fff; --text: #172033;
  --muted: #667085; --line: #d9dee8; --accent: #4f6bed; --accent-soft: #dfe6ff;
  --ok: #16835b; --warn: #b35c00; --bad: #c9362b; --shadow: 0 2px 9px #10182814; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg: #11151d; --panel: #191f2b;
  --text: #edf1f7; --muted: #a7b0c0; --line: #30394a; --accent: #8ea2ff;
  --accent-soft: #29345f; --ok: #4bd3a0; --warn: #ffb45d; --bad: #ff746b;
  --shadow: none; }} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 system-ui,
  -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width: 1800px; margin: 0 auto; padding: 28px; }}
h1 {{ margin: 0 0 6px; font-size: 25px; }}
a {{ color: var(--accent); }}
.subtitle, .note {{ color: var(--muted); }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
  gap: 12px; margin: 22px 0; }}
.card, .panel {{ background: var(--panel); border: 1px solid var(--line); border-radius: 9px;
  box-shadow: var(--shadow); }}
.card {{ padding: 13px 15px; }}
.card .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase;
  letter-spacing: .04em; }}
.card .value {{ margin-top: 3px; font-size: 20px; font-variant-numeric: tabular-nums; }}
.panel {{ overflow: hidden; }}
.toolbar {{ display: flex; align-items: center; gap: 8px; padding: 12px 14px;
  border-bottom: 1px solid var(--line); }}
.metric-toggle {{ display:flex; margin-right:auto; padding:2px; border:1px solid var(--line);
  border-radius:7px; background:var(--bg); }}
.metric-toggle button {{ border:0; padding:5px 10px; background:transparent; }}
.metric-toggle button.active {{ color:var(--panel); background:var(--accent); }}
button {{ border: 1px solid var(--line); background: var(--panel); color: var(--text);
  border-radius: 6px; padding: 6px 10px; cursor: pointer; }}
button:hover {{ border-color: var(--accent); }}
.columns, .node-row {{ display: grid; grid-template-columns: minmax(360px, 1fr) 130px
  minmax(190px, .45fr) 95px 135px; gap: 12px; align-items: center; }}
.columns {{ padding: 8px 14px; color: var(--muted); background: var(--bg);
  border-bottom: 1px solid var(--line); font-size: 12px; font-weight: 650; }}
.node-row {{ min-height: 38px; padding: 6px 14px; border-bottom: 1px solid var(--line); }}
.node-row:hover {{ background: color-mix(in srgb, var(--accent-soft) 35%, transparent); }}
summary.node-row {{ cursor: pointer; list-style: none; }}
summary.node-row::-webkit-details-marker {{ display: none; }}
summary.node-row .node-name::before {{ content: ""; display: inline-block; width: 7px;
  height: 7px; margin: 0 10px 1px 2px; border: solid var(--muted);
  border-width: 0 1.5px 1.5px 0; transform: rotate(-45deg); transition: transform .1s ease; }}
details[open] > summary.node-row .node-name::before {{ transform: rotate(45deg); }}
.leaf .node-name::before {{ border: 0; width: 7px; }}
.collapsible-panel > summary {{ display:flex; align-items:center; gap:10px; padding:14px;
  cursor:pointer; list-style:none; font-size:17px; font-weight:650; }}
.collapsible-panel > summary::-webkit-details-marker {{ display:none; }}
.collapsible-panel > summary::before {{ content:""; width:8px; height:8px;
  border:solid var(--muted); border-width:0 1.5px 1.5px 0; transform:rotate(-45deg);
  transition:transform .1s ease; }}
.collapsible-panel[open] > summary::before {{ transform:rotate(45deg); }}
.collapsible-panel .panel-content {{ padding:0 14px 14px; overflow:auto; }}
.warning {{ color:var(--warn); }}
.children {{ margin-left: 18px; border-left: 1px solid var(--line); }}
.node-name {{ min-width: 0; overflow-wrap: anywhere; }}
.kind {{ display: inline-block; margin-right: 7px; padding: 1px 5px; border-radius: 4px;
  color: var(--muted); background: var(--bg); font-size: 10px; text-transform: uppercase; }}
.status {{ margin-left: 7px; font-size: 11px; color: var(--muted); }}
.seconds, .cache, .telemetry {{ font-variant-numeric: tabular-nums; white-space: nowrap; }}
.seconds {{ text-align: right; font-weight: 650; }}
.share {{ display: grid; grid-template-columns: 1fr 49px; gap: 7px; align-items: center;
  color: var(--muted); font-variant-numeric: tabular-nums; }}
.bar {{ height: 8px; background: var(--accent-soft); border-radius: 99px; overflow: hidden; }}
.bar > span {{ display: block; height: 100%; background: var(--accent); border-radius: inherit; }}
.test-detail {{ padding: 12px 28px 16px 31px; border-bottom: 1px solid var(--line);
  background: color-mix(in srgb, var(--bg) 55%, transparent); }}
.detail-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  gap: 8px 20px; margin-bottom: 10px; }}
.detail-grid dt {{ color: var(--muted); font-size: 11px; text-transform: uppercase; }}
.detail-grid dd {{ margin: 1px 0 0; overflow-wrap: anywhere; }}
.telemetry-chart {{ margin-top: 14px; }}
.telemetry-chart + .telemetry-chart {{ padding-top: 14px; border-top: 1px solid var(--line); }}
.telemetry-chart-header {{ display: flex; align-items: baseline; justify-content: space-between;
  gap: 10px 18px; flex-wrap: wrap; margin-bottom: 5px; }}
.telemetry-chart-header h3 {{ margin: 0; font-size: 14px; }}
.telemetry-legend {{ display: flex; justify-content: flex-end; gap: 8px 16px; flex-wrap: wrap;
  color: var(--muted); font-size: 11px; }}
.telemetry-legend span {{ white-space: nowrap; }}
.telemetry-legend i {{ display: inline-block; width: 24px; margin: 0 6px 3px 0;
  border-top: 3px solid; }}
.telemetry-chart-svg {{ display: block; width: 100%; height: 210px; border: 1px solid var(--line);
  background: var(--bg); }}
table {{ width: 100%; border-collapse: collapse; margin-top: 9px; font-size: 12px; }}
th, td {{ padding: 5px 7px; border: 1px solid var(--line); text-align: left; }}
th {{ color: var(--muted); background: var(--panel); }}
.numeric {{ text-align: right; font-variant-numeric: tabular-nums; }}
code {{ font-family: ui-monospace, "Cascadia Code", monospace; font-size: .92em; }}
pre {{ overflow: auto; margin: 0; padding: 12px; background: var(--bg); font-size: 12px; }}
@media (max-width: 900px) {{ .columns {{ display: none; }} .node-row {{ grid-template-columns: 1fr 80px; }}
  .share, .cache, .telemetry {{ display: none; }} .children {{ margin-left: 8px; }} main {{ padding: 14px; }} }}
</style>
</head>
<body>
<main>
  <h1>CI test timing analysis: <code>{html.escape(job.name)}</code></h1>
  <div class="subtitle">{job_link} &middot; status {html.escape(job.status)} &middot;
    observed test durations can overlap across workers</div>
  <div class="cards">
    {_html_card("Job duration", _format_optional_seconds(job.duration))}
    {_html_metric_card("Test time", hierarchy.accumulated_seconds, hierarchy.cache_adjusted_seconds)}
    {_html_card("Tests", f"{len(results):,}")}
    {_html_card("Device cache policy", device_cache_policy or "unknown")}
    {_html_card("Shader compilations", f"{hierarchy.cached_shader_compilations:,} cached / {hierarchy.cold_shader_compilations:,} cold")}
    {_html_card("Shader compiler time", _format_seconds(hierarchy.shader_compile_seconds))}
    {_html_card("Pipeline creations", f"{cached_pipeline_creations:,} cached / {keyed_pipeline_cache_misses:,} misses / {unkeyed_pipeline_creations:,} unkeyed")}
    {_html_card("Pipeline creation time", _format_seconds(pipeline_create_seconds))}
    {_html_card("Forced device cleanup", f"{hierarchy.devices_forcibly_closed:,}")}
    {_html_card("Results", _format_status_counts(status_counts))}
    {system_cards}
  </div>
  <p class="note">{html.escape(_interpretation_warning(bool(telemetry_results), system_telemetry is not None))}</p>
  {telemetry_notice_html}
  <p class="note">Measured is the observed JUnit duration. Cache-adjusted removes directly observed
    keyed cold shader and pipeline work, allocates one estimated miss cost per backend and key equally
    among its reporting tests, and leaves unkeyed pipeline work intrinsic to the test. Bars show the
    selected metric relative to each row's immediate parent.</p>
  {telemetry_metadata}
  {system_panel}
  <section class="panel">
    <div class="toolbar">
      <div class="metric-toggle" role="group" aria-label="Hierarchy cost metric">
        <button type="button" class="active" data-metric="measured">Measured</button>
        <button type="button" data-metric="adjusted" title="Estimated warm execution plus equal shares of keyed shader and pipeline miss costs">Cache-adjusted</button>
      </div>
      <button type="button" id="expand-all">Expand all</button>
      <button type="button" id="collapse-all">Collapse all</button>
    </div>
    <div class="columns"><span>Folder, file, or test</span><span class="numeric" id="metric-column-label">Measured</span>
      <span>Share of parent</span><span>Shader cached/cold</span><span>Telemetry</span></div>
    <div id="test-hierarchy">{_render_html_node(hierarchy, results, cost_analysis, depth=0)}</div>
  </section>
  {shader_bucket_panel}
</main>
<script>
const metricButtons = document.querySelectorAll(".metric-toggle [data-metric]");
function sortMetricChildren(container, metric) {{
  const attribute = metric === "measured" ? "measuredSeconds" : "adjustedSeconds";
  const children = Array.from(container.children);
  children.sort((left, right) => {{
    const difference = Number(right.dataset[attribute]) - Number(left.dataset[attribute]);
    return difference || (left.dataset.sortName || "").localeCompare(right.dataset.sortName || "");
  }});
  children.forEach((child) => container.appendChild(child));
}}
function selectMetric(metric) {{
  const textAttribute = metric === "measured" ? "measuredText" : "adjustedText";
  const widthAttribute = metric === "measured" ? "measuredWidth" : "adjustedWidth";
  metricButtons.forEach((button) => button.classList.toggle("active", button.dataset.metric === metric));
  document.querySelectorAll(".metric-seconds, .metric-share, .metric-card-value").forEach((item) => {{
    item.textContent = item.dataset[textAttribute];
  }});
  document.querySelectorAll(".metric-bar").forEach((item) => {{
    item.style.width = `${{item.dataset[widthAttribute]}}%`;
  }});
  document.getElementById("metric-column-label").textContent =
    metric === "measured" ? "Measured" : "Cache-adjusted";
  const bucketLabel = document.getElementById("bucket-metric-label");
  if (bucketLabel) bucketLabel.textContent =
    metric === "measured" ? "Measured" : "Cache-adjusted";
  document.querySelectorAll("#test-hierarchy .children").forEach((container) =>
    sortMetricChildren(container, metric));
  const bucketBody = document.querySelector("#shader-bucket-table tbody");
  if (bucketBody) sortMetricChildren(bucketBody, metric);
}}
metricButtons.forEach((button) => button.addEventListener("click", () => {{
  const metric = button.dataset.metric;
  selectMetric(metric);
  history.replaceState(null, "", metric === "adjusted" ? "#cache-adjusted" : "#measured");
}}));
document.getElementById("expand-all").addEventListener("click", () =>
  document.querySelectorAll("details").forEach((item) => item.open = true));
document.getElementById("collapse-all").addEventListener("click", () =>
  document.querySelectorAll("details").forEach((item) => item.open = false));
selectMetric(window.location.hash === "#cache-adjusted" ? "adjusted" : "measured");
</script>
</body>
</html>
"""


def _html_card(label: str, value: str) -> str:
    return (
        '<div class="card"><div class="label">'
        + html.escape(label)
        + '</div><div class="value">'
        + html.escape(value)
        + "</div></div>"
    )


def _html_metric_card(label: str, measured: float, adjusted: float) -> str:
    return (
        '<div class="card"><div class="label">'
        + html.escape(label)
        + '</div><div class="value metric-card-value" '
        f'data-measured-text="{html.escape(_format_seconds(measured), quote=True)}" '
        f'data-adjusted-text="{html.escape(_format_seconds(adjusted), quote=True)}">'
        + html.escape(_format_seconds(measured))
        + "</div></div>"
    )


def _html_sidecar_metadata(telemetry: TelemetrySidecar) -> str:
    metadata = json.dumps(dict(telemetry.metadata), indent=2, sort_keys=True)
    return (
        '<details class="panel" style="margin:14px 0">'
        '<summary class="node-row"><span class="node-name">Sidecar metadata</span></summary>'
        f"<pre>{html.escape(metadata)}</pre></details>"
    )


def _render_html_shader_buckets(
    buckets: Sequence[ShaderBucket], results: Sequence[TestResult]
) -> str:
    if not buckets:
        return ""
    rows = ""
    for bucket in buckets[:100]:
        test_names = ", ".join(
            f"{results[index].location}::{results[index].name}" for index in bucket.test_indices
        )
        key_text = "Shaders:\n" + "\n".join(bucket.shader_keys)
        if bucket.pipeline_keys:
            key_text += "\n\nPipelines:\n" + "\n".join(bucket.pipeline_keys)
        rows += (
            f'<tr data-measured-seconds="{bucket.observed_seconds:.12g}" '
            f'data-adjusted-seconds="{bucket.cache_adjusted_seconds:.12g}" '
            f'data-sort-name="{html.escape(bucket.bucket_id, quote=True)}">'
            f"<td><code>{html.escape(bucket.bucket_id[:16])}</code></td>"
            f'<td class="numeric metric-seconds" '
            f'data-measured-text="{html.escape(_format_seconds(bucket.observed_seconds), quote=True)}" '
            f'data-adjusted-text="{html.escape(_format_seconds(bucket.cache_adjusted_seconds), quote=True)}" '
            f'title="Warm {_format_seconds(bucket.estimated_warm_seconds)}; allocated shader '
            f"{_format_seconds(bucket.allocated_shader_seconds)}; allocated pipeline "
            f'{_format_seconds(bucket.allocated_pipeline_seconds)}">'
            f"{html.escape(_format_seconds(bucket.observed_seconds))}</td>"
            f'<td class="numeric">{len(bucket.test_indices):,}</td>'
            f'<td class="numeric">{html.escape(_format_seconds(bucket.compile_seconds))}</td>'
            f'<td class="numeric">{html.escape(_format_seconds(bucket.pipeline_create_seconds))}</td>'
            f'<td class="numeric">{len(bucket.shader_keys):,}</td>'
            f'<td class="numeric">{len(bucket.pipeline_keys):,}</td>'
            f'<td class="numeric">{bucket.cached_compilations:,}/{bucket.cold_compilations:,}</td>'
            f'<td class="numeric">{bucket.cached_pipeline_creations:,}/'
            f"{bucket.keyed_pipeline_cache_misses:,}/{bucket.unkeyed_pipeline_creations:,}</td>"
            f'<td title="{html.escape(key_text, quote=True)}">{html.escape(test_names)}</td>'
            "</tr>"
        )
    truncated = (
        f'<p class="note">Showing the 100 most expensive of {len(buckets):,} buckets.</p>'
        if len(buckets) > 100
        else ""
    )
    return (
        '<details class="panel collapsible-panel" style="margin:14px 0">'
        f"<summary>Exact shader-artifact buckets ({len(buckets):,})</summary>"
        '<div class="panel-content"><p class="note" style="margin-top:0">Tests in one bucket '
        "compiled or loaded the same set of "
        "backend-and-shader-key identities. Pipeline keys and costs are summarized but do not split "
        "shader buckets. Observed test durations are summed once within each bucket.</p>"
        '<table id="shader-bucket-table"><thead><tr><th>Bucket</th><th id="bucket-metric-label">Measured</th><th>Tests</th><th>Shader compiler</th>'
        "<th>Pipeline creation</th><th>Shaders</th><th>Pipelines</th><th>Shader cached/cold</th>"
        "<th>Pipeline cached/miss/unkeyed</th><th>Tests (hover for keys)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>{truncated}</div></details>"
    )


def _render_html_system_telemetry(
    system_telemetry: SystemSidecar, summary: Mapping[str, Any]
) -> str:
    timestamps_value = system_telemetry.host.get("timestamps_ns")
    system = system_telemetry.host.get("system")
    test_processes = system_telemetry.host.get("test_processes")
    assert isinstance(timestamps_value, list)
    assert isinstance(system, dict)
    assert isinstance(test_processes, dict)
    host_timestamps = [int(timestamp) for timestamp in timestamps_value]
    gpu_timestamps_value = system_telemetry.gpu.get("timestamps_ns")
    gpu_timestamps = (
        [int(timestamp) for timestamp in gpu_timestamps_value]
        if isinstance(gpu_timestamps_value, list)
        else []
    )
    all_timestamps = host_timestamps + gpu_timestamps
    start = min(all_timestamps, default=0)
    end = max(all_timestamps, default=start)
    duration = max(1, end - start)
    plot_left = 92.0
    plot_top = 12.0
    plot_width = 816.0
    plot_height = 135.0
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    system_used = system.get("memory_used_bytes")
    process_rss = test_processes.get("rss_bytes")
    assert isinstance(system_used, list)
    assert isinstance(process_rss, list)

    def sample_indices(sample_count: int) -> list[int]:
        indices = list(range(sample_count))
        if len(indices) <= 400:
            return indices
        stride = math.ceil(len(indices) / 400)
        sampled = indices[::stride]
        if sampled[-1] != sample_count - 1:
            sampled.append(sample_count - 1)
        return sampled

    host_sample_indices = sample_indices(len(host_timestamps))
    gpu_sample_indices = sample_indices(len(gpu_timestamps))

    def numeric_peak(columns: Sequence[list[Any]]) -> float:
        return max(
            (
                float(value)
                for column in columns
                for value in column
                if isinstance(value, (int, float)) and math.isfinite(float(value))
            ),
            default=0.0,
        )

    def polyline(
        timestamps: list[int],
        column: list[Any],
        indices: list[int],
        scale_max: float,
        color: str,
        *,
        dashed: bool = False,
    ) -> str:
        segments: list[list[str]] = []
        current: list[str] = []
        for index in indices:
            value = column[index]
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                if current:
                    segments.append(current)
                    current = []
                continue
            x = plot_left + plot_width * (timestamps[index] - start) / duration
            ratio = max(0.0, min(1.0, float(value) / scale_max))
            y = plot_top + plot_height * (1.0 - ratio)
            current.append(f"{x:.2f},{y:.2f}")
        if current:
            segments.append(current)
        dash = ' stroke-dasharray="7 5"' if dashed else ""
        return "".join(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5"{dash} '
            f'points="{" ".join(segment)}"/>'
            for segment in segments
        )

    def axis_value(value: float, kind: str) -> str:
        if kind == "bytes":
            return _format_bytes(round(value))
        if kind == "percentage":
            return _format_percentage(value)
        return f"{round(value):,}"

    def y_axis(scale_max: float, kind: str, *, right: bool = False) -> str:
        result = ""
        for tick in range(5):
            ratio = tick / 4.0
            y = plot_top + plot_height * (1.0 - ratio)
            if not right:
                result += (
                    f'<line x1="{plot_left:.0f}" y1="{y:.2f}" x2="{plot_right:.0f}" '
                    f'y2="{y:.2f}" stroke="var(--line)" stroke-dasharray="3 5"/>'
                )
            label_x = plot_right + 9 if right else plot_left - 9
            anchor = "start" if right else "end"
            result += (
                f'<text x="{label_x:.0f}" y="{y + 4:.2f}" text-anchor="{anchor}" '
                f'fill="var(--muted)" font-size="11">'
                f"{html.escape(axis_value(scale_max * ratio, kind))}</text>"
            )
        return result

    x_axis = ""
    for tick in range(5):
        ratio = tick / 4.0
        x = plot_left + plot_width * ratio
        x_axis += (
            f'<line x1="{x:.2f}" y1="{plot_top:.0f}" x2="{x:.2f}" '
            f'y2="{plot_bottom:.0f}" stroke="var(--line)" stroke-dasharray="3 5"/>'
            f'<text x="{x:.2f}" y="170" text-anchor="middle" fill="var(--muted)" '
            f'font-size="11">{html.escape(_format_axis_seconds(duration * ratio / 1e9))}</text>'
        )

    def legend_item(label: str, color: str, *, dashed: bool = False) -> str:
        style = f"border-top-color:{color}"
        if dashed:
            style += ";border-top-style:dashed"
        return f'<span><i style="{style}"></i>{html.escape(label)}</span>'

    def chart(
        title: str,
        aria_label: str,
        left_label: str,
        left_scale: float,
        left_kind: str,
        legend: str,
        series: str,
        *,
        right_label: str | None = None,
        right_scale: float | None = None,
        right_kind: str = "count",
    ) -> str:
        right_axis = ""
        right_title = ""
        if right_label is not None and right_scale is not None:
            right_axis = y_axis(right_scale, right_kind, right=True)
            right_title = (
                f'<text transform="translate(988 80) rotate(90)" text-anchor="middle" '
                f'fill="var(--muted)" font-size="11">{html.escape(right_label)}</text>'
            )
        return f"""
    <div class="telemetry-chart">
      <div class="telemetry-chart-header">
        <h3>{html.escape(title)}</h3>
        <div class="telemetry-legend">{legend}</div>
      </div>
      <svg class="telemetry-chart-svg" viewBox="0 0 1000 205" role="img"
        aria-label="{html.escape(aria_label, quote=True)}">
        {y_axis(left_scale, left_kind)}
        {right_axis}
        {x_axis}
        <line x1="{plot_left:.0f}" y1="{plot_bottom:.0f}" x2="{plot_right:.0f}"
          y2="{plot_bottom:.0f}" stroke="var(--muted)"/>
        <text transform="translate(14 80) rotate(-90)" text-anchor="middle"
          fill="var(--muted)" font-size="11">{html.escape(left_label)}</text>
        {right_title}
        <text x="{plot_left + plot_width / 2:.0f}" y="198" text-anchor="middle"
          fill="var(--muted)" font-size="11">Elapsed time</text>
        {series}
      </svg>
    </div>"""

    gpu_devices_value = system_telemetry.gpu.get("devices")
    gpu_devices = (
        [
            (str(gpu_uuid), device)
            for gpu_uuid, device in gpu_devices_value.items()
            if isinstance(device, dict)
        ]
        if isinstance(gpu_devices_value, dict)
        else []
    )
    gpu_devices.sort(key=lambda item: (_telemetry_int(item[1], "index") or 0, item[0]))
    gpu_colors = ("var(--warn)", "var(--ok)", "#8b5cf6", "#0891b2")

    gpu_memory_columns: list[list[Any]] = []
    memory_legend = legend_item("System memory used", "var(--bad)") + legend_item(
        "Test-process RSS", "var(--accent)"
    )
    for device_number, (_, device) in enumerate(gpu_devices):
        memory = device.get("memory_used_bytes")
        if not isinstance(memory, list):
            continue
        gpu_memory_columns.append(memory)
        gpu_index = _telemetry_int(device, "index")
        color = gpu_colors[device_number % len(gpu_colors)]
        memory_legend += legend_item(f"GPU {gpu_index} VRAM", color, dashed=True)
    memory_scale = max(
        1.0, math.ceil(numeric_peak([system_used, process_rss, *gpu_memory_columns]) * 1.05)
    )
    memory_series = polyline(
        host_timestamps, system_used, host_sample_indices, memory_scale, "var(--bad)"
    ) + polyline(host_timestamps, process_rss, host_sample_indices, memory_scale, "var(--accent)")
    for device_number, memory in enumerate(gpu_memory_columns):
        memory_series += polyline(
            gpu_timestamps,
            memory,
            gpu_sample_indices,
            memory_scale,
            gpu_colors[device_number % len(gpu_colors)],
            dashed=True,
        )

    cpu_utilization = system.get("cpu_utilization")
    assert isinstance(cpu_utilization, list)
    utilization_legend = legend_item("System CPU", "var(--accent)")
    utilization_series = polyline(
        host_timestamps,
        cpu_utilization,
        host_sample_indices,
        1.0,
        "var(--accent)",
    )
    for device_number, (_, device) in enumerate(gpu_devices):
        utilization = device.get("utilization")
        if not isinstance(utilization, list):
            continue
        gpu_index = _telemetry_int(device, "index")
        color = gpu_colors[device_number % len(gpu_colors)]
        utilization_legend += legend_item(f"GPU {gpu_index}", color)
        utilization_series += polyline(
            gpu_timestamps,
            utilization,
            gpu_sample_indices,
            1.0,
            color,
        )

    open_resource_key = str(summary.get("open_resource_kind", "open_resources"))
    open_resources = test_processes.get(open_resource_key)
    process_count = test_processes.get("process_count")
    thread_count = test_processes.get("thread_count")
    assert isinstance(open_resources, list)
    assert isinstance(process_count, list)
    assert isinstance(thread_count, list)
    open_resource_label = open_resource_key.replace("_", " ")
    resource_scale = max(1.0, math.ceil(numeric_peak([open_resources]) * 1.05))
    process_shape_scale = max(1.0, math.ceil(numeric_peak([process_count, thread_count]) * 1.05))
    resource_legend = (
        legend_item(f"Test-process {open_resource_label}", "var(--bad)")
        + legend_item("Threads", "var(--accent)")
        + legend_item("Processes", "var(--ok)")
    )
    resource_series = (
        polyline(
            host_timestamps,
            open_resources,
            host_sample_indices,
            resource_scale,
            "var(--bad)",
        )
        + polyline(
            host_timestamps,
            thread_count,
            host_sample_indices,
            process_shape_scale,
            "var(--accent)",
        )
        + polyline(
            host_timestamps,
            process_count,
            host_sample_indices,
            process_shape_scale,
            "var(--ok)",
        )
    )

    charts = (
        chart(
            "Memory",
            "System, test-process, and GPU memory over elapsed time",
            "Memory",
            memory_scale,
            "bytes",
            memory_legend,
            memory_series,
        )
        + chart(
            "Utilization",
            "System CPU and GPU utilization over elapsed time",
            "Utilization",
            1.0,
            "percentage",
            utilization_legend,
            utilization_series,
        )
        + chart(
            "Test-process resources",
            "Test-process open resources, threads, and processes over elapsed time",
            open_resource_label.capitalize(),
            resource_scale,
            "count",
            resource_legend,
            resource_series,
            right_label="Processes / threads",
            right_scale=process_shape_scale,
        )
    )

    errors = system_telemetry.metadata.get("errors")
    errors_text = ""
    if isinstance(errors, list) and errors:
        errors_text = (
            f'<p class="note">Sampler errors: {html.escape("; ".join(map(str, errors)))}</p>'
        )
    gpu_rows = ""
    gpus = summary.get("gpus")
    if isinstance(gpus, dict):
        for gpu_uuid, gpu in gpus.items():
            if not isinstance(gpu, dict):
                continue
            gpu_rows += (
                "<tr>"
                f"<td>{html.escape(str(gpu.get('index', '-')))}</td>"
                f"<td>{html.escape(str(gpu.get('name') or gpu_uuid))}</td>"
                f'<td class="numeric">{html.escape(_format_percentage(_telemetry_float(gpu, "utilization_average")))}</td>'
                f'<td class="numeric">{html.escape(_format_percentage(_telemetry_float(gpu, "utilization_peak")))}</td>'
                f'<td class="numeric">{html.escape(_format_bytes(_telemetry_int(gpu, "memory_used_peak_bytes")))}</td>'
                "</tr>"
            )
    gpu_table = (
        "<table><thead><tr><th>GPU</th><th>Name</th><th>Average utilization</th>"
        "<th>Peak utilization</th><th>Peak memory used</th></tr></thead>"
        f"<tbody>{gpu_rows}</tbody></table>"
        if gpu_rows
        else '<p class="note">No GPU samples were available.</p>'
    )
    return f"""
  <section class="panel" style="margin:14px 0;padding:14px">
    <h2 style="margin:0 0 4px;font-size:17px">System telemetry timeline</h2>
    {charts}
    <div class="detail-grid" style="margin-top:10px">
      <div><dt>Host samples</dt><dd>{summary['host_sample_count']:,}</dd></div>
      <div><dt>GPU samples</dt><dd>{summary['gpu_sample_count']:,}</dd></div>
      <div><dt>System total</dt><dd>{html.escape(_format_bytes(_telemetry_int(summary, 'system_memory_total_bytes')))}</dd></div>
      <div><dt>Peak swap/pagefile used</dt><dd>{html.escape(_format_bytes(_telemetry_int(summary, 'system_swap_used_peak_bytes')))}</dd></div>
      <div><dt>Test-process CPU</dt><dd>{html.escape(_format_seconds(_telemetry_float(summary, 'test_process_cpu_seconds') or 0.0))}</dd></div>
      <div><dt>Average test-process cores</dt><dd>{html.escape(_format_optional_float(_telemetry_float(summary, 'test_process_average_cpu_cores')))}</dd></div>
      <div><dt>Peak test processes</dt><dd>{_telemetry_int(summary, 'test_process_count_peak') or 0:,}</dd></div>
      <div><dt>Peak test threads</dt><dd>{_telemetry_int(summary, 'test_process_thread_count_peak') or 0:,}</dd></div>
      <div><dt>Peak {html.escape(open_resource_label)}</dt><dd>{_telemetry_int(summary, 'open_resource_peak') or 0:,}</dd></div>
      <div><dt>{html.escape(open_resource_label.capitalize())} delta</dt><dd>{html.escape(_format_signed_count(_telemetry_int(summary, 'open_resource_delta')))}</dd></div>
    </div>
    {gpu_table}
    {errors_text}
  </section>"""


def _render_html_node(
    node: HierarchyNode,
    results: Sequence[TestResult],
    cost_analysis: CacheCostAnalysis,
    depth: int,
) -> str:
    row = _html_node_row(node)
    node_attributes = (
        f'data-measured-seconds="{node.accumulated_seconds:.12g}" '
        f'data-adjusted-seconds="{node.cache_adjusted_seconds:.12g}" '
        f'data-sort-name="{html.escape(node.name, quote=True)}"'
    )
    if node.kind == "test":
        assert node.test_index is not None
        result = results[node.test_index]
        if result.telemetry is None:
            return (
                f'<div class="node-row leaf" {node_attributes} '
                f'title="{html.escape(node.path, quote=True)}">{row}</div>'
            )
        return (
            f"<details {node_attributes}>"
            f'<summary class="node-row" title="{html.escape(node.path, quote=True)}">{row}</summary>'
            + _html_test_detail(result, cost_analysis.tests[node.test_index])
            + "</details>"
        )

    open_attribute = " open" if depth < 2 else ""
    children = "".join(
        _render_html_node(child, results, cost_analysis, depth + 1) for child in node.children
    )
    return (
        f"<details{open_attribute} {node_attributes}>"
        f'<summary class="node-row" title="{html.escape(node.path, quote=True)}">{row}</summary>'
        f'<div class="children">{children}</div></details>'
    )


def _html_node_row(node: HierarchyNode) -> str:
    relative_percent = max(0.0, min(100.0, node.relative_to_parent * 100.0))
    adjusted_relative_percent = max(0.0, min(100.0, node.cache_adjusted_relative_to_parent * 100.0))
    status = _format_status_counts(node.status_counts)
    if node.kind == "test":
        telemetry_text = "yes" if node.telemetry_test_count else "-"
    else:
        telemetry_text = f"{node.telemetry_test_count:,}/{node.test_count:,} tests"
    shader_text = (
        f"{node.cached_shader_compilations:,}/{node.cold_shader_compilations:,}"
        if node.telemetry_test_count
        else "-"
    )
    return (
        '<span class="node-name">'
        f'<span class="kind">{html.escape(node.kind)}</span>{html.escape(node.name)}'
        f'<span class="status">{html.escape(status)}</span></span>'
        f'<span class="seconds metric-seconds" data-measured-text="{html.escape(_format_seconds(node.accumulated_seconds), quote=True)}" '
        f'data-adjusted-text="{html.escape(_format_seconds(node.cache_adjusted_seconds), quote=True)}">'
        f"{html.escape(_format_seconds(node.accumulated_seconds))}</span>"
        '<span class="share"><span class="bar"><span class="metric-bar" '
        f'data-measured-width="{relative_percent:.3f}" data-adjusted-width="{adjusted_relative_percent:.3f}" '
        f'style="width:{relative_percent:.3f}%"></span></span><span class="metric-share" '
        f'data-measured-text="{relative_percent:.1f}%" data-adjusted-text="{adjusted_relative_percent:.1f}%">'
        f"{relative_percent:.1f}%</span></span>"
        f'<span class="cache">{shader_text}</span>'
        f'<span class="telemetry">{html.escape(telemetry_text)}</span>'
    )


def _html_test_detail(result: TestResult, cost: CacheAdjustedTestCost) -> str:
    telemetry = result.telemetry
    assert telemetry is not None
    details: list[tuple[str, str]] = [
        ("Source", result.location),
        ("Report", result.report),
        ("Worker", result.worker or "-"),
        ("Status", result.status),
        ("Wall time", _format_optional_seconds(result.wall_seconds)),
        ("Measured cost", _format_seconds(cost.measured_seconds)),
        ("Estimated warm execution", _format_seconds(cost.estimated_warm_seconds)),
        (
            "Observed cold shader work",
            _format_seconds(cost.observed_cold_shader_seconds),
        ),
        (
            "Observed keyed pipeline miss work",
            _format_seconds(cost.observed_keyed_pipeline_miss_seconds),
        ),
        ("Allocated shader cost", _format_seconds(cost.allocated_shader_seconds)),
        ("Allocated pipeline cost", _format_seconds(cost.allocated_pipeline_seconds)),
        ("Cache-adjusted cost", _format_seconds(cost.cache_adjusted_seconds)),
        ("PID", str(telemetry.get("pid", "-"))),
        ("Started at Unix ns", str(telemetry.get("started_at_unix_ns", "-"))),
        ("Finished at Unix ns", str(telemetry.get("finished_at_unix_ns", "-"))),
        (
            "Shader keys",
            str(
                len(
                    {
                        key
                        for item in result.shader_compilations
                        if (key := _telemetry_string(item, "key"))
                    }
                )
            ),
        ),
        (
            "Shader compilations",
            f"{result.cached_shader_compilations or 0} cached / "
            f"{result.cold_shader_compilations or 0} cold / "
            f"{_format_seconds(result.shader_compile_seconds)} compiler time",
        ),
        (
            "Pipeline keys",
            str(
                len(
                    {
                        key
                        for item in result.pipeline_creations
                        if (key := _telemetry_string(item, "key"))
                    }
                )
            ),
        ),
        (
            "Pipeline creations",
            f"{result.cached_pipeline_creations or 0} cached / "
            f"{result.keyed_pipeline_cache_misses or 0} misses / "
            f"{result.unkeyed_pipeline_creations or 0} unkeyed / "
            f"{_format_seconds(result.pipeline_create_seconds)} creation time",
        ),
        ("Devices forcibly closed", str(result.devices_forcibly_closed)),
        ("Post-teardown retained RSS", _format_signed_bytes(result.worker_rss_retained_bytes)),
    ]
    detail_html = "".join(
        f"<div><dt>{html.escape(label)}</dt><dd>{html.escape(value)}</dd></div>"
        for label, value in details
    )
    phase_rows = ""
    for phase in ("setup", "call", "teardown"):
        duration = _test_phase_duration(telemetry, phase)
        if duration is None:
            continue
        phase_rows += (
            f"<tr><td>{phase}</td>"
            f'<td class="numeric">{html.escape(_format_seconds(duration))}</td></tr>'
        )
    phase_table = (
        '<table><thead><tr><th>Phase</th><th class="numeric">Duration</th>'
        f"</tr></thead><tbody>{phase_rows}</tbody></table>"
        if phase_rows
        else ""
    )

    shader_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(compilation.get('phase', '-')))}</td>"
        f"<td>{html.escape(str(compilation.get('backend', '-')))}</td>"
        f"<td><code>{html.escape(str(compilation.get('key') or '-'))}</code></td>"
        f"<td>{html.escape(str(compilation.get('entry_point', '-')))}</td>"
        f"<td>{'yes' if bool(compilation.get('cached')) else 'no'}</td>"
        f'<td class="numeric">{html.escape(_format_seconds(_telemetry_float(compilation, "compile_seconds") or 0.0))}</td>'
        "</tr>"
        for compilation in result.shader_compilations
    )
    shader_table = (
        "<table><thead><tr><th>Phase</th><th>Backend</th><th>Shader key</th>"
        "<th>Entry point</th><th>Cached</th><th>Compiler</th></tr></thead>"
        f"<tbody>{shader_rows}</tbody></table>"
        if shader_rows
        else ""
    )

    pipeline_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(pipeline.get('phase', '-')))}</td>"
        f"<td>{html.escape(str(pipeline.get('backend', '-')))}</td>"
        f"<td><code>{html.escape(str(pipeline.get('key') or '-'))}</code></td>"
        f"<td>{html.escape(str(pipeline.get('type', '-')))}</td>"
        f"<td>{'yes' if bool(pipeline.get('cached')) else 'no'}</td>"
        f'<td class="numeric">{html.escape(_format_seconds(_telemetry_float(pipeline, "create_seconds") or 0.0))}</td>'
        "</tr>"
        for pipeline in result.pipeline_creations
    )
    pipeline_table = (
        "<table><thead><tr><th>Phase</th><th>Backend</th><th>Pipeline key</th>"
        "<th>Type</th><th>Cached</th><th>Creation</th></tr></thead>"
        f"<tbody>{pipeline_rows}</tbody></table>"
        if pipeline_rows
        else ""
    )

    return (
        f'<div class="test-detail"><dl class="detail-grid">{detail_html}</dl>'
        f"{phase_table}{shader_table}{pipeline_table}</div>"
    )


def _format_optional_seconds(seconds: float | None) -> str:
    return "unknown" if seconds is None else _format_seconds(seconds)


def _format_axis_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds:.3f} s"
    if seconds < 60:
        return f"{seconds:.2f} s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:04.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {minutes:02d}m {remainder:04.1f}s"


def _format_status_counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{status}={counts[status]}" for status in sorted(counts))


def _escape_table(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def _format_phases(result: TestResult) -> str:
    values = (result.setup_seconds, result.call_seconds, result.teardown_seconds)
    if all(value is None for value in values):
        return "-"
    return " / ".join("-" if value is None else f"{value:.3f}s" for value in values)


def _format_cache(result: TestResult) -> str:
    if result.cached_shader_compilations is None or result.cold_shader_compilations is None:
        return "-"
    return (
        f"{result.cached_shader_compilations}/{result.cold_shader_compilations}/"
        f"{result.shader_compile_seconds:.3f}s"
    )


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "not available"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    size = float(value)
    for unit in units[:-1]:
        if abs(size) < 1024.0:
            return f"{size:.0f} {unit}" if unit in ("B", "KiB") else f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} {units[-1]}"


def _format_signed_bytes(value: int | None) -> str:
    if value is None:
        return "not available"
    sign = "+" if value > 0 else ""
    return sign + _format_bytes(value)


def _format_percentage(value: float | int | None) -> str:
    return "not available" if value is None else f"{float(value) * 100.0:.1f}%"


def _format_optional_float(value: float | None) -> str:
    return "not available" if value is None else f"{value:.2f}"


def _format_signed_count(value: int | None) -> str:
    if value is None:
        return "not available"
    return f"{value:+,}" if value else "0"


def _interpretation_warning(has_telemetry: bool, has_system_telemetry: bool = False) -> str:
    if has_telemetry:
        warning = (
            "The sidecar attributes compilation-report shader and pipeline keys, cold-versus-cached "
            "work, compiler and pipeline timings, pytest phase durations, and forced device cleanup. "
            "Pipeline data is summarized within shader buckets but does not affect their grouping. "
            "In-memory program reuse can "
            "avoid producing a new compilation report, and parallel tests still share worker and GPU "
            "resources, so observed duration remains contextual rather than isolated marginal cost."
        )
    else:
        warning = (
            "These are observed JUnit durations from a shared-cache, parallel test run. JUnit does not "
            "record persistent-cache hits or misses, which test first compiled a shader, or xdist worker "
            "assignment. A slow result can include cache population that benefits later tests, while a "
            "fast result can be cache-warmed. Use this ranking to choose source code for inspection; do "
            "not treat it as isolated or marginal test cost."
        )
    if has_system_telemetry:
        warning += (
            " System telemetry values are sampled observations. CPU, memory, and GPU activity include "
            "other machine activity; summed test-process RSS can double-count shared pages, and retained "
            "RSS or open resources do not by themselves demonstrate a leak."
        )
    return warning


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "pipeline",
        help="numeric pipeline ID or full GitLab pipeline URL",
    )
    parser.add_argument("--job", required=True, help="exact name of the single job to analyze")
    parser.add_argument(
        "--project",
        help="numeric project ID or project path; inferred from a pipeline URL",
    )
    parser.add_argument(
        "--gitlab-url",
        help="GitLab server or API URL; defaults to the pipeline URL host",
    )
    parser.add_argument(
        "--token-env",
        help="environment variable holding a private token (CI_JOB_TOKEN uses JOB-TOKEN)",
    )
    parser.add_argument(
        "--report-glob",
        action="append",
        dest="report_globs",
        help=f"artifact report glob; repeatable (default: {DEFAULT_REPORT_GLOB})",
    )
    parser.add_argument(
        "--telemetry-path",
        default=DEFAULT_TELEMETRY_PATH,
        help=f"optional telemetry artifact path (default: {DEFAULT_TELEMETRY_PATH})",
    )
    parser.add_argument(
        "--system-path",
        default=DEFAULT_SYSTEM_TELEMETRY_PATH,
        help=(
            "optional system telemetry artifact path " f"(default: {DEFAULT_SYSTEM_TELEMETRY_PATH})"
        ),
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=REPOSITORY_ROOT,
        help="repository root used to resolve test source (default: this checkout)",
    )
    parser.add_argument("--top", type=int, default=30, help="number of rows per ranking")
    parser.add_argument("--format", choices=("markdown", "json", "html"), default="markdown")
    parser.add_argument("--output", type=Path, help="write the report to this path")
    return parser


def run(args: argparse.Namespace) -> str:
    if args.top <= 0:
        raise AnalysisError("--top must be greater than zero")
    target = parse_pipeline_target(args.pipeline, args.project, args.gitlab_url)
    client = GitLabClient(target.api_url, authentication_headers(args.token_env, target.api_url))
    encoded_project = urllib.parse.quote(target.project, safe="")
    jobs = client.get_json_pages(
        f"projects/{encoded_project}/pipelines/{target.pipeline_id}/jobs?include_retried=false"
    )
    job = select_job(jobs, args.job)
    artifact_data = client.get_bytes(f"projects/{encoded_project}/jobs/{job.id}/artifacts")
    reports = extract_reports(artifact_data, args.report_globs or [DEFAULT_REPORT_GLOB])

    source_index = SourceIndex(args.source_root)
    results = [
        result
        for report_name, report_data in reports
        for result in parse_junit_report(report_name, report_data, source_index)
    ]
    telemetry: TelemetrySidecar | None = None
    telemetry_files = _extract_artifact_files(artifact_data, [args.telemetry_path])
    if len(telemetry_files) > 1:
        raise AnalysisError(f"multiple telemetry files match {args.telemetry_path!r}")
    if telemetry_files:
        telemetry = parse_telemetry_sidecar(telemetry_files[0][1])
        results = apply_telemetry_sidecar(results, telemetry)
    system_telemetry: SystemSidecar | None = None
    system_files = _extract_artifact_files(artifact_data, [args.system_path])
    if len(system_files) > 1:
        raise AnalysisError(f"multiple system telemetry files match {args.system_path!r}")
    if system_files:
        system_telemetry = parse_system_sidecar(system_files[0][1])
    if not results:
        raise AnalysisError("the selected JUnit reports contain no test cases")

    if args.format == "json":
        return render_json(job, results, telemetry, system_telemetry)
    if args.format == "html":
        return render_html(job, results, telemetry, system_telemetry)
    return render_markdown(job, results, args.top, system_telemetry, telemetry)


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    try:
        output = run(args)
        if args.output:
            args.output.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 0
    except AnalysisError as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
