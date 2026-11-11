# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
BenchView integration for Falcor2 benchmark reporting.

The payload shape, batching, environment telemetry, HTTP client and retry
policy all come from ``slangpy.testing.benchmark``. Falcor2 supplies only its
own project identity and the metric builders that describe what a Falcor2
benchmark measures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, TypedDict

import pytest

from slangpy.testing.benchmark import (
    build_benchview_observation,
    build_benchview_submissions,
    build_metric,
    lock_gpu_clocks,
    unlock_gpu_clocks,
    submit_benchview_submissions,
    benchview_submission_url,
    BenchViewMetric,
    BenchViewObservation,
    BenchViewProject,
    BenchViewSubmission,
    get_machine_info,
    get_commit_info,
)

import falcor2.testing.helpers as helpers


FALCOR2_PROJECT_REPOSITORY = "https://github.com/NVlabs/falcor2"

FALCOR2_PROJECT = BenchViewProject(
    id="falcor2",
    name="Falcor2",
    suite_id="benchmarks",
    suite_name="Falcor2 benchmarks",
    repository=FALCOR2_PROJECT_REPOSITORY,
    producer_name="falcor2-pytest-benchmark",
    producer_version="1.0.0",
)

FALCOR2_SLANG_PROJECT = BenchViewProject(
    id="falcor2-slang",
    name="Falcor2-slang",
    suite_id=FALCOR2_PROJECT.suite_id,
    suite_name=FALCOR2_PROJECT.suite_name,
    repository="https://github.com/shader-slang/slang",
    producer_name=FALCOR2_PROJECT.producer_name,
    producer_version=FALCOR2_PROJECT.producer_version,
)


def benchmark_identity(
    project_id: str, compiler: dict[str, Any] | None
) -> tuple[BenchViewProject, dict[str, Any], dict[str, Any]]:
    """Select the history axis independently of pytest's benchmark selection."""
    project = {
        "falcor2": FALCOR2_PROJECT,
        "falcor2-slang": FALCOR2_SLANG_PROJECT,
    }[project_id]
    falcor = get_commit_info()
    if os.environ.get("CI_COMMIT_SHA") == falcor.get("id") and os.environ.get("CI_COMMIT_BRANCH"):
        falcor["branch"] = os.environ["CI_COMMIT_BRANCH"]
    if compiler is not None and compiler["falcor_commit"] != falcor["id"]:
        raise ValueError("Compiler provenance belongs to a different Falcor2 checkout")
    if project_id == "falcor2-slang":
        if compiler is None:
            raise ValueError("Slang history requires a compiler selected through ci.py setup")
        return project, {"version": falcor["id"]}, compiler
    return project, _get_falcor2_project_info(), falcor


# -----------------------------------------------------------------------------
# Metric builders
# -----------------------------------------------------------------------------


def _sanitize_component_id(raw_id: str) -> str:
    """Sanitize a breakdown component ID to match BenchView's regex: ^[a-z][a-z0-9._-]*$

    Stricter than the test ID pattern, which permits uppercase, ":" and "/".
    Shader program labels contain all three, so they need their own treatment.
    """
    sanitized = re.sub(r"[^a-z0-9._-]+", "-", raw_id.lower()).strip("-")
    sanitized = re.sub(r"^[^a-z]+", "", sanitized)
    return sanitized or "unknown"


def _build_breakdown_metric(
    metric_id: str,
    metric_name: str,
    components: list[tuple[str, str, float]],
) -> Optional[BenchViewMetric]:
    """Build a metric whose total is broken down into named components.

    ``components`` is a list of (id, display name, milliseconds). Components
    that contributed nothing are dropped, and the metric is omitted entirely
    when nothing at all was measured.
    """
    present = [
        {"id": component_id, "name": name, "distribution": {"samples": [value_ms]}}
        for component_id, name, value_ms in components
        if value_ms > 0
    ]
    if not present:
        return None

    total_ms = sum(value_ms for _, _, value_ms in components if value_ms > 0)
    metric = build_metric(metric_id, metric_name, [total_ms])
    metric["breakdown"] = {"mode": "complete", "components": present}
    return metric


def build_compile_time_breakdown_metric(
    *,
    slang_compile_ms: float,
    backend_compile_ms: float,
    pipeline_create_ms: float,
) -> Optional[BenchViewMetric]:
    """Build a compile_time metric broken down by compilation stage.

    Answers "which compilation stage is slow". Returns None when nothing
    compiled.
    """
    return _build_breakdown_metric(
        "compile_time",
        "Compile time",
        [
            ("slang", "Slang frontend", slang_compile_ms),
            ("backend", "Backend codegen", backend_compile_ms),
            ("pipeline", "Pipeline creation", pipeline_create_ms),
        ],
    )


def build_compile_time_by_program_metric(
    program_ms: dict[str, float],
) -> Optional[BenchViewMetric]:
    """Build a compile_time_by_program metric broken down by shader program.

    A BenchView metric carries at most one breakdown, so this exists alongside
    ``compile_time`` rather than replacing its breakdown. ``compile_time``
    answers "which compilation stage is slow"; this answers "which shader is
    slow". Both report the same total.
    """
    return _build_breakdown_metric(
        "compile_time_by_program",
        "Compile time by program",
        [
            (_sanitize_component_id(label), label, value_ms)
            for label, value_ms in sorted(program_ms.items())
        ],
    )


def build_falcor2_observation(
    *,
    test_name: str,
    filename: str,
    function_name: str,
    source_line: Optional[int],
    observed_at: datetime,
    metrics: list[BenchViewMetric],
    parameters: dict[str, Any],
    adapter_name: Optional[str],
    device_type: Optional[str],
    workload_parameter: Optional[str] = None,
) -> BenchViewObservation:
    """Build one observation, anchored to the repository root.

    Passing the repository root makes a source path outside it an error rather
    than a silent fallback to an absolute path, which would give every checkout
    its own disconnected graph.
    """
    observation = build_benchview_observation(
        filename=filename,
        function_name=function_name,
        display_name=test_name,
        parameters=parameters,
        observed_at=observed_at,
        metrics=metrics,
        adapter_name=adapter_name,
        source_line=source_line,
        metadata={"deviceType": device_type} if device_type else None,
        source_root=helpers.PROJECT_ROOT,
    )
    # BenchView stores one display name per test ID, shared by every backend.
    # A scene is a benchmark workload, so give it its own ID rather than
    # changing the name of the shared pytest function for each parameter case.
    if workload_parameter is not None:
        key = workload_parameter
        workload = parameters[key]
        normalized = str(workload).replace("\\", "/")
        # The contract restricts test IDs to ASCII without URL escape characters.
        # Hash the full path to retain identity even for spaces or Unicode names.
        workload_id = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        observation["test"]["id"] += "/" + key + "/" + workload_id
        label = Path(normalized).stem.replace("_", " ").replace("-", " ")
        observation["test"]["name"] = label[:1].upper() + label[1:]
    return observation


# -----------------------------------------------------------------------------
# Pytest plugin
# -----------------------------------------------------------------------------


#: Fraction of maximum clock to lock to. Below the thermal limit, so the clock
#: is sustainable for the length of a run rather than dropping partway through.
GPU_CLOCK_RATIO = 0.7

DEFAULT_BENCHMARK_ITERATIONS = 10


BENCHMARK_CONTEXT_KEY = "_benchmark_context"
BENCHVIEW_EXPORT_KIND = "falcor2-benchview-export"
BENCHVIEW_EXPORT_SCHEMA_VERSION = 1
BENCHVIEW_SUBMISSION_SCHEMA_ID = "benchmark-submission-v1"

# How many observations the terminal summary prints before truncating.
TERMINAL_SUMMARY_LIMIT = 20


def _format_metric_summary(metric: BenchViewMetric) -> str:
    """One metric as a terminal summary fragment.

    Reports the mean, plus the min for multi-sample metrics as the cleanest
    estimate of the underlying cost. The first sample is not representative:
    driver-level kernel compilation lands on it.
    """
    samples = metric["distribution"]["samples"]
    unit = metric["unit"]
    mean = sum(samples) / len(samples)
    if len(samples) > 1:
        return f"{metric['name']}: {mean:.2f} {unit} (min {min(samples):.2f}, n={len(samples)})"
    return f"{metric['name']}: {mean:.2f} {unit}"


class BenchmarkContext(TypedDict):
    observations: list[BenchViewObservation]
    execution_id: str
    gpu_clocks_locked: bool


def _get_benchmark_context(config: pytest.Config) -> Optional[BenchmarkContext]:
    return getattr(config, BENCHMARK_CONTEXT_KEY, None)


def _set_benchmark_context(config: pytest.Config, context: BenchmarkContext) -> None:
    setattr(config, BENCHMARK_CONTEXT_KEY, context)


def _get_falcor2_project_info() -> dict[str, Any]:
    """Project facts that identify what was measured.

    This feeds the run key's config digest, so every field here splits BenchView
    history when it changes. Keep it to things that genuinely change the
    measurement.
    """
    import slangpy

    return {
        "name": FALCOR2_PROJECT.id,
        "version": getattr(slangpy, "SGL_VERSION", None),
        "slang_build_tag": getattr(slangpy, "SLANG_BUILD_TAG", None),
    }


def _positive_int(value: str) -> int:
    """Parse an option value that must be one or greater."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _write_text_atomically(path: Path, text: str) -> None:
    """Write small text files without exposing a partially written final path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_submission_export(submissions: list[BenchViewSubmission]) -> dict[str, Any]:
    """Wrap BenchView submissions for sealed export through a host collector."""
    return {
        "schemaVersion": BENCHVIEW_EXPORT_SCHEMA_VERSION,
        "kind": BENCHVIEW_EXPORT_KIND,
        "submissionSchema": BENCHVIEW_SUBMISSION_SCHEMA_ID,
        "submissions": submissions,
    }


def _write_submission_export(
    submissions: list[BenchViewSubmission],
    *,
    payload_path: Path,
    ready_path: Optional[Path],
) -> None:
    """Write the sealed BenchView payload and ready marker expected by KernelVM."""
    payload = _build_submission_export(submissions)
    _write_text_atomically(
        payload_path,
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    if ready_path is not None:
        _write_text_atomically(ready_path, "ready\n")


def add_benchmark_options(parser: pytest.Parser) -> None:
    """Declare the benchmark options.

    A plain function rather than a hook on the plugin: the plugin is registered
    during ``pytest_configure``, which is too late for option parsing, so a
    conftest has to call this from its own ``pytest_addoption``.
    """
    group = parser.getgroup("benchmarking")
    group.addoption(
        "--benchmark-gpu-phases",
        action="store_true",
        default=False,
        help="Collect a separate instrumented scene pass with GPU phase breakdowns",
    )
    group.addoption(
        "--benchmark-project",
        choices=("falcor2", "falcor2-slang"),
        default=os.environ.get("CI_PERF_PROJECT", "falcor2"),
        help="Revision history to report; both projects run the same benchmarks",
    )
    group.addoption(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable benchmark data collection alongside tests",
    )
    group.addoption(
        "--benchmark-only",
        action="store_true",
        default=False,
        help="Run benchmarks without image assertions (implies --benchmark)",
    )
    group.addoption(
        "--benchmark-submit",
        type=str,
        default=None,
        metavar="REQUEST_ID",
        help="Submit benchmark results to BenchView with the given request ID "
        "(implies --benchmark)",
    )
    group.addoption(
        "--benchmark-api-url",
        type=str,
        default=None,
        metavar="URL",
        help="BenchView API base URL (or set BENCHVIEW_API_URL env var)",
    )
    group.addoption(
        "--benchmark-request-id",
        type=str,
        default=None,
        metavar="REQUEST_ID",
        help="Request ID to use when exporting benchmark submissions without "
        "posting them directly",
    )
    group.addoption(
        "--benchmark-export-submissions",
        type=Path,
        default=None,
        metavar="PATH",
        help="Write BenchView submission bodies to a sealed-export wrapper at PATH "
        "instead of relying on in-guest network access (implies --benchmark)",
    )
    group.addoption(
        "--benchmark-export-ready",
        type=Path,
        default=None,
        metavar="PATH",
        help=("Atomically write a ready marker after " "--benchmark-export-submissions succeeds"),
    )
    group.addoption(
        "--benchmark-iterations",
        type=_positive_int,
        default=DEFAULT_BENCHMARK_ITERATIONS,
        help=f"Number of benchmark iterations per test "
        f"(default: {DEFAULT_BENCHMARK_ITERATIONS}, minimum: 1)",
    )
    group.addoption(
        "--lock-gpu-clocks",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Lock GPU clocks for benchmark stability (requires nvidia-smi and, "
        "on Linux, passwordless sudo for the clock commands)",
    )


class BenchmarkPlugin:
    """Pytest plugin for benchmark data collection and BenchView submission."""

    def __init__(self, config: pytest.Config):
        self._config = config
        self._api_url: Optional[str] = None
        self._request_id: Optional[str] = None
        self._export_submissions: Optional[Path] = None
        self._export_ready: Optional[Path] = None
        self._project_id = "falcor2"
        self._compiler: Optional[dict[str, Any]] = None
        self._provenance: Optional[dict[str, Any]] = None
        self._skipped = False

    def pytest_configure(self, config: pytest.Config) -> None:
        """Validate configuration."""
        config.addinivalue_line(
            "markers",
            "benchmark_workload(parameter): parameter defining workload identity and name",
        )
        benchmark_submit = config.getoption("--benchmark-submit", default=None)
        benchmark_request_id = config.getoption("--benchmark-request-id", default=None)
        benchmark_export = config.getoption("--benchmark-export-submissions", default=None)
        benchmark_export_ready = config.getoption("--benchmark-export-ready", default=None)
        self._project_id = config.getoption("--benchmark-project", default="falcor2")

        # Both of these are meaningless without collection, so both imply it.
        # Without this, --benchmark-submit alone leaves the plugin inactive and
        # the run reports success having measured and sent nothing.
        if config.getoption("--benchmark-only") or benchmark_submit or benchmark_export is not None:
            config.option.benchmark = True

        api_url = config.getoption("--benchmark-api-url", default=None) or os.environ.get(
            "BENCHVIEW_API_URL"
        )
        request_id = benchmark_request_id or benchmark_submit

        if benchmark_request_id and benchmark_submit and benchmark_request_id != benchmark_submit:
            raise pytest.UsageError(
                "--benchmark-request-id must match --benchmark-submit when both are specified"
            )

        if benchmark_export is not None:
            if not request_id:
                raise pytest.UsageError(
                    "--benchmark-export-submissions requires --benchmark-request-id "
                    "or --benchmark-submit"
                )
            if (
                benchmark_export_ready is not None
                and benchmark_export.resolve() == benchmark_export_ready.resolve()
            ):
                raise pytest.UsageError(
                    "--benchmark-export-submissions and --benchmark-export-ready "
                    "must use different paths"
                )
            self._export_submissions = benchmark_export
            self._export_ready = benchmark_export_ready
        elif benchmark_export_ready is not None:
            raise pytest.UsageError(
                "--benchmark-export-ready requires --benchmark-export-submissions"
            )

        if benchmark_submit:
            if not api_url:
                raise pytest.UsageError(
                    "--benchmark-submit requires --benchmark-api-url or BENCHVIEW_API_URL"
                )
            try:
                benchview_submission_url(api_url)
            except Exception as exc:
                raise pytest.UsageError(f"Invalid --benchmark-api-url: {exc}") from exc

            if not os.environ.get("BENCHVIEW_API_KEY"):
                raise pytest.UsageError(
                    "--benchmark-submit requires BENCHVIEW_API_KEY environment variable"
                )

            self._api_url = api_url

        self._request_id = request_id

        if config.getoption("--benchmark", default=False):
            if hasattr(config, "workerinput"):
                raise pytest.UsageError(
                    "Benchmarks require single-process execution. "
                    "Use -n0 to disable pytest-xdist parallelization."
                )

    def pytest_sessionstart(self, session: pytest.Session) -> None:
        """Initialize benchmark context and optionally lock GPU clocks."""
        config = session.config

        if not config.getoption("--benchmark", default=False):
            return

        if self._export_ready is not None:
            self._export_ready.unlink(missing_ok=True)

        provenance_path = helpers.PROJECT_ROOT / "reports/benchmark-provenance.json"
        if provenance_path.exists():
            self._provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if self._provenance["falcor_commit"] != get_commit_info()["id"]:
                raise pytest.UsageError("Benchmark provenance belongs to a different checkout")
            self._compiler = self._provenance["compiler"]
        if self._compiler is not None or self._project_id != "falcor2":
            try:
                benchmark_identity(self._project_id, self._compiler)
                if self._compiler is not None:
                    import slangpy as spy

                    release = self._compiler.get("release")
                    if release and spy.SLANG_BUILD_TAG.removeprefix("v") != release[
                        "label"
                    ].removeprefix("v"):
                        raise ValueError("Loaded Slang compiler differs from the selected release")
                    self._compiler["build_tag"] = spy.SLANG_BUILD_TAG
            except (KeyError, ValueError) as exc:
                raise pytest.UsageError(str(exc)) from exc

        gpu_clocks_locked = False
        if config.getoption("--lock-gpu-clocks", default=False):
            # Asked for explicitly, so a failure is fatal rather than a warning.
            # Benchmarking on unlocked clocks produces numbers that look valid
            # and drift with whatever else the machine is doing.
            print("Locking GPU clocks for benchmark stability...")
            try:
                clocks = lock_gpu_clocks(0, GPU_CLOCK_RATIO, conservative=True)
            except Exception as exc:
                raise pytest.UsageError(
                    f"--lock-gpu-clocks was requested but locking failed: {exc}. "
                    "Clock control needs nvidia-smi and, on Linux, passwordless sudo "
                    "for the lock commands."
                ) from exc
            if clocks is None:
                raise pytest.UsageError(
                    "--lock-gpu-clocks was requested but no suitable clock combination "
                    "was found, so the run would measure unlocked clocks."
                )
            gpu_clocks_locked = True

        context: BenchmarkContext = {
            "observations": [],
            "execution_id": uuid.uuid4().hex,
            "gpu_clocks_locked": gpu_clocks_locked,
        }
        _set_benchmark_context(config, context)

    def pytest_runtest_logreport(self, report: pytest.TestReport) -> None:
        if report.skipped and self._config.getoption("--benchmark-only", default=False):
            self._skipped = True

    def pytest_sessionfinish(self, session: pytest.Session, exitstatus: int) -> None:
        """Submit observations to BenchView and unlock GPU clocks."""
        config = session.config
        context = _get_benchmark_context(config)

        if context is None:
            return

        if context["gpu_clocks_locked"]:
            print("Unlocking GPU clocks...")
            unlock_gpu_clocks(0)

        observations = context["observations"]

        # A partial session must not publish a complete-looking run or ready marker.
        if exitstatus != 0 or session.exitstatus != 0 or session.shouldstop:
            return
        if self._skipped:
            self._fail(
                session, "A selected benchmark was skipped; refusing to publish a partial run."
            )
            return

        # A benchmark session that measured nothing is a failure, not a pass.
        # Stay quiet when the session is already failing, so the real cause
        # keeps the exit code and stays at the top of the output, and when only
        # collecting, which legitimately measures nothing.
        if not observations:
            collecting = config.getoption("--collect-only", default=False)
            already_failing = session.exitstatus != 0 or session.shouldstop
            if not collecting and not already_failing:
                self._fail(
                    session,
                    "Benchmarking was requested but no benchmark observations were "
                    "recorded. Every benchmark was skipped or deselected; check that "
                    "benchmark assets are present and that the selection matches "
                    "benchmark tests.",
                )
            return

        if not self._request_id or (not self._api_url and self._export_submissions is None):
            return

        try:
            project, project_info, commit_info = benchmark_identity(
                self._project_id, self._compiler
            )
            if self._project_id == "falcor2-slang":
                for observation in observations:
                    observation.setdefault("case", {}).setdefault("dimensions", {})[
                        "compiler_integration"
                    ] = "rebuild"
            for observation in observations:
                observation.setdefault("metadata", {})["gpuClocksLocked"] = context[
                    "gpu_clocks_locked"
                ]
                if self._provenance is not None:
                    observation.setdefault("metadata", {})["provenance"] = self._provenance
            submissions = build_benchview_submissions(
                observations,
                request_id=self._request_id,
                execution_id=context["execution_id"],
                project_info=project_info,
                machine_info=get_machine_info(),
                commit_info=commit_info,
                project=project,
            )
            if (
                self._project_id == "falcor2-slang"
                and self._compiler
                and self._compiler.get("release")
            ):
                for submission in submissions:
                    submission["run"].setdefault("metadata", {})["release"] = self._compiler[
                        "release"
                    ]
        except Exception as exc:
            self._fail(session, f"BenchView submission build failed: {exc}")
            return

        if self._export_submissions is not None:
            try:
                _write_submission_export(
                    submissions,
                    payload_path=self._export_submissions,
                    ready_path=self._export_ready,
                )
            except Exception as exc:
                self._fail(session, f"BenchView submission export failed: {exc}")
                return
            print(f"\nExported BenchView submission to {self._export_submissions}")

        if not self._api_url:
            return

        print(f"\nSubmitting {len(observations)} benchmark observations to BenchView...")

        # A run told to submit results that fails to submit them must not exit
        # zero. Reported rather than raised because an exception escaping
        # pytest_sessionfinish surfaces as an INTERNALERROR traceback.
        try:
            receipts = submit_benchview_submissions(
                api_base_url=self._api_url,
                write_key=os.environ["BENCHVIEW_API_KEY"],
                submissions=submissions,
            )
        except Exception as exc:
            self._fail(session, f"BenchView submission failed: {exc}")
            return

        accepted = sum(1 for r in receipts if not r.get("duplicate", False))
        duplicates = sum(1 for r in receipts if r.get("duplicate", False))
        print(f"BenchView submission complete: {accepted} accepted, {duplicates} duplicates")

    @staticmethod
    def _fail(session: pytest.Session, message: str) -> None:
        """Record a benchmark infrastructure failure and make the session exit non-zero."""
        print(f"ERROR: {message}")
        session.exitstatus = pytest.ExitCode.INTERNAL_ERROR

    def pytest_terminal_summary(
        self, terminalreporter: Any, exitstatus: int, config: pytest.Config
    ) -> None:
        """Display benchmark summary table."""
        context = _get_benchmark_context(config)
        if context is None or not context["observations"]:
            return

        observations = context["observations"]
        terminalreporter.write_sep("=", "Benchmark Summary")
        terminalreporter.write_line(f"Collected {len(observations)} benchmark observations")
        terminalreporter.write_line("")

        for obs in observations[:TERMINAL_SUMMARY_LIMIT]:
            test_name = obs["test"]["name"]
            dimensions = obs.get("case", {}).get("dimensions", {})
            if dimensions:
                test_name += "[" + ",".join(str(v) for v in dimensions.values()) + "]"
            metrics_str = ", ".join(
                _format_metric_summary(m)
                for m in obs.get("metrics", [])
                if m["distribution"].get("samples")
            )
            terminalreporter.write_line(f"  {test_name}: {metrics_str}")

        if len(observations) > TERMINAL_SUMMARY_LIMIT:
            remaining = len(observations) - TERMINAL_SUMMARY_LIMIT
            terminalreporter.write_line(f"  ... and {remaining} more")
