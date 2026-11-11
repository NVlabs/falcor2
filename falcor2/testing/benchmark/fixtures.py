# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark fixtures for Falcor2 image tests.

Provides the benchmark_image_test fixture that wraps image_test to collect:
- GPU time for a complete frame, via hardware timestamp queries
- GPU time for individual dispatches, when a test opts into per-sample timing
- CPU command buffer build time
- Shader compilation times, optionally isolated by a dedicated compile probe
- Pipeline creation times

Compilation is measured against the benchmarked device only, and the fixture
fails a test if any compilation happens inside the timed window, because a
compile stall reported as render time is a silently wrong benchmark.

These metrics are submitted to BenchView when --benchmark-submit is specified.
"""

from __future__ import annotations

import re
import math
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Protocol, Union

import numpy as np
import pytest
import slangpy as spy
from slangpy import Bitmap, Tensor

from slangpy.testing.benchmark import BenchViewMetric, build_metric

from .plugin import (
    BENCHMARK_CONTEXT_KEY,
    DEFAULT_BENCHMARK_ITERATIONS,
    build_falcor2_observation,
    build_compile_time_breakdown_metric,
    build_compile_time_by_program_metric,
)
from ..image_test_plugin import ImageTest


DEFAULT_WARMUP_ITERATIONS = 2

RenderResult = Union["np.ndarray[Any, Any]", Bitmap, Tensor, spy.Texture]


class RenderFn(Protocol):
    """Protocol for render functions that draw one complete frame.

    ``cmd`` is the command encoder the fixture wraps in timestamp queries. When
    it is None the render function must submit its own work and return a
    readable image. ``sampler`` is only passed when the test opted into
    per-dispatch timing via ``sample_count``.
    """

    def __call__(
        self,
        cmd: spy.CommandEncoder | None = None,
        sampler: Optional["SampleTimer"] = None,
    ) -> RenderResult: ...


# -----------------------------------------------------------------------------
# Per-dispatch GPU timing
# -----------------------------------------------------------------------------


class SampleTimer:
    """Records GPU timestamps around individual dispatches inside one command buffer.

    A render function that draws N dispatches per frame can wrap each one to
    produce N samples of isolated dispatch cost, instead of a single aggregate
    frame time. The extra samples are what make small regressions and variance
    visible.

    Usage inside a render function:

        def render(cmd=None, sampler=None):
            for i in range(SPP):
                with sampler.sample(cmd):
                    sample, _ = pipeline.path_tracer(scene, camera, iteration=i, cmd=cmd)
                image = pipeline.accumulator(sample, cmd=cmd)
            return pipeline.tonemapper(image, cmd=cmd)
    """

    def __init__(self, device: spy.Device, sample_count: int):
        self._device = device
        self._sample_count = sample_count
        self._query_pool = device.create_query_pool(
            type=spy.QueryType.timestamp, count=sample_count * 2
        )
        self._recorded = 0

    @contextmanager
    def sample(self, cmd: spy.CommandEncoder | None) -> Iterator[None]:
        """Time the work recorded inside this context as one sample."""
        if cmd is None:
            # Warmup calls run without an encoder; nothing to time.
            yield
            return
        if self._recorded >= self._sample_count:
            raise RuntimeError(
                f"Render function recorded more than the declared {self._sample_count} "
                "samples; pass a matching sample_count to benchmark_image_test"
            )
        index = self._recorded
        cmd.write_timestamp(self._query_pool, index * 2)
        try:
            yield
        finally:
            cmd.write_timestamp(self._query_pool, index * 2 + 1)
            self._recorded += 1

    def read_sample_times_ms(self) -> list[float]:
        """Read back per-sample durations. Only valid after the device has idled."""
        if self._recorded == 0:
            return []
        frequency = float(self._device.info.timestamp_frequency)
        if frequency <= 0:
            return []
        timestamps = self._query_pool.get_results(0, self._recorded * 2)
        return [
            (float(timestamps[index + 1]) - float(timestamps[index])) / frequency * 1000.0
            for index in range(0, self._recorded * 2, 2)
        ]


# -----------------------------------------------------------------------------
# Device-scoped compilation accounting
# -----------------------------------------------------------------------------

# Per shader program, the number of entry point and pipeline reports already
# seen. Compilation reports only ever grow, so a cursor plus a later read gives
# exactly what was compiled in between.
CompilationCursor = tuple[tuple[int, int], ...]


def _compilation_cursor(device: spy.Device) -> CompilationCursor:
    """Snapshot how much compilation the device has reported so far."""
    return tuple(
        (len(report["entry_point_reports"]), len(report["pipeline_reports"]))
        for report in device.get_compilation_reports()
    )


def _compilation_delta(
    device: spy.Device,
    cursor: CompilationCursor,
) -> list[dict[str, Any]]:
    """Return everything the device compiled since the cursor was taken.

    Scoped to a single device on purpose. The benchmark session may have other
    devices alive, and their compilation must not be attributed to this test.
    """
    programs: list[dict[str, Any]] = []
    for program_index, report in enumerate(device.get_compilation_reports()):
        previous_entry_points, previous_pipelines = (
            cursor[program_index] if program_index < len(cursor) else (0, 0)
        )
        entry_points = report["entry_point_reports"][previous_entry_points:]
        pipelines = report["pipeline_reports"][previous_pipelines:]
        if entry_points or pipelines:
            programs.append(
                {
                    "label": str(report["label"]),
                    "entry_point_reports": entry_points,
                    "pipeline_reports": pipelines,
                }
            )
    return programs


def _compilation_stage_totals(
    programs: list[dict[str, Any]],
) -> tuple[float, float, float]:
    """Total Slang frontend, backend codegen and pipeline creation time in milliseconds."""
    slang_ms = 0.0
    backend_ms = 0.0
    pipeline_ms = 0.0
    for program in programs:
        for entry_point in program["entry_point_reports"]:
            slang_ms += float(entry_point["compile_slang_time"]) * 1000.0
            backend_ms += float(entry_point["compile_downstream_time"]) * 1000.0
        for pipeline in program["pipeline_reports"]:
            pipeline_ms += float(pipeline["create_time"]) * 1000.0
    return slang_ms, backend_ms, pipeline_ms


# Program labels are prefixed with a content hash of the linked program, for
# example
# "a4c7f759...20948:raygen_main, falcor2/rendernodes/reference_pathtracer.slang:_s".
# That hash changes whenever the shader changes, so it must not end up in a
# BenchView breakdown component id: the id has to stay stable across shader
# edits for compile time to be comparable over time.
_PROGRAM_LABEL_HASH_PREFIX = re.compile(r"^[0-9a-f]{16,}:")


def _program_label(raw_label: str) -> str:
    """Strip the unstable content hash from a compilation report label.

    Labels that reduce to the same text, such as several different modules that
    all expose only "compute_main", are aggregated into one component.
    """
    return _PROGRAM_LABEL_HASH_PREFIX.sub("", raw_label) or raw_label


def _compilation_program_totals(programs: list[dict[str, Any]]) -> dict[str, float]:
    """Total compile time in milliseconds attributed to each shader program.

    Sums the same stage components as _compilation_stage_totals, so compile_time
    and compile_time_by_program report an identical total.
    """
    totals: dict[str, float] = {}
    for program in programs:
        total_ms = 0.0
        for entry_point in program["entry_point_reports"]:
            total_ms += float(entry_point["compile_slang_time"]) * 1000.0
            total_ms += float(entry_point["compile_downstream_time"]) * 1000.0
        for pipeline in program["pipeline_reports"]:
            total_ms += float(pipeline["create_time"]) * 1000.0
        label = _program_label(program["label"])
        totals[label] = totals.get(label, 0.0) + total_ms
    return totals


def _describe_compilations(programs: list[dict[str, Any]]) -> str:
    """Human readable summary used when compilation contaminates a timed run."""
    lines: list[str] = []
    for program in programs:
        names = [str(entry["name"]) for entry in program["entry_point_reports"]]
        pipeline_count = len(program["pipeline_reports"])
        detail = ", ".join(names) if names else "no new entry points"
        lines.append(f"  {program['label']}: {detail} ({pipeline_count} pipeline(s))")
    return "\n".join(lines)


def _force_immediate_pipeline_compilation(modules: Sequence[spy.Module]) -> None:
    """Make pipeline creation happen eagerly rather than at first dispatch.

    SlangPy defaults to a deferred policy, which creates graphics pipelines
    lazily when a dispatch first needs them. That lets compilation cost leak
    into whichever dispatch happens to trigger it. Forcing immediate creation
    makes the compile measurement deterministic and keeps the timed window
    clean.
    """
    for module in modules:
        module.options["pipeline_compilation_policy"] = spy.PipelineCompilationPolicy.immediate


# -----------------------------------------------------------------------------
# Measurement
# -----------------------------------------------------------------------------


def _invoke_render(
    device: spy.Device,
    render_fn: RenderFn,
    cmd: spy.CommandEncoder | None,
    sample_count: Optional[int],
) -> tuple[RenderResult, Optional[SampleTimer]]:
    """Call the render function, handing it a sampler when it asked for one.

    A test that opts into per-dispatch timing gets a sampler on every call,
    including warmup and non-benchmark runs, so it never has to check for None.
    A sampler handed a null command encoder records nothing.
    """
    if sample_count:
        sampler = SampleTimer(device, sample_count)
        return render_fn(cmd=cmd, sampler=sampler), sampler
    return render_fn(cmd=cmd), None


def _measure_render_time(
    device: spy.Device,
    render_fn: RenderFn,
    sample_count: Optional[int] = None,
) -> tuple[float, float, list[float], RenderResult]:
    """
    Measure GPU execution time and CPU command buffer build time.

    Args:
        device: SGL device with timestamp query support
        render_fn: Function that accepts a command encoder and returns a rendered image
        sample_count: When set, the render function is also handed a SampleTimer
            and is expected to record this many per-dispatch samples

    Returns:
        Tuple of (gpu_time_ms, cpu_time_ms, sample_times_ms, rendered_image)
    """
    query_pool = device.create_query_pool(type=spy.QueryType.timestamp, count=2)

    # Measure CPU time for command buffer building
    cpu_start = time.perf_counter()

    command_encoder = device.create_command_encoder()
    command_encoder.write_timestamp(query_pool, 0)

    # Execute the render function with the command encoder
    image, sampler = _invoke_render(device, render_fn, command_encoder, sample_count)

    command_encoder.write_timestamp(query_pool, 1)
    command_buffer = command_encoder.finish()

    cpu_end = time.perf_counter()
    cpu_time_ms = (cpu_end - cpu_start) * 1000.0

    # Submit and wait for GPU
    device.submit_command_buffer(command_buffer)
    device.wait()

    # Get GPU timing results
    timestamps = query_pool.get_results(0, 2)
    frequency = float(device.info.timestamp_frequency)

    if not math.isfinite(frequency) or frequency <= 0:
        raise RuntimeError(f"GPU timestamp frequency must be positive and finite, got {frequency}")
    if len(timestamps) != 2:
        raise RuntimeError(f"Expected two GPU timestamps, got {len(timestamps)}")
    delta_ticks = timestamps[1] - timestamps[0]
    if not math.isfinite(delta_ticks) or delta_ticks < 0:
        raise RuntimeError(
            f"GPU timestamp duration must be nonnegative and finite, got {delta_ticks}"
        )
    gpu_time_ms = (delta_ticks / frequency) * 1000.0

    sample_times_ms = sampler.read_sample_times_ms() if sampler is not None else []
    if sample_count and len(sample_times_ms) != sample_count:
        raise AssertionError(
            f"Render function recorded {len(sample_times_ms)} of the declared "
            f"{sample_count} samples. A short distribution would be reported as "
            "sample_gpu_time without the missing dispatches; record one sample "
            "per dispatch or pass a matching sample_count."
        )

    return gpu_time_ms, cpu_time_ms, sample_times_ms, image


class BenchmarkImageTest:
    """
    Benchmark wrapper for image tests.

    Collects GPU render time, CPU command buffer build time, and compilation
    telemetry while optionally performing image comparison. Results are stored
    for BenchView submission.

    Usage:
        def test_render_scene(device_type, benchmark_image_test):
            device = helpers.get_device(device_type)
            scene = load_scene(device)
            pipeline = PathTracerPipeline.create(device)

            def render(cmd=None, sampler=None):
                pipeline.reset(cmd=cmd)
                return pipeline(scene, camera, cmd=cmd)

            benchmark_image_test(device, render, tolerance=0.001)
    """

    def __init__(
        self,
        request: pytest.FixtureRequest,
        image_test: ImageTest,
        config: pytest.Config,
    ):
        self._request = request
        self._image_test = image_test
        self._config = config
        self._benchmark_enabled = config.getoption("--benchmark", default=False)
        self._benchmark_only = config.getoption("--benchmark-only", default=False)
        self._iterations = config.getoption(
            "--benchmark-iterations", default=DEFAULT_BENCHMARK_ITERATIONS
        )

    def __call__(
        self,
        device: spy.Device,
        render_fn: RenderFn,
        *,
        iterations: Optional[int] = None,
        warmup_iterations: int = DEFAULT_WARMUP_ITERATIONS,
        sample_count: Optional[int] = None,
        compile_modules: Sequence[spy.Module] = (),
        compile_probe: Optional[Callable[[], None]] = None,
        tolerance: float = 1e-5,
        additional_name: Optional[str] = None,
        debug_output_normalize: bool = False,
        debug_output_stratify: bool = False,
    ) -> None:
        """
        Run benchmark measurements and optionally image comparison.

        Args:
            device: SGL device for GPU timing and compilation reports
            render_fn: Callable that accepts cmd=CommandEncoder and returns image
            iterations: Number of timed iterations (default: from --benchmark-iterations)
            warmup_iterations: Number of warmup iterations (default: 2)
            sample_count: Number of per-dispatch samples the render function
                records via the SampleTimer it is handed. Omit for tests that
                only report whole-frame time.
            compile_modules: SlangPy modules to switch to immediate pipeline
                compilation, so that compilation happens during the probe or
                warmup rather than inside the timed window.
            compile_probe: Minimal work that triggers the compilation this test
                wants to measure. When given, only compilation caused by the
                probe is reported, which is how a test isolates, for example,
                material shader compilation from the rest of the renderer. When
                omitted, compilation observed across all warmup iterations is
                reported instead.
            tolerance: Image comparison tolerance (passed to image_test)
            additional_name: Additional name for parameterized tests
            debug_output_normalize: Normalize debug PNG output
            debug_output_stratify: Stratify debug PNG output
        """
        if not self._benchmark_enabled:
            # Just run the image test directly (no cmd encoder needed)
            image, _ = _invoke_render(device, render_fn, None, sample_count)
            if isinstance(image, spy.Texture):
                image = image.to_numpy()
            self._image_test(
                image,
                tolerance=tolerance,
                additional_name=additional_name,
                debug_output_normalize=debug_output_normalize,
                debug_output_stratify=debug_output_stratify,
            )
            return

        # Use provided iterations or default from CLI
        if iterations is None:
            iterations = self._iterations

        _force_immediate_pipeline_compilation(compile_modules)

        # Compilation measurement. Either an explicit probe isolates the
        # compilation the test cares about, or we fall back to whatever the
        # warmup iterations compile.
        compile_cursor = _compilation_cursor(device)
        compile_programs: Optional[list[dict[str, Any]]] = None

        if compile_probe is not None:
            compile_probe()
            device.wait()
            compile_programs = _compilation_delta(device, compile_cursor)

        for _ in range(warmup_iterations):
            _invoke_render(device, render_fn, None, sample_count)
            device.wait()

        if compile_programs is None:
            compile_programs = _compilation_delta(device, compile_cursor)

        # Anything compiled from here on would be measured as render time.
        guard_cursor = _compilation_cursor(device)

        # Measured iterations
        gpu_times: list[float] = []
        cpu_times: list[float] = []
        sample_times: list[float] = []
        last_image: Optional[RenderResult] = None

        for _ in range(iterations):
            gpu_time_ms, cpu_time_ms, iteration_samples, image = _measure_render_time(
                device, render_fn, sample_count
            )
            gpu_times.append(gpu_time_ms)
            cpu_times.append(cpu_time_ms)
            sample_times.extend(iteration_samples)
            last_image = image

        leaked = _compilation_delta(device, guard_cursor)
        if leaked:
            raise AssertionError(
                "Shader or pipeline compilation occurred during the timed window, so "
                "the measured times include compile stalls and are not usable:\n"
                f"{_describe_compilations(leaked)}\n"
                "Increase warmup_iterations, or pass compile_modules/compile_probe so "
                "compilation is forced to happen before timing starts."
            )

        # Run image test assertion (unless benchmark-only mode)
        if not self._benchmark_only and last_image is not None:
            # Convert to numpy if needed for image test
            if isinstance(last_image, spy.Texture):
                last_image = last_image.to_numpy()
            self._image_test(
                last_image,
                tolerance=tolerance,
                additional_name=additional_name,
                debug_output_normalize=debug_output_normalize,
                debug_output_stratify=debug_output_stratify,
            )

        # Build and store observation
        self._store_observation(
            device=device,
            gpu_times=gpu_times,
            cpu_times=cpu_times,
            sample_times=sample_times,
            compile_programs=compile_programs,
            additional_name=additional_name,
        )

    def _store_observation(
        self,
        device: spy.Device,
        gpu_times: list[float],
        cpu_times: list[float],
        sample_times: list[float],
        compile_programs: list[dict[str, Any]],
        additional_name: Optional[str],
    ) -> None:
        """Store benchmark observation in the pytest config context."""
        context = getattr(self._config, BENCHMARK_CONTEXT_KEY, None)
        if context is None:
            return

        # gpu_time covers one complete frame including accumulation, tone
        # mapping and submission overhead; sample_gpu_time covers a single path
        # tracer dispatch in isolation. A BenchView graph is keyed on the metric
        # id, so the two must never share one.
        metrics: list[BenchViewMetric] = []

        if gpu_times:
            metrics.append(build_metric("gpu_time", "GPU time", gpu_times))

        # Per-dispatch GPU time, for tests that opted into sample timing
        if sample_times:
            metrics.append(build_metric("sample_gpu_time", "GPU time per sample", sample_times))

        # CPU time metric (command buffer building)
        if cpu_times:
            metrics.append(build_metric("cpu_time", "CPU time", cpu_times))

        # Compilation time, reported twice: broken down by stage, and broken
        # down by shader program. A BenchView metric carries one breakdown, and
        # both views answer questions the team needs answered.
        slang_ms, backend_ms, pipeline_ms = _compilation_stage_totals(compile_programs)
        compile_metric = build_compile_time_breakdown_metric(
            slang_compile_ms=slang_ms,
            backend_compile_ms=backend_ms,
            pipeline_create_ms=pipeline_ms,
        )
        if compile_metric is not None:
            metrics.append(compile_metric)

        program_metric = build_compile_time_by_program_metric(
            _compilation_program_totals(compile_programs)
        )
        if program_metric is not None:
            metrics.append(program_metric)

        # Extract test metadata
        node = self._request.node
        filename = str(node.path) if hasattr(node, "path") else str(node.fspath)
        function_name = getattr(node, "originalname", node.name)
        # Use base function name for test_name so device types share the same graph
        test_name = function_name
        if additional_name:
            test_name = f"{test_name}[{additional_name}]"

        # Get parameters for case dimensions
        parameters: dict[str, Any] = {}
        if hasattr(node, "callspec") and node.callspec:
            parameters = dict(node.callspec.params)

        # Get device info
        adapter_name = device.info.adapter_name if device else None
        device_type_name = None
        if device:
            device_type_name = device.desc.type.name

        # Build observation
        workload = node.get_closest_marker("benchmark_workload")
        observation = build_falcor2_observation(
            test_name=test_name,
            filename=filename,
            function_name=function_name,
            source_line=node.location[1] + 1 if hasattr(node, "location") else None,
            observed_at=datetime.now(timezone.utc),
            metrics=metrics,
            parameters=parameters,
            adapter_name=adapter_name,
            device_type=device_type_name,
            workload_parameter=workload.args[0] if workload else None,
        )

        context["observations"].append(observation)


@pytest.fixture
def benchmark_image_test(
    request: pytest.FixtureRequest,
    image_test: ImageTest,
) -> BenchmarkImageTest:
    """
    Fixture that provides benchmark-enabled image testing.

    When --benchmark is specified, collects:
    - GPU time for a complete frame, via timestamp queries (requires render_fn
      to accept a cmd= kwarg)
    - GPU time per dispatch, when the test passes sample_count and uses the
      SampleTimer it is handed
    - CPU command buffer build time
    - Compile time, broken down both by stage (Slang frontend, backend codegen,
      pipeline creation) and by shader program

    The fixture fails the test if anything compiles during the timed window.

    When --benchmark is not specified, behaves like a simple wrapper
    that calls the render function and passes the result to image_test.

    Usage:
        def test_render_scene(device_type, benchmark_image_test):
            device = helpers.get_device(device_type)
            scene = load_scene(device)
            pipeline = PathTracerPipeline.create(device)

            def render(cmd=None, sampler=None):
                pipeline.reset(cmd=cmd)
                return pipeline(scene, camera, cmd=cmd)

            benchmark_image_test(device, render, tolerance=0.001)
    """
    return BenchmarkImageTest(
        request=request,
        image_test=image_test,
        config=request.config,
    )
