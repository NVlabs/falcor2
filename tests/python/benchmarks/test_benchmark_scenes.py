# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The five full-engine scenes used by the local Falcor2 version sweeps.

Measure compilation on a fresh device per scene, then steady GPU frames
without profiling. Application module/shader disk caches must be disabled.
"""

from datetime import datetime, timezone
from pathlib import Path

import pytest
import slangpy as spy
from slangpy.testing.benchmark import build_metric

import falcor2 as f2
from falcor2.rendernodes import PathTracerPipeline
from falcor2.testing import helpers
from falcor2.testing.benchmark.fixtures import (
    _compilation_cursor,
    _compilation_delta,
    _compilation_stage_totals,
    _measure_render_time,
)
from falcor2.testing.benchmark.plugin import (
    BENCHMARK_CONTEXT_KEY,
    build_compile_time_breakdown_metric,
    build_falcor2_observation,
)
from falcor2.testing.benchmark.gpu_timing import build_gpu_metric, measure_frame


SCENES = {
    "emissive-strength": "data/scenes/emissive-strength.py",
    "sponza": "data/internal/sponza.py",
    "checker-material": "data/scenes/checker-material.py",
    "cornell-box": "data/scenes/cornell-box.py",
    "cornell-box-env": "data/scenes/cornell-box-env.py",
}


@pytest.mark.benchmark
@pytest.mark.benchmark_workload("scene")
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scene_name", list(SCENES))
def test_benchmark_scene(
    device_type: spy.DeviceType, scene_name: str, request: pytest.FixtureRequest
) -> None:
    """Report one scene's compilation total and steady GPU frame distribution."""
    assert (
        not helpers.MODULE_CACHE_ENABLED and not helpers.SHADER_CACHE_ENABLED
    ), "Scene compile benchmarks require disabled module and shader disk caches"
    path = Path(SCENES[scene_name])
    assert path.is_file(), f"Required benchmark scene is missing: {path}"
    iterations = request.config.getoption("--benchmark-iterations")
    context = getattr(request.config, BENCHMARK_CONTEXT_KEY)

    # Explicit profiler selection prevents a previously-created editor or
    # profiler from instrumenting the ordinary timing pass.
    profiler = spy.Profiler()
    profiler.enabled = False
    with profiler:
        device = helpers.get_device(device_type, use_cache=False)
        cursor = _compilation_cursor(device)
        scene = f2.Scene.load(device, str(path))
        scene.update()
        device.wait()
        camera = scene.active_camera
        assert camera is not None, f"Scene has no active camera: {path}"
        camera.width, camera.height = 1920, 1080
        pipeline = PathTracerPipeline.create(device)
        pipeline.spp = 1
        pipeline.path_tracer.max_depth = 3
        pipeline.path_tracer.enable_nee = True
        pipeline.path_tracer.enable_mis = True

        def render(cmd: spy.CommandEncoder | None = None):
            return pipeline(scene, camera, cmd=cmd)

        render()
        device.wait()
        for _ in range(8):
            render()
            device.wait()
        programs = _compilation_delta(device, cursor)
        guard = _compilation_cursor(device)

        gpu_times = []
        for _ in range(iterations):
            gpu, _, _, _ = _measure_render_time(device, render)
            gpu_times.append(gpu)
        assert not _compilation_delta(
            device, guard
        ), "Compilation occurred during runtime measurement"

        slang, backend, pipeline_ms = _compilation_stage_totals(programs)
        compile_metric = build_compile_time_breakdown_metric(
            slang_compile_ms=slang, backend_compile_ms=backend, pipeline_create_ms=pipeline_ms
        )
        assert compile_metric is not None, "Uncached benchmark produced no compilation timings"
        metrics = [compile_metric, build_metric("gpu_time", "Frame GPU time", gpu_times)]
        if request.config.getoption("--benchmark-gpu-phases"):
            # Keep the primary samples uninstrumented. Components describe only
            # frames from this separate pass, including their own enclosing total.
            frames = [measure_frame(device, pipeline, scene, camera) for _ in range(iterations)]
            assert not _compilation_delta(
                device, guard
            ), "Compilation occurred during diagnostic runtime measurement"
            metrics.append(build_gpu_metric(frames))
        context["observations"].append(
            build_falcor2_observation(
                test_name="test_benchmark_scene",
                filename=str(Path(__file__).resolve()),
                function_name="test_benchmark_scene",
                source_line=None,
                observed_at=datetime.now(timezone.utc),
                metrics=metrics,
                parameters={
                    "scene": scene_name,
                    "device_type": device_type.name,
                    "width": 1920,
                    "height": 1080,
                    "spp": 1,
                    "max_depth": 3,
                    "cache": "application-disk-disabled",
                },
                adapter_name=device.info.adapter_name,
                device_type=device_type.name,
                workload_parameter=request.node.get_closest_marker("benchmark_workload").args[0],
            )
        )
