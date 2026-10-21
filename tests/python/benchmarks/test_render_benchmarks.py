# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Benchmark tests for Falcor2 rendering performance.

These tests measure GPU render time, CPU command buffer build time, and shader
compilation metrics when run with --benchmark flag. Results can be submitted to
BenchView with --benchmark-submit.

All cases share one path tracer configuration and differ only in the scene and
the path tracer settings listed in CASES.

Run benchmarks:
    pytest tests/python/benchmarks --benchmark --shader-cache --image-tests -v

Submit to BenchView:
    pytest tests/python/benchmarks --benchmark --benchmark-submit=<request_id> --image-tests

Export for a sealed host collector:
    pytest tests/python/benchmarks --benchmark --benchmark-request-id=<request_id> --image-tests \
      --benchmark-export-submissions=perf-results/falcor2-pathtracer-scene-bench.json \
      --benchmark-export-ready=perf-results/falcor2-pathtracer-scene-bench.ready

Note: --image-tests is required because these use the benchmark_image_test fixture
which wraps image_test for comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import PathTracerPipeline
from falcor2.testing.benchmark import BenchmarkImageTest, RenderResult, SampleTimer

# Common test parameters
BENCHMARK_WIDTH = 1920
BENCHMARK_HEIGHT = 1080
BENCHMARK_SPP = 256

# Default camera position (looking at origin from distance)
DEFAULT_CAMERA_POSITION = spy.float3(0.0, 0.0, 3.0)
DEFAULT_CAMERA_TARGET = spy.float3(0.0, 0.0, 0.0)
DEFAULT_CAMERA_UP = spy.float3(0.0, 1.0, 0.0)

# Asset paths
HELMET_ASSET = Path("data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
ENVMAP_ASSET = Path("data/assets/envmaps/aerodynamics_workshop_512.hdr")
SPEC_GLOSS_ASSET = Path("data/assets/kronos/SpecGlossVsMetalRough/glTF/SpecGlossVsMetalRough.gltf")
MULTI_UV_ASSET = Path("data/assets/kronos/MultiUVTest/glTF/MultiUVTest.gltf")


@dataclass(frozen=True)
class BenchmarkCase:
    """One path tracer workload to measure."""

    scene: Path
    #: Environment map intensity, or None to render without one.
    envmap_intensity: Optional[float]
    max_depth: int
    enable_nee_mis: bool = True
    spp: int = BENCHMARK_SPP
    fov_y: float = 45.0
    camera_position: spy.float3 = DEFAULT_CAMERA_POSITION
    tolerance: float = 0.01


CASES = {
    # Direct lighting only, so the cost is dominated by primary rays and BSDF
    # evaluation rather than by light sampling.
    "pathtracer_helmet": BenchmarkCase(
        scene=HELMET_ASSET,
        envmap_intensity=None,
        max_depth=4,
        enable_nee_mis=False,
    ),
    "pathtracer_mis": BenchmarkCase(
        scene=HELMET_ASSET,
        envmap_intensity=1.0,
        max_depth=4,
    ),
    # Same scene and lighting as pathtracer_mis, isolating the cost of path
    # depth. Fewer samples because each one is roughly twice as expensive.
    "high_bounce_gi": BenchmarkCase(
        scene=HELMET_ASSET,
        envmap_intensity=1.0,
        max_depth=8,
        spp=BENCHMARK_SPP // 2,
        tolerance=0.02,
    ),
    "multi_material_scene": BenchmarkCase(
        scene=SPEC_GLOSS_ASSET,
        envmap_intensity=0.5,
        max_depth=3,
        fov_y=60.0,
        # Camera further back for larger scene.
        camera_position=spy.float3(0.0, 0.0, 5.0),
        tolerance=0.02,
    ),
    # Multi-UV meshes with normal maps, so texture sampling dominates.
    "texture_heavy": BenchmarkCase(
        scene=MULTI_UV_ASSET,
        envmap_intensity=0.8,
        max_depth=4,
    ),
}


def _setup_camera(scene: f2.Scene, case: BenchmarkCase) -> f2.Camera:
    """Create and position a camera looking at the scene origin."""
    camera = helpers.create_test_camera(
        scene,
        width=BENCHMARK_WIDTH,
        height=BENCHMARK_HEIGHT,
        fov_y=case.fov_y,
        position=case.camera_position,
    )
    direction = spy.math.normalize(DEFAULT_CAMERA_TARGET - case.camera_position)
    camera.entity.transform.rotation = spy.math.quat_from_look_at(direction, DEFAULT_CAMERA_UP)
    return camera


def _add_env_map(scene: f2.Scene, path: Path, intensity: float) -> None:
    """Add an environment map light to the scene."""
    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = str(path)
    # Historical baselines use RGB intensity; newer baselines use a scalar.
    env_map.intensity = type(env_map.intensity)(intensity)


@pytest.mark.benchmark
@pytest.mark.benchmark_workload("case_name")
@pytest.mark.parametrize(
    "device_type",
    # Only these backends have validated image references for this suite.
    # Keep the configured device selection, including an empty intersection.
    [
        device_type
        for device_type in helpers.DEFAULT_DEVICE_TYPES
        if device_type in (spy.DeviceType.vulkan, spy.DeviceType.cuda)
    ],
)
@pytest.mark.parametrize("case_name", list(CASES), ids=list(CASES))
def test_benchmark_pathtracer(
    case_name: str,
    device_type: spy.DeviceType,
    benchmark_image_test: BenchmarkImageTest,
) -> None:
    """Benchmark one path tracer workload end to end."""
    case = CASES[case_name]
    if not case.scene.exists():
        pytest.skip(f"Asset not found: {case.scene}")
    if case.envmap_intensity is not None and not ENVMAP_ASSET.exists():
        pytest.skip(f"Asset not found: {ENVMAP_ASSET}")

    # Uncached, so each case has its own compilation reports. A device shared
    # with earlier cases has already paid for the shared renderer shaders, and
    # compile_time would then describe the case that happened to run first.
    device = helpers.get_device(device_type, use_cache=False)
    scene = f2.Scene.load(device, str(case.scene))
    if case.envmap_intensity is not None:
        _add_env_map(scene, ENVMAP_ASSET, case.envmap_intensity)
    scene.update()

    camera = _setup_camera(scene, case)

    pipeline = PathTracerPipeline.create(device)
    pipeline.spp = case.spp
    pipeline.tone_map = True
    pipeline.tonemapper.auto_exposure = False
    pipeline.path_tracer.enable_nee = case.enable_nee_mis
    pipeline.path_tracer.enable_mis = case.enable_nee_mis
    pipeline.path_tracer.max_depth = case.max_depth

    def render(
        cmd: spy.CommandEncoder | None = None,
        sampler: SampleTimer | None = None,
    ) -> RenderResult:
        """Produce one complete, independent frame per call.

        These benchmarks measure whole-frame cost and do not use the sampler.
        PathTracerPipeline loops over all samples per accumulation internally,
        so there is no seam at which individual dispatches could be timed
        without reimplementing what the pipeline already does.
        """
        # Reset ensures we start fresh accumulation each benchmark iteration.
        pipeline.reset(cmd=cmd)
        result = pipeline(scene, camera, cmd=cmd)
        # With a command encoder the texture is not resolved yet, so hand it
        # back and let the fixture convert once the device has idled.
        if cmd is not None:
            return result
        return result.to_numpy()

    benchmark_image_test(
        device,
        render,
        # SlangPy creates graphics pipelines lazily by default, which lets
        # compilation cost land inside a timed iteration. Handing over the
        # scene-specialised path tracer module forces that work into warmup,
        # where it is measured as compile time rather than render time.
        compile_modules=[pipeline.path_tracer._get_module(scene)],
        tolerance=case.tolerance,
    )
