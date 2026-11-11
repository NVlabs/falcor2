# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendernodes.reference_pathtracer_node import (
    ReferencePathTracerNode,
    SchedulingMode,
    VisibilityRayMode,
)


class _LSSHitMaterial(f2.Material):
    def __init__(self) -> None:
        super().__init__()
        self.slang_type_name = "LSSHitMaterial"
        self._module: spy.SlangModule | None = None

    def required_capabilities(self) -> f2.SceneCapabilities:
        return f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)

    def update(self, ctx: f2.SceneUpdateContext) -> None:
        if self._module is None:
            self._module = self.scene.device.load_module("pathtracer/test_lss_closest_hit.slang")

    def required_module(self) -> spy.SlangModule | None:
        return self._module

    def write_to_cursor(self, cursor: spy.ShaderCursor) -> None:
        pass

    def get_this(self) -> dict[str, str]:
        return {"_type": self.slang_type_name}


def _force_procedural_lss(
    monkeypatch: pytest.MonkeyPatch,
    device: spy.Device,
    instance: f2.GeometryInstance,
    node: ReferencePathTracerNode,
    positions: np.ndarray,
    radii: np.ndarray,
) -> list[Any]:
    """Use real procedural intersection and closest-hit shaders with the fixture's scene buffers."""
    resources: list[Any] = []

    def build(inputs: list[Any], kind: spy.AccelerationStructureKind) -> spy.AccelerationStructure:
        desc = spy.AccelerationStructureBuildDesc({"inputs": inputs})
        sizes = device.get_acceleration_structure_sizes(desc)
        scratch = device.create_buffer(
            size=sizes.scratch_size, usage=spy.BufferUsage.unordered_access
        )
        structure = device.create_acceleration_structure(
            kind=kind, size=sizes.acceleration_structure_size
        )
        encoder = device.create_command_encoder()
        encoder.build_acceleration_structure(
            desc=desc, dst=structure, src=None, scratch_buffer=scratch
        )
        device.submit_command_buffer(encoder.finish())
        resources.extend([scratch, structure])
        return structure

    lower = (positions - radii[:, None]).reshape(-1, 2, 3).min(axis=1)
    upper = (positions + radii[:, None]).reshape(-1, 2, 3).max(axis=1)
    bounds = np.concatenate((lower, upper), axis=1).astype(np.float32)
    aabbs = device.create_buffer(usage=spy.BufferUsage.shader_resource, data=bounds)
    geometry = spy.AccelerationStructureBuildInputProceduralPrimitives(
        {
            "aabb_buffers": [aabbs],
            "aabb_stride": 24,
            "primitive_count": len(bounds),
            "flags": spy.AccelerationStructureGeometryFlags.opaque,
        }
    )
    blas = build([geometry], spy.AccelerationStructureKind.bottom_level)
    instances = device.create_acceleration_structure_instance_list(1)
    instances.write(
        0,
        {
            "transform": spy.float3x4(np.eye(4, dtype=np.float32)[:3]),
            "instance_id": int(instance.geometry_instance_id),
            "instance_mask": 255,
            "instance_contribution_to_hit_group_index": 0,
            "flags": spy.AccelerationStructureInstanceFlags.none,
            "acceleration_structure": blas.handle,
        },
    )
    tlas = build([instances.build_input_instances()], spy.AccelerationStructureKind.top_level)
    resources.extend([aabbs, instances])

    def bind(cursor: spy.ShaderCursor) -> None:
        cursor.g_scene.tlas = tlas

    original_get_render_func = node._get_render_func

    def get_render_func(module: Any, write_guide: dict[str, Any] | None = None) -> Any:
        groups = [
            spy.HitGroupDesc(
                {
                    "hit_group_name": "procedural_scatter",
                    "closest_hit_entry_point": "_scene_scatter_procedural_lss_closest_hit",
                    "intersection_entry_point": "procedural_lss_intersection",
                }
            ),
            spy.HitGroupDesc(
                {
                    "hit_group_name": "procedural_visibility",
                    "intersection_entry_point": "procedural_lss_intersection",
                }
            ),
        ]
        return (
            original_get_render_func(module, write_guide)
            .prelude("// Procedural LSS closest-hit test pipeline\n")
            .write(bind)
            .ray_tracing(
                hit_groups=groups,
                hit_group_names=["procedural_scatter", "procedural_visibility"],
                miss_entry_points=["_scene_scatter_miss", "_scene_visibility_miss"],
                max_recursion=2,
                max_ray_payload_size=128,
                max_attribute_size=8,
                flags=spy.RayTracingPipelineFlags.none,
            )
        )

    monkeypatch.setattr(node, "_get_render_func", get_render_func)
    return resources


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("scheduling_mode", [SchedulingMode.simple, SchedulingMode.ser])
@pytest.mark.parametrize("oblique", [False, True])
@pytest.mark.parametrize(
    "procedural,visibility_mode",
    [
        pytest.param(False, VisibilityRayMode.ray_query, id="native-ray-query"),
        pytest.param(False, VisibilityRayMode.trace_ray, id="native-trace-ray"),
        pytest.param(True, VisibilityRayMode.trace_ray, id="procedural-trace-ray"),
    ],
)
def test_lss_closest_hit_reconstruction(
    device_type: spy.DeviceType,
    device: spy.Device,
    visibility_mode: VisibilityRayMode,
    scheduling_mode: SchedulingMode,
    oblique: bool,
    procedural: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise native and procedural TraceRay shading, UV reads, and packed-hit reconstruction."""
    if device_type == spy.DeviceType.cuda and visibility_mode == VisibilityRayMode.ray_query:
        pytest.skip("CUDA has no inline visibility queries")
    if scheduling_mode == SchedulingMode.ser and not device.has_feature(
        spy.Feature.shader_execution_reordering
    ):
        pytest.skip("Shader execution reordering is unavailable")
    # Exercise endpoint-relative projection on oblique, small segments away from the origin.
    basis = np.eye(3)
    scale = 1.0
    offset = np.zeros(3)
    if oblique:
        forward = np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0)
        side = np.cross(forward, [0.0, 0.0, 1.0])
        side /= np.linalg.norm(side)
        basis = np.stack((forward, side, np.cross(forward, side)))
        scale = 0.01
        offset = np.array([0.123, -0.234, 0.345])
    positions = np.array([[0, 0, 0], [1, 0, 0], [2.5, 0, 0], [2.5, 0, 0], [4, 0, 0], [5, 0, 0]])
    scene = f2.Scene.create(device)
    geometry = scene.create_geometry(f2.StaticCurveGeometry)
    uv = np.array(
        [[0.2, 0.8], [0.9, 0.1], [0.1, 0.2], [0.9, 0.7], [0.3, 0.6], [0.8, 0.4]], dtype=np.float32
    )
    positions = np.asarray(positions @ basis * scale + offset, dtype=np.float32)
    radii = np.array([0.1, 0.1, 0.1, 0.2, 0.1, 0.2], dtype=np.float32) * scale
    geometry.set_curve_data(
        positions,
        radii,
        np.array([0, 1, 2, 3, 4, 5], dtype=np.uint32),
        vertex_attributes=[
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 0, uv, name="st")
        ],
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [scene.create_material(_LSSHitMaterial)]
    scene.update()
    if (
        not procedural
        and scene.requirements.lss_intersection_mode != f2.LSSIntersectionMode.hardware
    ):
        pytest.skip("Native linear swept sphere intersection is unavailable")
    node = ReferencePathTracerNode(device)
    node.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    node.scheduling_mode = scheduling_mode
    node.visibility_ray_mode = visibility_mode
    node.max_depth = 1
    node.enable_depth_of_field = False
    node.guide_output_specs = {}
    node.pixel_filter = f2.BoxReconstructionFilter(width=1)
    if procedural:
        # Retain custom acceleration-structure resources until the fixture's renders complete.
        resources = _force_procedural_lss(monkeypatch, device, instance, node, positions, radii)
    # Narrow views make every ray a known side/cap/pole hit, avoiding stochastic edge coverage.
    # Collapsed segments retain the object-space +Z parameterization even in the oblique fixture.
    cases = [
        (0.2, 0, (0.2 + 0.1) / 1.2),
        (-0.05, 0, (-0.05 + 0.1) / 1.2),
        (1.05, 0, (1.05 + 0.1) / 1.2),
        (4.5, 4, 0.6 / 1.3),
        (3.95, 4, 0.05 / 1.3),
        (5.1, 4, 1.2 / 1.3),
        (2.5, 2, (basis[2, 2] + 1) / 2),
        (2.6, 2, (0.1 * basis[0, 2] + np.sqrt(0.2**2 - 0.1**2) * basis[2, 2] + 0.2) / 0.4),
    ]
    for x, first, weight in cases:
        camera = helpers.create_test_camera(
            scene,
            width=8,
            height=8,
            fov_y=0.001,
            position=spy.float3(*(np.array([x, 0, 3]) @ basis * scale + offset)),
            rotation=spy.math.quat_from_look_at(spy.float3(*(-basis[2])), spy.float3(*basis[1])),
        )
        pixels = node(scene, camera, iteration=0, pixel_offset=spy.float2(0))[0].to_numpy()
        assert np.isfinite(pixels).all()
        expected = np.append(uv[first] * (1 - weight) + uv[first + 1] * weight, 1.0)
        np.testing.assert_allclose(
            pixels[..., :3], np.broadcast_to(expected, pixels[..., :3].shape), atol=1e-4, rtol=0
        )
    if procedural:
        device.wait()
        resources.clear()
