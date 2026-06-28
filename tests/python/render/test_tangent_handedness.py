# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import RenderMode, SceneShaderHelper

WIDTH = 8
HEIGHT = 8
SQRT_HALF = float(np.sqrt(0.5))


def _create_mirrored_quad() -> f2.ImporterMesh:
    mesh = f2.ImporterMesh.create(
        [np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)],
        {
            "position": np.array(
                [
                    [-1.0, -1.0, 0.0],
                    [1.0, -1.0, 0.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            "normal": np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
            "tex_coord": np.array(
                [
                    [0.0, 2.0],
                    [2.0, 2.0],
                    [2.0, 0.0],
                    [0.0, 0.0],
                ],
                dtype=np.float32,
            ),
        },
    )
    mesh.name = "mirrored_quad"
    mesh.add_tangents_from_uvs()

    subgeometries = list(mesh.subgeometries)
    subgeometries[0].name = "mirrored_quad"
    subgeometries[0].material_name = "debug_material"
    mesh.subgeometries = subgeometries
    return mesh


def _create_scene(device: spy.Device, transform_scale: spy.float3) -> f2.Scene:
    scene = f2.Scene.create(device)
    material = scene.create_material("StandardMaterial")

    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [1.0, 1.0, 0.0],
                [-1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        sub_mesh_indices=[np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)],
        normals=np.array([[0.0, 0.0, 1.0]] * 4, dtype=np.float32),
        tangents=np.array([[1.0, 0.0, 0.0]] * 4, dtype=np.float32),
        handedness=np.full((4,), -1.0, dtype=np.float32),
        texcoords=np.array(
            [
                [0.0, 2.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        name="mirrored_quad",
    )

    entity = scene.create_entity()
    transform = f2.Transform()
    transform.scale = transform_scale
    entity.transform = transform
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]

    scene.update()
    return scene


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_render_frame_handedness(device_type: spy.DeviceType) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("The editor debug shader uses ray queries and is not emitted for CUDA.")

    device = helpers.get_device(device_type)
    mesh = _create_mirrored_quad()

    np.testing.assert_allclose(
        np.asarray(mesh.handedness, dtype=np.float32),
        np.full((4,), -1.0, dtype=np.float32),
        atol=0.0,
    )

    # This exercises the full triangle SurfaceInteraction path. The mesh has imported negative
    # tangent-space handedness. Under a negative-X instance transform the world tangent flips, and
    # the transform determinant must also flip the handedness so the world bitangent stays stable.
    helper = SceneShaderHelper(device)

    def render_center_color(transform_scale: spy.float3, render_mode: RenderMode) -> np.ndarray:
        scene = _create_scene(device, transform_scale)
        camera = helpers.create_test_camera(
            scene,
            width=WIDTH,
            height=HEIGHT,
            fov_y=45.0,
            position=spy.float3(0.0, 0.0, 4.0),
            rotation=spy.math.quat_from_look_at(
                spy.float3(0.0, 0.0, -1.0),
                spy.float3(0.0, 1.0, 0.0),
            ),
        )
        scene.active_camera = camera
        scene.update()

        module = helper.get_module(scene, "falcor2.editor")
        output = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=WIDTH,
            height=HEIGHT,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=f"mirrored_{transform_scale.x}_{render_mode.name}",
        )

        module.render_debug_no_ids.type_conformances(scene.requirements.type_conformances).write(
            helper.bind_scene
        ).dispatch(
            thread_count=spy.uint3(WIDTH, HEIGHT, 1),
            camera=camera.get_uniforms(),
            render_mode=int(render_mode),
            color_output=output,
        )
        color = output.to_numpy()
        return color[HEIGHT // 2, WIDTH // 2, :3]

    # Identity transform: imported tangent points +X.
    tangent_color = render_center_color(spy.float3(1.0, 1.0, 1.0), RenderMode.shading_tangent)
    np.testing.assert_allclose(
        tangent_color,
        np.array([1.0, 0.5, 0.5], dtype=np.float32),
        atol=1e-4,
    )

    # Identity transform: imported negative handedness makes the bitangent point -Y.
    bitangent_color = render_center_color(spy.float3(1.0, 1.0, 1.0), RenderMode.shading_bitangent)
    np.testing.assert_allclose(
        bitangent_color,
        np.array([0.5, 0.0, 0.5], dtype=np.float32),
        atol=1e-4,
    )

    # Negative-X transform: the world tangent flips to -X.
    mirrored_tangent_color = render_center_color(
        spy.float3(-1.0, 1.0, 1.0), RenderMode.shading_tangent
    )
    np.testing.assert_allclose(
        mirrored_tangent_color,
        np.array([0.0, 0.5, 0.5], dtype=np.float32),
        atol=1e-4,
    )

    # Negative-X transform: transform handedness must also flip so the bitangent remains -Y.
    mirrored_bitangent_color = render_center_color(
        spy.float3(-1.0, 1.0, 1.0), RenderMode.shading_bitangent
    )
    np.testing.assert_allclose(
        mirrored_bitangent_color,
        np.array([0.5, 0.0, 0.5], dtype=np.float32),
        atol=1e-4,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_material_frame_handedness(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    # The normal map points halfway between local +Y and local +Z. That keeps the resulting frame
    # non-degenerate while making bitangent handedness visible as a sign change in the Y component.
    normal_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=1,
        height=1,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource,
        data=np.array(
            [[0.5, 0.5 + 0.5 * SQRT_HALF, 0.5 + 0.5 * SQRT_HALF, 0.0]],
            dtype=np.float32,
        ),
    )
    material_props = f2.Properties({"normal_texture": normal_texture, "normal_texture_scale": 1.0})
    scene = f2.Scene.create(device)
    standard_material = scene.create_material(f2.StandardMaterial, material_props)
    spec_gloss_material = scene.create_material(f2.StandardSpecGlossMaterial, material_props)
    openpbr_material = scene.create_material("OpenPBRMaterial", material_props)
    scene.update()

    signs = spy.Tensor.from_numpy(device, np.array([1.0, -1.0], dtype=np.float32))
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_tangent_handedness.slang")
    module = helper.get_module(scene, source_module)

    # This checks that Standard, SpecGloss, and OpenPBR preserve the input handedness when their
    # normal-map paths rebuild the material frame, and that MaterialX geomprop sees the same sign.
    material_probe = (
        module.material_frame_probe.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )

    material_result = material_probe(
        standard_material_id=int(standard_material.material_id),
        spec_gloss_material_id=int(spec_gloss_material.material_id),
        openpbr_material_id=int(openpbr_material.material_id),
        handedness=signs,
    ).to_numpy()

    np.testing.assert_allclose(
        material_result,
        np.array(
            [
                [SQRT_HALF, SQRT_HALF, SQRT_HALF, 1.0],
                [SQRT_HALF, SQRT_HALF, SQRT_HALF, -1.0],
            ],
            dtype=np.float32,
        ),
        atol=1e-5,
    )

    # This separately checks that the generated MaterialX normalmap implementation uses the
    # supplied bitangent instead of reconstructing an always-positive bitangent.
    materialx_probe = module.materialx_normalmap_probe.as_func().return_type(spy.Tensor)

    materialx_result = materialx_probe(signs).to_numpy()

    np.testing.assert_allclose(
        materialx_result,
        np.array([[1.0, SQRT_HALF], [-1.0, -SQRT_HALF]], dtype=np.float32),
        atol=1e-5,
    )
