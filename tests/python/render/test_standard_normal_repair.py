# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type",
    [f2.StandardMaterial, f2.StandardSpecGlossMaterial, f2.GLTFMaterial, f2.GLTFSpecGlossMaterial],
)
def test_standard_normal_map_crosses_geometric_horizon(
    device_type: spy.DeviceType, material_type: type, device: spy.Device
) -> None:
    # The interpolated normal is 60 degrees from the geometric normal. Every
    # texel is a valid positive-Z tangent normal, including the steep ones that
    # push the mapped normal across the geometric horizon.
    angles = np.deg2rad([-60.0, -30.0, 0.0, 29.9, 30.0, 30.1, 60.0, 89.0])
    texels = np.zeros((1, len(angles), 4), dtype=np.float32)
    texels[0, :, 0] = 0.5 + 0.5 * np.sin(angles)
    texels[0, :, 1] = 0.5
    texels[0, :, 2] = 0.5 + 0.5 * np.cos(angles)
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=len(angles),
        height=1,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource,
        data=texels,
    )
    scene = f2.Scene.create(device)
    props = {"normal_texture": texture, "double_sided": True}
    if material_type in (f2.StandardMaterial, f2.GLTFMaterial):
        props.update(base_color_factor=spy.float3(0.8), metallic_factor=0.0, roughness_factor=0.5)
    else:
        props.update(
            diffuse_factor=spy.float3(0.8), specular_factor=spy.float3(0.04), glossiness_factor=0.5
        )
    material = scene.create_material(material_type, f2.Properties(props))
    scene.update()
    helper = SceneShaderHelper(device)
    source = spy.Module.load_from_file(device, "render/test_standard_normal_repair.slang")
    module, requirements = helper.prepare(scene, source)
    probe = (
        module.standard_normal_repair_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    uv = np.array([[(i + 0.5) / len(angles), 0.5] for i in range(len(angles))], dtype=np.float32)

    for side in (1.0, -1.0):
        for handedness in (1.0, -1.0):
            output = probe(int(material.material_id), uv, side, handedness)
            cursor = output.cursor()
            cursor.load()
            results = [cursor[i].read() for i in range(len(angles))]
            normals = np.array([list(value["adjusted_normal"]) for value in results])
            original = np.array([list(value["original_normal"]) for value in results])
            assert original[-1, 2] * side < -0.5
            assert np.all(np.isfinite(normals))
            np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-5)
            assert np.all(normals[:, 2] * side > 0.0)
            # Reflection must stay on the visible side, including across the
            # mapped-normal horizon. The repaired normals must remain continuous.
            assert np.all(2.0 * normals[:, 2] ** 2 - 1.0 > -0.002)
            assert np.linalg.norm(normals[3] - normals[5]) < 0.01
            np.testing.assert_allclose(normals[:2], original[:2], atol=1e-5)
            for result, normal in zip(results, normals):
                tangent, bitangent = np.array(result["tangent"]), np.array(result["bitangent"])
                assert abs(normal @ tangent) < 1e-5
                assert abs(normal @ bitangent) < 1e-5
                assert np.sign(np.dot(np.cross(normal, tangent), bitangent)) == handedness * side
                assert np.min(np.array(result["eval"])) > 0.0

    # The repair must not turn a one-sided material into a two-sided material.
    material.double_sided = False
    scene.update()
    output = probe(int(material.material_id), uv, -1.0, 1.0)
    cursor = output.cursor()
    cursor.load()
    for i in range(len(angles)):
        result = cursor[i].read()
        assert result["adjusted_normal"].z > 0.0
        np.testing.assert_array_equal(np.array(result["eval"]), np.zeros(3))
