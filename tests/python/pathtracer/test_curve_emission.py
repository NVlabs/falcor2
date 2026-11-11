# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from pathtracer.test_pathtracer import _add_quad, _make_quad_camera, _render_mean


def _add_emissive_curve(
    scene: f2.Scene, *, z: float, radius: float, half_length: float, emission: spy.float3
) -> None:
    geometry = scene.create_geometry(f2.StaticCurveGeometry)
    geometry.set_curve_data(
        np.array([[-half_length, 0, z], [half_length, 0, z]], dtype=np.float32),
        np.full(2, radius, dtype=np.float32),
        np.array([0, 1], dtype=np.uint32),
    )
    material = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"emissive_factor": emission, "base_color_factor": spy.float3(0.0)}),
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_pathtracer_curve_emission_without_triangle_lights(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    emission = spy.float3(2.0, 0.5, 0.25)
    _add_emissive_curve(scene, z=0.0, radius=0.8, half_length=2.0, emission=emission)
    image = _render_mean(device, scene, _make_quad_camera(scene), max_depth=1, enable_nee=True)
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image, np.broadcast_to(np.asarray(emission), image.shape), atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_pathtracer_secondary_curve_emission_with_triangle_light(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    receiver = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"base_color_factor": spy.float3(1.0), "roughness_factor": 1.0}),
    )
    _add_quad(scene, z=0.0, normal_z=-1.0, material=receiver)
    # Behind the camera: only material-sampled secondary rays can see this emitter.
    _add_emissive_curve(
        scene, z=-6.0, radius=3.0, half_length=10.0, emission=spy.float3(2.0, 0.0, 0.0)
    )
    # A valid sampleable triangle light activates NEE coverage and triangle MIS.
    # Its green emission cannot contribute to the red curve signal being compared.
    triangle_light = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"emissive_factor": spy.float3(0.0, 2.0, 0.0)}),
    )
    _add_quad(scene, z=-1.0, normal_z=1.0, material=triangle_light, center_x=3.0, size=0.5)
    camera = _make_quad_camera(scene, width=16, height=16)
    red_means = []
    for enable_nee, enable_mis in ((False, False), (True, False), (True, True)):
        image = _render_mean(
            device,
            scene,
            camera,
            iterations=64,
            enable_nee=enable_nee,
            enable_mis=enable_mis,
            enable_russian_roulette=False,
            max_depth=2,
        )
        assert np.isfinite(image).all()
        red_means.append(float(image[..., 0].mean()))
    assert min(red_means) > 0.1, red_means
    np.testing.assert_allclose(red_means[1:], red_means[0], rtol=0.12, atol=0.01)
