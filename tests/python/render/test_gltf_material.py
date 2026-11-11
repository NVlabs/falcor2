# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


def _analytic_eval(device: spy.Device, scene: f2.Scene) -> Any:
    helper = SceneShaderHelper(device)
    module, requirements = helper.prepare(scene, "render/test_gltf_material.slang")
    return (
        module.analytic_material_eval.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
    )


def _ggx_cosine_factor(wi: np.ndarray, wo: np.ndarray, roughness: float) -> float:
    """Float64 isotropic GGX D * correlated Smith V * cos(theta_o), without Fresnel."""
    h = (wi + wo) / np.linalg.norm(wi + wo)
    alpha_squared = roughness**4
    d = alpha_squared / (np.pi * (1 + (alpha_squared - 1) * h[2] ** 2) ** 2)
    v = 0.5 / (
        wo[2] * np.sqrt(wi[2] ** 2 * (1 - alpha_squared) + alpha_squared)
        + wi[2] * np.sqrt(wo[2] ** 2 * (1 - alpha_squared) + alpha_squared)
    )
    return float(d * v * wo[2])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_analytic_specular_endpoints(device: spy.Device) -> None:
    """Clamping before strength and colored F0/scalar F90 survive material preparation."""
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.GLTFMaterial)
    material.base_color_factor = spy.float3(0)
    material.metallic_factor = 0
    material.roughness_factor = 0.5
    material.ior = 2
    material.specular_color_factor = spy.float3(16, 0.5, 0)
    material.specular_factor = 0.25
    scene.update()
    evaluate = _analytic_eval(device, scene)
    # ((2-1)/(2+1))^2 = 1/9; clamp tint first, THEN apply the 1/4 strength.
    f0 = np.array([1, 1 / 18, 0]) / 4
    for cosine in (1.0, 0.125):
        wi = np.array([np.sqrt(1 - cosine**2), 0, cosine])
        wo = wi * [-1, 1, 1]
        fresnel = f0 + (0.25 - f0) * (1 - cosine) ** 5
        expected = fresnel * _ggx_cosine_factor(wi, wo, 0.5)
        actual = evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.0, True)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)
    material.specular_factor = 0
    scene.update()
    np.testing.assert_allclose(
        evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.0, True),
        0,
        atol=1e-7,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_analytic_matched_ior_and_tir(device: spy.Device) -> None:
    """Equal bulk media have no interface reflection; exiting above critical angle reflects fully."""
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.GLTFMaterial)
    material.base_color_factor = spy.float3(0.75, 0.5, 0.25)
    material.metallic_factor = 0
    material.roughness_factor = 0.5
    material.transmission_factor = 1
    material.thickness_factor = 1
    material.ior = 1.5
    scene.update()
    evaluate = _analytic_eval(device, scene)
    wi = np.array([np.sqrt(0.75), 0, 0.5])
    wo = wi * [-1, 1, 1]
    np.testing.assert_allclose(
        evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.5, True),
        0,
        atol=1e-7,
    )
    # n_i/n_t = 1.5 and cos(theta_h)=1/2: Snell gives cos^2(theta_t)=-11/16.
    # Therefore F=1, independent of tint; a transmitted normal direction is also forbidden.
    reflected = evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.0, False)
    np.testing.assert_allclose(reflected, _ggx_cosine_factor(wi, wo, 0.5), rtol=3e-5)
    transmitted = evaluate(
        int(material.material_id), spy.float3(wi), spy.float3(0, 0, -1), 1.0, False
    )
    np.testing.assert_allclose(transmitted, 0, atol=1e-7)
    # A thin sheet also uses eta=1 internally, but must keep its air/material Fresnel.
    material.thickness_factor = 0
    scene.update()
    fresnel = 0.04 + 0.96 * (1 - wi[2]) ** 5
    thin = evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.5, True)
    np.testing.assert_allclose(thin, fresnel * _ggx_cosine_factor(wi, wo, 0.5), rtol=3e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_analytic_specgloss_attenuation(device: spy.Device) -> None:
    """SpecGloss preserves diffuse hue via max(F0) and squares 1-glossiness."""
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.GLTFSpecGlossMaterial)
    diffuse, f0 = np.array([0.75, 0.5, 0.25]), np.array([0.125, 0.5, 0.25])
    material.diffuse_factor = spy.float3(diffuse)
    material.specular_factor = spy.float3(f0)
    material.glossiness_factor = 0.5
    scene.update()
    evaluate = _analytic_eval(device, scene)
    for cosine in (1.0, 0.25):
        wi, wo = np.array([np.sqrt(1 - cosine**2), 0, cosine]), np.array([0, 0.6, 0.8])
        h = (wi + wo) / np.linalg.norm(wi + wo)
        fresnel = f0 + (1 - f0) * (1 - wi @ h) ** 5
        expected = diffuse * 0.5 * wo[2] / np.pi + fresnel * _ggx_cosine_factor(wi, wo, 0.5)
        actual = evaluate(int(material.material_id), spy.float3(wi), spy.float3(wo), 1.0, True)
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=1e-7)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_delta_sampling(device_type: spy.DeviceType, device: spy.Device) -> None:
    module = spy.Module(device.load_module("render/test_gltf_material.slang"))
    seeds = spy.Tensor.from_numpy(device, np.arange(512, dtype=np.uint32))
    for metallic, roughness, base in (
        (1.0, 0.0, 0.8),
        (0.0, 0.0, 0.8),
    ):
        for cosine in (1.0, 0.02):
            wi = spy.float3(float(np.sqrt(1.0 - cosine**2)), 0.0, cosine)
            result = module.sample_probe.return_type(spy.Tensor)(
                wi, spy.float3(base), metallic, roughness, seeds
            )
            cursor = result.cursor()
            cursor.load()
            valid = 0
            for i in range(512):
                sample = cursor[i].read()
                if not sample["valid"]:
                    continue
                valid += 1
                assert sample["wo"].z > 0.0
                assert np.all(np.isfinite(sample["weight"]))
                if sample["delta"]:
                    assert sample["pdf"] == 0.0
                else:
                    assert sample["pdf"] > 0.0
                    # Match the shared GGX consistency test's tolerance: at grazing
                    # angles reconstructing h from wi + wo loses float32 precision.
                    assert sample["pdf"] == pytest.approx(
                        sample["evaluated_pdf"], rel=1e-3, abs=1e-7
                    )
                if metallic == 1.0 and roughness == 0.0:
                    assert sample["delta"]
                    np.testing.assert_allclose(
                        sample["weight"], base + (1 - base) * (1 - cosine) ** 5, rtol=2e-5
                    )
            assert valid > 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_diffuse_gradient(device_type: spy.DeviceType, device: spy.Device) -> None:
    module = spy.Module(device.load_module("render/test_gltf_material.slang"))
    wi, wo = np.array([0.6, 0.0, 0.8]), np.array([-0.8, 0.0, 0.6])
    h = (wi + wo) / np.linalg.norm(wi + wo)
    fresnel = 0.04 + 0.96 * (1.0 - wi @ h) ** 5
    expected = (1.0 - fresnel) * wo[2] / np.pi
    actual = module.diffuse_gradient(spy.float3(wi), spy.float3(wo))
    np.testing.assert_allclose(actual, expected, rtol=2e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_material_uv_transform(device: spy.Device) -> None:
    """Check affine order and lower-left conversion, then invalidate the sparse record."""
    scene = f2.Scene.create(device, f2.SceneConfig(uv_origin=f2.UVOrigin.lower_left))
    material = scene.create_material(f2.GLTFMaterial)
    material.base_color_texture_offset = spy.float2(0.2, -0.1)
    material.base_color_texture_rotation = float(np.pi / 2)
    material.base_color_texture_uv_scale = spy.float2(-0.5, 1.5)
    scene.update()
    module, _ = SceneShaderHelper(device).prepare(scene, "render/test_gltf_material.slang")
    probe = module.transform_probe.as_func().return_type(spy.Tensor)
    uv = np.array([[0.2, 0.7], [0.8, 0.1]], dtype=np.float32)
    # Convert to authored upper-left UVs, scale, rotate clockwise 90 degrees,
    # translate, then convert back. The fixed landmarks detect reversed order/sign.
    expected = np.array([[0.65, 1.0], [1.55, 0.7]])
    np.testing.assert_allclose(probe(material, uv, 0).to_numpy(), expected, atol=2e-7)
    np.testing.assert_array_equal(probe(material, uv, 1).to_numpy(), uv)
    material.base_color_texture_offset = spy.float2(0)
    material.base_color_texture_rotation = 0
    material.base_color_texture_uv_scale = spy.float2(1)
    scene.update()
    np.testing.assert_array_equal(probe(material, uv, 0).to_numpy(), uv)
