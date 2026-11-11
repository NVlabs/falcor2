# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


_MATERIALX_LOBE_NORMALS = """<materialx version="1.39" colorspace="lin_rec709">
  <constant name="grazing_normal" type="vector3">
    <input name="value" type="vector3" value="0.99875234, 0.04993762, 0.0" />
  </constant>
  <constant name="aligned_normal" type="vector3">
    <input name="value" type="vector3" value="0.0, 1.0, 0.0" />
  </constant>
  <constant name="subsurface_normal" type="vector3">
    <input name="value" type="vector3" value="-0.99875234, 0.04993762, 0.0" />
  </constant>
  <constant name="hair_normal" type="vector3">
    <input name="value" type="vector3" value="0.0, 0.04993762, 0.99875234" />
  </constant>
  <constant name="direct_normal" type="vector3">
    <input name="value" type="vector3" value="0.0, 0.04993762, -0.99875234" />
  </constant>
  <constant name="curve_direction" type="vector3">
    <input name="value" type="vector3" value="1.0, 0.0, 0.0" />
  </constant>
  <nodedef name="ND_nested_dielectric" node="nested_dielectric" nodegroup="pbr">
    <input name="normal" type="vector3" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="NG_nested_dielectric" nodedef="ND_nested_dielectric">
    <dielectric_bsdf name="dielectric" type="BSDF">
      <input name="weight" type="float" value="1.0" />
      <input name="tint" type="color3" value="0.8, 0.8, 0.8" />
      <input name="roughness" type="vector2" value="0.3, 0.3" />
      <input name="normal" type="vector3" interfacename="normal" />
      <input name="scatter_mode" type="string" value="R" />
    </dielectric_bsdf>
    <output name="out" type="BSDF" nodename="dielectric" />
  </nodegraph>
  <nodedef name="ND_nested_surface" node="nested_surface" nodegroup="pbr">
    <input name="bsdf" type="BSDF" />
    <input name="direct_normal" type="vector3" />
    <output name="out" type="surfaceshader" />
  </nodedef>
  <nodegraph name="NG_nested_surface" nodedef="ND_nested_surface">
    <nested_dielectric name="direct" type="BSDF">
      <input name="normal" type="vector3" interfacename="direct_normal" />
    </nested_dielectric>
    <mix name="mixed" type="BSDF">
      <input name="fg" type="BSDF" interfacename="bsdf" />
      <input name="bg" type="BSDF" nodename="direct" />
      <input name="mix" type="float" value="0.0625" />
    </mix>
    <surface name="surface" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="mixed" />
    </surface>
    <output name="out" type="surfaceshader" nodename="surface" />
  </nodegraph>
  <sheen_bsdf name="grazing" type="BSDF">
    <input name="weight" type="float" value="1.0" />
    <input name="color" type="color3" value="0.8, 0.2, 0.1" />
    <input name="roughness" type="float" value="0.3" />
    <input name="normal" type="vector3" nodename="grazing_normal" />
  </sheen_bsdf>
  <oren_nayar_diffuse_bsdf name="aligned" type="BSDF">
    <input name="weight" type="float" value="1.0" />
    <input name="color" type="color3" value="0.1, 0.2, 0.8" />
    <input name="roughness" type="float" value="0.3" />
    <input name="normal" type="vector3" nodename="aligned_normal" />
  </oren_nayar_diffuse_bsdf>
  <subsurface_bsdf name="subsurface" type="BSDF">
    <input name="weight" type="float" value="1.0" />
    <input name="color" type="color3" value="0.2, 0.8, 0.2" />
    <input name="normal" type="vector3" nodename="subsurface_normal" />
  </subsurface_bsdf>
  <chiang_hair_bsdf name="hair" type="BSDF">
    <input name="normal" type="vector3" nodename="hair_normal" />
    <input name="curve_direction" type="vector3" nodename="curve_direction" />
  </chiang_hair_bsdf>
  <mix name="mixed_surface" type="BSDF">
    <input name="fg" type="BSDF" nodename="grazing" />
    <input name="bg" type="BSDF" nodename="aligned" />
    <input name="mix" type="float" value="0.5" />
  </mix>
  <mix name="mixed_subsurface" type="BSDF">
    <input name="fg" type="BSDF" nodename="mixed_surface" />
    <input name="bg" type="BSDF" nodename="subsurface" />
    <input name="mix" type="float" value="0.25" />
  </mix>
  <mix name="mixed" type="BSDF">
    <input name="fg" type="BSDF" nodename="mixed_subsurface" />
    <input name="bg" type="BSDF" nodename="hair" />
    <input name="mix" type="float" value="0.125" />
  </mix>
  <nested_surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="mixed" />
    <input name="direct_normal" type="vector3" nodename="direct_normal" />
  </nested_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


def _normal_repair_document(hair: bool) -> str:
    # Keep the authored parameter graphs, but test the fiber and surface domains separately.
    document = ET.fromstring(_MATERIALX_LOBE_NORMALS)
    surface = document.find("nested_surface")
    assert surface is not None
    if hair:
        # A normal with a tangent component must not rotate the physical fiber axis.
        normal = document.find("constant[@name='hair_normal']/input")
        assert normal is not None
        normal.set("value", "0.5, 0.04993762, 0.99875234")
        surface.tag = "surface"
        surface.remove(surface.find("input[@name='direct_normal']"))
    surface.find("input[@name='bsdf']").set("nodename", "hair" if hair else "mixed_subsurface")
    return ET.tostring(document, encoding="unicode")


_COMPATIBILITY_LAYERING_MODE = f2.MaterialXLayeringMode.bsdf_mix

_LAYERING_MODES = [
    pytest.param(f2.MaterialXLayeringMode.closure_tree, marks=pytest.mark.nightly),
    pytest.param(
        f2.MaterialXLayeringMode.bsdf_mix,
        marks=(
            pytest.mark.nightly
            if _COMPATIBILITY_LAYERING_MODE != f2.MaterialXLayeringMode.bsdf_mix
            else ()
        ),
    ),
]

_DEVICE_TYPES = [
    device_type
    for device_type in helpers.DEFAULT_DEVICE_TYPES
    if device_type in (spy.DeviceType.d3d12, spy.DeviceType.vulkan)
]


def _float3(value: Any) -> np.ndarray:
    return np.array([float(value.x), float(value.y), float(value.z)], dtype=np.float32)


@pytest.mark.parametrize("device_type", _DEVICE_TYPES[:1])
def test_materialx_default_per_lobe_normal_repair(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Exercise every normal/frame/sample assertion without overriding the default."""
    _check_materialx_per_lobe_normal_repair(device, layering_mode=None)


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("layering_mode", _LAYERING_MODES)
def test_materialx_per_lobe_normal_repair(
    device_type: spy.DeviceType, device: spy.Device, layering_mode: f2.MaterialXLayeringMode
) -> None:
    _check_materialx_per_lobe_normal_repair(device, layering_mode)


def _check_materialx_per_lobe_normal_repair(
    device: spy.Device, layering_mode: f2.MaterialXLayeringMode | None
) -> None:
    scene = f2.Scene.create(device)
    props = f2.Properties()
    props["mtlx_buffer"] = _normal_repair_document(hair=False)
    props["mtlx_node_name"] = "M"
    if layering_mode is not None:
        props["mtlx_layering_mode"] = layering_mode
    material = scene.create_material(f2.MaterialXMaterial, props)
    scene.update()

    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(
        device, "render/test_materialx_lobe_normal_repair.slang"
    )
    module, requirements = helper.prepare(scene, source_module)
    probe = (
        module.materialx_lobe_normal_repair_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )

    uv = np.array([(0.5, 0.5)], dtype=np.float32)
    wi = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    wo = np.array([(0.0, 1.0, 0.0)], dtype=np.float32)
    sample_count = 16
    sample_seeds = np.array([(7 + index, 0, 7) for index in range(sample_count)], dtype=np.uint32)
    sample_wi = np.repeat(wi, sample_count, axis=0)
    sample_wo = np.repeat(wo, sample_count, axis=0)
    probe_results = probe(
        seeds=sample_seeds,
        material_id=int(material.material_id),
        uv=np.repeat(uv, sample_count, axis=0),
        wi=sample_wi,
        wo=sample_wo,
    )
    probe_cursor = probe_results.cursor()
    probe_cursor.load()
    results = [probe_cursor[index].read() for index in range(sample_count)]
    first_result = results[0]

    assert first_result["no_hint_bsdf_count"] == 4
    assert first_result["adjusted_bsdf_count"] == 4
    no_hint_normals = np.stack([_float3(value) for value in first_result["no_hint_bsdf_normal"]])
    adjusted_normals = np.stack([_float3(value) for value in first_result["adjusted_bsdf_normal"]])

    aligned = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    grazing = np.array([0.99875234, 0.0, 0.04993762], dtype=np.float32)
    subsurface = np.array([-0.99875234, 0.0, 0.04993762], dtype=np.float32)
    direct = np.array([0.0, 0.99875234, 0.04993762], dtype=np.float32)

    # Layering modes store lobes in different orders, so identify them by their authored frames.
    assert np.min(np.linalg.norm(no_hint_normals - aligned, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(no_hint_normals - grazing, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(no_hint_normals - subsurface, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(no_hint_normals - direct, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(adjusted_normals - aligned, axis=1)) < 1e-5
    assert np.min(np.linalg.norm(adjusted_normals - grazing, axis=1)) > 0.1
    assert np.min(np.linalg.norm(adjusted_normals - subsurface, axis=1)) > 0.1
    assert np.min(np.linalg.norm(adjusted_normals - direct, axis=1)) > 0.1

    repaired_indices = [
        index
        for index, normal in enumerate(adjusted_normals)
        if np.linalg.norm(normal - aligned) > 0.1
    ]
    assert len(repaired_indices) == 3
    for normal in adjusted_normals[repaired_indices]:
        assert normal[2] > grazing[2]
        reflected = -aligned + 2.0 * np.dot(aligned, normal) * normal
        assert reflected[2] >= -1e-6

    tangents = np.stack([_float3(value) for value in first_result["adjusted_bsdf_tangent"]])
    assert np.all(
        np.abs(np.sum(adjusted_normals[repaired_indices] * tangents[repaired_indices], axis=1))
        < 1e-5
    )

    adjusted_eval = _float3(first_result["eval"])
    adjusted_pdf = float(first_result["pdf"])
    samples = [result["sample"] for result in results]
    sample_pdfs = np.array([float(sample["pdf"]) for sample in samples], dtype=np.float32)
    assert np.all(np.isfinite(sample_pdfs))
    assert np.all(sample_pdfs >= 0.0)
    successful_samples = [sample for sample in samples if sample["pdf"] > 0.0]
    assert np.all(np.isfinite(adjusted_eval)) and np.linalg.norm(adjusted_eval) > 0.0
    assert np.isfinite(adjusted_pdf) and adjusted_pdf > 0.0
    assert successful_samples
    for sample in successful_samples:
        sample_weight = _float3(sample["weight"])
        sampled_wo = _float3(sample["wo_ws"])
        assert np.all(np.isfinite(sample_weight)) and np.linalg.norm(sample_weight) > 0.0
        assert np.all(np.isfinite(sampled_wo)) and np.isclose(
            np.linalg.norm(sampled_wo), 1.0, atol=1e-5
        )


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
def test_materialx_hair_preserves_frame(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MaterialXMaterial,
        f2.Properties({"mtlx_buffer": _normal_repair_document(hair=True), "mtlx_node_name": "M"}),
    )
    scene.update()
    from falcor2.editor import SceneShaderHelper

    helper = SceneShaderHelper(device)
    source = spy.Module.load_from_file(device, "render/test_materialx_lobe_normal_repair.slang")
    module, requirements = helper.prepare(scene, source)
    probe = (
        module.materialx_lobe_normal_repair_probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    values = probe(
        seeds=np.array([(7, 0, 7)], dtype=np.uint32),
        material_id=int(material.material_id),
        uv=np.array([(0.5, 0.5)], dtype=np.float32),
        wi=np.array([(0, 1, 0)], dtype=np.float32),
        wo=np.array([(0, 1, 0)], dtype=np.float32),
    )
    cursor = values.cursor()
    cursor.load()
    result = cursor[0].read()
    assert result["no_hint_bsdf_count"] == result["adjusted_bsdf_count"] == 1
    expected = np.array([0.0, -0.99875234, 0.04993762], dtype=np.float32)
    np.testing.assert_allclose(_float3(result["no_hint_bsdf_normal"][0]), expected, atol=1e-5)
    np.testing.assert_allclose(_float3(result["adjusted_bsdf_normal"][0]), expected, atol=1e-5)
    tangent = _float3(result["adjusted_bsdf_tangent"][0])
    assert abs(np.dot(tangent, expected)) < 1e-5
