# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sampling contracts at the production material boundary, independent of benchmarks."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers

from .python_material_sample import PythonMaterialSample
from .scattering_tests import EstimatorMeanTest, SampleContractTest


_DATA_DIR = Path(__file__).resolve().parents[3] / "data"


def _materialx_surface(*, layered: bool, transmission: float = 0.0) -> str:
    return f"""<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" value="0.6, 0.2, 0.05" />
    <input name="specular" type="float" value="{1 if layered else 0}" />
    <input name="specular_roughness" type="float" value="0.6" />
    <input name="coat" type="float" value="{0.4 if layered else 0}" />
    <input name="coat_roughness" type="float" value="0.5" />
    <input name="transmission" type="float" value="{transmission}" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


@dataclass(frozen=True)
class _MaterialSamplingCase:
    name: str
    material_type: type | str
    properties: dict[str, Any]


_MATERIAL_CASES = [
    _MaterialSamplingCase("custom-diffuse", PythonMaterialSample, {}),
    _MaterialSamplingCase(
        "standard-mixed",
        f2.StandardMaterial,
        {
            "base_color_factor": spy.float3(0.6, 0.2, 0.05),
            "roughness_factor": 0.6,
        },
    ),
    _MaterialSamplingCase(
        "standard-transmission",
        f2.StandardMaterial,
        {
            "base_color_factor": spy.float3(0.6, 0.2, 0.05),
            "roughness_factor": 0.6,
            "specular_transmission_factor": 0.6,
            "transmission_factor": spy.float3(0.7, 0.8, 1.0),
            "thin_walled": True,
        },
    ),
    _MaterialSamplingCase(
        "standard-specgloss",
        f2.StandardSpecGlossMaterial,
        {
            "diffuse_factor": spy.float3(0.6, 0.2, 0.05),
            "glossiness_factor": 0.4,
            "specular_factor": spy.float3(0.15, 0.1, 0.05),
        },
    ),
    _MaterialSamplingCase(
        "gltf-layered",
        f2.GLTFMaterial,
        {
            "base_color_factor": spy.float3(0.6, 0.2, 0.05),
            "metallic_factor": 0.3,
            "roughness_factor": 0.6,
            "clearcoat_factor": 0.4,
            "clearcoat_roughness_factor": 0.5,
        },
    ),
    _MaterialSamplingCase(
        "gltf-specgloss",
        f2.GLTFSpecGlossMaterial,
        {
            "diffuse_factor": spy.float3(0.6, 0.2, 0.05),
            "glossiness_factor": 0.4,
            "specular_factor": spy.float3(0.15, 0.1, 0.05),
        },
    ),
    _MaterialSamplingCase(
        "openpbr-layered",
        "OpenPBRMaterial",
        {
            "base_color": spy.float3(0.6, 0.2, 0.05),
            "specular_roughness": 0.6,
            "coat_weight": 0.4,
            "coat_roughness": 0.5,
            "fuzz_weight": 0.2,
        },
    ),
    _MaterialSamplingCase(
        "mx-diffuse",
        f2.MaterialXMaterial,
        {
            "mtlx_buffer": _materialx_surface(layered=False),
            "mtlx_node_name": "M",
        },
    ),
    _MaterialSamplingCase(
        "mx-layered",
        f2.MaterialXMaterial,
        {
            "mtlx_buffer": _materialx_surface(layered=True),
            "mtlx_node_name": "M",
        },
    ),
    _MaterialSamplingCase(
        "mx-transmission",
        f2.MaterialXMaterial,
        {
            "mtlx_buffer": _materialx_surface(layered=True, transmission=0.6),
            "mtlx_node_name": "M",
        },
    ),
    _MaterialSamplingCase(
        "mdl-diffuse",
        f2.MDLMaterial,
        {
            "mdl_library_path": str(_DATA_DIR / "assets/mdl_sdk_examples"),
            "mdl_material_name": "tutorials::example_material",
            "mdl_class_compilation": True,
        },
    ),
    _MaterialSamplingCase(
        "mdl-layered",
        f2.MDLMaterial,
        {
            "mdl_library_path": str(_DATA_DIR / "assets/mdl_sdk_examples"),
            "mdl_material_name": "metal_single_diamond_plate::single_diamond_plate",
            "mdl_class_compilation": True,
        },
    ),
]


def _prepare_material(
    device: spy.Device,
    case: _MaterialSamplingCase,
    mode: f2.MaterialXLayeringMode | None = None,
) -> tuple[f2.Scene, f2.Material, spy.Module, SceneShaderHelper]:
    scene = f2.Scene.create(device)
    material = scene.create_material(case.material_type, f2.Properties(case.properties))
    if mode is not None:
        material.mtlx_layering_mode = mode
    scene.update()
    helper = SceneShaderHelper(device)
    module, _ = helper.prepare(
        scene, spy.Module(device.load_module("render/material_sampling_tests.slang"))
    )
    return scene, material, module, helper


def _target_bindings(material: f2.Material, degrees: int) -> dict[str, Any]:
    theta = np.deg2rad(degrees)
    return {
        "material_id": int(material.material_id),
        "uv": spy.float2(0.37, 0.61),
        "wi": spy.float3(np.sin(theta), np.cos(theta), 0.0),
    }


def _weight_policy_prelude(policy: str) -> str:
    return (
        "export static const MaterialSampleWeightPolicy MATERIAL_SAMPLE_WEIGHT_POLICY = "
        f"MaterialSampleWeightPolicy::{policy};"
    )


@pytest.mark.nightly_if(device_type=(spy.DeviceType.d3d12, spy.DeviceType.cuda))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("case", _MATERIAL_CASES, ids=lambda case: case.name)
def test_material_sample_contract(
    device: spy.Device,
    case: _MaterialSamplingCase,
) -> None:
    scene, material, module, helper = _prepare_material(device, case)
    for degrees in (0, 60, 85):
        test = SampleContractTest(
            device,
            module,
            "MaterialTarget",
            _target_bindings(material, degrees),
            helper.bind_scene,
            sample_count=4096,
            type_conformances=scene.requirements.type_conformances,
            prelude=_weight_policy_prelude("eval_consistent"),
        )
        assert test.run(), f"{case.name}, {degrees} degrees: {test.messages}"


@pytest.mark.nightly_if(device_type=(spy.DeviceType.d3d12, spy.DeviceType.cuda))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("case", _MATERIAL_CASES, ids=lambda case: case.name)
def test_material_component_layout_and_partition(
    device: spy.Device, case: _MaterialSamplingCase
) -> None:
    scene, material, module, helper = _prepare_material(device, case)
    layout_function = (
        module.component_layout.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
    )
    layout = int(
        layout_function(target={"_type": "MaterialTarget", **_target_bindings(material, 0)})
    )
    if case.name.startswith("standard-"):
        assert layout == 0x80000001
    elif case.name == "gltf-specgloss":
        assert layout == 0x80000002
    else:
        assert layout == 0
    if layout == 0:
        return
    for degrees in (0, 60, 85):
        test = SampleContractTest(
            device,
            module,
            "MaterialTarget",
            _target_bindings(material, degrees),
            helper.bind_scene,
            sample_count=4096,
            type_conformances=scene.requirements.type_conformances,
        )
        # The fixture graphs have fewer than 16 closures; unknown slots must contribute zero.
        count = 4 if layout == 0x80000001 else 2 if layout == 0x80000002 else 16
        for wo in (spy.float3(0.6, 0.8, 0.0), spy.float3(0.6, -0.8, 0.0)):
            residual = test.function("component_partition")(
                target={"_type": "MaterialTarget", **test.target_bindings},
                wo=wo,
                component_count=count,
            )
            np.testing.assert_allclose(np.asarray(residual), 0.0, atol=1e-6)


@pytest.mark.nightly_if(device_type=(spy.DeviceType.d3d12, spy.DeviceType.cuda))
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "case",
    [
        case
        for case in _MATERIAL_CASES
        if case.name in ("standard-mixed", "standard-transmission", "gltf-specgloss")
    ],
    ids=lambda case: case.name,
)
def test_material_component_layout_and_roughness(
    device: spy.Device, case: _MaterialSamplingCase
) -> None:
    scene, material, module, helper = _prepare_material(device, case)
    target = {"_type": "MaterialTarget", **_target_bindings(material, 60)}
    configure = (
        lambda fn: fn.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
    )
    layout = int(configure(module.component_layout)(target=target))
    assert layout == (0x80000002 if case.name == "gltf-specgloss" else 0x80000001)
    rough = configure(module.component_rough)
    assert rough(target=target, component=0, pdf=1e6, threshold=0.9)
    glossy = 1 if case.name == "gltf-specgloss" else 2
    assert not rough(target=target, component=glossy, pdf=0.01, threshold=0.9)
    assert rough(target=target, component=glossy, pdf=1e6, threshold=0.1)
    assert not rough(target=target, component=0xFFFFFFFF, pdf=1e6, threshold=0.9)
    assert rough(target=target, component=0xFFFFFFFF, pdf=0.01, threshold=0.9)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "mode", [f2.MaterialXLayeringMode.bsdf_mix, f2.MaterialXLayeringMode.closure_tree]
)
def test_materialx_compatibility_sample_contract(
    device: spy.Device, mode: f2.MaterialXLayeringMode
) -> None:
    case = next(case for case in _MATERIAL_CASES if case.name == "mx-layered")
    scene, material, module, helper = _prepare_material(device, case, mode)
    for degrees in (0, 60, 85):
        test = SampleContractTest(
            device,
            module,
            "MaterialTarget",
            _target_bindings(material, degrees),
            helper.bind_scene,
            sample_count=4096,
            type_conformances=scene.requirements.type_conformances,
            prelude=_weight_policy_prelude("eval_consistent"),
        )
        assert test.run(), f"{mode}, {degrees} degrees: {test.messages}"


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "name", ["custom-diffuse", "gltf-layered", "standard-mixed", "mdl-diffuse"]
)
@pytest.mark.parametrize("policy", ["native", "eval_consistent"])
def test_material_estimator_mean(device: spy.Device, name: str, policy: str) -> None:
    case = next(case for case in _MATERIAL_CASES if case.name == name)
    scene, material, module, helper = _prepare_material(device, case)
    test = EstimatorMeanTest(
        device,
        module,
        "MaterialTarget",
        _target_bindings(material, 60),
        helper.bind_scene,
        sample_count=262144,
        type_conformances=scene.requirements.type_conformances,
        prelude=_weight_policy_prelude(policy),
    )
    assert test.run(), test.messages


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("name", ["mdl-layered", "standard-mixed"])
def test_material_sample_weight_policy(device: spy.Device, name: str) -> None:
    case = next(case for case in _MATERIAL_CASES if case.name == name)
    scene, material, module, helper = _prepare_material(device, case)
    inputs = {
        "u_lobe": spy.Tensor.from_numpy(device, (np.arange(64, dtype=np.float32) + 0.5) / 64),
        "u_dir": spy.float2(0.3, 0.7),
        "aux_seed": 123,
    }
    results = {}
    for policy in (None, "native", "eval_consistent"):
        test = SampleContractTest(
            device,
            module,
            "MaterialTarget",
            _target_bindings(material, 60),
            helper.bind_scene,
            sample_count=4096,
            type_conformances=scene.requirements.type_conformances,
            prelude=_weight_policy_prelude(policy) if policy is not None else None,
        )
        assert test.run(check_weight=policy == "eval_consistent"), test.messages
        target = {"_type": "MaterialTarget", **test.target_bindings}
        events = test.function("sample_event")(target=target, input=inputs).to_numpy()
        weights = test.function("sample_event_weight")(target=target, input=inputs).to_numpy()
        results[policy] = events, weights
    default_events, default_weights = results[None]
    native_events, native_weights = results["native"]
    strict_events, strict_weights = results["eval_consistent"]
    np.testing.assert_allclose(default_events, native_events, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(default_weights, native_weights, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(native_events, strict_events, rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(native_weights[:, 3], strict_weights[:, 3], rtol=1e-6, atol=1e-7)
    if name == "mdl-layered":
        assert not np.allclose(native_weights[:, :3], strict_weights[:, :3], rtol=1e-3, atol=1e-8)
    else:
        np.testing.assert_allclose(native_weights, strict_weights, rtol=1e-6, atol=1e-7)
