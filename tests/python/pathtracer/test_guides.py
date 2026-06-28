# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.rendergraph import OutputPrelude
from falcor2.rendernodes import WRITE_GUIDE_INTERFACE


EXPECTED_GUIDE_OUTPUT_VALUE_TYPES = {
    "diffuse_albedo": "vector<float,4>",
    "material_color": "vector<float,4>",
    "specular_albedo": "vector<float,4>",
    "normals": "vector<float,4>",
    "roughness": "float",
    "depth": "float",
    "hardware_depth": "float",
    "specular_hit_distance": "float",
    "motion_vectors": "vector<float,2>",
    "emission": "vector<float,4>",
    "geometry_id": "vector<uint,4>",
}

EXPECTED_GUIDE_OUTPUT_FORMATS = {
    "diffuse_albedo": spy.Format.rgba16_float,
    "material_color": spy.Format.rgba16_float,
    "specular_albedo": spy.Format.rgba16_float,
    "normals": spy.Format.rgba16_float,
    "roughness": spy.Format.r16_float,
    "depth": spy.Format.r32_float,
    "hardware_depth": spy.Format.r32_float,
    "specular_hit_distance": spy.Format.r16_float,
    "motion_vectors": spy.Format.rg16_float,
    "emission": spy.Format.rgba16_float,
    "geometry_id": spy.Format.rgba32_uint,
}

EXPECTED_GUIDE_OUTPUT_CLEAR_VALUES = {
    "diffuse_albedo": (0.0, 0.0, 0.0, 1.0),
    "material_color": (0.0, 0.0, 0.0, 1.0),
    "specular_albedo": (0.0, 0.0, 0.0, 1.0),
    "normals": (0.0, 0.0, 1.0, 0.0),
    "roughness": (1.0, 0.0, 0.0, 0.0),
    "depth": (0.0, 0.0, 0.0, 0.0),
    "hardware_depth": (1.0, 0.0, 0.0, 0.0),
    "specular_hit_distance": (0.0, 0.0, 0.0, 0.0),
    "motion_vectors": (0.0, 0.0, 0.0, 0.0),
    "emission": (0.0, 0.0, 0.0, 0.0),
    "geometry_id": (
        0xFFFFFFFF,
        int(f2.GeometryInstanceID.invalid),
        0xFFFFFFFF,
        int(f2.MaterialID.invalid),
    ),
}


def load_linked_reference_module(device: spy.Device) -> spy.Module:
    scene = f2.Scene.create(device)
    scene.update()
    reqs = scene.requirements
    utils = spy.Module(device.load_module("falcor2.utils"))
    render_mod = spy.Module(scene.render_module)
    return spy.Module(
        device.load_module("falcor2.rendernodes.reference_pathtracer"),
        link=[utils, render_mod] + [spy.Module(module) for module in reqs.modules],
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_write_guide_interface_reflects_output_functions(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Interface member functions can be reflected by type and function name."""
    linked_module = load_linked_reference_module(device)
    interface_type = linked_module.layout.find_type_by_name(WRITE_GUIDE_INTERFACE)
    assert interface_type is not None
    assert interface_type.type_reflection.kind == spy.TypeReflection.Kind.interface

    reflected_outputs: dict[str, str] = {}
    for function_name in EXPECTED_GUIDE_OUTPUT_VALUE_TYPES:
        function = linked_module.layout.find_function_by_name_in_type(interface_type, function_name)
        assert function is not None
        assert function.static
        assert not function.have_return_value
        assert len(function.parameters) == 2
        assert function.parameters[0].name == "coord"
        assert function.parameters[0].type.full_name == "vector<uint,2>"
        assert function.parameters[1].name == "value"
        reflected_outputs[function.name] = function.parameters[1].type.full_name

        enabled_function = linked_module.layout.find_function_by_name_in_type(
            interface_type, f"{function_name}_enabled"
        )
        assert enabled_function is not None
        assert enabled_function.static
        assert enabled_function.have_return_value
        assert enabled_function.return_type.full_name == "bool"
        assert len(enabled_function.parameters) == 0

    assert reflected_outputs == EXPECTED_GUIDE_OUTPUT_VALUE_TYPES


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_write_guide_interface_reflects_output_specs(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Interface attributes provide guide output format and clear values."""
    linked_module = load_linked_reference_module(device)
    specs = tuple(OutputPrelude.create(linked_module, WRITE_GUIDE_INTERFACE).specs.values())

    assert [spec.name for spec in specs] == list(EXPECTED_GUIDE_OUTPUT_VALUE_TYPES)
    assert {spec.name: spec.format for spec in specs} == EXPECTED_GUIDE_OUTPUT_FORMATS
    for spec in specs:
        expected_clear = EXPECTED_GUIDE_OUTPUT_CLEAR_VALUES[spec.name]
        if "uint" in spec.format.name:
            assert tuple(spec.clear_value) == expected_clear
        else:
            np.testing.assert_allclose(
                tuple(spec.clear_value),
                expected_clear,
                atol=1e-6,
            )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hardware_depth_clear_value_is_far_plane(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    linked_module = load_linked_reference_module(device)
    specs = OutputPrelude.create(linked_module, WRITE_GUIDE_INTERFACE).specs

    hardware_depth = specs["hardware_depth"]
    assert hardware_depth.format == spy.Format.r32_float
    np.testing.assert_allclose(tuple(hardware_depth.clear_value), (1.0, 0.0, 0.0, 0.0))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_write_guide_prelude_matches_enabled_guides(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Generated guide prelude implements every output and stores enabled ones."""
    linked_module = load_linked_reference_module(device)
    depth = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.r32_float,
        width=2,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    motion_vectors = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    write_guide = {
        "depth": depth,
        "motion_vectors": motion_vectors,
    }

    prelude = OutputPrelude.create(linked_module, WRITE_GUIDE_INTERFACE).generate(write_guide)

    assert "public RWTexture2D<float> depth;" in prelude
    assert "public RWTexture2D<vector<float,2>> motion_vectors;" in prelude
    assert "public RWTexture2D<vector<float,4>> diffuse_albedo;" not in prelude
    assert "__g_falcor_OutputBlockIWriteGuide.depth[coord] = value;" in prelude
    assert "__g_falcor_OutputBlockIWriteGuide.motion_vectors[coord] = value;" in prelude
    assert "export struct WriteGuide : IWriteGuide" in prelude
    assert "public static bool depth_enabled()" in prelude
    assert "public static bool motion_vectors_enabled()" in prelude
    assert "public static bool diffuse_albedo_enabled()" in prelude
    assert "return true;" in prelude
    assert "return false;" in prelude
    assert "export static const bool WriteGuide_depth = true;" in prelude
    assert "export static const bool WriteGuide_diffuse_albedo = false;" in prelude
    assert "public static override void depth(vector<uint,2> coord, float value)" in prelude
    assert (
        "public static override void motion_vectors(vector<uint,2> coord, " "vector<float,2> value)"
    ) in prelude
    assert (
        "public static override void diffuse_albedo(vector<uint,2> coord, " "vector<float,4> value)"
    ) in prelude
