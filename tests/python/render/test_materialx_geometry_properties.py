# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers

from .geometry_property_helpers import (
    TRIANGLE_INDICES,
    TRIANGLE_POSITIONS,
    read_material_probe as _read_material_probe,
    transform_terms,
    expected_transforms,
)


@pytest.mark.nightly_if(language=("mdl",))
@pytest.mark.nightly_if_not_vk(asset=("materialx_texcoord_1.usda",))
@pytest.mark.parametrize(
    "language,device_type",
    [
        (language, t)
        for language in ("materialx", "mdl")
        for t in helpers.DEFAULT_DEVICE_TYPES
        if (language == "mdl" and t == helpers.DEFAULT_DEVICE_TYPES[0])
        or (language == "materialx" and t != spy.DeviceType.cuda)
    ],
)
@pytest.mark.parametrize(
    "asset,uv1_name,expected",
    [
        ("materialx_texcoord_1.usda", "st1", (0.8, 0.5, 0.0)),
        ("materialx_texcoord_1.usda", "lightmap", (0.26, 0.36, 0.0)),
        ("materialx_pref.usda", "st1", (0.34, 0.44, 0.54)),
    ],
)
def test_usd_materialx_geometry_properties(
    language: str,
    device_type: spy.DeviceType,
    device: spy.Device,
    asset: str,
    uv1_name: str,
    expected: tuple[float, float, float],
) -> None:
    options = f2.ImportOptions()
    options.usd_texcoord_primvar_names = [uv1_name, "st2", "st3", "st4", "st5", "st6", "st7"]
    config = f2.SceneConfig(uv_origin=f2.UVOrigin.lower_left)
    asset_path = Path("data/assets/test_usd_importer") / asset
    if language == "materialx":
        scene = f2.Scene.load(device, asset_path, import_options=options, config=config)
    else:
        # Replace before Scene creation so CUDA never compiles an unused MaterialX material.
        importer = f2.Importer.create(options)
        importer.import_asset(asset_path)
        importer.materials["/World/Material"].replace(
            f2.MDLMaterial,
            f2.Properties(
                {
                    "mdl_library_path": str(
                        Path(__file__).parent / "data/mdl_generated_material_pipeline"
                    ),
                    "mdl_material_name": (
                        "geometry_properties::Pref"
                        if asset == "materialx_pref.usda"
                        else "geometry_properties::UV"
                    ),
                }
            ),
        )
        scene = f2.Scene.from_importer(device, importer, config)
    instance = scene.components.find(type=f2.GeometryInstance)
    assert instance is not None
    material = instance.materials[0]
    if language == "materialx":
        assert isinstance(material, f2.MaterialXMaterial)
        material.mtlx_canonicalize_names = device_type == spy.DeviceType.vulkan
    else:
        assert isinstance(material, f2.MDLMaterial)
    # Importer integration exercises the build's default MaterialX backend.
    scene.update()

    # The shared probe uses weights (0.5, 0.2, 0.3) at triangle zero.
    actual = _read_material_probe(device, scene, instance)["material_color"]
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert not np.allclose(actual, (0.2, 0.3, 0.0))  # UV0 or current points.
    if asset == "materialx_pref.usda":
        assert not np.allclose(actual, (0.9, 0.8, 0.7))  # Missing-value fallback.


def _typed_transform_document() -> str:
    """Keep the uncovered typed reads and all transform kinds in one numeric emission probe."""
    nodes = [
        '<geompropvalue name="half" type="float"><input name="geomprop" type="string" value="half" /></geompropvalue>',
        '<geompropvalue name="integer" type="integer"><input name="geomprop" type="string" value="integer" /></geompropvalue>',
        '<geompropvalue name="flag" type="boolean"><input name="geomprop" type="string" value="flag" /></geompropvalue>',
        '<convert name="int_float" type="float"><input name="in" type="integer" nodename="integer" /></convert>',
        '<convert name="bool_float" type="float"><input name="in" type="boolean" nodename="flag" /></convert>',
        '<combine3 name="typed" type="vector3"><input name="in1" type="float" nodename="half" />'
        '<input name="in2" type="float" nodename="int_float" /><input name="in3" type="float" nodename="bool_float" /></combine3>',
        '<geompropvalue name="aux" type="vector3"><input name="geomprop" type="string" value="shaderOffset" /></geompropvalue>',
        '<geompropvalue name="missing" type="vector3"><input name="geomprop" type="string" value="missing" />'
        '<input name="default" type="vector3" value="0.125, 0.25, 0.5" /></geompropvalue>',
        '<normal name="object_normal" type="vector3"><input name="space" type="string" value="object" /></normal>',
        '<geompropvalue name="alias_normal" type="vector3"><input name="geomprop" type="string" value="Nobject" /></geompropvalue>',
    ]
    terms = ["aux", "missing", "object_normal", "alias_normal"]
    # Exercise both view-direction routes in the same probe, without more GPU graph variants.
    for index, space in enumerate(("world", "object", "model")):
        nodes.append(
            f'<geompropdef name="V{space}" type="vector3" geomprop="viewdirection" space="{space}" />'
        )
        for route in ("direct", "alias"):
            name = f"view_{space}_{route}"
            if route == "direct":
                nodes.append(
                    f'<viewdirection name="{name}" type="vector3"><input name="space" type="string" value="{space}" /></viewdirection>'
                )
            else:
                nodes.append(
                    f'<geompropvalue name="{name}" type="vector3"><input name="geomprop" type="string" value="V{space}" /></geompropvalue>'
                )
            nodes.append(
                f'<multiply name="weighted_{name}" type="vector3"><input name="in1" type="vector3" nodename="{name}" />'
                f'<input name="in2" type="float" value="{2 ** (2 * index + (route == "alias"))}" /></multiply>'
            )
            terms.append(f"weighted_{name}")
    # Keep every emission channel positive even when a view-direction component is negative.
    nodes.append(
        '<constant name="offset" type="vector3"><input name="value" type="vector3" value="256, 256, 256" /></constant>'
    )
    terms.append("offset")
    for kind, from_space, to_space, transform_weight in transform_terms():
        name = f"{kind}_{from_space}_{to_space}"
        nodes.append(
            f'<transform{kind} name="{name}" type="vector3">'
            '<input name="in" type="vector3" value="0.1, 0.2, 0.3" />'
            f'<input name="fromspace" type="string" value="{from_space}" />'
            f'<input name="tospace" type="string" value="{to_space}" /></transform{kind}>'
        )
        # Distinct weights prevent opposite directions or vector/normal transforms from masking a swap.
        nodes.append(
            f'<multiply name="weighted_{name}" type="vector3"><input name="in1" type="vector3" nodename="{name}" />'
            f'<input name="in2" type="float" value="{transform_weight}" /></multiply>'
        )
        terms.append(f"weighted_{name}")
    previous = "typed"
    for index, term in enumerate(terms):
        name = f"sum{index}"
        nodes.append(
            f'<add name="{name}" type="vector3"><input name="in1" type="vector3" nodename="{previous}" />'
            f'<input name="in2" type="vector3" nodename="{term}" /></add>'
        )
        previous = name
    nodes.append(
        f'<convert name="color" type="color3"><input name="in" type="vector3" nodename="{previous}" /></convert>'
        '<standard_surface name="surface" type="surfaceshader">'
        '<input name="base" type="float" value="0" /><input name="specular" type="float" value="0" />'
        '<input name="emission" type="float" value="1" />'
        '<input name="emission_color" type="color3" nodename="color" /></standard_surface>'
        '<surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="surface" /></surfacematerial>'
    )
    return '<materialx version="1.39" colorspace="lin_rec709">' + "".join(nodes) + "</materialx>"


def _mdl_transform_source() -> str:
    terms = []
    for kind, from_space, to_space, weight in transform_terms():
        expression = (
            f"state::transform_{kind}(state::coordinate_{from_space}, "
            f"state::coordinate_{to_space}, float3(0.1, 0.2, 0.3))"
        )
        # MDL does not promise normalized transform_normal results; match the MaterialX comparison explicitly.
        if kind == "normal":
            expression = f"math::normalize({expression})"
        terms.append(f"{weight} * {expression}")
    return (
        """mdl 1.10;
import ::state::*;
import ::math::*;
import ::df::*;
import ::scene::*;
export material Default(uniform float gain = 1) = material(
    surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: gain / 128 * color(
            float3(scene::data_lookup_float("half", 0.0),
                float(scene::data_lookup_int("integer", 0)), scene::data_isvalid("flag") ? 1.0 : 0.0)
            + scene::data_lookup_float3("shaderOffset", float3(0.0))
            + scene::data_lookup_float3("missing", float3(0.125, 0.25, 0.5)) + """
        + " + ".join(terms)
        + "))));"
    )


@pytest.mark.nightly_if(
    layering_mode=(f2.MaterialXLayeringMode.bsdf_mix, f2.MaterialXLayeringMode.closure_tree)
)
@pytest.mark.parametrize(
    "language,device_type,layering_mode",
    [
        (language, t, mode)
        for language in ("materialx", "mdl")
        for t in helpers.DEFAULT_DEVICE_TYPES
        for mode in [
            None,
            f2.MaterialXLayeringMode.closure_tree,
        ]
        if (language == "mdl" and t == helpers.DEFAULT_DEVICE_TYPES[0] and mode is None)
        or (language == "materialx" and t != spy.DeviceType.cuda)
    ],
)
def test_materialx_typed_and_transform_routes(
    language: str,
    device_type: spy.DeviceType,
    device: spy.Device,
    layering_mode: f2.MaterialXLayeringMode | None,
    tmp_path: Path,
) -> None:
    scene = f2.Scene.create(device)
    if language == "materialx":
        material_type = f2.MaterialXMaterial
        properties = {
            "mtlx_buffer": _typed_transform_document(),
            "mtlx_node_name": "M",
            "mtlx_canonicalize_names": device_type == spy.DeviceType.d3d12,
            "mtlx_editable_params": "surface_emission",
        }
    else:
        material_type = f2.MDLMaterial
        (tmp_path / "transforms.mdl").write_text(_mdl_transform_source(), encoding="utf-8")
        properties = {
            "mdl_library_path": str(tmp_path),
            "mdl_material_name": "transforms::Default",
            "debug_write_shader_path": str(tmp_path / "transforms.slang"),
        }
    material = scene.create_material(material_type, f2.Properties(properties))
    if language == "materialx" and layering_mode is not None:
        material.mtlx_layering_mode = layering_mode
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    # Float values interpolate; integer and Boolean values select the largest-weight vertex (vertex zero).
    attributes = [
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown,
            0,
            np.array([0.25, 0.5, 0.75], dtype=np.float16),
            name="half",
        ),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown, 0, np.array([2, 7, 9], dtype=np.int32), name="integer"
        ),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown,
            0,
            np.array([True, False, False], dtype=np.bool_),
            name="flag",
        ),
        f2.MeshVertexAttribute(
            f2.MeshAttributeSemantic.unknown,
            0,
            np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=np.float32),
            name="shaderOffset",
        ),
    ]
    # An oblique normal makes inverse-transpose direction observable under the nonuniform instance scale.
    object_normal = np.array([1.0, 2.0, 3.0])
    object_normal /= np.linalg.norm(object_normal)
    geometry.set_mesh_data(
        positions=TRIANGLE_POSITIONS,
        sub_mesh_indices=[TRIANGLE_INDICES],
        normals=np.tile(object_normal.astype(np.float32), (3, 1)),
        tangents=np.tile(np.array([1, 0, 0], dtype=np.float32), (3, 1)),
        handedness=np.ones(3, dtype=np.float32),
        vertex_attributes=attributes,
    )
    entity = scene.create_entity()
    transform = f2.Transform()
    transform.scale = spy.float3(2, 3, 4)
    transform.translation = spy.float3(0.3, -0.2, 0.1)
    # 60 degrees about (1, 1, 1) makes the probe's axial view oblique in object space.
    transform.rotation = spy.quatf(*(np.ones(3) / np.sqrt(12)), np.sqrt(3) / 2)
    entity.transform = transform
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.geometry_properties)

    scale = np.array([2, 3, 4])
    rotation = np.array([[2, -1, 2], [2, 2, -1], [-1, 2, 2]]) / 3
    linear = rotation @ np.diag(scale)
    inverse = np.linalg.inv(linear)
    expected = expected_transforms(True, linear)
    expected += np.array([0.45, 2, 1]) + [0.34, 0.44, 0.54]
    # Both intrinsic routes recover object space from the retained hit.
    if language == "materialx":
        expected += 2 * object_normal
    expected += [0.125, 0.25, 0.5]
    if language == "materialx":
        world_view = np.array([0.0, 0.0, -1.0])
        object_view = inverse @ world_view
        object_view /= np.linalg.norm(object_view)
        # Direct/alias weights sum to 3 for world, 12 for object, and 48 for model (= object).
        expected += 256 + 3 * world_view + 60 * object_view
    if language == "mdl":
        # The current MDL integration exposes BSDF albedo, not emission. Keep the weighted sum below one.
        expected /= 128
    output = "emissive_radiance" if language == "materialx" else "diffuse_reflection_albedo"
    np.testing.assert_allclose(
        _read_material_probe(device, scene, instance)[output],
        expected,
        rtol=1e-5,
        atol=1e-5,
    )
    # Exercise the promoted input without recompiling or adding another graph variant.
    material["inputs:surface_emission" if language == "materialx" else "gain"] = 0.5
    scene.update()
    np.testing.assert_allclose(
        _read_material_probe(device, scene, instance)[output],
        expected * 0.5,
        rtol=1e-5,
        atol=1e-5,
    )
