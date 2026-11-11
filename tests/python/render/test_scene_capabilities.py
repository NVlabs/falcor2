# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers


GEOMETRY = f2.SceneCapabilityFlags.geometry_properties
OPACITY = f2.SceneCapabilityFlags.opacity_evaluation
SOLID_TRANSMISSION = f2.SceneCapabilityFlags.solid_transmission


def test_capability_values() -> None:
    empty = f2.SceneCapabilities()
    geometry = f2.SceneCapabilities(GEOMETRY)
    both = geometry.merged(f2.SceneCapabilities(OPACITY))
    assert not empty.has(GEOMETRY)
    assert both.has(GEOMETRY | OPACITY)
    assert both.covers(geometry)
    assert not geometry.covers(f2.SceneCapabilities(OPACITY))
    assert geometry == f2.SceneCapabilities(GEOMETRY)
    assert geometry != both
    assert f2.SceneCapabilities.all().has(SOLID_TRANSMISSION)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_capabilities_follow_synchronized_scene(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    empty = f2.SceneCapabilities()
    assert scene.required_capabilities == empty
    material = scene.create_material(f2.StandardMaterial)
    scene.update()
    assert material.required_capabilities() == empty
    assert scene.required_capabilities == empty

    material.alpha_mode = f2.AlphaMode.mask
    assert scene.required_capabilities == empty
    scene.update()
    assert scene.required_capabilities == f2.SceneCapabilities(OPACITY)
    saved = scene.required_capabilities

    material.base_color_texture_texcoord = 1
    assert scene.required_capabilities == saved
    scene.update()
    assert scene.required_capabilities == f2.SceneCapabilities(GEOMETRY | OPACITY)

    material.base_color_texture_texcoord = 0
    assert scene.required_capabilities.has(GEOMETRY | OPACITY)
    scene.update()
    assert scene.required_capabilities == saved

    material.remove()
    assert scene.required_capabilities == saved
    scene.update()
    assert scene.required_capabilities == empty
    assert saved == f2.SceneCapabilities(OPACITY)


class _CustomMaterial(f2.Material):
    def __init__(self) -> None:
        super().__init__()
        self.slang_type_name = "InvalidMaterial"


class _GeometryMaterial(_CustomMaterial):
    def required_capabilities(self) -> f2.SceneCapabilities:
        return super().required_capabilities().merged(f2.SceneCapabilities(GEOMETRY))


class _OpacityMaterial(_CustomMaterial):
    def required_capabilities(self) -> f2.SceneCapabilities:
        return super().required_capabilities().merged(f2.SceneCapabilities(OPACITY))


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
@pytest.mark.parametrize(
    "material_type,flags",
    [
        (_GeometryMaterial, GEOMETRY | SOLID_TRANSMISSION),
        (_OpacityMaterial, OPACITY | SOLID_TRANSMISSION),
    ],
)
def test_custom_material_capability_opt_in(
    device: spy.Device, material_type: type[f2.Material], flags: f2.SceneCapabilityFlags
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(_CustomMaterial)
    scene.update()
    conservative = f2.SceneCapabilities(SOLID_TRANSMISSION)
    assert material.required_capabilities() == conservative
    assert scene.required_capabilities == conservative

    custom = scene.create_material(material_type)
    assert scene.required_capabilities == conservative
    scene.update()
    assert custom.required_capabilities().flags == flags
    assert scene.required_capabilities.flags == flags
    custom.remove()
    scene.update()
    assert scene.required_capabilities == conservative


def _mtlx_document(space: str) -> str:
    return f"""<materialx version="1.39">
    <position name="p" type="vector3"><input name="space" type="string" value="{space}" /></position>
    <convert name="c" type="color3"><input name="in" type="vector3" nodename="p" /></convert>
    <oren_nayar_diffuse_bsdf name="b" type="BSDF"><input name="color" type="color3" nodename="c" /></oren_nayar_diffuse_bsdf>
    <surface name="s" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="b" /></surface>
    <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="s" /></surfacematerial>
    </materialx>"""


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_materialx_capabilities_update_with_generated_code(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MaterialXMaterial,
        f2.Properties({"mtlx_buffer": _mtlx_document("world"), "mtlx_node_name": "M"}),
    )
    scene.update()
    assert not material.required_capabilities().has(GEOMETRY)
    assert not scene.required_capabilities.has(GEOMETRY)
    assert material.required_capabilities().has(SOLID_TRANSMISSION)
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)
    material.mtlx_buffer = _mtlx_document("object")
    assert not scene.required_capabilities.has(GEOMETRY)
    scene.update()
    assert material.required_capabilities().has(GEOMETRY)
    assert scene.required_capabilities.has(GEOMETRY)
    material.mtlx_buffer = _mtlx_document("world")
    scene.update()
    assert not scene.required_capabilities.has(GEOMETRY)


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
@pytest.mark.parametrize(
    "expression,needs_geometry",
    [
        ("color(0.3)", False),
        ("color(state::normal())", False),
        ("color(state::texture_coordinate(0))", True),
        ("color(state::texture_coordinate(1))", True),
        (
            "color(state::transform_point(state::coordinate_world, state::coordinate_object, state::position()))",
            True,
        ),
        ('color(scene::data_lookup_float3("Pref", float3(0.3)))', True),
    ],
)
def test_mdl_capabilities_from_sdk_metadata(
    device: spy.Device, expression: str, needs_geometry: bool
) -> None:
    module_name = "capability_" + hashlib.sha1(expression.encode()).hexdigest()
    source = f"""mdl 1.10;
    import ::df::*; import ::state::*; import ::scene::*;
    export material Default() = material(surface: material_surface(
        scattering: df::diffuse_reflection_bsdf(tint: {expression})));
    """
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MDLMaterial,
        f2.Properties(
            {
                "mdl_material_name": f"{module_name}::Default",
                "mdl_source": source,
            }
        ),
    )
    scene.update()
    assert material.required_capabilities().has(GEOMETRY) == needs_geometry
    assert scene.required_capabilities.has(GEOMETRY) == needs_geometry
    assert material.required_capabilities().has(SOLID_TRANSMISSION)
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)
    material.remove()
    scene.update()
    assert scene.required_capabilities == f2.SceneCapabilities()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type,texture_slots",
    [
        (
            "StandardMaterial",
            ("base_color", "metallic_roughness", "normal", "emissive", "transmission"),
        ),
        (
            "StandardSpecGlossMaterial",
            ("diffuse", "specular_glossiness", "normal", "emissive", "transmission"),
        ),
        ("UnlitMaterial", ("base_color",)),
        (
            "GLTFSpecGlossMaterial",
            ("diffuse", "specular_glossiness", "normal", "emissive"),
        ),
        (
            "GLTFMaterial",
            (
                "base_color",
                "metallic_roughness",
                "normal",
                "emissive",
                "transmission",
                "specular",
                "specular_color",
                "clearcoat",
                "clearcoat_roughness",
                "clearcoat_normal",
                "sheen_color",
                "sheen_roughness",
                "iridescence",
                "iridescence_thickness",
                "anisotropy",
                "diffuse_transmission",
                "diffuse_transmission_color",
            ),
        ),
        (
            "OpenPBRMaterial",
            (
                "base_weight",
                "base_color",
                "base_diffuse_roughness",
                "base_metalness",
                "specular_weight",
                "specular_color",
                "specular_roughness",
                "specular_roughness_anisotropy",
                "specular_ior",
                "coat_weight",
                "coat_color",
                "coat_roughness",
                "coat_roughness_anisotropy",
                "coat_ior",
                "coat_darkening",
                "fuzz_weight",
                "fuzz_color",
                "fuzz_roughness",
                "transmission_weight",
                "transmission_color",
                "transmission_depth",
                "transmission_scatter",
                "transmission_scatter_anisotropy",
                "thin_film_weight",
                "thin_film_thickness",
                "thin_film_ior",
                "emission_luminance",
                "emission_color",
                "normal",
                "opacity",
            ),
        ),
    ],
)
def test_builtin_geometry_capabilities(
    device: spy.Device, material_type: str, texture_slots: tuple[str, ...]
) -> None:
    # No geometry streams: every reported dependency must come from the material.
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    scene.update()
    empty = f2.SceneCapabilities()
    assert material.required_capabilities() == empty
    assert scene.required_capabilities == empty

    # glTF only needs geometry for selectors on effective (bound) textures.
    gltf_material = material_type in ("GLTFMaterial", "GLTFSpecGlossMaterial")
    unbound_selector_needs_geometry = not gltf_material
    for slot in texture_slots:
        property_name = f"{slot}_texture_texcoord"
        material[property_name] = 1
        assert scene.required_capabilities == empty
        scene.update()
        assert (
            material.required_capabilities().has(GEOMETRY) == unbound_selector_needs_geometry
        ), property_name
        assert (
            scene.required_capabilities.has(GEOMETRY) == unbound_selector_needs_geometry
        ), property_name
        material[property_name] = 0
        scene.update()
        assert material.required_capabilities() == empty, property_name
        assert scene.required_capabilities == empty, property_name

    property_name = f"{texture_slots[0]}_texture_texcoord"
    material[property_name] = 7
    scene.update()
    assert scene.required_capabilities.has(GEOMETRY) == unbound_selector_needs_geometry
    if gltf_material:
        texture = device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=1,
            height=1,
            usage=spy.TextureUsage.shader_resource,
            data=np.ones((1, 1, 4), dtype=np.float32),
        )
        material[f"{texture_slots[0]}_texture"] = texture
        assert scene.required_capabilities == empty
        scene.update()
        assert material.required_capabilities().has(GEOMETRY)
    assert scene.required_capabilities.has(GEOMETRY)
    material["use_vertex_color"] = True
    material[property_name] = 0
    scene.update()
    assert material.required_capabilities().has(GEOMETRY)
    assert scene.required_capabilities.has(GEOMETRY)

    if material_type == "OpenPBRMaterial":
        material["opacity_factor"] = 0.5
    else:
        material["alpha_mode"] = f2.AlphaMode.mask
    scene.update()
    assert scene.required_capabilities.has(GEOMETRY | OPACITY)
    material["use_vertex_color"] = False
    scene.update()
    assert material.required_capabilities() == f2.SceneCapabilities(OPACITY)
    assert scene.required_capabilities == f2.SceneCapabilities(OPACITY)
    material.remove()
    scene.update()
    assert scene.required_capabilities == empty


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
@pytest.mark.parametrize("material_type", [f2.StandardMaterial, f2.StandardSpecGlossMaterial])
def test_standard_solid_transmission_capability_follows_material_edits(
    device: spy.Device, material_type: type[f2.Material]
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.specular_transmission_factor = 1.0
    scene.update()
    assert material.required_capabilities().has(SOLID_TRANSMISSION)
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.thin_walled = True
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.thin_walled = False
    material.specular_transmission_factor = 0.0
    material.diffuse_transmission_factor = 1.0
    scene.update()
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.transmission_factor = spy.float3(0.0)
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.specular_transmission_factor = 1.0
    scene.update()
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
def test_gltf_solid_transmission_requires_thickness_and_transmission(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.GLTFMaterial)
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.thickness_factor = 1.0
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.transmission_factor = 1.0
    scene.update()
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.transmission_factor = 0.0
    material.diffuse_transmission_factor = 1.0
    scene.update()
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.thickness_factor = 0.0
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material.thickness_factor = 1.0
    material.feature_mask_filter = int(material.feature_mask_filter) & ~(1 << 7)  # VOLUME
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
def test_openpbr_solid_transmission_requires_weight_and_non_thin_geometry(
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material("OpenPBRMaterial")
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material["transmission_weight"] = 1.0
    scene.update()
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)

    material["geometry_thin_walled"] = True
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)

    material["geometry_thin_walled"] = False
    material["transmission_weight"] = 0.0
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
@pytest.mark.parametrize(
    "material_type", [f2.UnlitMaterial, f2.GLTFSpecGlossMaterial, f2.EmissionOnlyMaterial]
)
def test_non_transmitting_materials_do_not_require_solid_transmission(
    device: spy.Device, material_type: type[f2.Material]
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    scene.update()
    assert not material.required_capabilities().has(SOLID_TRANSMISSION)
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)


@pytest.mark.parametrize("device_type", [helpers.DEFAULT_DEVICE_TYPES[0]])
def test_unknown_material_conservatively_requires_solid_transmission(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(_CustomMaterial)
    scene.update()
    assert material.required_capabilities().has(SOLID_TRANSMISSION)
    assert scene.required_capabilities.has(SOLID_TRANSMISSION)
    material.remove()
    scene.update()
    assert not scene.required_capabilities.has(SOLID_TRANSMISSION)
