# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Callable
import base64
from io import BytesIO
import json
import struct

import numpy as np
import pytest
import slangpy as spy
from PIL import Image, PngImagePlugin

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


_SLOTS = {
    "GLTFMaterial": ["base_color", "normal", "metallic_roughness", "emissive"],
    "GLTFSpecGlossMaterial": ["diffuse", "normal", "specular_glossiness", "emissive"],
    "StandardMaterial": ["base_color", "normal", "metallic_roughness", "emissive", "transmission"],
    "StandardSpecGlossMaterial": [
        "diffuse",
        "normal",
        "specular_glossiness",
        "emissive",
        "transmission",
    ],
    "UnlitMaterial": ["base_color"],
    "OpenPBRMaterial": [
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
    ],
}


def _sampler_probe(device: spy.Device, scene: f2.Scene, material_type: str) -> Callable[..., Any]:
    if material_type in ("GLTFMaterial", "GLTFSpecGlossMaterial"):
        # Their four flat handle/selector pairs start at byte 16 in MaterialData.
        # Reading the uploaded slot also works when the scene selects an exact
        # generic material type instead of the flexible one.
        source = """
__exported import falcor2.render;
public float4 sample_binding(uint material_id, uint slot, float2 uv, float lod) {
    const MaterialData material = g_scene.material_system.get_material_data(MaterialID(material_id));
    TextureHandle handle = {};
    // The test enumerates color, normal, workflow, emissive;
    // the GPU pairs are color, workflow, normal, emissive.
    const uint gpu_slot = slot == 1u ? 2u : (slot == 2u ? 1u : slot);
    handle.data = material.payload.data[2u * gpu_slot];
    return handle.sample(ExplicitLodSampler(lod), uv, float4(-1.0));
}
"""
        helper = SceneShaderHelper(device)
        source_module = spy.Module(
            device.load_module_from_source(f"sampler_probe_{material_type}_sparse", source)
        )
        module, _ = helper.prepare(scene, source_module)
        function = module.sample_binding.as_func().write(helper.bind_scene)
        return lambda material, slot, uv, lod: function(int(material.material_id), slot, uv, lod)

    cases = []
    attributes = None
    if material_type == "OpenPBRMaterial":
        layout = scene.render_module.layout
        buffer_type = layout.find_type_by_name("StructuredBuffer<OpenPBRMaterial::Attributes>")
        attributes = layout.get_type_layout(buffer_type).element_type_layout
    for index, slot in enumerate(_SLOTS[material_type]):
        if material_type == "OpenPBRMaterial":
            if slot in ("normal", "opacity"):
                handle = f"material.{slot}_texture_handle.data"
            else:
                # Read the uploaded handle at its reflected offset; do not expose private shader fields.
                field = next(field for field in attributes.fields if field.name == slot)
                handle_field = next(
                    field for field in field.type_layout.fields if field.name == "texture_handle"
                )
                offset = field.offset + handle_field.offset
                handle = f"material.attributes_buffer.load<uint>({offset})"
        else:
            handle = f"material.{slot}_texture.data"
        cases.append(f"case {index}: handle.data = {handle}; break;")
    source = """
__exported import falcor2.render;
public float4 sample_binding(MATERIAL material, uint slot, float2 uv, float lod) {
    TextureHandle handle = {};
    switch (slot) { CASES }
    return handle.sample(ExplicitLodSampler(lod), uv, float4(-1.0));
}
""".replace(
        "MATERIAL", material_type
    ).replace(
        "CASES", "\n".join(cases)
    )
    helper = SceneShaderHelper(device)
    source_module = spy.Module(
        device.load_module_from_source(f"sampler_probe_{material_type}", source)
    )
    module, _ = helper.prepare(scene, source_module)
    return module.sample_binding.as_func().write(helper.bind_scene)


def _rgba(value: spy.float4) -> list[float]:
    return [value.x, value.y, value.z, value.w]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("image_source", ["external", "embedded"])
@pytest.mark.parametrize("image_mode", ["L", "LA", "RGBA"])
@pytest.mark.parametrize("linear_metadata", [False, True])
def test_gltf_texture_color_space_roles(
    tmp_path: Path,
    device_type: spy.DeviceType,
    image_source: str,
    image_mode: str,
    linear_metadata: bool,
) -> None:
    """glTF roles override PNG metadata without decoding alpha or data slots."""
    value = {"L": 56, "LA": (56, 102), "RGBA": (56, 56, 56, 102)}[image_mode]
    metadata = PngImagePlugin.PngInfo()
    if linear_metadata:
        metadata.add(b"gAMA", struct.pack(">I", 100000))
    image_path = tmp_path / "image.png"
    Image.new(image_mode, (2, 2), value).save(image_path, pnginfo=metadata)
    uri = image_path.name
    if image_source == "embedded":
        uri = "data:image/png;base64," + base64.b64encode(image_path.read_bytes()).decode("ascii")
    texture = {"index": 0}
    document = {
        "asset": {"version": "2.0"},
        "extensionsUsed": ["KHR_materials_pbrSpecularGlossiness", "KHR_materials_unlit"],
        "images": [{"uri": uri}],
        "textures": [{"source": 0}],
        "materials": [
            {
                "name": "specgloss",
                "extensions": {
                    "KHR_materials_pbrSpecularGlossiness": {
                        "diffuseTexture": texture,
                        "specularGlossinessTexture": texture,
                    }
                },
                "emissiveTexture": texture,
                "emissiveFactor": [1.0, 1.0, 1.0],
                "normalTexture": texture,
            },
            {
                "name": "standard",
                "pbrMetallicRoughness": {
                    "baseColorTexture": texture,
                    "metallicRoughnessTexture": texture,
                },
                "emissiveTexture": texture,
                "emissiveFactor": [1.0, 1.0, 1.0],
            },
            {
                "name": "unlit",
                "pbrMetallicRoughness": {"baseColorTexture": texture},
                "extensions": {"KHR_materials_unlit": {}},
            },
        ],
    }
    path = tmp_path / "color_spaces.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")
    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, path)
    if image_source == "embedded":
        for material, slots in [
            (scene.materials[0], ["diffuse", "specular_glossiness", "emissive", "normal"]),
            (scene.materials[1], ["base_color", "metallic_roughness", "emissive"]),
            (scene.materials[2], ["base_color"]),
        ]:
            for slot in slots:
                assert material[f"{slot}_texture_bytes"].data == image_path.read_bytes()
                assert material[f"{slot}_texture"] is None
    scene.update()
    if image_source == "embedded":
        for material, slots in [
            (scene.materials[0], ["diffuse", "specular_glossiness", "emissive", "normal"]),
            (scene.materials[1], ["base_color", "metallic_roughness", "emissive"]),
            (scene.materials[2], ["base_color"]),
        ]:
            for slot in slots:
                assert material[f"{slot}_texture"] is None
    encoded = 56 / 255
    decoded = ((encoded + 0.055) / 1.055) ** 2.4
    alpha = 1.0 if image_mode == "L" else 102 / 255
    checks = [
        (
            scene.materials[0],
            "GLTFSpecGlossMaterial",
            ["diffuse", "specular_glossiness", "emissive"],
            ["normal"],
        ),
        (
            scene.materials[1],
            "GLTFMaterial",
            ["base_color", "emissive"],
            ["metallic_roughness"],
        ),
        (scene.materials[2], "UnlitMaterial", ["base_color"], []),
    ]
    for material, material_type, color_slots, data_slots in checks:
        probe = _sampler_probe(device, scene, material_type)
        for slot in color_slots + data_slots:
            expected_rgb = decoded if slot in color_slots else encoded
            np.testing.assert_allclose(
                _rgba(probe(material, _SLOTS[material_type].index(slot), spy.float2(0.5), 0.0)),
                [expected_rgb, expected_rgb, expected_rgb, alpha],
                atol=2e-4,
                err_msg=f"{material_type}.{slot}",
            )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type",
    [
        "StandardMaterial",
        "StandardSpecGlossMaterial",
        "GLTFMaterial",
        "UnlitMaterial",
        "OpenPBRMaterial",
    ],
)
def test_material_texture_sampler_properties(
    device_type: spy.DeviceType, material_type: str
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    slot = "diffuse" if material_type == "StandardSpecGlossMaterial" else "base_color"
    assert material[f"{slot}_texture_sampler"] is None
    sampler = device.create_sampler(min_filter=spy.TextureFilteringMode.point)
    material[f"{slot}_texture_sampler"] = sampler
    assert material[f"{slot}_texture_sampler"] == sampler


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_texture_sampler_propagation(tmp_path: Path, device_type: spy.DeviceType) -> None:
    Image.new("RGBA", (2, 2), (255, 0, 0, 255)).save(tmp_path / "image.png")
    document = {
        "asset": {"version": "2.0"},
        "images": [{"uri": "image.png"}],
        "samplers": [{"minFilter": 9728, "magFilter": 9728, "wrapS": 33071, "wrapT": 33648}],
        "textures": [{"source": 0, "sampler": 0}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
    }
    path = tmp_path / "samplers.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")
    scene = f2.Scene.load(helpers.get_device(device_type), path)
    sampler = scene.materials[0]["base_color_texture_sampler"]
    assert sampler.desc.min_filter == spy.TextureFilteringMode.point
    assert sampler.desc.mag_filter == spy.TextureFilteringMode.point
    assert sampler.desc.address_u == spy.TextureAddressingMode.clamp_to_edge
    assert sampler.desc.address_v == spy.TextureAddressingMode.mirror_repeat
    assert sampler.desc.max_lod == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("material_type", list(_SLOTS))
@pytest.mark.parametrize("source", ["texture", "path"])
def test_material_sampler_live_bindings(
    tmp_path: Path, device_type: spy.DeviceType, material_type: str, source: str
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    material = scene.create_material(material_type)
    pixels = np.array(
        [[[1, 0, 0, 0], [0, 1, 0, 1]], [[0, 0, 1, 1], [1, 1, 1, 0]]], dtype=np.float32
    )
    texture = device.create_texture(
        format=spy.Format.rgba32_float,
        width=2,
        height=2,
        usage=spy.TextureUsage.shader_resource,
        data=pixels,
    )
    path = tmp_path / "binding.png"
    Image.fromarray((pixels * 255).astype(np.uint8)).save(path)
    for slot in _SLOTS[material_type]:
        assert material[f"{slot}_texture_sampler"] is None
        material[f"{slot}_texture" + ("_path" if source == "path" else "")] = (
            path if source == "path" else texture
        )
    if material_type in ("GLTFMaterial", "GLTFSpecGlossMaterial"):
        # The sparse GPU payload omits emissive textures while emission is off.
        material.emissive_factor = spy.float3(1.0)
    scene.update()
    probe = _sampler_probe(device, scene, material_type)
    generation = scene.requirements_generation
    texture_count = scene.texture_manager.stats.regular_texture_count
    if source == "texture":
        assert texture_count == 1
    nearest = device.create_sampler(
        min_filter=spy.TextureFilteringMode.point,
        mag_filter=spy.TextureFilteringMode.point,
        address_u=spy.TextureAddressingMode.clamp_to_edge,
    )
    for index, slot in enumerate(_SLOTS[material_type]):
        np.testing.assert_allclose(
            _rgba(probe(material, index, spy.float2(0.6, 0.25), 0.0)),
            [0.3, 0.7, 0, 0.7],
            atol=0.002,
        )
        material[f"{slot}_texture_sampler"] = nearest
        scene.update()
        np.testing.assert_allclose(
            _rgba(probe(material, index, spy.float2(0.6, 0.25), 0.0)), [0, 1, 0, 1], atol=0.002
        )
        # Changing one binding must not affect the next binding of the same image.
        if index + 1 < len(_SLOTS[material_type]):
            np.testing.assert_allclose(
                _rgba(probe(material, index + 1, spy.float2(0.6, 0.25), 0.0)),
                [0.3, 0.7, 0, 0.7],
                atol=0.002,
                err_msg=f"after {slot}, next {_SLOTS[material_type][index + 1]}",
            )
        material[f"{slot}_texture_sampler"] = scene.texture_manager.default_sampler
        scene.update()
        np.testing.assert_allclose(
            _rgba(probe(material, index, spy.float2(0.6, 0.25), 0.0)),
            [0.3, 0.7, 0, 0.7],
            atol=0.002,
        )
    assert scene.texture_manager.stats.regular_texture_count == texture_count
    assert scene.requirements_generation == generation


def _write_sampler_fixture(
    tmp_path: Path, image_source: str, workflow: str = "metallic_roughness"
) -> tuple[Path, list[dict[str, int]]]:
    # Each quadrant has a distinct color; the final mip is their linear average.
    pixels = np.zeros((8, 8, 4), dtype=np.uint8)
    pixels[:4, :4] = [255, 0, 0, 255]
    pixels[:4, 4:] = [0, 255, 0, 255]
    pixels[4:, :4] = [0, 0, 255, 255]
    pixels[4:, 4:] = [255, 255, 255, 255]
    path = tmp_path / "image.png"
    Image.fromarray(pixels).save(path)
    samplers = [
        {
            "minFilter": mode,
            "magFilter": 9728 if mode in (9728, 9984, 9986) else 9729,
            "wrapS": 33071,
            "wrapT": 33071,
        }
        for mode in (9728, 9729, 9984, 9985, 9986, 9987)
    ]
    samplers += [
        {"minFilter": 9728, "magFilter": 9728, "wrapS": wrap, "wrapT": wrap}
        for wrap in (10497, 33648)
    ]
    # Valid but not exactly expressible by CUDA's single within-level filter.
    samplers += [{"minFilter": 9984, "magFilter": 9729}]
    document = {
        "asset": {"version": "2.0"},
        "images": [{"uri": path.name}],
        "samplers": samplers,
        "textures": [{"source": 0, "sampler": index} for index in range(len(samplers))]
        + [{"source": 0}],
        "materials": [
            {"name": str(index), "pbrMetallicRoughness": {"baseColorTexture": {"index": index}}}
            for index in range(len(samplers) + 1)
        ],
    }
    if workflow == "specular_glossiness":
        document["extensionsUsed"] = ["KHR_materials_pbrSpecularGlossiness"]
        for material in document["materials"]:
            pbr = material.pop("pbrMetallicRoughness")
            material["extensions"] = {
                "KHR_materials_pbrSpecularGlossiness": {"diffuseTexture": pbr["baseColorTexture"]}
            }
    png = path.read_bytes()
    output = tmp_path / "samplers.gltf"
    if image_source == "embedded":
        document["images"] = [{"bufferView": 0, "mimeType": "image/png"}]
        document["buffers"] = [{"byteLength": len(png)}]
        document["bufferViews"] = [{"buffer": 0, "byteLength": len(png)}]
        text = json.dumps(document).encode("utf-8")
        text += b" " * (-len(text) % 4)
        binary = png + b"\0" * (-len(png) % 4)
        output = output.with_suffix(".glb")
        output.write_bytes(
            struct.pack("<4sII", b"glTF", 2, 28 + len(text) + len(binary))
            + struct.pack("<I4s", len(text), b"JSON")
            + text
            + struct.pack("<I4s", len(binary), b"BIN\0")
            + binary
        )
    else:
        if image_source == "data_uri":
            document["images"] = [
                {"uri": "data:image/png;base64," + base64.b64encode(png).decode("ascii")}
            ]
        output.write_text(json.dumps(document), encoding="utf-8")
    return output, samplers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("image_source", ["external", "embedded", "data_uri"])
@pytest.mark.parametrize("workflow", ["metallic_roughness", "specular_glossiness"])
def test_gltf_sampler_modes_and_shared_image(
    tmp_path: Path, device_type: spy.DeviceType, image_source: str, workflow: str
) -> None:
    path, samplers = _write_sampler_fixture(tmp_path, image_source, workflow)
    device = helpers.get_device(device_type)
    imported = f2.GltfImporter().load_scene(path)
    scene = f2.Scene.from_importer_scene(device, imported)
    scene.update()
    material_type = "GLTFSpecGlossMaterial" if workflow == "specular_glossiness" else "GLTFMaterial"
    slot = "diffuse" if workflow == "specular_glossiness" else "base_color"
    slot_index = _SLOTS[material_type].index(slot)
    sample = _sampler_probe(device, scene, material_type)
    # Samplers do not duplicate image storage, including GLB/data URI sources.
    assert scene.texture_manager.stats.regular_texture_count == 1
    for index, authored in enumerate(samplers):
        material = scene.materials.find(str(index))
        sampler = material[f"{slot}_texture_sampler"]
        min_filter = authored["minFilter"]
        point = min_filter in (9728, 9984, 9986)
        assert sampler.desc.min_filter == (
            spy.TextureFilteringMode.point if point else spy.TextureFilteringMode.linear
        )
        assert sampler.desc.mag_filter == (
            spy.TextureFilteringMode.point
            if authored["magFilter"] == 9728
            else spy.TextureFilteringMode.linear
        )
        assert imported.textures[index].use_mipmaps == (min_filter not in (9728, 9729))
        expected_mip = (
            spy.TextureFilteringMode.point
            if min_filter in (9984, 9985)
            else spy.TextureFilteringMode.linear
        )
        assert sampler.desc.mip_filter == expected_mip
        assert imported.textures[index].mip_filter == (
            f2.TextureFilterMode.nearest
            if min_filter in (9984, 9985)
            else f2.TextureFilterMode.linear
        )
        if index < 6:
            # LOD 3 is the 1x1 average unless the binding forbids mipmapping.
            expected = [1, 0, 0, 1] if index < 2 else [0.5, 0.5, 0.5, 1]
            np.testing.assert_allclose(
                _rgba(sample(material, slot_index, spy.float2(0.25, 0.25), 3.0)),
                expected,
                atol=0.008,
            )
            # Between the 2x2 and 1x1 levels: nearest chooses red, linear blends.
            expected = [0.8, 0.2, 0.2, 1] if index in (4, 5) else [1, 0, 0, 1]
            np.testing.assert_allclose(
                _rgba(sample(material, slot_index, spy.float2(0.25, 0.25), 2.4)),
                expected,
                atol=0.008,
            )
    for index, expected in [(0, [0, 1, 0, 1]), (6, [1, 0, 0, 1]), (7, [0, 1, 0, 1])]:
        material = scene.materials.find(str(index))
        np.testing.assert_allclose(
            _rgba(sample(material, slot_index, spy.float2(1.25, 0.25), 0.0)), expected, atol=0.002
        )
    for index, expected in [(0, [0, 0, 1, 1]), (6, [1, 0, 0, 1]), (7, [0, 0, 1, 1])]:
        material = scene.materials.find(str(index))
        np.testing.assert_allclose(
            _rgba(sample(material, slot_index, spy.float2(0.25, 1.25), 0.0)), expected, atol=0.002
        )
    for index, expected in [(0, [1, 0, 0, 1]), (6, [0, 1, 0, 1]), (7, [1, 0, 0, 1])]:
        material = scene.materials.find(str(index))
        np.testing.assert_allclose(
            _rgba(sample(material, slot_index, spy.float2(-0.25, 0.25), 0.0)), expected, atol=0.002
        )
    assert scene.materials.find(str(len(samplers)))[f"{slot}_texture_sampler"] is None
    if device_type == spy.DeviceType.cuda:
        # Document the existing CUDA backend limitation, not a software emulation.
        mixed = scene.materials.find("8")
        np.testing.assert_allclose(
            _rgba(sample(mixed, slot_index, spy.float2(0.51, 0.25), -1.0)),
            [0, 1, 0, 1],
            atol=0.002,
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_all_material_sampler_mappings(tmp_path: Path, device_type: spy.DeviceType) -> None:
    path, _ = _write_sampler_fixture(tmp_path, "external")
    document = json.loads(path.read_text(encoding="utf-8"))
    binding = {"index": 0}
    document["extensionsUsed"] = ["KHR_materials_pbrSpecularGlossiness"]
    document["materials"] = [
        {
            "name": "pbr",
            "alphaMode": "MASK",
            "pbrMetallicRoughness": {
                "baseColorTexture": binding,
                "metallicRoughnessTexture": binding,
            },
            "normalTexture": binding,
            "emissiveTexture": binding,
        },
        {
            "name": "specgloss",
            "normalTexture": binding,
            "emissiveTexture": binding,
            "extensions": {
                "KHR_materials_pbrSpecularGlossiness": {
                    "diffuseTexture": binding,
                    "specularGlossinessTexture": binding,
                }
            },
        },
    ]
    path.write_text(json.dumps(document), encoding="utf-8")
    scene = f2.Scene.load(helpers.get_device(device_type), path)
    scene.update()
    sampler = scene.materials.find("pbr")["base_color_texture_sampler"]
    assert sampler.desc.min_filter == spy.TextureFilteringMode.point
    for name, slots in [
        ("pbr", ["base_color", "metallic_roughness", "normal", "emissive"]),
        ("specgloss", ["diffuse", "specular_glossiness", "normal", "emissive"]),
    ]:
        for slot in slots:
            assert scene.materials.find(name)[f"{slot}_texture_sampler"] == sampler
    # The scene default and one shared authored sampler.
    assert scene.texture_manager.stats.total_sampler_count == 2


def _write_embedded_color_scene(path: Path, name: str, color: tuple[int, int, int, int]) -> bytes:
    stream = BytesIO()
    Image.new("RGBA", (2, 2), color).save(stream, format="PNG")
    data = stream.getvalue()
    document = {
        "asset": {"version": "2.0"},
        "images": [
            {
                "name": f"{name} image",
                "uri": "data:image/png;base64," + base64.b64encode(data).decode("ascii"),
            }
        ],
        "textures": [{"source": 0}],
        "materials": [{"name": name, "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
    }
    path.write_text(json.dumps(document), encoding="utf-8")
    return data


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize(
    "mode, material_type",
    [
        (f2.GLTFMaterialMode.standard_material, "StandardMaterial"),
        (f2.GLTFMaterialMode.openpbr, "OpenPBRMaterial"),
    ],
)
def test_embedded_texture_import_modes(
    tmp_path: Path, device_type: spy.DeviceType, mode: f2.GLTFMaterialMode, material_type: str
) -> None:
    path = tmp_path / "mode.gltf"
    encoded = _write_embedded_color_scene(path, "Color", (255, 0, 0, 255))
    options = f2.ImportOptions()
    options.gltf_material_mode = mode
    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, path, import_options=options)
    material = scene.materials.find("Color")
    assert material["base_color_texture_bytes"].data == encoded
    assert material["base_color_texture"] is None

    scene.update()
    sample = _sampler_probe(device, scene, material_type)
    slot = _SLOTS[material_type].index("base_color")
    np.testing.assert_allclose(_rgba(sample(material, slot, spy.float2(0.5), 0.0)), [1, 0, 0, 1])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_embedded_texture_outlives_importer_and_append_uses_new_bytes(
    tmp_path: Path, device_type: spy.DeviceType
) -> None:
    path = tmp_path / "changing.gltf"
    first_data = _write_embedded_color_scene(path, "First", (255, 0, 0, 255))
    device = helpers.get_device(device_type)
    imported = f2.GltfImporter().load_scene(path)
    scene = f2.Scene.from_importer_scene(device, imported)
    del imported

    second_data = _write_embedded_color_scene(path, "Second", (0, 0, 255, 255))
    scene.append(f2.GltfImporter().load_scene(path))
    first = scene.materials.find("First")
    second = scene.materials.find("Second")
    assert first.base_color_texture_bytes.data == first_data
    assert second.base_color_texture_bytes.data == second_data
    assert first.base_color_texture is None
    assert second.base_color_texture is None

    scene.update()
    sample = _sampler_probe(device, scene, "GLTFMaterial")
    slot = _SLOTS["GLTFMaterial"].index("base_color")
    np.testing.assert_allclose(_rgba(sample(first, slot, spy.float2(0.5), 0.0)), [1, 0, 0, 1])
    np.testing.assert_allclose(_rgba(sample(second, slot, spy.float2(0.5), 0.0)), [0, 0, 1, 1])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_embedded_and_file_textures_load_in_one_update(
    tmp_path: Path, device_type: spy.DeviceType
) -> None:
    path = tmp_path / "mixed.gltf"
    _write_embedded_color_scene(path, "Embedded", (255, 0, 0, 255))
    Image.new("RGBA", (2, 2), (0, 0, 255, 255)).save(tmp_path / "file.png")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["images"].append({"uri": "file.png"})
    document["textures"].append({"source": 1})
    document["materials"].append(
        {"name": "File", "pbrMetallicRoughness": {"baseColorTexture": {"index": 1}}}
    )
    path.write_text(json.dumps(document), encoding="utf-8")

    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, path)
    assert scene.materials.find("Embedded").base_color_texture_bytes is not None
    assert scene.materials.find("File").base_color_texture_path == tmp_path / "file.png"
    scene.update()

    sample = _sampler_probe(device, scene, "GLTFMaterial")
    slot = _SLOTS["GLTFMaterial"].index("base_color")
    np.testing.assert_allclose(
        _rgba(sample(scene.materials.find("Embedded"), slot, spy.float2(0.5), 0.0)),
        [1, 0, 0, 1],
    )
    np.testing.assert_allclose(
        _rgba(sample(scene.materials.find("File"), slot, spy.float2(0.5), 0.0)),
        [0, 0, 1, 1],
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_embedded_texture_decoding_is_on_demand(
    tmp_path: Path, device_type: spy.DeviceType
) -> None:
    corrupt = b"invalid embedded PNG"
    document = {
        "asset": {"version": "2.0"},
        "images": [
            {
                "name": "Broken image",
                "uri": "data:image/png;base64," + base64.b64encode(corrupt).decode("ascii"),
            }
        ],
        "textures": [{"source": 0}],
        "materials": [
            {"name": "Referenced", "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}
        ],
    }
    path = tmp_path / "corrupt.gltf"
    path.write_text(json.dumps(document), encoding="utf-8")
    device = helpers.get_device(device_type)

    scene = f2.Scene.load(device, path)
    assert scene.materials.find("Referenced").base_color_texture_bytes.data == corrupt
    with pytest.raises(RuntimeError, match="Broken image"):
        scene.update()

    overridden = f2.Scene.load(device, path)
    material = overridden.materials.find("Referenced")
    material.base_color_texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba8_unorm,
        width=1,
        height=1,
        usage=spy.TextureUsage.shader_resource,
    )
    overridden.update()

    removed = f2.Scene.load(device, path)
    removed.materials.find("Referenced").remove()
    removed.update()

    document["materials"] = [{"name": "Unused"}]
    path.write_text(json.dumps(document), encoding="utf-8")
    unused = f2.Scene.load(device, path)
    unused.update()
