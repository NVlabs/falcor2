# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import struct
import zlib

import numpy as np
from PIL import Image, PngImagePlugin
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers

from .geometry_property_helpers import TRIANGLE_INDICES, TRIANGLE_POSITIONS, read_material_probe


def _write_test_material(tmp_path: Path, textured_default: bool) -> None:
    for name, color in [("red", (255, 0, 0)), ("blue", (0, 0, 255))]:
        Image.new("RGB", (4, 4), color).save(tmp_path / f"{name}.png")
    default = (
        'texture_2d("red.png", gamma: tex::gamma_srgb)' if textured_default else "texture_2d()"
    )
    (tmp_path / "texture_parameters.mdl").write_text(
        f"""mdl 1.10;
import ::df::*;
import ::tex::*;

color sample_or_fallback(uniform texture_2d image) = tex::texture_isvalid(image)
    ? tex::lookup_color(image, float2(0.5)) : color(0.125);

export material Test(uniform texture_2d first = {default}, uniform texture_2d second = {default})
    = material(surface: material_surface(scattering: df::diffuse_reflection_bsdf(
        tint: (sample_or_fallback(first) + sample_or_fallback(second)) * 0.5)));
""",
        encoding="utf-8",
    )


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("textured_default", [False, True], ids=["empty", "shared-default"])
def test_mdl_texture_parameters(
    device_type: spy.DeviceType, textured_default: bool, tmp_path: Path
) -> None:
    _write_test_material(tmp_path, textured_default)
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MDLMaterial,
        f2.Properties(
            {
                "mdl_library_path": str(tmp_path),
                "mdl_material_name": "texture_parameters::Test",
                "first": str(tmp_path / "red.png"),
                "second": str(tmp_path / "blue.png"),
            }
        ),
    )
    assert "first" in material and "second" in material
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(positions=TRIANGLE_POSITIONS, sub_mesh_indices=[TRIANGLE_INDICES])
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]

    def check_color(expected: tuple[float, float, float]) -> None:
        scene.update()
        handles = material.build_texture_list()
        assert len(handles) == len({handle.data for handle in handles})
        actual = read_material_probe(device, scene, instance)["diffuse_reflection_albedo"]
        np.testing.assert_allclose(actual, expected, atol=1e-5)

    # Distinct overrides must not alias each other, even with identical defaults.
    check_color((0.5, 0.0, 0.5))
    module_name = material.required_module().name
    material["first"] = ""
    check_color((0.0625, 0.0625, 0.5625))
    material["second"] = ""
    check_color((0.125, 0.125, 0.125))
    embedded_blue = f2.Blob((tmp_path / "blue.png").read_bytes(), "mdl-blue.png")
    material["first:texture_bytes"] = embedded_blue
    check_color((0.0625, 0.0625, 0.5625))
    assert material["first:texture"] is None
    assert material["first:texture_bytes"].name == "mdl-blue.png"
    assert any(handle.path is None for handle in material.build_texture_list())
    material["first:texture_bytes"] = None
    material["first"] = str(tmp_path / "blue.png")
    material["second"] = str(tmp_path / "red.png")
    check_color((0.5, 0.0, 0.5))
    assert material.required_module().name == module_name

    # A codegen property edit must preserve the edited paths and resource slots.
    material.debug_write_shader_path = str(tmp_path / "regenerated.slang")
    material.mdl_library_path = f"{tmp_path};{tmp_path}"
    check_color((0.5, 0.0, 0.5))
    assert material["first"] == str(tmp_path / "blue.png")
    assert (tmp_path / "regenerated.slang").is_file()


@pytest.mark.parametrize("textured_default", [False, True], ids=["empty", "srgb-default"])
@pytest.mark.parametrize(
    "color_spaces, image_mode, gamma, expected_srgb",
    [
        (("sRGB", "raw"), "RGB", None, (True, False)),
        (("auto", "raw"), "RGB", None, (True, False)),
        (("auto", "sRGB"), "RGB", 1.0, (False, True)),
        (("auto", "raw"), "RGB", 1.0 / 2.2, (True, False)),
        (("auto", "raw"), "L", None, (False, False)),
        (("auto", "raw"), "I;16", None, (False, False)),
        (("auto", "raw"), "RGB16", None, (False, False)),
        (("auto", "raw"), "RGBA", None, (True, False)),
        (("auto", "raw"), "RGB16", "sRGB", (False, False)),
    ],
    ids=[
        "explicit",
        "auto-rgb8",
        "auto-rgb8-linear-tag",
        "auto-srgb-metadata",
        "auto-grey8",
        "auto-grey16",
        "auto-rgb16",
        "auto-rgba8",
        "auto-srgb-tag-rgb16",
    ],
)
def test_usd_mdl_texture_color_spaces(
    tmp_path: Path,
    textured_default: bool,
    color_spaces: tuple[str, str],
    image_mode: str,
    gamma: float | str | None,
    expected_srgb: tuple[bool, bool],
) -> None:
    from falcor2.pxr import Sdf, Usd, UsdGeom, UsdShade

    _write_test_material(tmp_path, textured_default)
    if image_mode == "RGBA":
        color = (128, 128, 128, 255)
    elif image_mode == "RGB":
        color = (128, 128, 128)
    else:
        color = 128 * 257 if image_mode == "I;16" else 128
    metadata = PngImagePlugin.PngInfo()
    if gamma == "sRGB":
        metadata.add(b"sRGB", b"\0")
    elif gamma is not None:
        metadata.add(b"gAMA", round(gamma * 100000).to_bytes(4, "big"))
    if image_mode == "RGB16":
        # Pillow and the SGL writer cannot write untagged 16-bit RGB PNGs.
        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data))
            )

        rows = (b"\0" + struct.pack(">12H", *([128 * 257] * 12))) * 4
        (tmp_path / "grey.png").write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 16, 2, 0, 0, 0))
            + b"".join(chunk(kind, data) for kind, data, *_ in metadata.chunks)
            + chunk(b"IDAT", zlib.compress(rows))
            + chunk(b"IEND", b"")
        )
    else:
        Image.new(image_mode, (4, 4), color).save(tmp_path / "grey.png", pnginfo=metadata)
    stage_path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(stage_path))
    usd_material = UsdShade.Material.Define(stage, "/Material")
    shader = UsdShade.Shader.Define(stage, "/Material/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("texture_parameters.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("Test", "mdl")
    for name, colorspace in zip(("first", "second"), color_spaces):
        texture = shader.CreateInput(name, Sdf.ValueTypeNames.Asset)
        texture.Set(Sdf.AssetPath("grey.png"))
        texture.GetAttr().SetColorSpace(colorspace)
    usd_material.CreateSurfaceOutput("mdl").ConnectToSource(
        shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    )
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr(TRIANGLE_POSITIONS.tolist())
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    mesh.CreateSubdivisionSchemeAttr("none")
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(usd_material)
    stage.GetRootLayer().Save()

    device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
    scene = f2.Scene.load(device, stage_path)
    material = scene.materials[0]
    assert isinstance(material, f2.MDLMaterial)
    assert material["first:srgb"] is expected_srgb[0]
    assert material["second:srgb"] is expected_srgb[1]
    scene.update()
    # One image with distinct color-space interpretations must retain both handles.
    grey_handles = [
        handle for handle in material.build_texture_list() if handle.path == tmp_path / "grey.png"
    ]
    assert len(grey_handles) == len(set(expected_srgb))
    if len(grey_handles) == 2:
        assert grey_handles[0] != grey_handles[1]
    instance = scene.components.find(type=f2.GeometryInstance)
    actual = read_material_probe(device, scene, instance)["diffuse_reflection_albedo"]
    raw = 128 / 255
    linear = ((raw + 0.055) / 1.055) ** 2.4
    expected = sum(linear if srgb else raw for srgb in expected_srgb) / 2
    # Scalar image channel expansion is separate from transfer-function selection.
    np.testing.assert_allclose(actual[0], expected, atol=2e-4)
    if image_mode in {"RGB", "RGBA", "RGB16"}:
        np.testing.assert_allclose(actual, [expected] * 3, atol=2e-4)


def test_mdl_mixed_texture_sources(tmp_path: Path) -> None:
    """Embedded BSDF lookup data must survive alongside an editable image and regeneration."""
    _write_test_material(tmp_path, False)
    (tmp_path / "mixed_sources.mdl").write_text(
        """mdl 1.10;
import ::df::*;
import ::tex::*;

export material Test(uniform texture_2d image = texture_2d()) = let {
    color editable = tex::texture_isvalid(image)
        ? tex::lookup_color(image, float2(0.5)) : color(0.125);
    color fixed = tex::lookup_color(texture_2d("red.png", tex::gamma_linear), float2(0.5));
} in material(surface: material_surface(scattering: df::sheen_bsdf(
    roughness: 0.3, tint: (editable + fixed) * 0.5, multiscatter_tint: color(1))));
""",
        encoding="utf-8",
    )
    device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
    scene = f2.Scene.create(device)
    material = scene.create_material(
        f2.MDLMaterial,
        f2.Properties(
            {
                "mdl_library_path": str(tmp_path),
                "mdl_material_name": "mixed_sources::Test",
                "image": str(tmp_path / "blue.png"),
            }
        ),
    )

    def read_lookup_data() -> list[np.ndarray]:
        scene.update()
        handles = material.build_texture_list()
        assert all(handle.is_valid() and handle.is_finalized() for handle in handles)
        textures = [handle.texture for handle in handles]
        assert any(texture.type == spy.TextureType.texture_2d for texture in textures)
        lookup_data = [
            texture.to_numpy() for texture in textures if texture.type == spy.TextureType.texture_3d
        ]
        assert lookup_data
        assert all(np.isfinite(data).all() and np.any(data != 0) for data in lookup_data)
        return lookup_data

    before = read_lookup_data()
    material["image"] = ""
    after_clear = read_lookup_data()
    material.debug_write_shader_path = str(tmp_path / "regenerated.slang")
    material.mdl_library_path = f"{tmp_path};{tmp_path}"
    after_regeneration = read_lookup_data()
    assert material["image"] == ""
    assert (tmp_path / "regenerated.slang").is_file()
    for actual in [after_clear, after_regeneration]:
        assert len(actual) == len(before)
        for expected_data, actual_data in zip(before, actual):
            np.testing.assert_array_equal(actual_data, expected_data)
