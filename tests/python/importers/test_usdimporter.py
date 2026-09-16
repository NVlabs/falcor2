# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import struct
import zipfile
from pathlib import Path

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers

DATA = Path(__file__).parent.parent.parent.parent / "data"
_DDS_PATH = DATA.parent / "external/slangpy/data/test_images/dds/bc1-unorm.dds"

_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9Wl2aX8AAAAASUVORK5CYII="
)

_PACKAGED_SCENE = b"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root"
{
    def Mesh "Quad"
    {
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        rel material:binding = </Root/Looks/Material>
        uniform token subdivisionScheme = "none"
    }

    def Scope "Looks"
    {
        def Material "Material"
        {
            token outputs:falcor:surface.connect = </Root/Looks/Material/FalcorSurface.outputs:surface>
            token outputs:surface.connect = </Root/Looks/Material/Surface.outputs:surface>

            def Shader "FalcorSurface"
            {
                uniform token info:id = "FalcorStandardMaterial"
                float inputs:alpha_cutoff = 0.5
                float inputs:alpha_factor = 1
                token inputs:alpha_mode = "opaque"
                color3f inputs:base_color_factor = (1, 1, 1)
                asset inputs:base_color_texture_path = @falcor.png@
                bool inputs:double_sided = 0
                color3f inputs:emissive_factor = (0, 0, 0)
                float inputs:ior = 1.5
                float inputs:metallic_factor = 0
                uint inputs:metallic_texture_channel = 2
                asset inputs:normal_texture_path = @normal.dds@
                float inputs:roughness_factor = 1
                uint inputs:roughness_texture_channel = 1
                token outputs:surface
            }

            def Shader "Surface"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </Root/Looks/Material/Texture.outputs:rgb>
                token outputs:surface
            }

            def Shader "Texture"
            {
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @texture.png@
                float2 inputs:st.connect = </Root/Looks/Material/Primvar.outputs:result>
                float3 outputs:rgb
            }

            def Shader "Primvar"
            {
                uniform token info:id = "UsdPrimvarReader_float2"
                token inputs:varname = "st"
                float2 outputs:result
            }
        }
    }

    def DomeLight "Sky"
    {
        asset inputs:texture:file = @environment.png@
    }
}
"""


def _aligned_zip_info(archive: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name)
    info.compress_type = zipfile.ZIP_STORED
    local_data_offset = archive.fp.tell() + 30 + len(name.encode("utf-8"))
    extra_size = (-local_data_offset) % 64
    if 0 < extra_size < 4:
        extra_size += 64
    if extra_size:
        info.extra = struct.pack("<HH", 0xFFFF, extra_size - 4) + bytes(extra_size - 4)
    return info


def _write_package(path: Path) -> None:
    with zipfile.ZipFile(path, "w", allowZip64=True) as archive:
        archive.writestr(_aligned_zip_info(archive, "scene.usda"), _PACKAGED_SCENE)
        archive.writestr(_aligned_zip_info(archive, "texture.png"), _PNG)
        archive.writestr(_aligned_zip_info(archive, "falcor.png"), _PNG)
        archive.writestr(_aligned_zip_info(archive, "normal.dds"), _DDS_PATH.read_bytes())
        archive.writestr(_aligned_zip_info(archive, "environment.png"), _PNG)


def test_format_specific_load_and_generic_import_scene() -> None:
    path = DATA / "assets/cornell-box/usdpreviewsurface/cornell-box.usda"
    usd_importer = f2.UsdImporter()
    scene = usd_importer.load_scene(path)
    assert scene is not None
    assert f2.import_scene(path) is not None

    for mesh in scene.meshes:
        assert mesh.positions is not None
        assert mesh.normals is not None
        assert mesh.tangents is not None
        assert mesh.handedness is not None
        tc0 = mesh.texcoords(0)
        assert tc0 is not None
        assert mesh.positions.shape[0] == mesh.vertex_count
        assert mesh.normals.shape[0] == mesh.vertex_count
        assert mesh.tangents.shape[0] == mesh.vertex_count
        assert mesh.handedness.shape[0] == mesh.vertex_count
        assert tc0.shape[0] == mesh.vertex_count
        # Note: Individual vertex access is no longer available in the new stream-based API
        # The above assertions verify that all attribute arrays have the correct length

        for subgeo in mesh.subgeometries:
            indices1D = subgeo.indices_numpy.flatten()
            for i, tri in enumerate(subgeo.indices):
                assert (subgeo.indices_numpy[i] == tri).all()
                assert spy.math.all(spy.bool3([indices1D[i * 3 + j] for j in range(3)] == tri))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_usdz_with_packaged_images(
    tmp_path: Path, device_type: spy.DeviceType, device: spy.Device
) -> None:
    package_path = tmp_path / "scene.usdz"
    _write_package(package_path)

    imported = f2.UsdImporter().load_scene(package_path.resolve())
    packaged_assets = {str(asset.path): bytes(asset.data) for asset in imported.assets}

    def get_asset_data(name: str) -> bytes:
        matches = [data for path, data in packaged_assets.items() if path.endswith(f"[{name}]")]
        assert len(matches) == 1
        return matches[0]

    assert get_asset_data("texture.png") == _PNG
    assert get_asset_data("falcor.png") == _PNG
    assert get_asset_data("normal.dds") == _DDS_PATH.read_bytes()
    assert get_asset_data("environment.png") == _PNG
    assert len(imported.lights) == 1

    scene = f2.Scene.load(device, package_path)
    assert scene is not None
    scene.update()

    material = scene.materials[0]
    assert material.base_color_texture is not None
    assert material.base_color_texture.format == spy.Format.rgba8_unorm_srgb
    assert material.base_color_texture_path == Path()
    assert material.normal_texture is not None
    assert material.normal_texture.format == spy.Format.bc1_unorm
    assert material.normal_texture_path == Path()

    env_map = scene.components.find(type=f2.EnvMapLight)
    assert env_map is not None
    assert env_map.env_map_path == Path()


def test_light_shaping_properties(tmp_path: Path) -> None:
    scene_path = tmp_path / "shaped_lights.usda"
    scene_path.write_text(
        """#usda 1.0

def SphereLight "Sphere" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    bool inputs:enableColorTemperature = 1
    float inputs:colorTemperature = 3200
    float inputs:shaping:cone:angle = 40
    float inputs:shaping:cone:softness = 0.25
    float inputs:shaping:focus = 4
}

def RectLight "Rect" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:shaping:cone:angle = 30
    float inputs:shaping:cone:softness = 0.5
}
""",
        encoding="utf-8",
    )

    scene = f2.UsdImporter().load_scene(scene_path)
    assert scene is not None
    assert len(scene.lights) == 2
    light = next(light for light in scene.lights if light.name == "/Sphere")
    assert light.enable_color_temperature
    assert light.color_temperature == pytest.approx(3200.0)
    assert light.enable_shaping
    assert light.shaping_cone_angle == pytest.approx(40.0)
    assert light.shaping_cone_softness == pytest.approx(0.25)
    assert light.shaping_focus == pytest.approx(4.0)

    light.enable_shaping = False
    light.shaping_cone_angle = 35.0
    light.shaping_cone_softness = 0.5
    light.shaping_focus = 2.0
    light.enable_color_temperature = False
    light.color_temperature = 4500.0
    assert not light.enable_color_temperature
    assert light.color_temperature == pytest.approx(4500.0)
    assert not light.enable_shaping
    assert light.shaping_cone_angle == pytest.approx(35.0)
    assert light.shaping_cone_softness == pytest.approx(0.5)
    assert light.shaping_focus == pytest.approx(2.0)

    rect = next(light for light in scene.lights if light.name == "/Rect")
    assert rect.type == f2.ImporterLight.Type.rectangular
    assert rect.enable_shaping
    assert rect.shaping_cone_angle == pytest.approx(30.0)
    assert rect.shaping_cone_softness == pytest.approx(0.5)
    assert rect.shaping_focus == pytest.approx(0.0)


def test_render_light_geometry_properties(tmp_path: Path) -> None:
    scene_path = tmp_path / "render_light_geometry.usda"
    scene_path.write_text(
        """#usda 1.0

def RectLight "KarmaVisible"
{
    bool karma:light:renderlightgeo = 1
}

def RectLight "FalcorVisible"
{
    custom bool falcor:geometryVisible = 1
}

def RectLight "FalcorHidden"
{
    bool karma:light:renderlightgeo = 1
    custom bool falcor:geometryVisible = 0
}
""",
        encoding="utf-8",
    )

    scene = f2.UsdImporter().load_scene(scene_path)
    assert scene is not None
    assert len(scene.lights) == 3
    lights = {light.name: light for light in scene.lights}
    assert lights["/KarmaVisible"].geometry_visible
    assert lights["/FalcorVisible"].geometry_visible
    assert not lights["/FalcorHidden"].geometry_visible

    lights["/FalcorVisible"].geometry_visible = False
    assert not lights["/FalcorVisible"].geometry_visible


def test_generic_import_routes_usdz_to_usd_importer(tmp_path: Path) -> None:
    path = tmp_path / "invalid.usdz"
    path.write_bytes(b"not a USDZ archive")

    with pytest.raises(Exception) as exc_info:
        f2.import_scene(path)

    assert "Unknown scene file extension" not in str(exc_info.value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
