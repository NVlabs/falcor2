# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import base64
import struct
import zipfile
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers

DATA = Path(__file__).parent.parent.parent.parent / "data"
_DDS_PATH = DATA.parent / "external/slangpy/data/test_images/dds/bc1-unorm.dds"


@pytest.fixture
def preview_material(tmp_path: Path) -> tuple:
    from falcor2.pxr import Sdf, Usd, UsdGeom, UsdShade

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    material = UsdShade.Material.Define(stage, "/Material")
    surface = UsdShade.Shader.Define(stage, "/Material/Surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    )
    texture = UsdShade.Shader.Define(stage, "/Material/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture_path = tmp_path / "texture.png"
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    pixels = np.full((2, 2, 4), 64, dtype=np.uint8)
    spy.Bitmap(pixels).write(texture_path)
    for channel, value_type in (
        ("rgb", Sdf.ValueTypeNames.Float3),
        ("r", Sdf.ValueTypeNames.Float),
        ("g", Sdf.ValueTypeNames.Float),
        ("b", Sdf.ValueTypeNames.Float),
        ("a", Sdf.ValueTypeNames.Float),
    ):
        texture.CreateOutput(channel, value_type)
    mesh = UsdGeom.Mesh.Define(stage, "/Mesh")
    mesh.CreatePointsAttr([(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return stage, surface, texture, texture_path


def _load_preview(preview: tuple) -> tuple:
    stage, _, _, _ = preview
    device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
    scene = f2.Scene.from_importer_scene(device, f2.UsdImporter().import_stage(stage))
    material = next(m for m in scene.materials if isinstance(m, f2.StandardMaterial))
    return scene, material


def test_preview_packed_metallic_roughness(preview_material: tuple) -> None:
    from falcor2.pxr import Sdf

    _, surface, texture, texture_path = preview_material
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
    for name, channel in (("metallic", "b"), ("roughness", "g")):
        value = surface.CreateInput(name, Sdf.ValueTypeNames.Float)
        value.Set(0.0)  # A connected input must replace this fallback.
        value.ConnectToSource(texture.GetOutput(channel))
    scene, material = _load_preview(preview_material)
    assert material.metallic_roughness_texture_path == texture_path
    assert float(material.metallic_factor) == 1
    assert float(material.roughness_factor) == 1
    assert (material.metallic_texture_channel, material.roughness_texture_channel) == (2, 1)


@pytest.mark.parametrize(
    "opacity,threshold,textured,expected_mode,expected_factor",
    [
        (None, 0, True, f2.AlphaMode.opaque, 1),
        (0.25, 0, False, f2.AlphaMode.blend, 0.25),
        (0, 0, True, f2.AlphaMode.blend, 0),
        (0.5, 0.5, True, f2.AlphaMode.opaque, 1),
        (0.25, 0.5, True, f2.AlphaMode.mask, 0),
        ("alpha", 0.5, True, f2.AlphaMode.mask, 1),
        ("alpha", 0, True, f2.AlphaMode.blend, 1),
        ("scaled_alpha", 0.25, True, f2.AlphaMode.mask, 0.5),
    ],
)
def test_preview_opacity(
    preview_material: tuple,
    opacity: str | float | None,
    threshold: float,
    textured: bool,
    expected_mode: f2.AlphaMode,
    expected_factor: float,
) -> None:
    from falcor2.pxr import Sdf

    _, surface, texture, _ = preview_material
    if textured:
        surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            texture.GetOutput("rgb")
        )
    surface.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(threshold)
    if opacity is not None:
        value = surface.CreateInput("opacity", Sdf.ValueTypeNames.Float)
        if isinstance(opacity, str):
            value.ConnectToSource(texture.GetOutput("a"))
            if opacity == "scaled_alpha":
                texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1, 1, 1, 0.5))
        else:
            value.Set(opacity)
    scene, material = _load_preview(preview_material)
    assert material.alpha_mode == expected_mode
    assert material.alpha_factor == pytest.approx(expected_factor)
    if expected_mode == f2.AlphaMode.mask:
        assert material.alpha_cutoff == pytest.approx(threshold)
    scene.update()
    assert scene.required_capabilities.has(f2.SceneCapabilityFlags.opacity_evaluation) == (
        expected_mode != f2.AlphaMode.opaque
    )


@pytest.mark.parametrize(
    "case", ["independent", "red", "bias", "scale", "procedural", "constant_with_texture"]
)
def test_preview_unsupported_opacity(preview_material: tuple, case: str) -> None:
    from falcor2.pxr import Sdf, UsdShade

    stage, surface, texture, _ = preview_material
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.GetOutput("rgb")
    )
    opacity = surface.CreateInput("opacity", Sdf.ValueTypeNames.Float)
    opacity.ConnectToSource(texture.GetOutput("a"))
    if case == "independent":
        other = UsdShade.Shader.Define(stage, "/Material/Other")
        other.CreateIdAttr("UsdUVTexture")
        other.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture.GetInput("file").Get())
        opacity.ConnectToSource(other.CreateOutput("a", Sdf.ValueTypeNames.Float))
    elif case == "red":
        opacity.ConnectToSource(texture.GetOutput("r"))
    elif case == "bias":
        texture.CreateInput("bias", Sdf.ValueTypeNames.Float4).Set((0, 0, 0, 0.1))
    elif case == "scale":
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set((1, 1, 1, 2))
    elif case == "procedural":
        texture.CreateIdAttr("UsdPrimvarReader_float")
    else:
        opacity.DisconnectSource()
        opacity.Set(0.25)
    with pytest.raises(RuntimeError, match="opacity"):
        _load_preview(preview_material)


def test_preview_imported_mask_clips_rays(preview_material: tuple) -> None:
    from falcor2.pxr import Sdf
    from falcor2.rendernodes import ReferencePathTracerNode

    _, surface, texture, _ = preview_material
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.GetOutput("rgb")
    )
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set((1, 0, 0))
    surface.CreateInput("opacity", Sdf.ValueTypeNames.Float).ConnectToSource(texture.GetOutput("a"))
    surface.CreateInput("opacityThreshold", Sdf.ValueTypeNames.Float).Set(0.5)
    scene, material = _load_preview(preview_material)
    device = helpers.get_device(helpers.DEFAULT_DEVICE_TYPES[0])
    camera = helpers.create_test_camera(
        scene,
        width=8,
        height=8,
        fov_y=35,
        position=spy.float3(0, 0, 2),
        rotation=spy.math.quat_from_look_at(spy.float3(0, 0, -1), spy.float3(0, 1, 0)),
    )
    renderer = ReferencePathTracerNode.create(device)
    renderer.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    renderer.enable_nee = False
    renderer.max_depth = 1
    renderer.use_background_color = True
    renderer.background_color = spy.float3(0)
    renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
    assert material.alpha_mode == f2.AlphaMode.mask
    for opaque in (False, True):
        if opaque:
            material.alpha_mode = f2.AlphaMode.opaque
        scene.update()
        image, _ = renderer(scene, camera, iteration=0, pixel_offset=spy.float2(0))
        color = image.to_numpy()[4, 4, :3]
        if opaque:
            assert color[0] > 0.5
            assert color[1:] == pytest.approx((0, 0), abs=1e-4)
        else:
            assert color == pytest.approx((0, 0, 0), abs=1e-4)


@pytest.mark.parametrize("scale", [None, (2, 3, 4, 0.25)])
def test_textured_emission_scale(preview_material: tuple, scale: tuple | None) -> None:
    from falcor2.pxr import Sdf

    _, surface, texture, texture_path = preview_material
    if scale is not None:
        texture.CreateInput("scale", Sdf.ValueTypeNames.Float4).Set(scale)
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.GetOutput("rgb")
    )
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
        texture.GetOutput("rgb")
    )
    scene, imported = _load_preview(preview_material)
    assert np.asarray(imported.emissive_factor) == pytest.approx(scale[:3] if scale else (1, 1, 1))
    assert imported.emissive_texture_path == texture_path


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_scene_max_mip_count(tmp_path: Path, device_type: spy.DeviceType) -> None:
    from falcor2.pxr import Usd

    image_path = tmp_path / "texture.png"
    spy.Bitmap(np.full((16, 16, 4), 128, dtype=np.uint8)).write(image_path)
    path = tmp_path / "scene.usda"
    stage = Usd.Stage.CreateNew(str(path))
    stage.GetRootLayer().customLayerData = {"falcor:maxMipCount": 4}
    stage.DefinePrim("/Root", "Xform")
    stage.GetRootLayer().Save()

    device = helpers.get_device(device_type)
    for cap, expected in ((5, 4), (3, 3)):
        scene = f2.Scene.load(device, path, config=f2.SceneConfig(max_mip_count=cap))
        assert scene.config.max_mip_count == expected
        manager = scene.texture_manager
        handle = manager.load_texture(image_path)
        assert handle.texture.width == handle.texture.height == 1 << (expected - 1)
        assert manager.load_texture(image_path, max_mip_count=None) == handle
        assert manager.load_texture(image_path, max_mip_count=expected) == handle
        for override, width in ((2, 2), (0, 16)):
            overridden = manager.load_texture(
                image_path, max_mip_count=override, load_deferred=True
            )
            assert overridden != handle
            manager.update()
            assert overridden.texture.width == overridden.texture.height == width
            assert manager.load_texture(image_path, max_mip_count=override) == overridden


def test_importer_property_factories(tmp_path: Path) -> None:
    options = f2.ImportOptions()
    options.mesh_refinement_level = 2
    options.curve_refinement_level = 7
    options.recompute_normals = True
    options.force_rectangle_lights_to_geometry = True
    assert f2.ImporterMesh.make_properties(options)["refinement_level"] == 2
    assert f2.ImporterCurve.make_properties(options)["refinement_level"] == 7
    assert f2.ImporterLight.make_properties(options, f2.ImporterLight.Type.rectangular)[
        "convert_to_mesh"
    ]
    assert not f2.ImporterLight.make_properties(options, f2.ImporterLight.Type.dome)[
        "convert_to_mesh"
    ]
    scene_path = tmp_path / "properties.usda"
    scene_path.write_text(
        """#usda 1.0
def Mesh "Triangle" {
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    uniform token subdivisionScheme = "none"
}
def DomeLight "Dome" {}
""",
        encoding="utf-8",
    )
    scene = f2.UsdImporter().load_scene(scene_path, options)
    mesh = scene.meshes[0]
    assert mesh.properties["recompute_normals"]
    mesh.properties["refinement_level"] = 0
    assert mesh.properties["refinement_level"] == 0
    scene.lights[0].properties["convert_to_mesh"] = True
    assert scene.lights[0].properties["convert_to_mesh"]


# One RGB pixel with an explicit sRGB chunk, used for packaged color textures.
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAAAXNSR0IArs4c6QAAAAxJREFUeJxjaGhoAAADBAGBS9PSEAAAAABJRU5ErkJggg=="
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
                uint inputs:nested_priority = 5
                float3 inputs:volume_sigma_a = (12, 16, 20)
                float3 inputs:volume_sigma_s = (0.1, 0.2, 0.3)
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
    for objects in (imported.nodes, imported.meshes, imported.materials, imported.lights):
        for obj in objects:
            if obj.source_asset_ref is not None:
                assert obj.source_asset_ref.asset_path == str(package_path.resolve())
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
    assert material.nested_priority == 5
    np.testing.assert_allclose(material.volume_sigma_a, [12, 16, 20])
    np.testing.assert_allclose(material.volume_sigma_s, [0.1, 0.2, 0.3])
    assert material.source_asset_ref is not None
    assert material.source_asset_ref.format == "usd"
    assert material.source_asset_ref.object_path == "/Root/Looks/Material"
    assert material in scene.find_objects_by_source_asset_ref(material.source_asset_ref)

    source_asset_path = str(package_path.resolve())
    quad_objects = scene.find_objects_by_source_asset_ref(
        f2.SourceAssetRef("usd", source_asset_path, "/Root/Quad")
    )
    assert len([obj for obj in quad_objects if isinstance(obj, f2.Entity)]) == 1
    assert len([obj for obj in quad_objects if isinstance(obj, f2.GeometryInstance)]) == 1

    sky_objects = scene.find_objects_by_source_asset_ref(
        f2.SourceAssetRef("usd", source_asset_path, "/Root/Sky")
    )
    assert len([obj for obj in sky_objects if isinstance(obj, f2.Entity)]) == 1
    assert len([obj for obj in sky_objects if isinstance(obj, f2.Light)]) == 1
    assert material.base_color_texture is not None
    assert material.base_color_texture.format == spy.Format.rgba8_unorm_srgb
    assert material.base_color_texture_path == Path()
    assert material.normal_texture is not None
    assert material.normal_texture.format == spy.Format.bc1_unorm
    assert material.normal_texture_path == Path()

    env_map = scene.components.find(type=f2.EnvMapLight)
    assert env_map is not None
    assert env_map.env_map_path == Path()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_source_lookup_distinguishes_assets(
    tmp_path: Path, device_type: spy.DeviceType, device: spy.Device
) -> None:
    first_path = tmp_path / "first.usda"
    second_path = tmp_path / "second.usda"
    content = '#usda 1.0\ndef Camera "Camera" {}\n'
    first_path.write_text(content)
    second_path.write_text(content)

    scene = f2.Scene.load(device, first_path)
    scene.append(second_path)
    first_source = f2.SourceAssetRef("usd", str(first_path.resolve()), "/Camera")
    second_source = f2.SourceAssetRef("usd", str(second_path.resolve()), "/Camera")
    first_objects = scene.find_objects_by_source_asset_ref(first_source)
    second_objects = scene.find_objects_by_source_asset_ref(second_source)
    for objects, source in ((first_objects, first_source), (second_objects, second_source)):
        assert len(objects) == 2
        assert sum(isinstance(obj, f2.Entity) for obj in objects) == 1
        assert sum(isinstance(obj, f2.Camera) for obj in objects) == 1
        assert all(obj.source_asset_ref == source for obj in objects)
    assert all(obj not in second_objects for obj in first_objects)


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

def CylinderLight "Cylinder" (
    prepend apiSchemas = ["ShapingAPI"]
)
{
    float inputs:length = 3
    float inputs:radius = 0.75
    float inputs:shaping:cone:angle = 45
    bool treatAsLine = 1
}
""",
        encoding="utf-8",
    )

    scene = f2.UsdImporter().load_scene(scene_path)
    assert scene is not None
    assert len(scene.lights) == 3
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

    cylinder = next(light for light in scene.lights if light.name == "/Cylinder")
    assert cylinder.type == f2.ImporterLight.Type.cylinder
    assert cylinder.radius == pytest.approx(0.75)
    assert cylinder.length == pytest.approx(3.0)
    assert cylinder.treat_as_line
    assert cylinder.enable_shaping
    assert cylinder.shaping_cone_angle == pytest.approx(45.0)


def test_render_light_geometry_properties(tmp_path: Path) -> None:
    scene_path = tmp_path / "render_light_geometry.usda"
    scene_path.write_text(
        """#usda 1.0

def RectLight "KarmaVisible"
{
    bool karma:light:renderlightgeo = 1
}

def RectLight "UsdVisible"
{
    custom bool inputs:enableInPrimaryRay = 1
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
    assert len(scene.lights) == 4
    lights = {light.name: light for light in scene.lights}
    assert lights["/KarmaVisible"].geometry_visible
    assert lights["/UsdVisible"].geometry_visible
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


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_usd_texcoord_primvar_mapping_file_entry_points(
    tmp_path: Path, device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene_path = tmp_path / "mapped_texcoords.usda"
    scene_path.write_text(
        """#usda 1.0

def Mesh "MappedTexcoords"
{
    uniform token subdivisionScheme = "none"
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
    texCoord2f[] primvars:lightmap = [(4, 1), (5, 1), (4, 2)] (
        interpolation = "vertex"
    )
}
""",
        encoding="utf-8",
    )

    options = f2.ImportOptions()
    defaults = [f"st{index}" for index in range(1, 8)]
    assert list(options.usd_texcoord_primvar_names) == defaults
    with pytest.raises((TypeError, ValueError)):
        options.usd_texcoord_primvar_names = ["lightmap"]
    names = defaults.copy()
    names[0] = "lightmap"
    options.usd_texcoord_primvar_names = names
    assert list(options.usd_texcoord_primvar_names) == names

    for imported in (
        f2.UsdImporter().load_scene(scene_path, options),
        f2.import_scene(scene_path, options),
    ):
        uv1 = imported.meshes[0].texcoords(1)
        assert uv1 is not None
        np.testing.assert_array_equal(uv1[0], [4.0, 1.0])

    scene = f2.Scene.load(device, scene_path, options)
    geometry = scene.geometries[0]
    uv1 = next(
        attribute
        for attribute in geometry.vertex_attributes(0)
        if attribute.semantic == f2.MeshAttributeSemantic.tex_coord and attribute.index == 1
    )
    assert uv1.data[0, 0] == 4.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_held_animation_interpolation() -> None:
    from falcor2.pxr import Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    translate = UsdGeom.Xform.Define(stage, "/Animated").AddTranslateOp()
    translate.Set((0, 0, 0), 0)
    translate.Set((10, 0, 0), 24)
    imported = f2.UsdImporter().import_stage(stage)
    assert len(imported.animation.channels) == 1
    assert imported.animation.channels[0].interpolation_mode == f2.AnimationInterpolationMode.step
    assert not any(node.name.endswith(("_pivot", "_pivot_inverse")) for node in imported.nodes)


@pytest.mark.parametrize("animated", [False, True])
def test_pivot_transform(animated: bool) -> None:
    from falcor2.pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Y")
    camera = UsdGeom.Camera.Define(stage, "/Camera")
    api = UsdGeom.XformCommonAPI(camera)
    api.SetPivot(Gf.Vec3f(1, 2, 3))
    api.SetRotate(Gf.Vec3f(0, 0, 45))
    if animated:
        api.SetPivot(Gf.Vec3f(1, 2, 3), 0)
        api.SetPivot(Gf.Vec3f(2, 3, 4), 24)
        api.SetRotate(Gf.Vec3f(0, 0, 45), time=0)
        api.SetRotate(Gf.Vec3f(0, 0, 90), time=24)

    imported = f2.UsdImporter().import_stage(stage)
    nodes = {node.name: node for node in imported.nodes}
    authored = nodes["/Camera"]
    matrix = authored.transform.to_numpy()
    if animated:
        pivot, inverse = nodes["/Camera_pivot"], nodes["/Camera_pivot_inverse"]
        assert imported.nodes[authored.parent].name == pivot.name
        assert imported.nodes[inverse.parent].name == authored.name
        assert inverse.camera_index >= 0
        assert authored.animation_rotation >= 0
        for node, sign in ((pivot, 1), (inverse, -1)):
            assert node.animation_translation >= 0
            channel = imported.animation.channels[node.animation_translation]
            np.testing.assert_allclose(channel.values, np.array([1, 2, 3, 2, 3, 4]) * sign)
        matrix = pivot.transform.to_numpy() @ matrix @ inverse.transform.to_numpy()
    else:
        assert "/Camera_pivot" not in nodes
        assert authored.camera_index >= 0
    np.testing.assert_allclose(matrix, np.array(camera.GetLocalTransformation(0)).T, atol=2e-6)


def test_inherited_mesh_primvars() -> None:
    from falcor2.pxr import Sdf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    parent = UsdGeom.Scope.Define(stage, "/Parent")
    mesh = UsdGeom.Mesh.Define(stage, "/Parent/Mesh")
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
    mesh.CreateFaceVertexCountsAttr([3])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    api = UsdGeom.PrimvarsAPI(parent)
    api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "constant").Set([(0.25, 0.75)])
    api.CreatePrimvar("normals", Sdf.ValueTypeNames.Normal3fArray, "constant").Set([(0, 1, 0)])
    api.CreatePrimvar("weight", Sdf.ValueTypeNames.FloatArray, "constant").Set([0.7])

    result = f2.UsdImporter().import_stage(stage).meshes[0]
    np.testing.assert_allclose(result.texcoords(0), [(0.25, 0.75)] * 3)
    np.testing.assert_allclose(result.normals, [(0, 1, 0)] * 3)
    np.testing.assert_allclose(np.asarray(result.find_stream("weight")).ravel(), [0.7] * 3)
