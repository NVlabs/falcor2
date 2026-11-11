# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.pxr import Usd, UsdGeom


def _probe(
    device: spy.Device, scene: f2.Scene, origins: np.ndarray, directions: np.ndarray
) -> list[Any]:
    return _make_probe(device, scene)(origins, directions)


def _make_probe(
    device: spy.Device, scene: f2.Scene
) -> Callable[[np.ndarray, np.ndarray], list[Any]]:
    # Reuse shader setup across animation updates with unchanged geometry/material types.
    helper = SceneShaderHelper(device)
    source = spy.Module(
        device.load_module("render/test_usd_ribbons.slang"),
        link=[spy.Module.load_from_file(device, "falcor2/ui/kernels/scene_picker.slang")],
    )
    module, requirements = helper.prepare(scene, source)
    ray = f2.SceneRayTracingSetup.RayDesc()
    ray.name = "intersect"
    ray.has_miss = True
    ray.has_closest_hit = True
    setup = f2.SceneRayTracingSetup.create(scene, [ray])
    function = (
        module.ribbon_probe.constants({"ENABLE_TRIANGLE_BACKFACE_CULLING": True})
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
        .ray_tracing(
            hit_groups=setup.hit_groups,
            hit_group_names=setup.sbt_hit_group_names,
            miss_entry_points=setup.sbt_miss_entry_points,
            max_recursion=1,
            max_ray_payload_size=128,
            flags=setup.pipeline_flags,
        )
    )

    def probe(origins: np.ndarray, directions: np.ndarray) -> list[Any]:
        result = function(
            spy.Tensor.from_numpy(device, origins), spy.Tensor.from_numpy(device, directions)
        )
        cursor = result.cursor()
        cursor.load()
        return [cursor[i].read() for i in range(len(origins))]

    return probe


# Exercise the mesh integration on each backend; UV policy itself is CPU-tested.
# One additional case covers authored UVs and the scene's V flip.
@pytest.mark.parametrize(
    "device_type,uv_origin,authored_uv",
    [(backend, f2.UVOrigin.lower_left, False) for backend in helpers.DEFAULT_DEVICE_TYPES]
    + [(backend, f2.UVOrigin.upper_left, True) for backend in helpers.DEFAULT_DEVICE_TYPES[:1]],
)
def test_usd_ribbon_hits_material_uv_and_transform(
    device_type: spy.DeviceType,
    device: spy.Device,
    tmp_path: Path,
    uv_origin: f2.UVOrigin,
    authored_uv: bool,
) -> None:
    path = tmp_path / "ribbon.usda"
    st = (
        'texCoord2f[] primvars:st = [(0.2,0.3),(0.8,0.7)] (interpolation = "vertex")'
        if authored_uv
        else ""
    )
    path.write_text(
        """#usda 1.0
(metersPerUnit = 1)
def Material "Paint" {
 token outputs:surface.connect = </Paint/Surface.outputs:surface>
 def Shader "Surface" {
  uniform token info:id = "UsdPreviewSurface"
  color3f inputs:diffuseColor = (0.3,0.5,0.7)
  color3f inputs:emissiveColor = (0.2,0.4,0.6)
  token outputs:surface
 }
}
def BasisCurves "Strip" {
 uniform token type = "linear"
 int[] curveVertexCounts = [2]
 point3f[] points = [(0,0,0), (2,0,0)]
 normal3f[] normals = [(0,0,1)] (interpolation = "constant")
 float[] widths = [2] (interpolation = "constant")
 rel material:binding = </Paint>
 double3 xformOp:translate = (3,0,0)
 uniform token[] xformOpOrder = ["xformOp:translate"]
"""
        + st
        + "\n}\n",
        encoding="utf-8",
    )
    scene = f2.Scene.load(device, path, f2.ImportOptions(), f2.SceneConfig(uv_origin=uv_origin))
    scene.update()
    instance = scene.components.find_all(type=f2.GeometryInstance)[0]
    assert isinstance(instance.geometry, f2.StaticMeshGeometry)
    instance.materials[0].enable_backface_culling = True
    scene.update()
    origins = np.array(
        [[3.5, 0.5, 1], [3.5, 1.1, 1], [3.5, -2, 0.1], [3.5, 0.5, -1]], dtype=np.float32
    )
    directions = np.array([[0, 0, -1], [0, 0, -1], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    hits = _probe(device, scene, origins, directions)
    assert [h["valid"] for h in hits] == [True, False, False, False]
    np.testing.assert_allclose(hits[0]["position"], [3.5, 0.5, 0], atol=1e-5)
    np.testing.assert_allclose(hits[0]["normal"], [0, 0, 1], atol=1e-5)
    uv = [0.35, 0.4] if authored_uv else [0.75, 0.25]
    if uv_origin == f2.UVOrigin.upper_left:
        uv[1] = 1 - uv[1]
    np.testing.assert_allclose(hits[0]["uv"], uv, atol=1e-5)
    np.testing.assert_allclose(hits[0]["emission"], [0.2, 0.4, 0.6], atol=1e-5)
    assert np.isfinite(hits[0]["tangent"]).all()
    assert np.linalg.norm(hits[0]["tangent"]) == pytest.approx(1, abs=1e-5)
    # Material sidedness applies through the ordinary mesh traversal path.
    instance.materials[0].enable_backface_culling = False
    scene.update()
    assert _probe(device, scene, origins[3:], directions[3:])[0]["valid"]

    if not authored_uv and uv_origin == f2.UVOrigin.lower_left:
        transform = f2.Transform()
        transform.translation = spy.float3(3, 0, 0)
        transform.scale = spy.float3(-1, 1, 1)
        instance.entity.transform = transform
        scene.update()
        mirrored = _probe(
            device, scene, np.array([[2.5, 0.5, 1]], dtype=np.float32), directions[:1]
        )[0]
        assert mirrored["valid"]
        np.testing.assert_allclose(mirrored["position"], [2.5, 0.5, 0], atol=1e-5)
        assert np.isfinite(mirrored["normal"]).all()
        path.write_text(
            path.read_text().replace(
                "rel material:binding", "bool doubleSided = true\n rel material:binding"
            ),
            encoding="utf-8",
        )
        two_sided = f2.Scene.load(device, path)
        two_sided.update()
        material = two_sided.components.find_all(type=f2.GeometryInstance)[0].materials[0]
        assert not material.enable_backface_culling
        assert _probe(device, two_sided, origins[3:], directions[3:])[0]["valid"]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_usd_ribbon_generated_uv_texture(device_type: spy.DeviceType, device: spy.Device) -> None:
    from PIL import Image

    asset = Path(__file__).resolve().parents[3] / "data/assets/test_usd_ribbons/textured.usda"
    scene = f2.Scene.load(
        device, asset, f2.ImportOptions(), f2.SceneConfig(uv_origin=f2.UVOrigin.lower_left)
    )
    scene.update()
    origins = np.array(
        [[x, y + offset, 1] for y in [3, 1.5, 0] for x in [1, 5] for offset in [-0.2, 0.2]],
        dtype=np.float32,
    )
    directions = np.tile(np.array([0, 0, -1], dtype=np.float32), (len(origins), 1))
    hits = _probe(device, scene, origins, directions)
    # RenderMan-style coordinates, verified against Karma: U across the width,
    # V along the spine. Compare actual material emission with decoded, bilinear
    # texture samples, independently of the tessellator.
    pixels = np.asarray(Image.open(asset.with_name("uv_grid.png")), dtype=np.float64) / 255
    pixels = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    height, width = pixels.shape[:2]
    for i, hit in enumerate(hits):
        x = origins[i, 0]
        offset = [-0.2, 0.2][i % 2]
        uv = np.array([0.5 + offset / 0.65, x / 6])
        texel = np.array([uv[0] * width, (1 - uv[1]) * height]) - 0.5
        ix, iy = np.floor(texel).astype(int)
        tx, ty = texel - [ix, iy]
        expected = (
            (1 - tx) * (1 - ty) * pixels[iy % height, ix % width]
            + tx * (1 - ty) * pixels[iy % height, (ix + 1) % width]
            + (1 - tx) * ty * pixels[(iy + 1) % height, ix % width]
            + tx * ty * pixels[(iy + 1) % height, (ix + 1) % width]
        )
        assert hit["valid"]
        np.testing.assert_allclose(hit["uv"], uv, atol=1e-6)
        np.testing.assert_allclose(hit["emission"], expected, atol=2e-3)
        np.testing.assert_allclose(hit["tangent"], [0, 1, 0], atol=1e-5)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_usd_ribbon_motion_sample_submission(device: spy.Device) -> None:
    stage = Usd.Stage.CreateInMemory()
    stage.SetTimeCodesPerSecond(2)
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    curve = UsdGeom.BasisCurves.Define(stage, "/Ribbon")
    curve.CreateTypeAttr("linear")
    curve.CreateCurveVertexCountsAttr([2])
    curve.CreatePointsAttr().Set([(0, 0, 0), (2, 0, 0)], 0)
    curve.GetPointsAttr().Set([(0, 0, 2), (2, 0, 2)], 2)
    curve.CreateNormalsAttr([(0, 0, 1)])
    curve.SetNormalsInterpolation("constant")
    curve.CreateWidthsAttr([1])
    curve.SetWidthsInterpolation("constant")
    options = f2.ImportOptions()
    options.import_curve_motion = True
    imported = f2.UsdImporter().import_stage(stage, options)
    mesh = imported.meshes[0]
    assert mesh.motion_times == [0, 1]
    scene = f2.Scene.from_importer_scene(device, imported)
    scene.update()
    assert any(isinstance(g, f2.DynamicMeshGeometry) for g in scene.geometries)
    origins = np.array([[1, 0.25, 4]], dtype=np.float32)
    directions = np.array([[0, 0, -1]], dtype=np.float32)
    before = _probe(device, scene, origins, directions)[0]
    assert before["valid"]
    for time in (0, 0.5, 1):
        scene.time = time
        scene.update()
        result = _probe(device, scene, origins, directions)[0]
        assert result["valid"]
        np.testing.assert_allclose(result["position"], [1, 0.25, 2 * time], atol=1e-5)
