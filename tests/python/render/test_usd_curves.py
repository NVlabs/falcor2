# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


def _probe(device: spy.Device, scene: f2.Scene, name: str, *arrays: np.ndarray) -> list[Any]:
    helper = SceneShaderHelper(device)
    module, requirements = helper.prepare(scene, "render/test_usd_curves.slang")
    function = (
        module[name]
        .as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    result = function(*(spy.Tensor.from_numpy(device, array) for array in arrays))
    cursor = result.cursor()
    cursor.load()
    return [cursor[i].read() for i in range(arrays[0].shape[0])]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_curve_vertex_uv_side_and_cap_hits(device_type: spy.DeviceType, device: spy.Device) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip(
            "Inline ray queries are unavailable in CUDA compute; packed UVs are covered by curve_uv_probe"
        )
    scene = f2.Scene.create(device)
    geometry = scene.create_geometry(f2.StaticCurveGeometry)
    uv = np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float32)
    geometry.set_curve_data(
        np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        np.full(2, 0.1, dtype=np.float32),
        np.array([0, 1], dtype=np.uint32),
        vertex_attributes=[
            f2.MeshVertexAttribute(f2.MeshAttributeSemantic.tex_coord, 0, uv, name="st")
        ],
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [scene.create_material(f2.StandardMaterial)]
    scene.update()
    origins = np.array([[0.2, 0, 1], [-0.05, 0, 1], [1.05, 0, 1]], dtype=np.float32)
    results = _probe(device, scene, "curve_ray_probe", origins)
    for result, origin in zip(results, origins):
        assert result["valid"]
        parameter = (float(origin[0]) + 0.1) / 1.2
        assert result["curve_parameters"][0] == pytest.approx(parameter, abs=1e-5)
        np.testing.assert_allclose(
            result["uv"], uv[0] * (1 - parameter) + uv[1] * parameter, atol=1e-5
        )
        np.testing.assert_allclose(result["uv0"], uv[0], atol=1e-6)
        np.testing.assert_allclose(result["uv1"], uv[1], atol=1e-6)
        assert result["additional_mask"] == 0
        np.testing.assert_allclose(np.asarray(result["position"])[:2], origin[:2], atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_collapsed_curve_hits(device_type: spy.DeviceType, device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    geometry = scene.create_geometry(f2.StaticCurveGeometry)
    center = np.array([2, 3, 4], dtype=np.float32)
    geometry.set_curve_data(
        np.tile(center, (2, 1)),
        np.array([0.25, 1.0], dtype=np.float32),
        np.array([0, 1], dtype=np.uint32),
    )
    instance = scene.create_entity().create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [scene.create_material(f2.StandardMaterial)]
    scene.update()
    # Off-axis and pole hits exercise reconstruction and the fallback shading frame.
    offsets = np.array([[0.36, 0.48, 3], [0, 0, 3]], dtype=np.float32)
    ids = np.full(len(offsets), int(instance.geometry_instance_id), dtype=np.uint32)
    results = _probe(device, scene, "curve_procedural_probe", ids, center + offsets)
    if device_type != spy.DeviceType.cuda:
        results += _probe(device, scene, "curve_ray_probe", center + offsets)
    expected_offsets = np.array([[0.36, 0.48, 0.8], [0, 0, 1]])
    for i, result in enumerate(results):
        offset = expected_offsets[i % 2]
        assert result["valid"]
        parameters = np.asarray(result["curve_parameters"])
        assert np.isfinite(parameters).all()
        np.testing.assert_allclose(result["position"], center + offset, atol=1e-5)
        np.testing.assert_allclose(result["normal"], offset, atol=1e-5)
        frame = result["frame"]
        basis = np.array([frame["tangent"], frame["bitangent"], frame["normal"]])
        assert np.isfinite(basis).all()
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-5)
        np.testing.assert_allclose(frame["normal"], offset, atol=1e-5)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("origin", [f2.UVOrigin.upper_left, f2.UVOrigin.lower_left])
def test_usd_curves_uv_origin_and_default_strands(
    device_type: spy.DeviceType,
    device: spy.Device,
    direct: bool,
    origin: f2.UVOrigin,
    tmp_path: Path,
) -> None:
    path = tmp_path / "curves.usda"
    path.write_text(
        """#usda 1.0
def BasisCurves "Fur" {
    uniform token type = "cubic"
    uniform token basis = "bezier"
    uniform token wrap = "nonperiodic"
    int[] curveVertexCounts = [4, 4]
    point3f[] points = [(0,0,0), (1,0,0), (2,0,0), (3,0,0), (0,1,0), (1,1,0), (2,1,0), (3,1,0)]
    float[] widths = [0.1] (interpolation = "constant")
    texCoord2f[] primvars:st = [(0.2,0.7), (0.8,0.3)] (interpolation = "uniform")
}
""",
        encoding="utf-8",
    )
    options = f2.ImportOptions()
    options.curve_refinement_level = 0 if direct else 7
    scene = f2.Scene.load(device, path, options, f2.SceneConfig(uv_origin=origin))
    scene.update()
    instance = scene.components.find_all(type=f2.GeometryInstance)[0]
    ids = np.full(2, int(instance.geometry_instance_id), dtype=np.uint32)
    primitive_indices = np.array([0, 3 if direct else 8], dtype=np.uint32)
    parameters = np.full((2, 2), 0.25, dtype=np.float32)
    results = _probe(device, scene, "curve_uv_probe", ids, primitive_indices, parameters)
    expected = np.array([[0.2, 0.7], [0.8, 0.3]], dtype=np.float32)
    if origin == f2.UVOrigin.upper_left:
        expected[:, 1] = 1 - expected[:, 1]
    for result, uv in zip(results, expected):
        np.testing.assert_allclose(result["uv"], uv, atol=1e-6)
    # Defaults use V=0.5 and need only one check per device/tessellation mode.
    if origin != f2.UVOrigin.lower_left:
        return
    path.write_text(
        path.read_text().replace(
            '    texCoord2f[] primvars:st = [(0.2,0.7), (0.8,0.3)] (interpolation = "uniform")\n',
            "",
        ),
        encoding="utf-8",
    )
    defaults = f2.Scene.load(device, path, options)
    defaults.update()
    default_instance = defaults.components.find_all(type=f2.GeometryInstance)[0]
    ids[:] = int(default_instance.geometry_instance_id)
    results = _probe(device, defaults, "curve_uv_probe", ids, primitive_indices, parameters)
    for result in results:
        np.testing.assert_allclose(result["uv"], [0.25 / (3 if direct else 8), 0.5], atol=1e-6)
