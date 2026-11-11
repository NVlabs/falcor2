# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Material-level emission eligibility must precede per-placement light storage."""

from pathlib import Path
import re

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper


_DEVICE_TYPES = [
    device_type
    for device_type in helpers.DEFAULT_DEVICE_TYPES
    if device_type in (spy.DeviceType.d3d12, spy.DeviceType.vulkan)
]


def _counts(scene: f2.Scene) -> tuple[int, int]:
    helper = SceneShaderHelper(scene.device)
    module, _ = helper.prepare(scene, "render/test_materialx_emission.slang")
    result = module.emissive_triangle_counts.as_func().write(helper.bind_scene)()
    return int(result.x), int(result.y)


def _scene(
    device: spy.Device, properties: dict[str, object]
) -> tuple[f2.Scene, f2.MaterialXMaterial]:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.MaterialXMaterial, f2.Properties(properties))
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
        normals=np.array([[0, 0, 1]] * 3, dtype=np.float32),
        texcoords=np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32),
        sub_mesh_indices=[np.array([[0, 1, 2]], dtype=np.uint32)],
    )
    for _ in range(4):
        instance = scene.create_entity().create_component(f2.GeometryInstance)
        instance.geometry = geometry
        instance.materials = [material]
    return scene, material


def _document(inputs: str, nodes: str = "", category: str = "standard_surface") -> str:
    return f"""<materialx version="1.39">
  {nodes}
  <{category} name="surface" type="surfaceshader">
    {inputs}
  </{category}>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>"""


_COMPOUND = """<nodedef name="ND_test_emitter" node="test_emitter">
  <input name="strength" type="float" value="0" />
  <output name="out" type="surfaceshader" />
</nodedef>
<nodegraph name="NG_test_emitter" nodedef="ND_test_emitter">
  <standard_surface name="inner" type="surfaceshader">
    <input name="emission" type="float" interfacename="strength" />
  </standard_surface>
  <output name="out" type="surfaceshader" nodename="inner" />
</nodegraph>"""


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize(
    "kind",
    [
        "arithmetic",
        "compound",
        "shared_compound",
        "color",
        "color_product",
        "luminance",
        "dot",
        "dot_color",
    ],
)
def test_materialx_generic_emission_edits(device: spy.Device, kind: str) -> None:
    if kind == "arithmetic":
        document = _document(
            '<input name="emission" type="float" nodename="difference" />',
            """<constant name="control" type="float"><input name="value" type="float" value="1" /></constant>
<subtract name="difference" type="float">
  <input name="in1" type="float" nodename="control" />
  <input name="in2" type="float" value="1" />
</subtract>""",
        )
        parameter, values = "control_value", (1.0, 2.0, 1.0)
    elif kind in ("dot", "dot_color"):
        is_color = kind == "dot_color"
        value_type = "color3" if is_color else "float"
        surface_input = "emission_color" if is_color else "emission"
        zero = "0, 0, 0" if is_color else "0"
        document = _document(
            ('<input name="emission" type="float" value="1" />' if is_color else "")
            + f'<input name="{surface_input}" type="{value_type}" nodename="reroute" />',
            f"""<constant name="control" type="{value_type}"><input name="value" type="{value_type}" value="{zero}" /></constant>
<dot name="reroute" type="{value_type}">
  <input name="in" type="{value_type}" nodename="control" />
  <input name="note" type="string" value="Emission control" />
</dot>""",
        )
        parameter = "control_value"
        values = (
            (spy.float3(0), spy.float3(0, 1, 0), spy.float3(0)) if is_color else (0.0, 1.0, 0.0)
        )
    elif kind == "luminance":
        document = _document(
            '<input name="emission" type="float" nodename="channel" />',
            """<luminance name="luma" type="color3">
  <input name="in" type="color3" value="0, 0, 0" />
</luminance>
<extract name="channel" type="float">
  <input name="in" type="color3" nodename="luma" />
  <input name="index" type="integer" value="0" />
</extract>""",
        )
        parameter, values = "luma_in", (spy.float3(0), spy.float3(0, 1, 0), spy.float3(0))
    elif kind == "compound":
        document = _document(
            '<input name="strength" type="float" value="0" />', _COMPOUND, "test_emitter"
        )
        parameter, values = "surface_strength", (0.0, 1.0, 0.0)
    elif kind == "shared_compound":
        document = _document(
            """<input name="bg" type="surfaceshader" nodename="dark" />
<input name="fg" type="surfaceshader" nodename="edited" />
<input name="mix" type="float" value="0.5" />""",
            _COMPOUND
            + """<test_emitter name="dark" type="surfaceshader"><input name="strength" type="float" value="0" /></test_emitter>
<test_emitter name="edited" type="surfaceshader"><input name="strength" type="float" value="0" /></test_emitter>""",
            "mix",
        )
        parameter, values = "edited_strength", (0.0, 1.0, 0.0)
    elif kind == "color_product":
        document = _document(
            '<input name="emission" type="float" value="1" />'
            '<input name="emission_color" type="color3" nodename="product" />',
            """<multiply name="product" type="color3">
  <input name="in1" type="color3" value="1, 0, 0" />
  <input name="in2" type="color3" value="0, 1, 0" />
</multiply>""",
        )
        parameter = "product_in2"
        # Each operand is nonzero, but disjoint channels produce an exactly black product.
        values = (spy.float3(0, 1, 0), spy.float3(1, 0, 0), spy.float3(0, 1, 0))
    else:
        document = _document(
            '<input name="emission" type="float" value="1" />'
            '<input name="emission_color" type="color3" value="0, 0, 0" />'
        )
        parameter = "surface_emission_color"
        values = (spy.float3(0), spy.float3(0, 1, 0), spy.float3(0))
    scene, material = _scene(
        device,
        {
            "mtlx_buffer": document,
            "mtlx_node_name": "M",
            "mtlx_editable_params": parameter,
            "mtlx_canonicalize_names": True,
        },
    )
    compiled = None
    for value, count in zip(values, (0, 4, 0)):
        material[f"inputs:{parameter}"] = value
        scene.update()
        assert _counts(scene) == (count, count)
        if compiled is None:
            compiled = material.required_module()
        assert material.required_module() is compiled


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("weight,color,expected", [(1, "0,0,0", 0), (1e-12, "1,1,1", 4)])
def test_materialx_exact_emission_zero(
    device: spy.Device, weight: float, color: str, expected: int
) -> None:
    scene, _ = _scene(
        device,
        {
            "mtlx_buffer": _document(
                f'<input name="emission" type="float" value="{weight}" />'
                f'<input name="emission_color" type="color3" value="{color}" />'
            ),
            "mtlx_node_name": "M",
        },
    )
    scene.update()
    assert _counts(scene)[0] == expected


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("operation", ["multiply", "mix"])
def test_materialx_spatial_emission_uniform_gate(
    device: spy.Device, tmp_path: Path, operation: str
) -> None:
    texture = tmp_path / "emission.png"
    spy.Bitmap(np.full((2, 2, 3), 255, dtype=np.uint8)).write(texture)
    operands = (
        '<input name="in1" type="float" nodename="image" />'
        '<input name="in2" type="float" nodename="control" />'
        if operation == "multiply"
        else '<input name="fg" type="float" nodename="image" />'
        '<input name="bg" type="float" value="0" />'
        '<input name="mix" type="float" nodename="control" />'
    )
    document = _document(
        '<input name="emission" type="float" nodename="reroute" />',
        f"""<image name="image" type="float">
  <input name="file" type="filename" value="{texture.as_posix()}" />
</image>
<constant name="control" type="float"><input name="value" type="float" value="0" /></constant>
<{operation} name="gate" type="float">{operands}</{operation}>
<dot name="reroute" type="float">
  <input name="in" type="float" nodename="gate" />
  <input name="note" type="string" value="Spatial emission gate" />
</dot>""",
    )
    shader_path = tmp_path / "material.slang"
    scene, material = _scene(
        device,
        {
            "mtlx_buffer": document,
            "mtlx_node_name": "M",
            "mtlx_editable_params": "control_value",
            "debug_write_shader_path": str(shader_path),
        },
    )
    for value, count in ((0.0, 0), (1.0, 4), (0.0, 0)):
        material["inputs:control_value"] = value
        scene.update()
        assert _counts(scene) == (count, count)
    # Only the standalone query is inspected: the full shading graph must retain its texture.
    source = shader_path.read_text(encoding="utf-8")
    query = source.split("bool mx_emission_is_nonzero(", 1)[1].split("}", 1)[0]
    assert "lod_sampler" not in query
    assert "SurfaceInteraction" not in query


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
def test_materialx_geometry_emission_remains_eligible(device: spy.Device) -> None:
    scene, _ = _scene(
        device,
        {
            "mtlx_buffer": _document(
                '<input name="emission" type="float" nodename="attribute" />',
                """<geompropvalue name="attribute" type="float">
  <input name="geomprop" type="string" value="missing_emission" />
  <input name="default" type="float" value="0" />
</geompropvalue>""",
            ),
            "mtlx_node_name": "M",
        },
    )
    scene.update()
    # A missing attribute's zero default is not a proof about every possible hit.
    assert _counts(scene) == (4, 0)


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
def test_materialx_emission_ignores_misleading_graph_name(device: spy.Device) -> None:
    definitions = _COMPOUND.replace("NG_test_emitter", "NG_standard_surface_surfaceshader_100")
    definitions = definitions.replace(
        '<standard_surface name="inner" type="surfaceshader">'
        '\n    <input name="emission" type="float" interfacename="strength" />'
        "\n  </standard_surface>",
        '<uniform_edf name="light" type="EDF"><input name="color" type="color3" value="1,1,1" /></uniform_edf>'
        '<surface name="inner" type="surfaceshader"><input name="edf" type="EDF" nodename="light" /></surface>',
    )
    scene, _ = _scene(
        device,
        {
            "mtlx_buffer": _document(
                '<input name="strength" type="float" value="0" />', definitions, "test_emitter"
            ),
            "mtlx_node_name": "M",
            "mtlx_canonicalize_names": True,
        },
    )
    scene.update()
    assert _counts(scene) == (4, 4)


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("weight", [None, 0, 1])
def test_materialx_constant_emission_candidates(device: spy.Device, weight: int | None) -> None:
    inputs = "" if weight is None else f'<input name="emission" type="float" value="{weight}" />'
    scene, _ = _scene(device, {"mtlx_buffer": _document(inputs), "mtlx_node_name": "M"})
    scene.update()
    assert _counts(scene) == ((4, 4) if weight else (0, 0))


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize(
    "mode,canonicalize,large,category",
    [
        (None, False, False, "standard_surface"),
        (None, True, False, "standard_surface"),
        (None, True, True, "standard_surface"),
        (None, True, False, "open_pbr_surface"),
        (f2.MaterialXLayeringMode.closure_tree, True, False, "standard_surface"),
        (f2.MaterialXLayeringMode.bsdf_mix, True, False, "standard_surface"),
    ],
)
def test_materialx_emission_edits(
    device: spy.Device,
    tmp_path: Path,
    mode: f2.MaterialXLayeringMode | None,
    canonicalize: bool,
    large: bool,
    category: str,
) -> None:
    parameter = "emission" if category == "standard_surface" else "emission_luminance"
    shader_path = tmp_path / "material.slang"
    properties = {
        "mtlx_buffer": _document(
            f'<input name="{parameter}" type="float" value="0" />', category=category
        ),
        "mtlx_node_name": "M",
        "mtlx_editable_params": "*" if large else f"surface_{parameter}",
        "mtlx_canonicalize_names": canonicalize,
        "debug_write_shader_path": str(shader_path),
    }
    if mode is not None:
        properties["mtlx_layering_mode"] = mode
    scene, material = _scene(device, properties)
    compiled = None
    for weight in (0, 1, 0):
        material[f"inputs:surface_{parameter}"] = float(weight)
        scene.update()
        assert _counts(scene) == ((4, 4) if weight else (0, 0))
        if compiled is None:
            compiled = material.required_module()
        assert material.required_module() is compiled
    shader = shader_path.read_text(encoding="utf-8")
    payload_size = int(re.search(r"#define MATERIALX_PAYLOAD_SIZE (\d+)", shader)[1])
    payload_limit = int(re.search(r"#define MATERIAL_PAYLOAD_LIMIT (\d+)", shader)[1])
    # Scene materials currently use a zero-byte inline payload allowance.
    # Exercise both the single-float and fully editable buffer layouts here;
    # the native codegen integration test covers nonzero inline allowances.
    assert payload_size > payload_limit
    if large:
        assert payload_size > 64


@pytest.mark.parametrize("device_type", _DEVICE_TYPES)
@pytest.mark.parametrize("factor_texture", [False, True])
@pytest.mark.parametrize("texture_value", [0, 255])
def test_materialx_textured_emission_candidates(
    device: spy.Device, tmp_path: Path, factor_texture: bool, texture_value: int
) -> None:
    texture = tmp_path / "emission.png"
    spy.Bitmap(np.full((4, 4, 3), texture_value, dtype=np.uint8)).write(texture)
    image_type = "float" if factor_texture else "color3"
    nodes = f"""<image name="image" type="{image_type}">
      <input name="file" type="filename" value="{texture.as_posix()}" />
    </image>"""
    inputs = (
        '<input name="emission" type="float" nodename="image" />'
        if factor_texture
        else '<input name="emission" type="float" value="0" />'
        '<input name="emission_color" type="color3" nodename="image" />'
    )
    properties = {"mtlx_buffer": _document(inputs, nodes), "mtlx_node_name": "M"}
    if not factor_texture:
        properties["mtlx_editable_params"] = "surface_emission"
    scene, material = _scene(device, properties)
    for weight in (1,) if factor_texture else (0, 1, 0):
        if not factor_texture:
            material["inputs:surface_emission"] = float(weight)
        scene.update()
        # A black texture cannot be classified from one sample. Keep its
        # candidate storage unless a proven uniform zero disables emission.
        assert _counts(scene) == (4 if weight else 0, 4 if weight and texture_value else 0)
