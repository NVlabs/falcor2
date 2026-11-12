# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import AccumulationMode, PathTracerPipeline, ReferencePathTracerNode
from falcor2.testing.image_test_plugin import ImageTest


PROJECT_DIR = Path(__file__).resolve().parents[3]
MATERIAL_AOV_DOCUMENT = PROJECT_DIR / "tests/data/materialx/additional_graph_outputs.mtlx"
MARBLE_AOV_DOCUMENT = PROJECT_DIR / "data/assets/test_material_aovs/marble_blue_veins.mtlx"
MDL_AOV_LIBRARY = PROJECT_DIR / "data/assets/mdl_sdk_examples"
MDL_MARBLE_LIBRARY = PROJECT_DIR / "data/assets/test_material_aovs"
MATERIALX_MDL_LIBRARY = PROJECT_DIR / "external/MaterialX/source/MaterialXGenMdl/mdl"
SHADER_BALL_SCENE = PROJECT_DIR / "data/assets/standard_shader_ball/standard_shader_ball_scene.usda"
MATERIAL_AOV_DEVICE_TYPES = [
    pytest.param(
        device_type,
        marks=pytest.mark.nightly if device_type == spy.DeviceType.cuda else (),
    )
    for device_type in helpers.DEFAULT_DEVICE_TYPES
]


def _skip_materialx_on_cuda(device_type: spy.DeviceType) -> None:
    if device_type == spy.DeviceType.cuda:
        pytest.skip("MaterialXMaterial is disabled on CUDA")


def _add_quad(scene: f2.Scene, material: f2.Material, *, uv1: bool = False) -> None:
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array(
            [
                [-0.7, -0.7, 0.0],
                [0.7, -0.7, 0.0],
                [-0.7, 0.7, 0.0],
                [0.7, 0.7, 0.0],
            ],
            dtype=np.float32,
        ),
        sub_mesh_indices=[np.array([[0, 2, 1], [2, 3, 1]], dtype=np.uint32)],
        normals=np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (4, 1)),
        tangents=np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (4, 1)),
        handedness=np.ones((4,), dtype=np.float32),
        texcoords=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
        vertex_attributes=(
            [
                f2.MeshVertexAttribute(
                    f2.MeshAttributeSemantic.tex_coord,
                    1,
                    np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32),
                )
            ]
            if uv1
            else []
        ),
        name="material_aov_quad",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]


def _create_material_aov_scene(
    device: spy.Device,
    *,
    document: Path | str = MATERIAL_AOV_DOCUMENT,
    node_name: str = "stormtrooper_dust_surface",
    layering_mode: f2.MaterialXLayeringMode | None = None,
    uv1: bool = False,
    scene: f2.Scene | None = None,
    debug_write_shader_path: Path | None = None,
) -> tuple[f2.Scene, f2.Material, f2.Camera]:
    scene = scene if scene is not None else f2.Scene.create(device)
    properties = f2.Properties(
        {
            ("mtlx_path" if isinstance(document, Path) else "mtlx_buffer"): str(document),
            "mtlx_node_name": node_name,
        }
    )
    if layering_mode is not None:
        properties["mtlx_layering_mode"] = layering_mode
    if debug_write_shader_path is not None:
        properties["debug_write_shader_path"] = str(debug_write_shader_path)
    material = scene.create_material(f2.MaterialXMaterial, properties)
    _add_quad(scene, material, uv1=uv1)
    camera = helpers.create_test_camera(
        scene,
        width=12,
        height=10,
        fov_y=35,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(
            spy.float3(0.0, 0.0, 1.0),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    return scene, material, camera


def _create_mdl_aov_scene(
    device: spy.Device,
    *,
    debug_write_shader_path: Path | None = None,
    material_name: str = "::tutorials_aov::green",
    library_path: str = str(MDL_AOV_LIBRARY),
    scene: f2.Scene | None = None,
) -> tuple[f2.Scene, f2.Material, f2.Camera]:
    scene = scene if scene is not None else f2.Scene.create(device)
    properties = f2.Properties(
        {
            "mdl_library_path": library_path,
            "mdl_material_name": material_name,
            "mdl_target_material_type": "::tutorials_aov::aov_material",
            "mdl_class_compilation": True,
        }
    )
    if debug_write_shader_path is not None:
        properties["debug_write_shader_path"] = str(debug_write_shader_path)
    material = scene.create_material(f2.MDLMaterial, properties)
    _add_quad(scene, material)
    camera = helpers.create_test_camera(
        scene,
        width=12,
        height=10,
        fov_y=35,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(
            spy.float3(0.0, 0.0, 1.0),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    return scene, material, camera


def _create_marble_shader_ball_scene(
    device: spy.Device,
    width: int,
    height: int,
    backend: str,
    layering_mode: f2.MaterialXLayeringMode | None = None,
    debug_write_shader_path: Path | None = None,
) -> tuple[f2.Scene, f2.Camera, f2.GeometryInstance]:
    scene = f2.Scene.load(device, SHADER_BALL_SCENE)
    if backend == "materialx":
        properties = {
            "mtlx_path": str(MARBLE_AOV_DOCUMENT),
            "mtlx_node_name": "Marble_3D",
        }
        if layering_mode is not None:
            properties["mtlx_layering_mode"] = layering_mode
        material = scene.create_material(
            f2.MaterialXMaterial,
            f2.Properties(properties),
        )
    elif backend == "mdl":
        assert layering_mode is None
        properties = f2.Properties(
            {
                "mdl_library_path": f"{MDL_MARBLE_LIBRARY};{MATERIALX_MDL_LIBRARY}",
                "mdl_material_name": "::marble_blue_veins::Marble_3D",
                "mdl_class_compilation": True,
            }
        )
        if debug_write_shader_path is not None:
            properties["debug_write_shader_path"] = str(debug_write_shader_path)
        material = scene.create_material(
            f2.MDLMaterial,
            properties,
        )
    else:
        raise ValueError(f"Unknown marble material backend: {backend}")
    surface = next(
        instance
        for instance in scene.components.find_all(type=f2.GeometryInstance)
        if instance.geometry.name.endswith("/shader_ball/material_surface")
    )
    surface.materials = [material]
    camera = scene.components.find(type=f2.Camera)
    assert camera is not None
    camera.width = width
    camera.height = height
    return scene, camera, surface


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize("backend", ["materialx-inline", "materialx-buffered", "mdl"])
def test_material_aov_writes_without_instance_setup(
    device_type: spy.DeviceType, device: spy.Device, backend: str, tmp_path: Path
) -> None:
    shader_path = tmp_path / "material.slang"
    if backend == "mdl":
        scene, material, _ = _create_mdl_aov_scene(device, debug_write_shader_path=shader_path)
        name = "example"
        expected = [0, 1, 0, 0]
    else:
        _skip_materialx_on_cuda(device_type)
        scene = f2.Scene.create(device)
        buffered = backend == "materialx-buffered"
        # A connected sum keeps enough editable parameters to exceed the inline payload.
        nodes = []
        previous = ""
        for index in range(64 if buffered else 1):
            first_input = f'nodename="{previous}"' if index else 'value="0.25"'
            nodes.append(
                f'<add name="sum{index}" type="float">'
                f'<input name="in1" type="float" {first_input} />'
                '<input name="in2" type="float" value="0" /></add>'
            )
            previous = f"sum{index}"
        document = (
            '<materialx version="1.39"><oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />'
            '<surface name="surface" type="surfaceshader">'
            '<input name="bsdf" type="BSDF" nodename="diffuse" /></surface>'
            + "".join(nodes)
            + f'<output name="mask" type="float" nodename="{previous}" /></materialx>'
        )
        material = scene.create_material(
            f2.MaterialXMaterial,
            f2.Properties(
                {
                    "mtlx_buffer": document,
                    "mtlx_node_name": "surface",
                    "mtlx_editable_params": "*" if buffered else "",
                    "debug_write_shader_path": str(shader_path),
                }
            ),
        )
        _add_quad(scene, material)
        name = "mask"
        expected = [0.25, 0, 0, 0]

    scene.update()
    source = shader_path.read_text(encoding="utf-8")
    if backend == "mdl":
        writer = source.index("write_material_aovs_impl")
        assert source.index("if (!(Writer::generic_data_enabled", writer) < source.index(
            "var state = init_shading_state", writer
        )
    else:
        macros = dict(
            re.findall(
                r"^\s*#define (MATERIALX_PAYLOAD_SIZE|MATERIAL_PAYLOAD_LIMIT) (\d+)$",
                source,
                re.MULTILINE,
            )
        )
        assert (
            int(macros["MATERIALX_PAYLOAD_SIZE"]) > int(macros["MATERIAL_PAYLOAD_LIMIT"])
        ) == buffered
    instance = scene.components.find(type=f2.GeometryInstance)
    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "pathtracer/test_material_aovs.slang")
    module, requirements = helper.prepare(scene, source_module, material_aovs=(name,))
    probe = (
        module.probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    geometry_id = spy.Tensor.from_numpy(
        device, np.array([int(instance.geometry_instance_id)], dtype=np.uint32)
    )
    output_id = scene.material_aov_registry.find_output(name).id
    np.testing.assert_allclose(probe(geometry_id, output_id, True).to_numpy()[0], expected)
    np.testing.assert_array_equal(probe(geometry_id, output_id, False).to_numpy()[0], [-1] * 4)
    if backend == "materialx-buffered":
        assert material["inputs:sum0_in1"] == pytest.approx(0.25)
        material["inputs:sum0_in1"] = 0.75
        scene.update()
        np.testing.assert_allclose(
            probe(geometry_id, output_id, True).to_numpy()[0], [0.75, 0, 0, 0]
        )

    # Disable only AOV support; even an accepting writer must receive no writes.
    disabled_module, requirements = helper.prepare(
        scene,
        source_module,
        policy=f2.SceneCapabilityPolicy.specified,
        specified_capabilities=scene.required_capabilities.merged(
            f2.SceneCapabilities(f2.SceneCapabilityFlags.geometry_properties)
        ),
    )
    disabled_probe = (
        disabled_module.probe.as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    np.testing.assert_array_equal(
        disabled_probe(geometry_id, output_id, True).to_numpy()[0], [-1] * 4
    )


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_material_aov_default_renders_through_guide_outputs(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """The primary texture witness uses the build's default MaterialX backend."""
    _skip_materialx_on_cuda(device_type)
    _check_material_aov_guide_outputs(device, f2.ContainerSpec.auto(), layering_mode=None)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize(
    "output_spec,layering_mode",
    [
        pytest.param(f2.ContainerSpec.tensor(), None, id="default-tensor"),
    ],
)
def test_material_aov_renders_through_guide_outputs(
    device_type: spy.DeviceType,
    device: spy.Device,
    output_spec: f2.ContainerSpec,
    layering_mode: f2.MaterialXLayeringMode | None,
) -> None:
    _skip_materialx_on_cuda(device_type)
    _check_material_aov_guide_outputs(device, output_spec, layering_mode=layering_mode)


def _check_material_aov_guide_outputs(
    device: spy.Device,
    output_spec: f2.ContainerSpec,
    *,
    layering_mode: f2.MaterialXLayeringMode | None,
) -> None:
    scene, material, camera = _create_material_aov_scene(device, layering_mode=layering_mode)
    scene.update()
    updates = []
    connection = scene.updated.connect(lambda *args: updates.append(args))
    path_tracer = ReferencePathTracerNode.create(device)
    path_tracer.guide_output_specs = {
        "dust_mask_output": output_spec,
        "metallic": f2.ContainerSpec.auto(),
    }

    _, guides = path_tracer(scene, camera, iteration=0)

    dust_mask = guides["dust_mask_output"]
    assert isinstance(dust_mask, (spy.Texture, spy.Tensor))
    values = dust_mask.to_numpy()
    if isinstance(dust_mask, spy.Tensor):
        assert values.shape == (10, 12, 1)
        values = values[..., 0]
    assert values.shape == (10, 12)
    assert f2.Container.format(dust_mask) == spy.Format.r32_float
    covered = values > 0.0
    assert covered.any()
    np.testing.assert_allclose(values[covered], 0.35, atol=1e-6)
    assert (values == 0.0).any()
    assert guides["metallic"] is not None
    assert path_tracer._aov_output_info["dust_mask_output"][0] == (
        scene.material_aov_registry.find_output("dust_mask_output").id
    )
    previous = dust_mask
    output_info = path_tracer._aov_output_info
    indexed_outputs = path_tracer._indexed_outputs
    _, guides = path_tracer(scene, camera, iteration=1)
    assert guides["dust_mask_output"] is previous
    assert path_tracer._aov_output_info is output_info
    assert path_tracer._indexed_outputs is indexed_outputs

    # Explicit low-level targets determine the writes, not the previous dispatch's targets.
    path_tracer._make_write_guide_targets({"metallic": guides["metallic"]})
    assert path_tracer._indexed_outputs == ()
    path_tracer._make_write_guide_targets(guides)
    assert path_tracer._indexed_outputs[0].target is previous
    resized_guides = path_tracer._get_guide_outputs(6, 5, scene)
    path_tracer._make_write_guide_targets(resized_guides)
    assert path_tracer._indexed_outputs[0].target is not previous

    path_tracer.guide_output_specs = {"dust_mask_output": None}
    _, guides = path_tracer(scene, camera)
    assert guides["dust_mask_output"] is None
    assert path_tracer._guides["dust_mask_output"] is None
    assert not path_tracer._aov_output_info
    assert not updates
    material.mtlx_canonicalize_names = not material.mtlx_canonicalize_names
    scene.update()
    assert len(updates) == 1
    path_tracer.guide_output_specs = {"dust_mask_output": output_spec}
    _, guides = path_tracer(scene, camera)
    assert np.any(guides["dust_mask_output"].to_numpy() != 0)
    assert len(updates) == 1
    assert connection.connected


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", MATERIAL_AOV_DEVICE_TYPES)
@pytest.mark.nightly_if_not_vk
def test_mdl_material_aov_renders_from_custom_material_struct(
    device_type: spy.DeviceType,
    device: spy.Device,
    tmp_path: Path,
) -> None:
    shader_path = tmp_path / "mdl_aov.slang"
    # Exercise scene-data callbacks through the AOV entry point using the existing render case.
    (tmp_path / "scene_data_aov.mdl").write_text(
        """mdl 1.9;
import ::tutorials_aov::*;
import ::scene::*;
export tutorials_aov::aov_material green() = tutorials_aov::aov_material(
    example: scene::data_lookup_float3("missing_aov", float3(0, 1, 0)),
    scattering: tutorials_aov::green().scattering
);
""",
        encoding="utf-8",
    )
    scene, _, camera = _create_mdl_aov_scene(
        device,
        debug_write_shader_path=shader_path,
        material_name="::scene_data_aov::green",
        library_path=f"{tmp_path};{MDL_AOV_LIBRARY}",
    )
    scene.update()

    assert scene.material_aov_registry.has_output("example")
    path_tracer = ReferencePathTracerNode.create(device)
    path_tracer.guide_output_specs = {"example": f2.ContainerSpec.auto()}
    _, guides = path_tracer(scene, camera)

    values = guides["example"].to_numpy()
    covered = np.any(values != 0.0, axis=-1)
    assert covered.any()
    np.testing.assert_allclose(
        values[covered, :3], np.tile((0.0, 1.0, 0.0), (covered.sum(), 1)), atol=1e-6
    )
    np.testing.assert_allclose(values[covered, 3], 0.0, atol=1e-6)
    source = shader_path.read_text(encoding="utf-8")
    output_id = path_tracer._aov_output_info["example"][0]
    assert f"material_aov_{output_id}(state)" in source
    assert "material_aov_magic" not in source
    assert source.count("_ns::init(state);") == 1


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_material_aov_binding_lifecycle(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    registry = scene.material_aov_registry
    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {"mask": f2.ContainerSpec.auto()}
    assert registry.output_generation == 0
    with pytest.raises(RuntimeError, match="nor registered in the current scene"):
        node._get_guide_outputs(1, 1, scene)
    assert registry.outputs == []
    assert not node._aov_output_info

    registry.register_output("unrelated", spy.DataType.float32, 4)
    assert not node._aov_output_info
    output_id = registry.register_output("mask", spy.DataType.float32, 1)
    node._get_guide_outputs(1, 1, scene)
    assert node._aov_output_info == {"mask": (output_id, 1)}
    output_info = node._aov_output_info
    assert registry.output_generation == 2

    # Configuration itself is unrestricted; failed binding cannot commit a partial cache.
    node.guide_output_specs = {"mask": f2.ContainerSpec.auto(), "": f2.ContainerSpec.auto()}
    with pytest.raises(RuntimeError, match="nor registered in the current scene"):
        node._get_guide_outputs(1, 1, scene)
    assert node._aov_output_info is output_info
    node.guide_output_specs = {"mask": f2.ContainerSpec.auto()}
    node._get_guide_outputs(1, 1, scene)
    assert node._aov_output_info is output_info

    node.guide_output_specs["mask"] = None
    node._get_guide_outputs(1, 1, scene)
    assert not node._aov_output_info
    assert registry.output_generation == 2
    assert registry.has_output("mask")

    # Container lifetime follows in-place removals, without requiring a render.
    node.guide_output_specs = {"roughness": f2.ContainerSpec.auto()}
    node._get_guide_outputs(1, 1, scene)
    assert node._guides["roughness"] is not None
    assert "roughness" in node._guide_outputs_needing_clear
    node.guide_output_specs.pop("roughness")
    node._get_guide_outputs(1, 1, scene)
    assert "roughness" not in node._guides
    assert "roughness" not in node._guide_outputs_needing_clear

    # New declarations invalidate the fixed-guide collision check.
    registry.register_output("roughness", spy.DataType.float32, 1)
    node.guide_output_specs = {"roughness": f2.ContainerSpec.auto()}
    with pytest.raises(RuntimeError, match="both a fixed guide and a material-provided AOV"):
        node._get_guide_outputs(1, 1, scene)

    # Switching catalogs must discard bindings even if their revision is identical.
    other_scene = f2.Scene.create(device)
    other_registry = other_scene.material_aov_registry
    other_registry.register_output("mask", spy.DataType.float32, 3)
    other_registry.register_output("other", spy.DataType.float32, 1)
    other_registry.register_output("third", spy.DataType.float32, 1)
    assert other_registry.output_generation == registry.output_generation
    node.guide_output_specs = {"mask": f2.ContainerSpec.auto()}
    node._get_guide_outputs(1, 1, scene)
    node._get_guide_outputs(1, 1, other_scene)
    assert node._aov_output_info == {"mask": (0, 3)}
    assert f2.Container.format(node._guides["mask"]) == spy.Format.rgba32_float


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_pipeline_resolves_material_aov_after_normal_scene_update(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    _skip_materialx_on_cuda(device_type)
    scene, _, camera = _create_material_aov_scene(device)
    updates = []
    connection = scene.updated.connect(lambda *args: updates.append(args))
    pipeline = PathTracerPipeline.create(device)
    pipeline.spp = 2
    pipeline.tone_map = False
    pipeline.guide_output_specs = {"dust_mask_output": f2.ContainerSpec.auto()}

    _, guides = pipeline(scene, camera, output_guides=True)

    assert pipeline.accumulator._sample_count == 2
    dust_mask = guides["dust_mask_output"]
    assert isinstance(dust_mask, spy.Texture)
    values = dust_mask.to_numpy()
    covered = values > 0.0
    assert covered.any()
    np.testing.assert_allclose(values[covered], 0.35, atol=1e-6)
    registry = scene.material_aov_registry
    assert pipeline.path_tracer._aov_output_info["dust_mask_output"][0] == (
        registry.find_output("dust_mask_output").id
    )
    assert len(updates) == 1
    assert connection.connected
    pipeline.guide_output_specs = {}
    assert registry.has_output("dust_mask_output")


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_pipeline_preserves_arbitrary_guide_output_specs(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    """Material AOV specs survive normal forwarding and DLSS fixed-guide preparation."""
    from falcor2.rendernodes import PathTracerPipeline

    pipeline = PathTracerPipeline.create(device)
    dust_spec = f2.ContainerSpec.tensor()
    pipeline.guide_output_specs = {
        "depth": f2.ContainerSpec.auto(),
        "dust_mask_output": dust_spec,
    }

    assert pipeline.guide_output_specs["dust_mask_output"] is dust_spec
    assert pipeline.guide_output_specs["depth"] is not None
    dlss_specs = pipeline._merge_material_aov_guide_specs(
        {"normals": f2.ContainerSpec.texture2d(spy.Format.rgba16_float)}
    )
    assert dlss_specs["dust_mask_output"] is dust_spec
    assert "depth" not in dlss_specs
    assert "normals" in dlss_specs


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", [spy.DeviceType.d3d12])
@pytest.mark.parametrize(
    ("backend", "layering_mode"),
    [
        pytest.param(
            "materialx",
            None,
            id="materialx-default",
        ),
        pytest.param(
            "materialx",
            f2.MaterialXLayeringMode.closure_tree,
            id="materialx-closure_tree",
            marks=pytest.mark.nightly,
        ),
        pytest.param("mdl", None, id="mdl"),
    ],
)
def test_material_aov_marble_blue_veins_on_aswf_shader_ball(
    device_type: spy.DeviceType,
    backend: str,
    layering_mode: f2.MaterialXLayeringMode | None,
    image_test: ImageTest,
    tmp_path: Path,
) -> None:
    if device_type not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip(f"{device_type} is not available in this test environment")
    device = helpers.get_device(device_type)
    shader_path = tmp_path / "mdl_marble_aov.slang" if backend == "mdl" else None
    scene, camera, surface = _create_marble_shader_ball_scene(
        device,
        width=320,
        height=180,
        backend=backend,
        layering_mode=layering_mode,
        debug_write_shader_path=shader_path,
    )
    pipeline = PathTracerPipeline.create(device)
    pipeline.accumulation_mode = AccumulationMode.off
    pipeline.tone_map = False
    # Keep the uniform pixel footprint used by the material AOV reference images.
    pipeline.path_tracer.pixel_filter = f2.BoxReconstructionFilter(width=1)
    pipeline.guide_output_specs = {
        "material_color": f2.ContainerSpec.auto(),
        "geometry_id": f2.ContainerSpec.auto(),
        "blue_vein_mask": f2.ContainerSpec.auto(),
    }

    _, guides = pipeline(scene, camera, output_guides=True)

    material_color = guides["material_color"].to_numpy()[..., :3]
    geometry_id = guides["geometry_id"].to_numpy()
    mask = guides["blue_vein_mask"].to_numpy()
    surface_pixels = geometry_id[..., 1] == np.uint32(int(surface.geometry_instance_id))
    masked_surface = surface_pixels & (mask > 0.5)
    unmasked_surface = surface_pixels & (mask < 0.5)

    assert surface_pixels.any()
    assert masked_surface.any()
    assert unmasked_surface.any()
    assert (mask[~surface_pixels] == 0.0).all()
    assert set(np.unique(mask)) == {0.0, 1.0}
    blue_bias = material_color[..., 2] - material_color[..., 0]
    assert blue_bias[masked_surface].mean() > blue_bias[unmasked_surface].mean() + 0.05
    mask_rgb = np.repeat(mask[..., np.newaxis], 3, axis=-1)
    reference_variant = None
    if backend == "materialx" and layering_mode is None:
        material = surface.materials[0]
        assert isinstance(material, f2.MaterialXMaterial)
        reference_variant = f"default-mode-{material.mtlx_layering_mode.value}"
    image_test(
        np.concatenate((material_color, mask_rgb), axis=1),
        tolerance=1e-5,
        additional_name=reference_variant,
    )
    if shader_path is not None:
        source = shader_path.read_text(encoding="utf-8")
        output_id = pipeline.path_tracer._aov_output_info["blue_vein_mask"][0]
        assert f"material_aov_{output_id}(state)" in source
        assert source.count("_ns::init(state);") == 1


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
@pytest.mark.parametrize("backend", ["materialx", "mdl"])
def test_material_aov_generation_is_independent_of_consumers(
    device_type: spy.DeviceType,
    device: spy.Device,
    backend: str,
    tmp_path: Path,
) -> None:
    scene = f2.Scene.create(device)
    registry = scene.material_aov_registry
    assert registry.find_output("dust_mask_output" if backend == "materialx" else "example") is None
    shader_path = tmp_path / "material.slang"
    if backend == "materialx":
        _skip_materialx_on_cuda(device_type)
        scene, material, camera = _create_material_aov_scene(
            device,
            document=PROJECT_DIR / "tests/data/materialx/material_aov_independent_output.mtlx",
            node_name="surface_output",
            uv1=True,
            scene=scene,
            debug_write_shader_path=shader_path,
        )
        name = "dust_mask_output"
    else:
        scene, material, camera = _create_mdl_aov_scene(
            device,
            material_name="::tutorials_aov::scaled",
            scene=scene,
            debug_write_shader_path=shader_path,
        )
        material["scale"] = 0.7
        name = "example"

    registry = scene.material_aov_registry
    scene.update()
    initial_module = material.required_module()
    initial_generation = scene.requirements_generation
    output_id = registry.find_output(name).id
    initial_catalog_generation = registry.output_generation
    source = shader_path.read_text(encoding="utf-8")
    assert "write_material_aovs_impl" in source
    if backend == "mdl":
        assert f"material_aov_{output_id}(state)" in source
    # Updating a regenerated material rewrites this diagnostic file, even when its
    # Slang module cache returns the same module. Consumer edits must leave it alone.
    shader_path.write_text("no material update", encoding="utf-8")
    textures = material.build_texture_list()
    if backend == "materialx":
        # The texture is used only by the AOV and is ready before any consumer exists.
        assert len(textures) == 1
        assert all(texture.is_finalized() for texture in textures)
    else:
        assert material["scale"] == pytest.approx(0.7)
    path_tracer = ReferencePathTracerNode.create(device)
    path_tracer.scene_capability_policy = f2.SceneCapabilityPolicy.required

    for enabled in (True, False, True):
        path_tracer.guide_output_specs = {name: f2.ContainerSpec.auto() if enabled else None}
        scene.update()
        _, guides = path_tracer(scene, camera)
        assert shader_path.read_text(encoding="utf-8") == "no material update"
        assert material.required_module() is initial_module
        assert scene.requirements_generation == initial_generation
        assert material.build_texture_list() == textures
        assert registry.find_output(name).id == output_id
        assert registry.output_generation == initial_catalog_generation
        assert path_tracer.capabilities.has(f2.SceneCapabilityFlags.material_aovs) == enabled
        if enabled:
            assert path_tracer.capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
            assert np.any(guides[name].to_numpy() != 0)
        if backend == "mdl":
            assert material["scale"] == pytest.approx(0.7)

    # Removing the producer leaves the declaration attachable; unwritten outputs clear to zero.
    instance = scene.components.find(type=f2.GeometryInstance)
    instance.materials = [scene.create_material(f2.StandardMaterial)]
    material.remove()
    scene.update()
    assert registry.find_output(name).id == output_id
    _, guides = path_tracer(scene, camera)
    assert np.all(guides[name].to_numpy() == 0)
    assert registry.output_generation == initial_catalog_generation
    # Even an output without a remaining producer conservatively retains geometry access.
    assert scene.required_capabilities == f2.SceneCapabilities()
    assert path_tracer.capabilities.has(f2.SceneCapabilityFlags.geometry_properties)
    assert path_tracer.capabilities.has(f2.SceneCapabilityFlags.material_aovs)
    path_tracer.scene_capabilities = f2.SceneCapabilities()
    with pytest.raises(RuntimeError, match="omits required capabilities"):
        path_tracer(scene, camera)
    path_tracer.scene_capability_policy = f2.SceneCapabilityPolicy.required
    replacement = scene.create_material(type(material), material.properties)
    instance.materials = [replacement]
    scene.update()
    assert registry.find_output(name).id == output_id
    assert registry.output_generation == initial_catalog_generation
    _, guides = path_tracer(scene, camera)
    assert np.any(guides[name].to_numpy() != 0)


@pytest.mark.nightly
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_unused_aov_guide_binding_stays_cheap(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """Count catalog access locally; no timing benchmark or production profiling hooks."""
    registry = f2.Scene.create(device).material_aov_registry
    accesses = []

    class Catalog:
        @property
        def output_generation(self) -> int:
            accesses.append("revision")
            return registry.output_generation

        def has_output(self, name: str) -> bool:
            accesses.append(("has", name))
            return registry.has_output(name)

        def find_output(self, name: str) -> f2.AOVDesc | None:
            accesses.append(("find", name))
            return registry.find_output(name)

    class Scene:
        @property
        def material_aov_registry(self) -> Catalog:
            accesses.append("registry")
            return catalog

    catalog = Catalog()
    scene = Scene()
    node = ReferencePathTracerNode.create(device)
    node.guide_output_specs = {}
    assert not accesses
    for _ in range(2):
        guides = node._get_guide_outputs(1, 1, scene)
        node._make_write_guide_targets(guides)
    assert not accesses
    node.guide_output_specs = {"roughness": f2.ContainerSpec.auto()}
    for _ in range(2):
        guides = node._get_guide_outputs(1, 1, scene)
        node._make_write_guide_targets(guides)
    assert accesses == ["registry", "revision"] * 2  # Empty catalog needs no name lookup.
    assert node._indexed_outputs == ()

    registry.register_output("mask", spy.DataType.float32, 1)
    accesses.clear()
    for _ in range(2):
        node._get_guide_outputs(1, 1, scene)
    assert accesses == ["registry", "revision", ("has", "roughness"), "registry", "revision"]
    node.guide_output_specs["mask"] = f2.ContainerSpec.auto()
    guides = node._get_guide_outputs(1, 1, scene)
    node._make_write_guide_targets(guides)
    info, indexed = node._aov_output_info, node._indexed_outputs
    registry.register_output("unrelated", spy.DataType.float32, 2)
    accesses.clear()
    for _ in range(2):
        guides = node._get_guide_outputs(1, 1, scene)
        node._make_write_guide_targets(guides)
    assert accesses == ["registry", "revision", ("has", "roughness"), "registry", "revision"]
    assert node._aov_output_info is info
    assert node._indexed_outputs is indexed
    del node.guide_output_specs["mask"]
    guides = node._get_guide_outputs(1, 1, scene)
    node._make_write_guide_targets(guides)
    assert not node._aov_output_info and not node._indexed_outputs
    assert "mask" not in node._guides
    node.guide_output_specs.clear()
    accesses.clear()
    node._get_guide_outputs(1, 1, scene)
    assert not accesses
