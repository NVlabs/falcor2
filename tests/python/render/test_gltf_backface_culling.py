# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
import falcor2.ui as ui
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import ReferencePathTracerNode


def _write_exact_opposite_fixture(
    tmp_path: Path,
    *,
    primitive_order: tuple[str, ...],
    front_z: float = 0.0,
    back_double_sided: bool = False,
    front_double_sided: bool = False,
    back_transmission: bool = False,
    same_material: bool = False,
    single_mesh: bool = False,
    unreferenced_back: bool = False,
    isolated_back_copy_x: float | None = None,
    skinned_back_copy_x: float | None = None,
) -> Path:
    """Write the coincident opposite-winding pattern found in TexVerse assets."""
    positions = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [-1.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype="<f4",
    )
    back_positions = positions.copy()
    front_positions = positions.copy()
    front_positions[:, 0] += 2.0
    front_positions[:, 2] = front_z
    back_normals = np.tile(np.array([0.0, 0.0, 1.0], dtype="<f4"), (4, 1))
    front_normals = np.tile(np.array([0.0, 0.0, -1.0], dtype="<f4"), (4, 1))
    back_indices = np.array([0, 1, 2, 2, 1, 3], dtype="<u2")
    front_indices = np.array([0, 2, 1, 2, 3, 1], dtype="<u2")

    chunks = (
        back_positions,
        back_normals,
        back_indices,
        front_positions,
        front_normals,
        front_indices,
    )
    offsets: list[int] = []
    buffer_data = bytearray()
    for chunk in chunks:
        offsets.append(len(buffer_data))
        buffer_data.extend(chunk.tobytes())

    suffix = (
        f"{'-'.join(primitive_order)}-{front_z:g}-{back_double_sided}-{front_double_sided}-"
        f"{back_transmission}-{same_material}-{single_mesh}-{unreferenced_back}-"
        f"{isolated_back_copy_x}-{skinned_back_copy_x}"
    ).replace(".", "_")
    buffer_path = tmp_path / f"exact_opposite-{suffix}.bin"
    buffer_path.write_bytes(buffer_data)
    scene_path = tmp_path / f"exact_opposite-{suffix}.gltf"

    primitives = {
        "back": {"attributes": {"POSITION": 0, "NORMAL": 1}, "indices": 2, "material": 0},
        "front": {
            "attributes": {"POSITION": 3, "NORMAL": 4},
            "indices": 5,
            "material": 0 if same_material else 1,
        },
    }
    if single_mesh:
        nodes = [{"name": "combined", "mesh": 0}]
    else:
        nodes = [
            {
                "name": name,
                "mesh": index,
                **({"translation": [-2.0, 0.0, 0.0]} if name == "front" else {}),
            }
            for index, name in enumerate(primitive_order)
        ]
    if isolated_back_copy_x is not None:
        nodes.append(
            {
                "name": "isolated_back",
                "mesh": primitive_order.index("back"),
                "translation": [isolated_back_copy_x, 0.0, 0.0],
            }
        )

    skins: list[dict[str, Any]] = []
    if skinned_back_copy_x is not None:
        joint_node_index = len(nodes) + 1
        nodes.extend(
            [
                {
                    "name": "skinned_back",
                    "mesh": primitive_order.index("back"),
                    "skin": 0,
                    "translation": [skinned_back_copy_x, 0.0, 0.0],
                },
                {"name": "joint"},
            ]
        )
        skins.append({"name": "skin", "joints": [joint_node_index]})

    scene_path.write_text(
        json.dumps(
            {
                "asset": {"version": "2.0"},
                "buffers": [{"uri": buffer_path.name, "byteLength": len(buffer_data)}],
                "bufferViews": [
                    {
                        "buffer": 0,
                        "byteOffset": offset,
                        "byteLength": chunk.nbytes,
                        "target": (
                            34963 if chunk is back_indices or chunk is front_indices else 34962
                        ),
                    }
                    for offset, chunk in zip(offsets, chunks, strict=True)
                ],
                "accessors": [
                    {
                        "bufferView": 0,
                        "componentType": 5126,
                        "count": 4,
                        "type": "VEC3",
                        "min": [-1.0, -1.0, 0.0],
                        "max": [1.0, 1.0, 0.0],
                    },
                    {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
                    {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"},
                    {
                        "bufferView": 3,
                        "componentType": 5126,
                        "count": 4,
                        "type": "VEC3",
                        "min": front_positions.min(axis=0).tolist(),
                        "max": front_positions.max(axis=0).tolist(),
                    },
                    {"bufferView": 4, "componentType": 5126, "count": 4, "type": "VEC3"},
                    {"bufferView": 5, "componentType": 5123, "count": 6, "type": "SCALAR"},
                ],
                "materials": [
                    {
                        "name": "Back",
                        "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 0.0, 1.0]},
                        "emissiveFactor": [1.0, 1.0, 1.0],
                        "alphaMode": "OPAQUE",
                        "doubleSided": back_double_sided,
                        **(
                            {
                                "extensions": {
                                    "KHR_materials_transmission": {"transmissionFactor": 0.0}
                                }
                            }
                            if back_transmission
                            else {}
                        ),
                    },
                    {
                        "name": "Front",
                        "pbrMetallicRoughness": {"baseColorFactor": [0.0, 0.0, 0.0, 1.0]},
                        "emissiveFactor": [1.0, 0.0, 0.0],
                        "alphaMode": "OPAQUE",
                        "doubleSided": front_double_sided,
                    },
                ],
                "meshes": (
                    [
                        {
                            "name": "combined",
                            "primitives": [primitives[name] for name in primitive_order],
                        }
                    ]
                    if single_mesh
                    else [
                        {"name": name, "primitives": [primitives[name]]} for name in primitive_order
                    ]
                )
                + (
                    [{"name": "unreferenced", "primitives": [primitives["back"]]}]
                    if unreferenced_back
                    else []
                ),
                "nodes": nodes,
                **({"extensionsUsed": ["KHR_materials_transmission"]} if back_transmission else {}),
                **({"skins": skins} if skins else {}),
                "scenes": [{"nodes": list(range(len(nodes)))}],
                "scene": 0,
            }
        ),
        encoding="utf-8",
    )
    return scene_path


def _create_primary_emission_renderer(device: spy.Device) -> ReferencePathTracerNode:
    renderer = ReferencePathTracerNode.create(device)
    renderer.output_spec = f2.ContainerSpec.texture2d(spy.Format.rgba32_float)
    renderer.enable_nee = False
    renderer.enable_mis = False
    renderer.enable_russian_roulette = False
    renderer.max_depth = 1
    renderer.use_background_color = True
    renderer.background_color = spy.float3(0.0)
    return renderer


def _render_gltf_fixture_center(
    device: spy.Device,
    renderer: ReferencePathTracerNode,
    scene_path: Path,
) -> tuple[f2.Scene, np.ndarray]:
    scene = f2.Scene.load(device, scene_path)
    # Make scattering two-sided explicitly, independently of imported culling eligibility.
    for material in scene.materials:
        material.double_sided = True
    scene.update()
    camera = helpers.create_test_camera(
        scene,
        width=8,
        height=8,
        fov_y=35.0,
        position=spy.float3(0.0, 0.0, -2.0),
        rotation=spy.math.quat_from_look_at(
            spy.float3(0.0, 0.0, 1.0),
            spy.float3(0.0, 1.0, 0.0),
        ),
    )
    return scene, _render_scene_center(scene, renderer, camera)


def _render_scene_center(
    scene: f2.Scene,
    renderer: ReferencePathTracerNode,
    camera: f2.Camera,
) -> np.ndarray:
    renderer.pixel_filter = f2.BoxReconstructionFilter(width=0)
    image, _ = renderer(
        scene,
        camera,
        iteration=0,
        pixel_offset=spy.float2(0.0),
    )
    return image.to_numpy()[4, 4, :3]


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_gltf_triangle_backface_culling_is_opt_in_and_respects_thin_sidedness(
    tmp_path: Path,
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    programmatic_scene = f2.Scene.create(device)
    programmatic_material = programmatic_scene.create_material(f2.StandardMaterial)
    assert programmatic_material.enable_backface_culling is False

    renderer = _create_primary_emission_renderer(device)
    assert renderer.enable_triangle_backface_culling is False

    def render(
        *,
        enabled: bool,
        back_double_sided: bool = False,
        back_transmission: bool = False,
        primitive_order: tuple[str, ...] = ("back",),
        front_double_sided: bool = False,
        single_mesh: bool = False,
    ) -> tuple[f2.Scene, np.ndarray]:
        renderer.enable_triangle_backface_culling = enabled
        return _render_gltf_fixture_center(
            device,
            renderer,
            _write_exact_opposite_fixture(
                tmp_path,
                primitive_order=primitive_order,
                back_double_sided=back_double_sided,
                back_transmission=back_transmission,
                front_double_sided=front_double_sided,
                single_mesh=single_mesh,
            ),
        )

    default_scene, default_pixel = render(enabled=False)
    default_material = default_scene.materials.find("Back")
    assert default_material is not None
    assert default_material.enable_backface_culling is True
    np.testing.assert_allclose(default_pixel, [1.0, 1.0, 1.0], rtol=0.0, atol=1e-4)

    _, culled_pixel = render(enabled=True)
    np.testing.assert_allclose(culled_pixel, [0.0, 0.0, 0.0], rtol=0.0, atol=1e-4)

    double_sided_scene, double_sided_pixel = render(enabled=True, back_double_sided=True)
    double_sided_material = double_sided_scene.materials.find("Back")
    assert double_sided_material is not None
    assert double_sided_material.enable_backface_culling is False
    np.testing.assert_allclose(double_sided_pixel, [1.0, 1.0, 1.0], rtol=0.0, atol=1e-4)

    transmission_scene, transmission_pixel = render(enabled=True, back_transmission=True)
    transmission_material = transmission_scene.materials.find("Back")
    assert transmission_material is not None
    assert transmission_material.enable_backface_culling is True
    np.testing.assert_allclose(transmission_pixel, [0.0, 0.0, 0.0], rtol=0.0, atol=1e-4)
    _, two_sided_transmission_pixel = render(
        enabled=True, back_transmission=True, back_double_sided=True
    )
    np.testing.assert_allclose(two_sided_transmission_pixel, [1.0, 1.0, 1.0], rtol=0.0, atol=1e-4)

    mixed_scene, mixed_pixel = render(
        enabled=True,
        primitive_order=("back", "front"),
        front_double_sided=True,
        single_mesh=True,
    )
    assert len(mixed_scene.components.find_all(type=f2.GeometryInstance)) == 1
    np.testing.assert_allclose(mixed_pixel, [1.0, 1.0, 1.0], rtol=0.0, atol=1e-4)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("enabled", [False, True])
def test_gltf_backface_culling_keeps_picking_two_sided(
    tmp_path: Path,
    device: spy.Device,
    enabled: bool,
) -> None:
    scene_path = _write_exact_opposite_fixture(
        tmp_path, primitive_order=("back", "front"), front_z=0.25
    )
    renderer = _create_primary_emission_renderer(device)
    renderer.enable_triangle_backface_culling = enabled
    scene, pixel = _render_gltf_fixture_center(device, renderer, scene_path)
    np.testing.assert_allclose(
        pixel, [1.0, 0.0, 0.0] if enabled else [1.0, 1.0, 1.0], rtol=0.0, atol=1e-4
    )

    camera = scene.components.find_all(type=f2.Camera)[0]
    picker = ui.ScenePicker(device)
    command_encoder = device.create_command_encoder()
    picker.render(command_encoder, scene, camera)
    device.submit_command_buffer(command_encoder.finish())
    picked = picker.pick_entity(scene, spy.uint2(4, 4))
    assert picked is not None
    assert picked.name == "back"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_renderer_backface_culling_toggles_without_scene_updates(
    tmp_path: Path, device: spy.Device
) -> None:
    renderer = _create_primary_emission_renderer(device)
    two_sided_renderer = _create_primary_emission_renderer(device)
    scene, _ = _render_gltf_fixture_center(
        device, renderer, _write_exact_opposite_fixture(tmp_path, primitive_order=("back",))
    )
    camera = scene.components.find_all(type=f2.Camera)[0]
    instance = scene.components.find_all(type=f2.GeometryInstance)[0]
    scene.update()
    instance_id = instance.geometry_instance_id
    for enabled in (True, False, True):
        renderer.enable_triangle_backface_culling = enabled
        assert not scene.update() & f2.SceneUpdateFlags.render_state
        np.testing.assert_allclose(
            _render_scene_center(scene, renderer, camera),
            [0.0, 0.0, 0.0] if enabled else [1.0, 1.0, 1.0],
            atol=1e-4,
        )
        np.testing.assert_allclose(
            _render_scene_center(scene, two_sided_renderer, camera), [1.0, 1.0, 1.0], atol=1e-4
        )
        assert instance.geometry_instance_id == instance_id


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("enabled", [False, True])
def test_runtime_backface_culling_updates_shared_instances(
    tmp_path: Path, device: spy.Device, enabled: bool
) -> None:
    renderer = _create_primary_emission_renderer(device)
    renderer.enable_triangle_backface_culling = enabled
    scene, _ = _render_gltf_fixture_center(
        device,
        renderer,
        _write_exact_opposite_fixture(
            tmp_path, primitive_order=("back",), isolated_back_copy_x=4.0
        ),
    )
    cameras = [scene.components.find_all(type=f2.Camera)[0]]
    cameras.append(
        helpers.create_test_camera(
            scene,
            width=8,
            height=8,
            fov_y=35.0,
            position=spy.float3(4.0, 0.0, -2.0),
            rotation=spy.math.quat_from_look_at(
                spy.float3(0.0, 0.0, 1.0), spy.float3(0.0, 1.0, 0.0)
            ),
        )
    )
    scene.update()
    instances = scene.components.find_all(type=f2.GeometryInstance)
    assert len(instances) == 2
    instance_ids = [instance.geometry_instance_id for instance in instances]
    material = scene.materials.find("Back")

    for culling in (False, True, False):
        material.enable_backface_culling = culling
        flags = scene.update()
        assert flags & f2.SceneUpdateFlags.render_state
        assert [instance.geometry_instance_id for instance in instances] == instance_ids
        expected = [0.0, 0.0, 0.0] if enabled and culling else [1.0, 1.0, 1.0]
        for camera in cameras:
            np.testing.assert_allclose(
                _render_scene_center(scene, renderer, camera), expected, atol=1e-4
            )

    # An unused material's policy change must not invalidate any render instance.
    scene.materials.find("Front").enable_backface_culling = False
    assert not scene.update() & f2.SceneUpdateFlags.render_state


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_runtime_backface_culling_keeps_mixed_instance_conservative(
    tmp_path: Path, device: spy.Device
) -> None:
    renderer = _create_primary_emission_renderer(device)
    renderer.enable_triangle_backface_culling = True
    scene, pixel = _render_gltf_fixture_center(
        device,
        renderer,
        _write_exact_opposite_fixture(
            tmp_path, primitive_order=("back", "front"), single_mesh=True, front_double_sided=True
        ),
    )
    np.testing.assert_allclose(pixel, [1.0, 1.0, 1.0], atol=1e-4)
    camera = scene.components.find_all(type=f2.Camera)[0]
    instance = scene.components.find_all(type=f2.GeometryInstance)[0]
    instance_id = instance.geometry_instance_id
    front = scene.materials.find("Front")
    back = scene.materials.find("Back")
    for culling in (True, False):
        front.enable_backface_culling = culling
        assert scene.update() & f2.SceneUpdateFlags.render_state
        assert instance.geometry_instance_id == instance_id
        np.testing.assert_allclose(
            _render_scene_center(scene, renderer, camera),
            [0.0, 0.0, 0.0] if culling else [1.0, 1.0, 1.0],
            atol=1e-4,
        )
    back.enable_backface_culling = False
    assert not scene.update() & f2.SceneUpdateFlags.render_state
    np.testing.assert_allclose(
        _render_scene_center(scene, renderer, camera), [1.0, 1.0, 1.0], atol=1e-4
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_runtime_backface_culling_handles_material_assignment_and_removal(
    tmp_path: Path, device: spy.Device
) -> None:
    renderer = _create_primary_emission_renderer(device)
    renderer.enable_triangle_backface_culling = True
    scene, _ = _render_gltf_fixture_center(
        device,
        renderer,
        _write_exact_opposite_fixture(tmp_path, primitive_order=("back",)),
    )
    camera = scene.components.find_all(type=f2.Camera)[0]
    instance = scene.components.find_all(type=f2.GeometryInstance)[0]
    back = scene.materials.find("Back")
    front = scene.materials.find("Front")
    front.enable_backface_culling = False
    for material, expected in ((front, [1.0, 0.0, 0.0]), (back, [0.0, 0.0, 0.0])):
        instance.materials = [material]
        scene.update()
        np.testing.assert_allclose(
            _render_scene_center(scene, renderer, camera), expected, atol=1e-4
        )

    # Probe traversal directly because missing materials are rejected by the path tracer's shading.
    helper = SceneShaderHelper(device)
    source = spy.Module(
        device.load_module("render/test_gltf_backface_culling.slang"),
        link=[spy.Module.load_from_file(device, "falcor2/ui/kernels/scene_picker.slang")],
    )
    result = spy.Tensor.from_numpy(device, np.zeros(1, dtype=np.uint32))

    def is_visible() -> bool:
        module, requirements = helper.prepare(scene, source)
        ray_desc = f2.SceneRayTracingSetup.RayDesc()
        ray_desc.name = "intersect"
        ray_desc.has_miss = True
        ray_desc.has_closest_hit = True
        setup = f2.SceneRayTracingSetup.create(scene, [ray_desc])
        probe = (
            module.probe_backface.constants({"ENABLE_TRIANGLE_BACKFACE_CULLING": True})
            .type_conformances(requirements.type_conformances)
            .write(helper.bind_scene)
            .ray_tracing(
                hit_groups=setup.hit_groups,
                hit_group_names=setup.sbt_hit_group_names,
                miss_entry_points=setup.sbt_miss_entry_points,
                max_recursion=1,
                max_ray_payload_size=128,
                flags=setup.pipeline_flags,
            )
        )
        probe(_result=result)
        return bool(result.to_numpy()[0])

    assert not is_visible()
    for materials, visible in (
        ([back, back], False),
        ([back, front], True),
        ([], False),
        ([None], True),
        ([back], False),
    ):
        instance.materials = materials
        scene.update()
        assert is_visible() == visible, materials
    back.remove()
    scene.update()
    assert is_visible()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "material_type", ["gltf_pbrMetallicRoughness", "gltf_pbrSpecularGlossiness"]
)
@pytest.mark.parametrize(
    "double_sided,has_transmission,expected_culling",
    [
        (False, False, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
    ],
)
def test_gltf_material_promotion_sets_backface_culling(
    device: spy.Device,
    material_type: str,
    double_sided: bool,
    has_transmission: bool,
    expected_culling: bool,
) -> None:
    imported = f2.ImporterScene()
    material = f2.ImporterMaterial()
    material.name = "Surface"
    material.params = f2.Properties(
        {
            "_type": material_type,
            "doubleSided": double_sided,
            "hasTransmission": has_transmission,
        }
    )
    imported.materials = [material]

    scene = f2.Scene.from_importer_scene(device, imported)
    if material_type == "gltf_pbrMetallicRoughness":
        # Native MR is thin; a transmission declaration does not override sidedness.
        expected_culling = not double_sided
    assert scene.materials.find("Surface").enable_backface_culling == expected_culling


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("enable_backface_culling", [False, True])
def test_explicit_runtime_material_keeps_its_backface_culling_policy(
    device: spy.Device,
    enable_backface_culling: bool,
) -> None:
    imported = f2.ImporterScene()
    material = f2.ImporterMaterial()
    material.name = "Surface"
    material.params = f2.Properties(
        {
            "_scene_material_type": "StandardMaterial",
            "_type": "gltf_pbrMetallicRoughness",
            "doubleSided": enable_backface_culling,
            "enable_backface_culling": enable_backface_culling,
        }
    )
    imported.materials = [material]

    scene = f2.Scene.from_importer_scene(device, imported)
    assert scene.materials.find("Surface").enable_backface_culling == enable_backface_culling
