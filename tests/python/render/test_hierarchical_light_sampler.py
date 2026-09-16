# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.editor import SceneShaderHelper
from falcor2.rendernodes import ReferencePathTracerNode


def _add_emissive_quad(scene: f2.Scene) -> tuple[f2.StandardMaterial, f2.Entity]:
    material = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"emissive_factor": spy.float3(1.0, 0.5, 0.25)}),
    )
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    geometry.set_mesh_data(
        positions=np.array(
            [
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        sub_mesh_indices=[np.array([[0, 1, 2], [2, 1, 3]], dtype=np.uint32)],
        normals=np.tile(np.array([0.0, 0.0, 1.0], dtype=np.float32), (4, 1)),
        name="quad",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    return material, entity


def _add_component_lights(scene: f2.Scene) -> tuple[f2.PointLight, f2.ConstantLight]:
    point_entity = scene.create_entity()
    point_light = point_entity.create_component(f2.PointLight)
    point_light.intensity = spy.float3(1.0)

    environment_entity = scene.create_entity()
    environment_light = environment_entity.create_component(f2.ConstantLight)
    environment_light.radiance = spy.float3(2.0)
    return point_light, environment_light


def _add_emissive_triangle(
    scene: f2.Scene,
    positions: np.ndarray,
    emissive_factor: spy.float3,
) -> tuple[f2.StandardMaterial, f2.Entity]:
    material = scene.create_material(
        f2.StandardMaterial,
        f2.Properties({"emissive_factor": emissive_factor}),
    )
    geometry = scene.create_geometry(f2.StaticMeshGeometry)
    edge0 = positions[1] - positions[0]
    edge1 = positions[2] - positions[0]
    normal = np.cross(edge0, edge1)
    normal_length = np.linalg.norm(normal)
    normal = normal / normal_length if normal_length > 0.0 else np.array([0.0, 0.0, 1.0])
    geometry.set_mesh_data(
        positions=positions.astype(np.float32),
        sub_mesh_indices=[np.array([[0, 1, 2]], dtype=np.uint32)],
        normals=np.tile(normal.astype(np.float32), (3, 1)),
        name="triangle",
    )
    entity = scene.create_entity()
    instance = entity.create_component(f2.GeometryInstance)
    instance.geometry = geometry
    instance.materials = [material]
    return material, entity


def _create_hierarchical_shader_function(
    module: Any,
    helper: SceneShaderHelper,
    scene: f2.Scene,
    sampler: f2.HierarchicalLightSampler,
    name: str,
) -> Any:
    def bind_sampler(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["hierarchical_light_sampler"] = sampler

    return (
        getattr(module, name)
        .as_func()
        .prelude(sampler.shader_specialization_source)
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_sampler)
        .return_type(spy.Tensor)
    )


def _add_emissive_triangle_specs(
    scene: f2.Scene,
    triangle_specs: list[tuple[np.ndarray, float]],
) -> None:
    for positions, intensity in triangle_specs:
        _add_emissive_triangle(scene, positions, spy.float3(intensity))


def _coincident_emissive_triangle_specs() -> list[tuple[np.ndarray, float]]:
    lower_left = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]])
    upper_right = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 0.0], [1.0, -1.0, 0.0]])
    return [
        (lower_left, 1.0),
        (lower_left[[0, 2, 1]], 2.0),
        (upper_right, 4.0),
        (upper_right[[0, 2, 1]], 8.0),
    ]


def _spatial_emissive_triangle_specs() -> list[tuple[np.ndarray, float]]:
    base = np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [-0.5, 0.5, 0.0]])
    specs = [
        (base + np.array([x, 0.25 * x, 0.0]), intensity)
        for x, intensity in zip(
            (-6.0, -3.0, -1.0, 1.0, 3.0, 6.0),
            (1e-3, 1.0, 1e3, 4.0, 16.0, 64.0),
            strict=True,
        )
    ]
    specs[4] = (specs[4][0][[0, 2, 1]], specs[4][1])
    specs.append((base + np.array([9.0, 0.0, 0.0]), 0.0))
    return specs


def _asymmetric_emissive_triangle_specs() -> list[tuple[np.ndarray, float]]:
    base = np.array([[-0.25, -0.25, 0.0], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]])
    specs = []
    for exponent in range(12):
        x = float((1 << exponent) - 1)
        positions = base + np.array([x, 0.125 * float(exponent % 3), 0.0])
        specs.append((positions, x * x + 16.0))
    return specs


def test_hierarchical_sampler_exposes_supported_options() -> None:
    sampler = f2.HierarchicalLightSampler()
    initial_shader_generation = sampler.shader_generation
    assert sampler.split_heuristic == f2.EmissiveTriangleTreeSplitHeuristic.binned_sah
    assert sampler.leaf_sampling_mode == f2.EmissiveTriangleTreeLeafSamplingMode.uniform
    assert "EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE = 0" in sampler.shader_specialization_source
    assert f2.HierarchicalLightSampler().shader_generation != initial_shader_generation

    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.equal
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    contextual_shader_generation = sampler.shader_generation
    assert contextual_shader_generation != initial_shader_generation
    assert "EMISSIVE_TRIANGLE_TREE_LEAF_SAMPLING_MODE = 1" in sampler.shader_specialization_source
    assert sampler.split_heuristic == f2.EmissiveTriangleTreeSplitHeuristic.equal
    assert (
        sampler.leaf_sampling_mode == f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    )
    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh
    assert sampler.split_heuristic == f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    assert sampler.shader_generation == contextual_shader_generation

    for removed_property in (
        "use_bounding_cone",
        "use_lighting_cone",
        "disable_node_flux",
        "use_uniform_triangle_sampling",
    ):
        assert not hasattr(sampler, removed_property)


def test_hierarchical_sampler_clamps_build_options() -> None:
    sampler = f2.HierarchicalLightSampler()
    sampler.max_triangle_count_per_leaf = 0
    assert sampler.max_triangle_count_per_leaf == 1
    sampler.max_triangle_count_per_leaf = 65
    assert sampler.max_triangle_count_per_leaf == 64

    sampler.bin_count = 1
    assert sampler.bin_count == 2
    sampler.bin_count = 129
    assert sampler.bin_count == 128


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_option_invalidation(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_quad(scene)
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1

    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    assert not sampler.update(scene)
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    assert not sampler.update(scene)
    sampler.allow_refitting = False
    assert not sampler.update(scene)
    sampler.allow_refitting = False
    assert not sampler.update(scene)

    sampler.bin_count = 32
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2
    sampler.bin_count = 32
    assert not sampler.update(scene)

    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.equal
    assert sampler.update(scene)
    assert sampler.rebuild_count == 3
    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.equal
    assert not sampler.update(scene)

    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh
    assert sampler.update(scene)
    assert sampler.rebuild_count == 4
    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh
    assert not sampler.update(scene)

    sampler.max_triangle_count_per_leaf = 1
    assert sampler.update(scene)
    assert sampler.rebuild_count == 5
    sampler.max_triangle_count_per_leaf = 1
    assert not sampler.update(scene)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_rebuilds_refits_and_updates_component_lights(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material, entity = _add_emissive_quad(scene)
    sampler = f2.HierarchicalLightSampler()
    sampler.max_triangle_count_per_leaf = 1

    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 0
    assert sampler.node_count == 3
    assert sampler.leaf_count == 2
    assert sampler.max_depth == 1
    assert not sampler.update(scene)

    transform = f2.Transform()
    transform.translation = spy.float3(2.0, 0.0, 0.0)
    entity.transform = transform
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 1

    material.emissive_factor = spy.float3(2.0, 1.0, 0.5)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 2

    point_light, environment_light = _add_component_lights(scene)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 2

    point_light.intensity = spy.float3(4.0)
    environment_light.radiance = spy.float3(8.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 2

    sampler.max_triangle_count_per_leaf = 2
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2
    assert sampler.refit_count == 2
    assert sampler.node_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_rebuilds_only_when_active_triangle_mapping_changes(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    material, _ = _add_emissive_quad(scene)
    sampler = f2.HierarchicalLightSampler()

    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 0

    material.emissive_factor = spy.float3(2.0, 1.0, 0.5)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 1
    assert sampler.refit_count == 1

    material.emissive_factor = spy.float3(0.0)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 2
    assert sampler.refit_count == 1
    assert sampler.node_count == 0

    material.emissive_factor = spy.float3(1.0, 0.5, 0.25)
    scene.update()
    assert sampler.update(scene)
    assert sampler.rebuild_count == 3
    assert sampler.refit_count == 1
    assert sampler.node_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reference_pathtracer_exposes_hierarchical_sampler(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    node = ReferencePathTracerNode.create(device)
    sampler = f2.HierarchicalLightSampler()
    node.light_sampler = sampler
    assert node.light_sampler is sampler
    assert node.light_sampler.slang_type_name == "HierarchicalLightSampler"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_evaluates_complete_scene_pdfs(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_quad(scene)
    _add_component_lights(scene)
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)

    def bind_sampler(cursor: Any) -> None:
        helper.bind_scene(cursor)
        cursor["hierarchical_light_sampler"] = sampler

    environment_pdf = (
        module.hierarchical_environment_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_sampler)
        .return_type(spy.Tensor)
    )
    directions = spy.Tensor.from_numpy(
        device,
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        environment_pdf(directions).to_numpy(),
        np.array([1.0 / (8.0 * np.pi)], dtype=np.float32),
        rtol=1e-6,
    )

    emissive_pdf = (
        module.hierarchical_emissive_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(bind_sampler)
        .return_type(spy.Tensor)
    )
    triangle_ids = spy.Tensor.from_numpy(
        device,
        np.array([0, 1, 2, np.iinfo(np.uint32).max], dtype=np.uint32),
    )
    pdf = emissive_pdf(triangle_ids).to_numpy()
    assert np.isfinite(pdf).all()
    assert (pdf[:2] > 0.0).all()
    np.testing.assert_array_equal(pdf[2:], np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_refit_matches_rebuild_and_sampling_pdf(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    triangle_specs = [
        (
            np.array([[-2.0, -1.0, 0.0], [-1.0, -1.0, 0.0], [-2.0, 0.0, 0.0]]),
            spy.float3(1.0, 0.25, 0.125),
        ),
        (
            np.array([[-0.5, -1.0, 0.2], [0.5, -1.0, 0.4], [-0.5, 0.0, 0.2]]),
            spy.float3(2.0, 0.5, 0.25),
        ),
        (
            np.array([[1.0, -1.0, 0.1], [2.0, -1.0, 0.1], [1.0, 0.0, 0.3]]),
            spy.float3(4.0, 1.0, 0.5),
        ),
        (
            np.array([[-0.5, 0.5, 0.6], [0.5, 0.5, 0.5], [-0.5, 1.5, 0.6]]),
            spy.float3(8.0, 2.0, 1.0),
        ),
    ]
    materials: list[f2.StandardMaterial] = []
    entities: list[f2.Entity] = []
    for positions, emissive_factor in triangle_specs:
        material, entity = _add_emissive_triangle(scene, positions, emissive_factor)
        materials.append(material)
        entities.append(entity)

    scene.update()
    refit_sampler = f2.HierarchicalLightSampler()
    refit_sampler.max_triangle_count_per_leaf = 1
    rebuild_sampler = f2.HierarchicalLightSampler()
    rebuild_sampler.max_triangle_count_per_leaf = 1
    rebuild_sampler.allow_refitting = False
    assert refit_sampler.update(scene)
    assert rebuild_sampler.update(scene)
    assert refit_sampler.refit_dispatch_count == refit_sampler.max_depth + 1
    assert refit_sampler.node_buffer_recreation_count == 1

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    refit_sample_pdf = _create_hierarchical_shader_function(
        module, helper, scene, refit_sampler, "hierarchical_emissive_triangle_tree_sample_pdf"
    )
    rebuild_sample_pdf = _create_hierarchical_shader_function(
        module, helper, scene, rebuild_sampler, "hierarchical_emissive_triangle_tree_sample_pdf"
    )
    refit_eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, refit_sampler, "hierarchical_emissive_triangle_tree_pdf"
    )
    rebuild_eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, rebuild_sampler, "hierarchical_emissive_triangle_tree_pdf"
    )
    refit_bounds_min_flux = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        refit_sampler,
        "hierarchical_emissive_triangle_tree_node_bounds_min_flux",
    )
    rebuild_bounds_min_flux = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        rebuild_sampler,
        "hierarchical_emissive_triangle_tree_node_bounds_min_flux",
    )
    refit_bounds_max_cone = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        refit_sampler,
        "hierarchical_emissive_triangle_tree_node_bounds_max_cone",
    )
    refit_cone_direction = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        refit_sampler,
        "hierarchical_emissive_triangle_tree_node_cone_direction",
    )
    refit_leaf_data = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        refit_sampler,
        "hierarchical_emissive_triangle_tree_node_leaf_data",
    )
    refit_triangle_id = _create_hierarchical_shader_function(
        module,
        helper,
        scene,
        refit_sampler,
        "hierarchical_emissive_triangle_tree_triangle_id",
    )
    sample_values = spy.Tensor.from_numpy(
        device,
        ((np.arange(64, dtype=np.float32) + 0.5) / 64.0),
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.arange(4, dtype=np.uint32))
    node_indices = spy.Tensor.from_numpy(
        device,
        np.arange(refit_sampler.node_count, dtype=np.uint32),
    )
    translation = np.zeros(3, dtype=np.float32)

    def compare_samplers() -> None:
        refit_samples = refit_sample_pdf(sample_values).to_numpy()
        rebuild_samples = rebuild_sample_pdf(sample_values).to_numpy()
        assert (refit_samples[:, 0] == 1.0).all()
        np.testing.assert_allclose(refit_samples[:, 2], refit_samples[:, 3], rtol=1e-6)
        np.testing.assert_allclose(rebuild_samples[:, 2], rebuild_samples[:, 3], rtol=1e-6)
        # CPU construction and GPU refitting evaluate and quantize conservative
        # cone merges on different floating-point backends. Their selection
        # probabilities may differ slightly at a quantization boundary.
        np.testing.assert_allclose(refit_samples, rebuild_samples, rtol=5e-3, atol=1e-6)
        np.testing.assert_allclose(
            refit_eval_pdf(triangle_ids).to_numpy(),
            rebuild_eval_pdf(triangle_ids).to_numpy(),
            rtol=5e-3,
            atol=1e-6,
        )

        bounds_min_flux = refit_bounds_min_flux(node_indices).to_numpy()
        rebuild_node_min_flux = rebuild_bounds_min_flux(node_indices).to_numpy()
        bounds_max_cone = refit_bounds_max_cone(node_indices).to_numpy()
        cone_direction = refit_cone_direction(node_indices).to_numpy()
        leaf_data = refit_leaf_data(node_indices).to_numpy()
        tree_triangle_ids = refit_triangle_id(triangle_ids).to_numpy()
        assert np.isfinite(bounds_min_flux).all()
        assert np.isfinite(bounds_max_cone).all()
        assert np.isfinite(cone_direction).all()
        assert (bounds_min_flux[:, :3] <= bounds_max_cone[:, :3]).all()
        np.testing.assert_allclose(bounds_min_flux[:, 3], rebuild_node_min_flux[:, 3], rtol=1e-6)

        for node_index in np.flatnonzero(leaf_data[:, 1] > 0):
            triangle_offset, triangle_count = leaf_data[node_index]
            leaf_triangle_ids = tree_triangle_ids[
                triangle_offset : triangle_offset + triangle_count
            ]
            leaf_positions = np.concatenate(
                [
                    triangle_specs[int(triangle_id)][0].astype(np.float32) + translation
                    for triangle_id in leaf_triangle_ids
                ]
            )
            assert (bounds_min_flux[node_index, :3] <= leaf_positions.min(axis=0)).all()
            assert (bounds_max_cone[node_index, :3] >= leaf_positions.max(axis=0)).all()
            for triangle_id in leaf_triangle_ids:
                positions = triangle_specs[int(triangle_id)][0]
                normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
                normal /= np.linalg.norm(normal)
                assert (
                    np.dot(cone_direction[node_index], normal.astype(np.float32)) + 1e-5
                    >= bounds_max_cone[node_index, 3]
                )

    compare_samplers()

    for entity in entities:
        transform = f2.Transform()
        transform.translation = spy.float3(0.25, -0.5, 0.75)
        entity.transform = transform
    translation = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    scene.update()
    assert refit_sampler.update(scene)
    assert rebuild_sampler.update(scene)
    assert refit_sampler.rebuild_count == 1
    assert refit_sampler.refit_count == 1
    assert refit_sampler.node_buffer_recreation_count == 1
    assert rebuild_sampler.rebuild_count == 2
    compare_samplers()

    materials[1].emissive_factor = spy.float3(3.0, 0.75, 0.375)
    scene.update()
    assert refit_sampler.update(scene)
    assert rebuild_sampler.update(scene)
    assert refit_sampler.rebuild_count == 1
    assert refit_sampler.refit_count == 2
    assert refit_sampler.node_buffer_recreation_count == 1
    assert rebuild_sampler.rebuild_count == 3
    compare_samplers()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_builds_valid_trees_for_all_heuristics(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    configurations = (
        (f2.EmissiveTriangleTreeSplitHeuristic.equal, 16, 2),
        (f2.EmissiveTriangleTreeSplitHeuristic.binned_sah, 2, 2),
        (f2.EmissiveTriangleTreeSplitHeuristic.binned_sah, 128, 2),
        (f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh, 2, 2),
        (f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh, 128, 2),
        (f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh, 128, 64),
    )
    for triangle_specs in (
        _coincident_emissive_triangle_specs(),
        _spatial_emissive_triangle_specs(),
    ):
        scene = f2.Scene.create(device)
        _add_emissive_triangle_specs(scene, triangle_specs)
        scene.update()

        helper = SceneShaderHelper(device)
        source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
        module = helper.get_module(scene, source_module)
        triangle_count = sum(intensity > 0.0 for _, intensity in triangle_specs)

        for split_heuristic, bin_count, max_triangle_count_per_leaf in configurations:
            sampler = f2.HierarchicalLightSampler()
            sampler.split_heuristic = split_heuristic
            sampler.max_triangle_count_per_leaf = max_triangle_count_per_leaf
            sampler.bin_count = bin_count
            assert sampler.update(scene)
            assert sampler.node_count == 2 * sampler.leaf_count - 1

            node_indices = spy.Tensor.from_numpy(
                device,
                np.arange(sampler.node_count, dtype=np.uint32),
            )
            bounds_min_flux = _create_hierarchical_shader_function(
                module,
                helper,
                scene,
                sampler,
                "hierarchical_emissive_triangle_tree_node_bounds_min_flux",
            )(node_indices).to_numpy()
            bounds_max_cone = _create_hierarchical_shader_function(
                module,
                helper,
                scene,
                sampler,
                "hierarchical_emissive_triangle_tree_node_bounds_max_cone",
            )(node_indices).to_numpy()
            cone_direction = _create_hierarchical_shader_function(
                module,
                helper,
                scene,
                sampler,
                "hierarchical_emissive_triangle_tree_node_cone_direction",
            )(node_indices).to_numpy()
            leaf_data = _create_hierarchical_shader_function(
                module,
                helper,
                scene,
                sampler,
                "hierarchical_emissive_triangle_tree_node_leaf_data",
            )(node_indices).to_numpy()

            assert np.isfinite(bounds_min_flux).all()
            assert np.isfinite(bounds_max_cone).all()
            assert np.isfinite(cone_direction).all()
            assert (bounds_min_flux[:, :3] <= bounds_max_cone[:, :3]).all()
            assert (bounds_min_flux[:, 3] > 0.0).all()
            assert ((bounds_max_cone[:, 3] >= -1.0) & (bounds_max_cone[:, 3] <= 1.0)).all()

            leaf_triangle_counts = leaf_data[:, 1]
            assert (leaf_triangle_counts <= max_triangle_count_per_leaf).all()
            assert np.count_nonzero(leaf_triangle_counts) == sampler.leaf_count
            leaf_mask = leaf_triangle_counts > 0
            assert (
                leaf_data[leaf_mask, 0] + leaf_triangle_counts[leaf_mask] <= triangle_count
            ).all()

            triangle_indices = spy.Tensor.from_numpy(
                device,
                np.arange(triangle_count, dtype=np.uint32),
            )
            triangle_ids = _create_hierarchical_shader_function(
                module,
                helper,
                scene,
                sampler,
                "hierarchical_emissive_triangle_tree_triangle_id",
            )(triangle_indices).to_numpy()
            np.testing.assert_array_equal(
                np.sort(triangle_ids),
                np.arange(triangle_count, dtype=np.uint32),
            )

            for node_index in np.flatnonzero(leaf_mask):
                triangle_offset, leaf_triangle_count = leaf_data[node_index]
                leaf_triangle_ids = triangle_ids[
                    triangle_offset : triangle_offset + leaf_triangle_count
                ]
                leaf_positions = np.concatenate(
                    [
                        triangle_specs[int(triangle_id)][0].astype(np.float32)
                        for triangle_id in leaf_triangle_ids
                    ]
                )
                source_bounds_min = leaf_positions.min(axis=0)
                source_bounds_max = leaf_positions.max(axis=0)
                assert (bounds_min_flux[node_index, :3] <= source_bounds_min).all()
                assert (bounds_max_cone[node_index, :3] >= source_bounds_max).all()

                for triangle_id in leaf_triangle_ids:
                    positions = triangle_specs[int(triangle_id)][0]
                    normal = np.cross(positions[1] - positions[0], positions[2] - positions[0])
                    normal /= np.linalg.norm(normal)
                    assert (
                        np.dot(cone_direction[node_index], normal.astype(np.float32)) + 1e-5
                        >= bounds_max_cone[node_index, 3]
                    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "leaf_mode",
    (
        f2.EmissiveTriangleTreeLeafSamplingMode.uniform,
        f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance,
    ),
)
def test_hierarchical_sampler_saoh_sample_matches_pdf(
    device_type: spy.DeviceType,
    device: spy.Device,
    leaf_mode: f2.EmissiveTriangleTreeLeafSamplingMode,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_triangle_specs(scene, _spatial_emissive_triangle_specs()[:4])
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.binned_saoh
    sampler.max_triangle_count_per_leaf = 2
    sampler.leaf_sampling_mode = leaf_mode
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    sample_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_sample_pdf"
    )
    sample_values = spy.Tensor.from_numpy(
        device,
        ((np.arange(64, dtype=np.float32) + 0.5) / 64.0),
    )
    samples = sample_pdf(sample_values).to_numpy()
    assert (samples[:, 0] == 1.0).all()
    np.testing.assert_allclose(samples[:, 2], samples[:, 3], rtol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "leaf_mode",
    (
        f2.EmissiveTriangleTreeLeafSamplingMode.uniform,
        f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance,
    ),
)
def test_hierarchical_sampler_asymmetric_tree_sample_pdf_matches_exactly(
    device_type: spy.DeviceType,
    device: spy.Device,
    leaf_mode: f2.EmissiveTriangleTreeLeafSamplingMode,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_triangle_specs(scene, _asymmetric_emissive_triangle_specs())
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.split_heuristic = f2.EmissiveTriangleTreeSplitHeuristic.binned_sah
    sampler.bin_count = 128
    sampler.max_triangle_count_per_leaf = 1
    sampler.leaf_sampling_mode = leaf_mode
    assert sampler.update(scene)
    assert sampler.max_depth >= 7

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    sample_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_sample_pdf"
    )
    sample_values = spy.Tensor.from_numpy(
        device,
        ((np.arange(4096, dtype=np.float32) + 0.5) / 4096.0),
    )
    samples = sample_pdf(sample_values).to_numpy()
    assert (samples[:, 0] == 1.0).all()
    assert np.unique(samples[:, 1]).size == len(_asymmetric_emissive_triangle_specs())
    np.testing.assert_array_equal(samples[:, 2], samples[:, 3])


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize(
    "leaf_mode",
    (
        f2.EmissiveTriangleTreeLeafSamplingMode.uniform,
        f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance,
    ),
)
def test_hierarchical_sampler_rejects_back_facing_hemisphere(
    device_type: spy.DeviceType,
    device: spy.Device,
    leaf_mode: f2.EmissiveTriangleTreeLeafSamplingMode,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_quad(scene)
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.max_triangle_count_per_leaf = 1
    sampler.leaf_sampling_mode = leaf_mode
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    hemisphere_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_hemisphere_pdf"
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))
    pdf = hemisphere_pdf(triangle_ids).to_numpy()
    assert np.isfinite(pdf).all()
    assert (pdf[:, 0] > 0.0).all()
    np.testing.assert_array_equal(pdf[:, 1], np.zeros(2, dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_contextual_leaf_pdf_is_finite_near_vertex(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    _add_emissive_quad(scene)
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    sampler.max_triangle_count_per_leaf = 2
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_near_vertex_pdf"
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))
    pdf = eval_pdf(triangle_ids).to_numpy()
    assert np.isfinite(pdf).all()
    assert (pdf >= 0.0).all()


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_contextual_leaf_uses_triangle_distance(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    emissive_factor = spy.float3(1.0)
    _add_emissive_triangle(
        scene,
        np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [-1.0, 1.0, 0.0]]),
        emissive_factor,
    )
    _add_emissive_triangle(
        scene,
        np.array([[3.0, -1.0, 0.0], [5.0, -1.0, 0.0], [3.0, 1.0, 0.0]]),
        emissive_factor,
    )
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    sampler.max_triangle_count_per_leaf = 2
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_pdf"
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))
    np.testing.assert_allclose(
        eval_pdf(triangle_ids).to_numpy(),
        np.array([25.0 / 41.0, 16.0 / 41.0], dtype=np.float32),
        rtol=1e-6,
    )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_contextual_leaf_handles_near_degenerate_triangle(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    emissive_factor = spy.float3(1.0)
    _add_emissive_triangle(
        scene,
        np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, -0.5 + 1e-7, 0.0]]),
        emissive_factor,
    )
    _add_emissive_triangle(
        scene,
        np.array([[1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        emissive_factor,
    )
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    sampler.max_triangle_count_per_leaf = 2
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_pdf"
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))
    pdf = eval_pdf(triangle_ids).to_numpy()
    assert np.isfinite(pdf).all()
    assert (pdf >= 0.0).all()
    np.testing.assert_allclose(pdf.sum(), 1.0, rtol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_hierarchical_sampler_contextual_leaf_excludes_zero_area_triangle(
    device_type: spy.DeviceType,
    device: spy.Device,
) -> None:
    scene = f2.Scene.create(device)
    emissive_factor = spy.float3(1.0)
    _add_emissive_triangle(
        scene,
        np.array([[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, -0.5, 0.0]]),
        emissive_factor,
    )
    _add_emissive_triangle(
        scene,
        np.array([[1.0, -1.0, 0.0], [2.0, -1.0, 0.0], [1.0, 0.0, 0.0]]),
        emissive_factor,
    )
    scene.update()

    sampler = f2.HierarchicalLightSampler()
    sampler.leaf_sampling_mode = f2.EmissiveTriangleTreeLeafSamplingMode.contextual_importance
    sampler.max_triangle_count_per_leaf = 2
    assert sampler.update(scene)

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_sampler.slang")
    module = helper.get_module(scene, source_module)
    eval_pdf = _create_hierarchical_shader_function(
        module, helper, scene, sampler, "hierarchical_emissive_triangle_tree_pdf"
    )
    triangle_ids = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))
    pdf = eval_pdf(triangle_ids).to_numpy()
    assert np.isfinite(pdf).all()
    assert (pdf >= 0.0).all()
    np.testing.assert_array_equal(pdf, np.array([0.0, 1.0], dtype=np.float32))


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_reference_pathtracer_renders_with_hierarchical_sampler(
    device_type: spy.DeviceType,
    device: spy.Device,
    helmet_scene: f2.Scene,
) -> None:
    node = ReferencePathTracerNode.create(device)
    node.light_sampler = f2.HierarchicalLightSampler()
    camera = helpers.create_test_camera(helmet_scene, width=16, height=16, fov_y=45)
    color = spy.Tensor.empty(device, (16, 16), spy.float4)

    node._render(helmet_scene, camera, color, iteration=0)
    data = color.to_numpy()

    assert np.isfinite(data).all()
    assert data.max() > 0.0
