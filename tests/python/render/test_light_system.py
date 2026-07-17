# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_environment_light_distribution(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    scene.update()

    helper = SceneShaderHelper(device)
    source_module = spy.Module.load_from_file(device, "render/test_light_system.slang")
    module = helper.get_module(scene, source_module)
    direction_probe = (
        module.environment_direction_pdf.as_func()
        .type_conformances(scene.requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    directions = spy.Tensor.from_numpy(device, np.array([[0.0, 0.0, 1.0]], dtype=np.float32))

    # No environment lights evaluate to zero without accessing a distribution.
    np.testing.assert_allclose(direction_probe(directions).to_numpy(), 0.0, atol=0.0)

    entity = scene.create_entity()
    light = entity.create_component(f2.ConstantLight)
    light.radiance = spy.float3(1.0)
    scene.update()

    # A single environment light delegates directly to the light's PDF without
    # constructing or binding a selection distribution.
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([1.0 / (4.0 * np.pi)], dtype=np.float32),
        rtol=1e-6,
    )

    distant_entity = scene.create_entity()
    distant_light = distant_entity.create_component(f2.DistantLight)
    distant_light.radiance = spy.float3(1.0)
    distant_light.cutoff_angle = 60.0

    scene.update()

    indices = spy.Tensor.from_numpy(device, np.array([0, 1], dtype=np.uint32))

    def read_selection_pdf() -> np.ndarray:
        probe = (
            module.environment_light_selection_pdf.as_func()
            .type_conformances(scene.requirements.type_conformances)
            .write(helper.bind_scene)
            .return_type(spy.Tensor)
        )
        return probe(indices).to_numpy()

    # The constant light covers 4*pi steradians while a 60-degree distant
    # light covers pi steradians, giving selection probabilities 4:1.
    np.testing.assert_allclose(
        read_selection_pdf(),
        np.array([0.8, 0.2], dtype=np.float32),
        rtol=1e-6,
    )

    directions = spy.Tensor.from_numpy(
        device,
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], dtype=np.float32),
    )
    # The first direction is inside the distant-light cone; the second only
    # has the constant-light PDF. Both include the light-selection PDF.
    np.testing.assert_allclose(
        direction_probe(directions).to_numpy(),
        np.array([0.4 / np.pi, 0.2 / np.pi], dtype=np.float32),
        rtol=1e-6,
    )

    # A fully black environment still needs a valid distribution so sampling
    # does not produce a zero selection probability.
    light.radiance = spy.float3(0.0)
    distant_light.radiance = spy.float3(0.0)
    scene.update()
    np.testing.assert_allclose(
        read_selection_pdf(),
        np.array([0.5, 0.5], dtype=np.float32),
        rtol=1e-6,
    )
