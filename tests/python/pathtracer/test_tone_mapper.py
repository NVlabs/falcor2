# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tone_mapper_node_maps_to_display_range(
    device_type: spy.DeviceType, device: spy.Device, helmet_scene: f2.Scene
) -> None:
    """TonemapperNode applies ACES filmic tonemapping into a fresh output tensor."""
    from falcor2.rendernodes import TonemapperNode
    from falcor2.rendernodes import ReferencePathTracerNode

    cam = helpers.create_test_camera(helmet_scene, width=4, height=4, fov_y=45)
    input_tensor, _ = ReferencePathTracerNode.create(device)(helmet_scene, cam, iteration=0)
    node = TonemapperNode.create(device)

    output = node(input_tensor).to_numpy()

    assert output.min() >= 0.0
    assert output.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_tonemapper_auto_output_spec_matches_float3_input(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    """TonemapperNode defaults to matching the input container format."""
    from falcor2.rendernodes import TonemapperNode

    node = TonemapperNode.create(device)
    input_tensor = spy.Tensor.empty(device, (4, 4), spy.float3)

    output = node(input_tensor)

    assert output.shape == (4, 4)
    assert output.dtype.full_name == "vector<float,3>"
