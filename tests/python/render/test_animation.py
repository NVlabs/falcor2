# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy
import falcor2 as f2
from falcor2.testing.helpers import DEFAULT_DEVICE_TYPES


# Channel references and paused edits are independent of the GPU backend.
@pytest.mark.parametrize("device_type", DEFAULT_DEVICE_TYPES[:1])
def test_paused_channel_edits(device: spy.Device) -> None:
    scene = f2.Scene.create(device)
    animation = scene.create_animation()
    channel = f2.AnimationChannel()
    channel.times = [0.0, 1.0]
    channel.values = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
    channel.in_tangents = channel.out_tangents = [0.0] * 6
    animation.channels = [channel]
    entity = scene.create_entity()
    component = entity.create_component(f2.TransformAnimation)
    component.animation = animation
    component.translation_channel = 0
    scene.time = 0.5
    scene.update()
    assert entity.transform.translation.x == pytest.approx(1.0)
    assert not (scene.update() & f2.SceneUpdateFlags.animation)
    assert animation.channels[0] is channel

    for attribute, value, expected in [
        ("values", [0.0, 0.0, 0.0, 4.0, 0.0, 0.0], 2.0),
        ("times", [0.0, 2.0], 1.0),
        ("interpolation_mode", f2.AnimationInterpolationMode.cubic_spline, 0.625),
        ("out_tangents", [4.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1.75),
        ("in_tangents", [0.0, 0.0, 0.0, 4.0, 0.0, 0.0], 1.375),
    ]:
        setattr(channel, attribute, value)
        assert scene.update() & f2.SceneUpdateFlags.animation
        assert entity.transform.translation.x == pytest.approx(expected)
    assert scene.time == 0.5

    replacement = f2.AnimationChannel()
    replacement.times = [0.0]
    replacement.values = [8.0, 0.0, 0.0]
    animation.channels = [replacement]
    assert scene.update() & f2.SceneUpdateFlags.animation
    assert entity.transform.translation.x == pytest.approx(8.0)
    channel.values = [1.0] * 6
    assert not (scene.update() & f2.SceneUpdateFlags.animation)

    component.animation = None
    assert component.animation is None
    scene.update()
    replacement.value_type = f2.AnimationValueType.float
    assert scene.update() & f2.SceneUpdateFlags.animation
    assert entity.transform.translation.x == pytest.approx(8.0)
    # Retained Python channels must remain editable after their native owner dies.
    del component, entity, animation, scene
    replacement.values = [7.0, 8.0, 9.0]
    assert replacement.values == [7.0, 8.0, 9.0]
