# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


@pytest.fixture
def device(device_type: spy.DeviceType) -> spy.Device:
    return helpers.get_device(device_type)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_geometry_instance_id_wrapper(device: spy.Device) -> None:
    assert f2.GeometryInstanceID.invalid is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_geometry_instance_exposes_typed_id(device: spy.Device) -> None:
    scene = f2.Scene.create(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
    scene.update()

    component = next(
        component for component in scene.components if isinstance(component, f2.GeometryInstance)
    )

    assert isinstance(component.geometry_instance_id, f2.GeometryInstanceID)
    assert component.geometry_instance_id != f2.GeometryInstanceID.invalid
