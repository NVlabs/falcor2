# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy

import falcor2 as f2
import falcor2.testing.helpers as helpers
from falcor2.tools.materials.material_tools import MaterialVisualizer


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_material_visualizer_material_id_requires_active_scene(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    scene = f2.Scene.create(device)
    material = scene.create_material(f2.StandardMaterial)
    scene.update()

    assert MaterialVisualizer._material_id(scene, material) == int(material.material_id)

    other_scene = f2.Scene.create(device)
    with pytest.raises(ValueError, match="active scene"):
        MaterialVisualizer._material_id(other_scene, material)

    pending_material = scene.create_material(f2.StandardMaterial)
    with pytest.raises(ValueError, match="owned by the scene"):
        MaterialVisualizer._material_id(scene, pending_material)
