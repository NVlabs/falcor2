# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import slangpy as spy
import falcor2.testing.helpers as helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_device(device_type: spy.DeviceType) -> None:
    """create_device returns a working Device."""
    from falcor2.editor import create_device

    device = create_device(device_type, enable_debug_layers=True)
    assert device is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_scene(device_type: spy.DeviceType) -> None:
    """load_scene returns a Scene with materials/geometries."""
    from falcor2.editor import create_device, load_scene

    device = create_device(device_type, enable_debug_layers=True)
    scene = load_scene(device, "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf")
    assert len(scene.materials) > 0
    assert len(scene.geometries) > 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_save_image(device_type: spy.DeviceType, workspace_tmp_path: Path) -> None:
    """save_image writes a valid image file."""
    from falcor2.editor import create_device, save_image

    device = create_device(device_type, enable_debug_layers=True)
    tensor = spy.Tensor.empty(device, (16, 16), spy.float4)
    out_path = workspace_tmp_path / "test.png"
    save_image(tensor, str(out_path))
    assert out_path.exists()
    assert out_path.stat().st_size > 0
