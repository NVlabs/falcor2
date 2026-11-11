# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import slangpy as spy

from falcor2.editor.utils import get_module_and_shader_cache_root
from falcor2.minitracer import tools
from falcor2.testing import helpers


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_device_uses_falcor_defaults(device_type: spy.DeviceType, tmp_path: Path) -> None:
    device = tools.create_device(device_type, additional_include_paths=[tmp_path])
    try:
        desc = device.desc
        assert desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.fast
        assert tmp_path in [Path(path) for path in desc.compiler_options.include_paths]
        cache_root = get_module_and_shader_cache_root()
        assert desc.module_cache_path == cache_root / "module"
        assert desc.shader_cache_path == cache_root / "shader"
    finally:
        device.close()
