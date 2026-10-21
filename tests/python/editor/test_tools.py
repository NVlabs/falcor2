# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest
import slangpy as spy
import falcor2.testing.helpers as helpers


def test_app_cache_root_uses_user_cache_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from falcor2.editor.utils import get_module_and_shader_cache_root

    if os.name == "nt":
        monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
        assert get_module_and_shader_cache_root() == tmp_path / "falcor2" / "app"
    elif sys.platform == "darwin":
        assert (
            get_module_and_shader_cache_root()
            == Path.home() / "Library" / "Caches" / "falcor2" / "app"
        )
    else:
        monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
        assert get_module_and_shader_cache_root() == tmp_path / "falcor2" / "app"


def test_create_device_dict_defaults() -> None:
    from falcor2.editor.utils import (
        _prepare_device_desc,
        get_module_and_shader_cache_root,
        get_slang_include_paths,
    )

    options = {"type": spy.DeviceType.cuda, "compiler_options": {"include_paths": []}}
    desc = _prepare_device_desc(options)
    cache_root = get_module_and_shader_cache_root()

    assert desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.fast
    assert [str(path) for path in desc.compiler_options.include_paths] == get_slang_include_paths()
    assert desc.module_cache_path == cache_root / "module"
    assert desc.shader_cache_path == cache_root / "shader"
    assert options["compiler_options"]["include_paths"] == []


def test_create_device_preserves_explicit_float_mode() -> None:
    from falcor2.editor.utils import _prepare_device_desc

    desc = _prepare_device_desc(
        {"compiler_options": {"floating_point_mode": spy.SlangFloatingPointMode.precise}}
    )

    assert desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.precise


def test_create_device_dict_explicit_none_disables_caches() -> None:
    from falcor2.editor.utils import _prepare_device_desc

    desc = _prepare_device_desc({"module_cache_path": None, "shader_cache_path": None})

    assert desc.module_cache_path is None
    assert desc.shader_cache_path is None
    assert desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.fast


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_create_device(device_type: spy.DeviceType) -> None:
    """create_device returns a working Device."""
    from falcor2.editor import create_device

    device = create_device(device_type, enable_debug_layers=True)
    assert device is not None
    assert device.desc.compiler_options.floating_point_mode == spy.SlangFloatingPointMode.fast


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_load_scene(device_type: spy.DeviceType) -> None:
    """load_scene returns a Scene with materials/geometries."""
    from falcor2.editor import create_device, load_scene

    device = create_device(device_type, enable_debug_layers=True)
    scene = load_scene(device, "data/assets/kronos/Avocado/glTF-Binary/Avocado.glb")
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
