# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.ngx as fngx
import falcor2.testing.helpers as helpers
from falcor2.rendernodes import DLSSFrameGenNode, DLSSRayReconNode, DLSSSuperResNode


def _make_texture(
    device: spy.Device,
    width: int,
    height: int,
    format_value: spy.Format = spy.Format.rgba32_float,
) -> spy.Texture:
    return device.create_texture(
        type=spy.TextureType.texture_2d,
        format=format_value,
        width=width,
        height=height,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )


def _make_tensor(
    device: spy.Device,
    width: int,
    height: int,
    format_value: spy.Format = spy.Format.rgba32_float,
) -> spy.Tensor:
    return f2.Container.empty(device, f2.ContainerSpec.tensor(format_value, (height, width)))


def _expected_rgba(height: int, width: int, rgba: tuple[float, float, float, float]) -> np.ndarray:
    expected = np.zeros((height, width, 4), dtype=np.float32)
    expected[..., 0] = rgba[0]
    expected[..., 1] = rgba[1]
    expected[..., 2] = rgba[2]
    expected[..., 3] = rgba[3]
    return expected


def test_frame_jitter_bounds() -> None:
    from falcor2.utils.jitter import frame_jitter

    for index in range(64):
        jitter_x, jitter_y = frame_jitter(index)
        assert -0.5 <= jitter_x <= 0.5
        assert -0.5 <= jitter_y <= 0.5


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_rr_get_optimal_specs_returns_color_and_guide_specs(
    device_type: spy.DeviceType,
    device: spy.Device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    node = DLSSRayReconNode.create(device)
    fake_settings = fngx.DLSSDOptimalSettings()
    fake_settings.render_width = 853
    fake_settings.render_height = 480
    monkeypatch.setattr(node, "get_optimal_settings", lambda _w, _h: fake_settings)

    color_spec, guide_specs = node.get_optimal_specs(1280, 720)
    expected_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )

    assert color_spec == f2.ContainerSpec.texture2d(expected_format, (480, 853))
    assert set(guide_specs) == set(DLSSRayReconNode.GUIDE_NAMES)
    assert all(spec.dims == (480, 853) for spec in guide_specs.values())


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_rr_forward_rejects_missing_guides(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    node = DLSSRayReconNode.create(device)
    color_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )
    color = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=color_format,
        width=853,
        height=480,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=853, height=480)

    with pytest.raises(ValueError, match="Missing DLSSRR guide"):
        node(color, {}, camera, 1280, 720)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_rr_forward_copies_texture_output_to_tensor(
    device_type: spy.DeviceType,
    device: spy.Device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-texture output specs copy the NGX texture output into the requested container."""
    width = 8
    height = 4
    node = DLSSRayReconNode.create(device)
    node.output_spec = f2.ContainerSpec.tensor(spy.Format.rgba32_float)

    color = _make_tensor(device, width, height)
    guides = {name: _make_tensor(device, width, height) for name in DLSSRayReconNode.GUIDE_NAMES}
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=width, height=height)

    expected = _expected_rgba(height, width, (0.25, 0.50, 0.75, 1.00))

    class FakeFeature:
        def evaluate(self, desc: fngx.DLSSRREvaluateDesc) -> None:
            desc.output.copy_from_numpy(expected)

    def fake_ensure_feature(_color: spy.Texture, _output: spy.Texture) -> None:
        node._feature = FakeFeature()

    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)

    output = node(color, guides, camera, width, height)

    assert isinstance(output, spy.Tensor)
    assert tuple(output.shape) == (height, width)
    np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-6)


def test_dlss_rr_forward_copies_texture_output_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DLSS-RR can copy its texture output into the torch render layout."""
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    width = 8
    height = 4
    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    node = DLSSRayReconNode.create(device)
    node.output_spec = f2.ContainerSpec.torch(spy.Format.rgba32_float)

    color = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    guides = {
        name: device.create_texture(
            type=spy.TextureType.texture_2d,
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            mip_count=1,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        )
        for name in DLSSRayReconNode.GUIDE_NAMES
    }
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=width, height=height)

    expected = _expected_rgba(height, width, (0.75, 0.50, 0.25, 1.00))

    class FakeFeature:
        def evaluate(self, desc: fngx.DLSSRREvaluateDesc) -> None:
            desc.output.copy_from_numpy(expected)

    def fake_ensure_feature(_color: spy.Texture, _output: spy.Texture) -> None:
        node._feature = FakeFeature()

    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)

    output = node(color, guides, camera, width, height)

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (4, height, width)
    actual = output.permute(1, 2, 0).contiguous().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_sr_forward_copies_texture_output_to_tensor(
    device_type: spy.DeviceType,
    device: spy.Device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 8
    height = 4
    node = DLSSSuperResNode.create(device)
    node.output_spec = f2.ContainerSpec.tensor(spy.Format.rgba32_float)
    color = _make_tensor(device, width, height)
    guides = {name: _make_tensor(device, width, height) for name in DLSSSuperResNode.GUIDE_NAMES}
    expected = _expected_rgba(height, width, (0.10, 0.20, 0.30, 1.00))

    class FakeEvaluateDesc:
        pass

    class FakeFeature:
        def evaluate(self, *args: object) -> None:
            desc = args[-1]
            desc.output.copy_from_numpy(expected)

    def fake_ensure_feature(_color: spy.Texture, _output: spy.Texture) -> None:
        node._feature = FakeFeature()

    monkeypatch.setattr(fngx, "DLSSSREvaluateDesc", FakeEvaluateDesc, raising=False)
    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)

    output = node(color, guides, width, height)

    assert isinstance(output, spy.Tensor)
    assert tuple(output.shape) == (height, width)
    np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-6)


def test_dlss_sr_forward_copies_texture_output_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    width = 8
    height = 4
    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    node = DLSSSuperResNode.create(device)
    node.output_spec = f2.ContainerSpec.torch(spy.Format.rgba32_float)
    color = _make_texture(device, width, height)
    guides = {name: _make_texture(device, width, height) for name in DLSSSuperResNode.GUIDE_NAMES}
    expected = _expected_rgba(height, width, (0.30, 0.20, 0.10, 1.00))

    class FakeEvaluateDesc:
        pass

    class FakeFeature:
        def evaluate(self, *args: object) -> None:
            desc = args[-1]
            desc.output.copy_from_numpy(expected)

    def fake_ensure_feature(_color: spy.Texture, _output: spy.Texture) -> None:
        node._feature = FakeFeature()

    monkeypatch.setattr(fngx, "DLSSSREvaluateDesc", FakeEvaluateDesc, raising=False)
    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)

    output = node(color, guides, width, height)

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (4, height, width)
    actual = output.permute(1, 2, 0).contiguous().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_sr_forward_rejects_missing_guides(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    node = DLSSSuperResNode.create(device)
    color_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )
    color = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=color_format,
        width=853,
        height=480,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )

    with pytest.raises(ValueError, match="Missing DLSSSR guide"):
        node(color, {}, 1280, 720)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_frame_generation_forward_rejects_missing_guides(
    device_type: spy.DeviceType, device: spy.Device
) -> None:
    node = DLSSFrameGenNode.create(device)
    color_format = (
        spy.Format.rgba32_float
        if device.desc.type == spy.DeviceType.cuda
        else spy.Format.rgba16_float
    )
    backbuffer = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=color_format,
        width=1280,
        height=720,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=1280, height=720)

    with pytest.raises(ValueError, match="Missing DLSSG guide"):
        node(backbuffer, {}, camera)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_dlss_frame_generation_forward_copies_texture_output_to_tensor(
    device_type: spy.DeviceType,
    device: spy.Device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    width = 8
    height = 4
    node = DLSSFrameGenNode.create(device)
    node.output_spec = f2.ContainerSpec.tensor(spy.Format.rgba32_float)
    backbuffer = _make_tensor(device, width, height)
    guides = {
        "hardware_depth": _make_tensor(device, width, height),
        "motion_vectors": _make_tensor(device, width, height),
    }
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=width, height=height)
    expected = _expected_rgba(height, width, (0.60, 0.40, 0.20, 1.00))

    class FakeEvaluateDesc:
        pass

    class FakeFeature:
        def evaluate(self, *args: object) -> None:
            desc = args[-1]
            desc.output_interpolated_frame.copy_from_numpy(expected)

    def fake_ensure_feature(_backbuffer: spy.Texture, _depth: spy.Texture) -> None:
        node._feature = FakeFeature()

    def fake_camera_desc(*_args: object) -> object:
        return object()

    monkeypatch.setattr(fngx, "DLSSGEvaluateDesc", FakeEvaluateDesc, raising=False)
    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)
    monkeypatch.setattr(node, "_make_camera_desc", fake_camera_desc)

    output = node(backbuffer, guides, camera)

    assert isinstance(output, spy.Tensor)
    assert tuple(output.shape) == (height, width)
    np.testing.assert_allclose(output.to_numpy(), expected, atol=1e-6)


def test_dlss_frame_generation_forward_copies_texture_output_to_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    try:
        import torch
    except ImportError:
        pytest.skip("PyTorch not available")

    if spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA device not available")

    width = 8
    height = 4
    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    node = DLSSFrameGenNode.create(device)
    node.output_spec = f2.ContainerSpec.torch(spy.Format.rgba32_float)
    backbuffer = _make_texture(device, width, height)
    guides = {
        "hardware_depth": _make_texture(device, width, height),
        "motion_vectors": _make_texture(device, width, height),
    }
    scene = f2.Scene.create(device)
    camera = helpers.create_test_camera(scene, width=width, height=height)
    expected = _expected_rgba(height, width, (0.20, 0.40, 0.60, 1.00))

    class FakeEvaluateDesc:
        pass

    class FakeFeature:
        def evaluate(self, *args: object) -> None:
            desc = args[-1]
            desc.output_interpolated_frame.copy_from_numpy(expected)

    def fake_ensure_feature(_backbuffer: spy.Texture, _depth: spy.Texture) -> None:
        node._feature = FakeFeature()

    def fake_camera_desc(*_args: object) -> object:
        return object()

    monkeypatch.setattr(fngx, "DLSSGEvaluateDesc", FakeEvaluateDesc, raising=False)
    monkeypatch.setattr(node, "_ensure_feature", fake_ensure_feature)
    monkeypatch.setattr(node, "_make_camera_desc", fake_camera_desc)

    output = node(backbuffer, guides, camera)

    assert isinstance(output, torch.Tensor)
    assert tuple(output.shape) == (4, height, width)
    actual = output.permute(1, 2, 0).contiguous().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
