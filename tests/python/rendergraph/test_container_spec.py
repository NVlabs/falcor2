# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.rendergraph import Container, ContainerSpec, container_torch, AUTO
from falcor2.rendergraph.image_format import (
    channel_count,
    format_to_typename,
    slangtype_to_format,
)

ALL_DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_empty_requires_resolved_dims(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    with pytest.raises(ValueError, match="fully resolved"):
        Container.empty(device, ContainerSpec.tensor(format=spy.Format.rgba32_float))


def test_container_spec_resolved_merges_dims():
    resolved = ContainerSpec.tensor(format=spy.Format.rgba32_float).resolved(
        ContainerSpec.tensor(dims=(4, 5))
    )

    assert resolved.container_type is spy.Tensor
    assert resolved.format == spy.Format.rgba32_float
    assert resolved.dims == (4, 5)


def test_container_spec_with_dims_returns_updated_copy():
    spec = ContainerSpec.tensor(format=spy.Format.rgba32_float)
    updated = spec.with_dims((4, 5))

    assert spec.dims is AUTO
    assert updated.container_type is spy.Tensor
    assert updated.format == spy.Format.rgba32_float
    assert updated.dims == (4, 5)


def test_container_spec_with_format_returns_updated_copy():
    spec = ContainerSpec.tensor(dims=(4, 5))
    updated = spec.with_format(spy.Format.rgb32_float)

    assert updated.container_type is spy.Tensor
    assert updated.format == spy.Format.rgb32_float
    assert updated.dims == (4, 5)


def test_format_to_typename_maps_supported_formats():
    assert format_to_typename(spy.Format.r32_float) == "vector<float,1>"
    assert format_to_typename(spy.Format.rg32_float) == "vector<float,2>"
    assert format_to_typename(spy.Format.rgb32_float) == "vector<float,3>"
    assert format_to_typename(spy.Format.rgba32_float) == "vector<float,4>"
    assert format_to_typename(spy.Format.r32_sint) == "vector<int,1>"
    assert format_to_typename(spy.Format.rg32_sint) == "vector<int,2>"
    assert format_to_typename(spy.Format.rgb32_sint) == "vector<int,3>"
    assert format_to_typename(spy.Format.rgba32_sint) == "vector<int,4>"
    assert format_to_typename(spy.Format.r32_uint) == "vector<uint,1>"
    assert format_to_typename(spy.Format.rg32_uint) == "vector<uint,2>"
    assert format_to_typename(spy.Format.rgb32_uint) == "vector<uint,3>"
    assert format_to_typename(spy.Format.rgba32_uint) == "vector<uint,4>"


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_create_tensor_container_uses_dims_and_format(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    spec = ContainerSpec.tensor(format=spy.Format.rgba32_float, dims=(4, 5))

    tensor = Container.empty(device, spec)

    assert isinstance(tensor, spy.Tensor)
    assert tuple(int(dim) for dim in tensor.shape) == (4, 5)
    assert tensor.dtype.full_name == "vector<float,4>"


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_spec_from_tensor_uses_dims(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    tensor = spy.Tensor.empty(device, shape=(3, 2), dtype="float3")

    spec = ContainerSpec.from_container(tensor)

    assert spec.container_type is spy.Tensor
    assert spec.format == spy.Format.rgb32_float
    assert spec.dims == (3, 2)
    assert Container.dims(tensor) == (3, 2)
    assert Container.format(tensor) == spy.Format.rgb32_float


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_texture_helpers_use_height_width_dims(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    texture = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=5,
        height=4,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
        label="test_texture",
    )

    spec = ContainerSpec.from_container(texture)

    assert spec.container_type is spy.Texture
    assert spec.format == spy.Format.rgba32_float
    assert spec.dims == (4, 5)
    assert Container.dims(texture) == (4, 5)
    assert Container.format(texture) == spy.Format.rgba32_float
    assert Container.is_spec(
        texture,
        ContainerSpec.texture2d(spy.Format.rgba32_float, (4, 5)).resolved(
            ContainerSpec.from_container(texture)
        ),
    )


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_create_temp_reuses_matching_tensor(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    spec = ContainerSpec.tensor(format=spy.Format.rgba32_float, dims=(4, 5))
    current = Container.empty(device, spec)

    reused = Container.create_temp(device, spec, current=current)

    assert reused is current


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_create_temp_like_reuses_matching_texture(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    target = Container.empty(device, ContainerSpec.texture2d(spy.Format.rgba32_float, (4, 5)))
    current = Container.empty_like(target)

    reused = Container.create_temp_like(device, target, current=current)

    assert reused is current


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_empty_like_supports_texture_format_override(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    source = Container.empty(device, ContainerSpec.texture2d(spy.Format.r32_float, (4, 5)))

    updated = Container.empty_like(source, override_format=spy.Format.rgba32_float)

    assert isinstance(updated, spy.Texture)
    assert Container.format(updated) == spy.Format.rgba32_float
    assert Container.dims(updated) == (4, 5)


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_slangtype_to_format_maps_supported_dtypes(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="float").dtype)
        == spy.Format.r32_float
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="float2").dtype)
        == spy.Format.rg32_float
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="float3").dtype)
        == spy.Format.rgb32_float
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="float4").dtype)
        == spy.Format.rgba32_float
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="int").dtype)
        == spy.Format.r32_sint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="int2").dtype)
        == spy.Format.rg32_sint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="int3").dtype)
        == spy.Format.rgb32_sint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="int4").dtype)
        == spy.Format.rgba32_sint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="uint").dtype)
        == spy.Format.r32_uint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="uint2").dtype)
        == spy.Format.rg32_uint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="uint3").dtype)
        == spy.Format.rgb32_uint
    )
    assert (
        slangtype_to_format(spy.Tensor.empty(device, shape=(1,), dtype="uint4").dtype)
        == spy.Format.rgba32_uint
    )


def test_channel_count_maps_supported_formats():
    assert channel_count(spy.Format.r32_float) == 1
    assert channel_count(spy.Format.rg32_float) == 2
    assert channel_count(spy.Format.rgb32_float) == 3
    assert channel_count(spy.Format.rgba32_float) == 4
    assert channel_count(spy.Format.r32_sint) == 1
    assert channel_count(spy.Format.rg32_sint) == 2
    assert channel_count(spy.Format.rgb32_sint) == 3
    assert channel_count(spy.Format.rgba32_sint) == 4
    assert channel_count(spy.Format.r32_uint) == 1
    assert channel_count(spy.Format.rg32_uint) == 2
    assert channel_count(spy.Format.rgb32_uint) == 3
    assert channel_count(spy.Format.rgba32_uint) == 4


def test_torch_format_conversion_round_trip():
    torch = pytest.importorskip("torch")

    assert container_torch.format_to_torch(spy.Format.rgba32_float) == torch.float32
    assert container_torch.torch_to_format(torch.float32, 4) == spy.Format.rgba32_float
    assert container_torch.format_to_torch(spy.Format.rgba32_sint) == torch.int32
    assert container_torch.torch_to_format(torch.int32, 4) == spy.Format.rgba32_sint


def test_torch_allocate_uses_chw_layout():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch container allocation")

    tensor = container_torch.allocate((4, 5), spy.Format.rgba32_float)

    assert tuple(int(dim) for dim in tensor.shape) == (4, 4, 5)
    assert tensor.dtype == torch.float32


def test_torch_spec_inference_uses_dims_and_chw_layout():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch container allocation")

    tensor = torch.empty((1, 4, 5), dtype=torch.float32, device="cuda")

    spec = ContainerSpec.from_container(tensor)

    assert spec.container_type is torch.Tensor
    assert spec.format == spy.Format.r32_float
    assert spec.dims == (4, 5)
    assert Container.dims(tensor) == (4, 5)
    assert Container.format(tensor) == spy.Format.r32_float


def test_torch_render_layout_converts_chw_to_hwc():
    torch = pytest.importorskip("torch")

    tensor = torch.empty((4, 5, 6), dtype=torch.float32)

    render_layout = Container.to_render_layout(tensor)

    assert tuple(int(dim) for dim in render_layout.shape) == (5, 6, 4)


@pytest.mark.parametrize(
    ("format_value", "clear_value", "expected_values"),
    [
        (spy.Format.r16_float, spy.float4(0.25, 0.5, 0.75, 1.0), (0.25,)),
        (spy.Format.rg16_float, spy.float4(0.25, 0.5, 0.75, 1.0), (0.25, 0.5)),
        (
            spy.Format.rgba16_float,
            spy.float4(0.25, 0.5, 0.75, 1.0),
            (0.25, 0.5, 0.75, 1.0),
        ),
        (spy.Format.r32_float, spy.float4(0.25, 0.5, 0.75, 1.0), (0.25,)),
        (spy.Format.rg32_float, spy.float4(0.25, 0.5, 0.75, 1.0), (0.25, 0.5)),
        (
            spy.Format.rgb32_float,
            spy.float4(0.25, 0.5, 0.75, 1.0),
            (0.25, 0.5, 0.75),
        ),
        (
            spy.Format.rgba32_float,
            spy.float4(0.25, 0.5, 0.75, 1.0),
            (0.25, 0.5, 0.75, 1.0),
        ),
        (spy.Format.r32_sint, spy.int4(-1, 2, -3, 4), (-1,)),
        (spy.Format.rg32_sint, spy.int4(-1, 2, -3, 4), (-1, 2)),
        (spy.Format.rgb32_sint, spy.int4(-1, 2, -3, 4), (-1, 2, -3)),
        (spy.Format.rgba32_sint, spy.int4(-1, 2, -3, 4), (-1, 2, -3, 4)),
        (spy.Format.r32_uint, spy.uint4(1, 2, 3, 4), (1,)),
        (spy.Format.rg32_uint, spy.uint4(1, 2, 3, 4), (1, 2)),
        (spy.Format.rgb32_uint, spy.uint4(1, 2, 3, 4), (1, 2, 3)),
        (spy.Format.rgba32_uint, spy.uint4(1, 2, 3, 4), (1, 2, 3, 4)),
    ],
)
@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_clear_supports_tensor_nonzero_value(
    device_type: spy.DeviceType,
    format_value: spy.Format,
    clear_value: object,
    expected_values: tuple[float | int, ...],
):
    device = helpers.get_device(device_type)
    tensor = Container.empty(device, ContainerSpec.tensor(format_value, (2, 3)))

    Container.clear(tensor, clear_value)

    data = tensor.to_numpy()
    assert data.shape == (2, 3, len(expected_values))
    for channel, expected in enumerate(expected_values):
        if "int" in format_value.name:
            assert data[..., channel].min() == expected
            assert data[..., channel].max() == expected
        else:
            assert data[..., channel].min() == pytest.approx(expected, abs=1e-3)
            assert data[..., channel].max() == pytest.approx(expected, abs=1e-3)


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_clear_requires_vector_clear_value(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    tensor = Container.empty(device, ContainerSpec.tensor(spy.Format.rgba32_float, (2, 3)))

    with pytest.raises(TypeError, match="float4 clear value"):
        Container.clear(tensor, (0.25, 0.5, 0.75, 1.0))


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_clear_requires_float4_for_float_format(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    tensor = Container.empty(device, ContainerSpec.tensor(spy.Format.rgba32_float, (2, 3)))

    with pytest.raises(TypeError, match="float4 clear value"):
        Container.clear(tensor, spy.float3(1.0, 2.0, 3.0))


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_clear_requires_uint4_for_uint_format(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    tensor = Container.empty(device, ContainerSpec.tensor(spy.Format.rgba32_uint, (2, 3)))

    with pytest.raises(TypeError, match="uint4 clear value"):
        Container.clear(tensor, spy.float4(1.0, 2.0, 3.0, 4.0))


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_container_clear_requires_int4_for_sint_format(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    tensor = Container.empty(device, ContainerSpec.tensor(spy.Format.rgba32_sint, (2, 3)))

    with pytest.raises(TypeError, match="int4 clear value"):
        Container.clear(tensor, spy.float4(1.0, 2.0, 3.0, 4.0))


@pytest.mark.parametrize(
    ("clear_value", "expected_values"),
    [
        (spy.float4(0.25, 0.5, 0.75, 1.0), (0.25,)),
        (spy.float4(0.25, 0.5, 0.75, 1.0), (0.25, 0.5)),
        (spy.float4(0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75)),
        (spy.float4(0.25, 0.5, 0.75, 1.0), (0.25, 0.5, 0.75, 1.0)),
    ],
)
def test_container_clear_supports_torch_chw_value(
    clear_value: object,
    expected_values: tuple[float, ...],
):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or spy.DeviceType.cuda not in helpers.DEFAULT_DEVICE_TYPES:
        pytest.skip("CUDA is required for torch container clearing")

    device = helpers.get_torch_device(spy.DeviceType.cuda, use_cache=False)
    tensor = torch.empty((len(expected_values), 2, 3), dtype=torch.float32, device="cuda")
    command_encoder = device.create_command_encoder()

    Container.clear(tensor, clear_value, command_encoder=command_encoder)
    device.submit_command_buffer(command_encoder.finish())
    torch.cuda.synchronize()

    for channel, expected in enumerate(expected_values):
        assert torch.allclose(tensor[channel], torch.full((2, 3), expected, device="cuda"))


def test_torch_empty_like_supports_override_format():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch container allocation")

    source = torch.empty((1, 4, 5), dtype=torch.float32, device="cuda")

    updated = Container.empty_like(source, override_format=spy.Format.rgba32_float)

    assert tuple(int(dim) for dim in updated.shape) == (4, 4, 5)
    assert updated.dtype == torch.float32


def test_torch_is_like_requires_same_device():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for torch device checks")

    a = torch.empty((4, 4, 5), dtype=torch.float32, device="cuda:0")
    b = torch.empty((4, 4, 5), dtype=torch.float32, device="cuda:0")

    assert Container.is_like(a, b)
