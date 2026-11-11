# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import slangpy as spy
from slangpy.core.native import get_value_signature

import falcor2.testing.helpers as helpers
from falcor2.rendergraph import (
    Container,
    ContainerSpec,
    IndexedOutput,
    OutputOperation,
    OutputPrelude,
)

ALL_DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES
INDEXED_OUTPUT_DEVICE_TYPES = [
    pytest.param(
        device_type,
        marks=pytest.mark.nightly if device_type == spy.DeviceType.cuda else (),
    )
    for device_type in helpers.DEFAULT_DEVICE_TYPES
]

TEST_MODULE = """
import slangpy;

[__AttributeUsage(_AttributeTargets.Function)]
struct OutputSpecAttribute
{
    string format;
    string clear_value;
}

public interface ITestWrite
{
    [OutputSpec("rgba32_float", "0,0,0,1")]
    public static void color(uint2 coord, float4 value);
    public static bool color_enabled();
    [OutputSpec("r32_float", "1")]
    public static void depth(uint2 coord, float value);
    public static bool depth_enabled();
    [OutputSpec("r32_float", "0")]
    public static bool ignored_metadata() { return false; }
    public static void generic_data(uint2 coord, uint id, float4 value);
    public static bool generic_data_enabled(uint id);
}

public interface TestWriteNoPrefix
{
    public static void color(uint2 coord, float4 value);
}

extern struct TestWrite : ITestWrite;
extern struct CustomTestWrite : ITestWrite;
extern static const bool TestWrite_color = false;
extern static const bool TestWrite_depth = false;
extern static const bool CustomTestWrite_color = false;

public void dummy()
{
}

public void write_mixed(uint2 coord, uint id, float4 value)
{
    TestWrite::color(coord, value);
    TestWrite::generic_data(coord, id, value);
}
"""

FIXED_GENERIC_MODULE = """
import slangpy;

[__AttributeUsage(_AttributeTargets.Function)]
struct OutputSpecAttribute
{
    string format;
    string clear_value;
    string operation;
}

public interface IFixedGenericWrite
{
    [OutputSpec("rgba32_float", "0", "increment")]
    public static void generic_data(uint2 coord, float4 value);
    public static bool generic_data_enabled();
}

extern struct FixedGenericWrite : IFixedGenericWrite;
"""

TEST_INT_MODULE = """
import slangpy;

[__AttributeUsage(_AttributeTargets.Function)]
struct OutputSpecAttribute
{
    string format;
    string clear_value;
}

public interface IIntWrite
{
    [OutputSpec("rgba32_sint", "-1,2,-3,4")]
    public static void ids(uint2 coord, int4 value);
    public static bool ids_enabled();
}

public void dummy()
{
}
"""

TEST_INCREMENT_MODULE = """
import slangpy;

[__AttributeUsage(_AttributeTargets.Function)]
struct OutputSpecAttribute
{
    string format;
    string clear_value;
    string operation;
}

public interface ITestIncrement
{
    [OutputSpec("r32_uint", "0", "increment")]
    public static void count(uint2 coord, uint value);
    public static bool count_enabled();
}

extern struct TestIncrement : ITestIncrement;
extern static const bool TestIncrement_count = false;

public void count_nan(uint2 coord, float3 value)
{
    if (any(isnan(value)))
        TestIncrement::count(coord, 1u);
}
"""


def _module(device: spy.Device) -> spy.Module:
    return spy.Module(device.load_module_from_source("output_prelude_test", TEST_MODULE))


def _int_module(device: spy.Device) -> spy.Module:
    return spy.Module(device.load_module_from_source("output_prelude_int_test", TEST_INT_MODULE))


def _increment_module(device: spy.Device) -> spy.Module:
    return spy.Module(
        device.load_module_from_source("output_prelude_increment_test", TEST_INCREMENT_MODULE)
    )


def _texture(device: spy.Device, format: spy.Format) -> spy.Texture:
    return device.create_texture(
        type=spy.TextureType.texture_2d,
        format=format,
        width=3,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )


def _fixed_generic_module(device: spy.Device) -> spy.Module:
    return spy.Module(
        device.load_module_from_source("output_prelude_fixed_generic_test", FIXED_GENERIC_MODULE)
    )


def _indexed_target(
    device: spy.Device,
    component_count: int,
    container_type: type[spy.Texture] | type[spy.Tensor],
) -> spy.Texture | spy.Tensor:
    format_value = OutputPrelude.indexed_output_format(component_count)
    if container_type is spy.Texture:
        return _texture(device, format_value)
    return Container.empty(
        device,
        ContainerSpec.tensor(format=format_value, dims=(2, 3)),
    )


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_read_specs_reflects_function_attributes(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    module = _module(device)

    specs = OutputPrelude.read_specs(module, "ITestWrite")

    assert [spec.name for spec in specs] == ["color", "depth"]
    assert [spec.format for spec in specs] == [spy.Format.rgba32_float, spy.Format.r32_float]
    assert tuple(specs[0].clear_value) == (0.0, 0.0, 0.0, 1.0)
    assert tuple(specs[1].clear_value) == (1.0, 0.0, 0.0, 0.0)
    assert all(spec.operation == OutputOperation.write for spec in specs)
    assert OutputPrelude.create(module, "ITestWrite").specs == {
        "color": specs[0],
        "depth": specs[1],
    }


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_read_specs_supports_signed_integer_clear_values(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    module = _int_module(device)

    specs = OutputPrelude.read_specs(module, "IIntWrite")

    assert len(specs) == 1
    assert specs[0].name == "ids"
    assert specs[0].format == spy.Format.rgba32_sint
    assert specs[0].clear_value == spy.int4(-1, 2, -3, 4)


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_generates_texture_and_tensor_outputs(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    module = _module(device)
    color = _texture(device, spy.Format.rgba32_float)
    depth = spy.Tensor.empty(device, shape=(2, 3), dtype="float")

    targets = {"color": color, "depth": depth}
    output_prelude = OutputPrelude.create(module, "ITestWrite")
    prelude = output_prelude.generate(targets)

    assert "public RWTexture2D<vector<float,4>> color;" in prelude
    assert "import slangpy;" not in prelude
    assert "public WTensor<float, 2> depth;" in prelude
    assert "public struct __falcor_OutputBlockITestWrite" in prelude
    assert "__g_falcor_OutputBlockITestWrite.color[coord] = value;" in prelude
    assert (
        "__g_falcor_OutputBlockITestWrite.depth.store(int(coord.y), int(coord.x), value);"
        in prelude
    )
    assert "export struct TestWrite : ITestWrite = __falcor_OutputImplITestWrite;" in prelude
    assert "public static bool color_enabled()" in prelude
    assert "public static bool depth_enabled()" in prelude
    assert "return true;" in prelude
    assert "export static const bool TestWrite_color = true;" in prelude
    assert "export static const bool TestWrite_depth = true;" in prelude
    expected_signature = get_value_signature(targets)
    assert output_prelude.signature(targets) == "ITestWrite\nTestWrite\n" + expected_signature


@pytest.mark.nightly_if(device_type=(spy.DeviceType.cuda,))
@pytest.mark.parametrize("device_type", INDEXED_OUTPUT_DEVICE_TYPES)
def test_output_prelude_writes_indexed_texture_and_tensor_outputs(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    module = _module(device)
    output_prelude = OutputPrelude.create(module, "ITestWrite")

    indexed_outputs = []
    output_ids = iter((2, 5, 9, 14, 20, 27, 35, 44))
    for container_type in (spy.Texture, spy.Tensor):
        for component_count in range(1, 5):
            output_id = next(output_ids)
            indexed_outputs.append(
                IndexedOutput(
                    id=output_id,
                    name=f"mask.with-invalid/slang-chars-{container_type.__name__}-{component_count}",
                    component_count=component_count,
                    target=_indexed_target(device, component_count, container_type),
                )
            )

    indexed_outputs.reverse()
    fixed = _indexed_target(device, 4, spy.Tensor)
    targets = {"color": fixed}
    prelude = output_prelude.generate(targets, indexed_outputs)
    assert output_prelude.signature(targets, indexed_outputs) == output_prelude.signature(
        targets, list(reversed(indexed_outputs))
    )
    assert "mask.with-invalid" not in prelude
    for output in indexed_outputs:
        assert f"indexed_output_{output.id}" in prelude
        assert f"case {output.id}u:" in prelude
    sorted_ids = sorted(output.id for output in indexed_outputs)
    assert [prelude.index(f"indexed_output_{output_id};") for output_id in sorted_ids] == sorted(
        prelude.index(f"indexed_output_{output_id};") for output_id in sorted_ids
    )
    assert "default:" in prelude
    assert "return false;" in prelude

    for output in indexed_outputs:
        Container.clear(output.target, spy.float4(9.0))
    command_encoder = device.create_command_encoder()
    output_prelude.clear_indexed(indexed_outputs, command_encoder=command_encoder)
    device.submit_command_buffer(command_encoder.finish())

    write_mixed = module.write_mixed.prelude(prelude).write(
        lambda cursor: output_prelude.bind(cursor, targets, indexed_outputs)
    )
    value = spy.float4(1.0, 2.0, 3.0, 4.0)
    for output in indexed_outputs:
        write_mixed(spy.uint2(1, 0), output.id, value)
    np.testing.assert_allclose(fixed.to_numpy()[0, 1], (1.0, 2.0, 3.0, 4.0))
    write_mixed(spy.uint2(1, 0), 0xFFFFFFFF, spy.float4(9.0, 9.0, 9.0, 9.0))

    for output in indexed_outputs:
        target_component_count = spy.get_format_info(Container.format(output.target)).channel_count
        expected = np.zeros(target_component_count, dtype=np.float32)
        expected[: output.component_count] = np.arange(1, output.component_count + 1)
        actual = np.asarray(output.target.to_numpy()[0, 1]).reshape(-1)
        np.testing.assert_allclose(actual, expected, atol=1e-6, err_msg=output.name)
        unwritten = np.asarray(output.target.to_numpy()[1, 2]).reshape(-1)
        np.testing.assert_allclose(
            unwritten,
            np.zeros(target_component_count, dtype=np.float32),
            atol=1e-6,
            err_msg=f"clear {output.name}",
        )


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_output_prelude_keeps_fixed_output_named_generic_data(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    module = _fixed_generic_module(device)
    output_prelude = OutputPrelude.create(module, "IFixedGenericWrite")
    target = _indexed_target(device, 4, spy.Tensor)
    targets = {"generic_data": target}
    prelude = output_prelude.generate(targets)
    assert "generic_data(vector<uint,2> coord, vector<float,4> value)" in prelude
    assert "generic_data(uint id" not in prelude


def test_output_prelude_indexed_output_default_formats() -> None:
    assert [OutputPrelude.indexed_output_format(count) for count in range(1, 5)] == [
        spy.Format.r32_float,
        spy.Format.rg32_float,
        spy.Format.rgba32_float,
        spy.Format.rgba32_float,
    ]
    with pytest.raises(ValueError, match="one through four"):
        OutputPrelude.indexed_output_format(0)
    with pytest.raises(ValueError, match="one through four"):
        OutputPrelude.indexed_output_format(True)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES[:1])
def test_output_prelude_rejects_invalid_indexed_outputs(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    output_prelude = OutputPrelude.create(_module(device), "ITestWrite")
    scalar_target = _texture(device, spy.Format.r32_float)

    outputs = (IndexedOutput(3, "scalar", 1, scalar_target),)
    signature = output_prelude.signature({}, outputs)
    cached = output_prelude._indexed_sorted
    assert output_prelude.signature({}, outputs) == signature
    assert output_prelude._indexed_sorted is cached
    mutable_outputs = list(outputs)
    assert output_prelude.signature({}, mutable_outputs) == signature
    mutable_outputs.append(outputs[0])
    with pytest.raises(ValueError, match="Duplicate indexed output ID"):
        output_prelude.signature({}, mutable_outputs)
    assert output_prelude._indexed_input is None
    output_prelude.signature({}, outputs)
    output_prelude.signature({})
    assert output_prelude._indexed_input is None
    assert output_prelude._indexed_sorted == ()

    with pytest.raises(ValueError, match="Duplicate indexed output ID"):
        output_prelude.generate(
            {},
            [
                IndexedOutput(3, "first", 1, scalar_target),
                IndexedOutput(3, "second", 1, scalar_target),
            ],
        )
    with pytest.raises(ValueError, match="Duplicate indexed output name"):
        output_prelude.generate(
            {},
            [
                IndexedOutput(3, "same", 1, scalar_target),
                IndexedOutput(4, "same", 1, scalar_target),
            ],
        )
    with pytest.raises(ValueError, match="one through four"):
        output_prelude.generate({}, [IndexedOutput(3, "wide", 5, scalar_target)])
    with pytest.raises(TypeError, match="cannot preserve 2 float components"):
        output_prelude.generate({}, [IndexedOutput(3, "too_narrow", 2, scalar_target)])
    with pytest.raises(TypeError, match="cannot preserve 1 float components"):
        output_prelude.generate(
            {},
            [IndexedOutput(3, "integer", 1, _texture(device, spy.Format.r32_uint))],
        )


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_generates_incrementing_texture_and_tensor_outputs(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    module = _increment_module(device)
    output_prelude = OutputPrelude.create(module, "ITestIncrement")

    spec = output_prelude.specs["count"]
    assert spec.format == spy.Format.r32_uint
    assert tuple(spec.clear_value) == (0, 0, 0, 0)
    assert spec.operation == OutputOperation.increment

    texture = _texture(device, spy.Format.r32_uint)
    texture_targets = {"count": texture}
    texture_prelude = output_prelude.generate(texture_targets)
    assert (
        'public [format("r32ui")] RWTexture2D<uint> '
        "__g_falcor_OutputBlockITestIncrement_count;" in texture_prelude
    )
    assert (
        "__g_falcor_OutputBlockITestIncrement_count[coord] = "
        "__g_falcor_OutputBlockITestIncrement_count[coord] + value;" in texture_prelude
    )
    count_texture_nan = module.count_nan.prelude(texture_prelude).write(
        lambda cursor: output_prelude.bind(cursor, texture_targets)
    )
    Container.clear(texture, spec.clear_value)
    count_texture_nan(spy.uint2(0, 0), spy.float3(0.0))
    count_texture_nan(spy.uint2(0, 0), spy.float3(float("nan"), 0.0, 0.0))
    count_texture_nan(spy.uint2(0, 0), spy.float3(float("nan"), 0.0, 0.0))
    assert texture.to_numpy()[0, 0] == 2

    tensor = spy.Tensor.empty(device, shape=(2, 3), dtype="uint")
    tensor_targets = {"count": tensor}
    tensor_prelude = output_prelude.generate(tensor_targets)
    assert "public RWTensor<uint, 2> __g_falcor_OutputBlockITestIncrement_count;" in tensor_prelude
    assert (
        "__g_falcor_OutputBlockITestIncrement_count.store(int(coord.y), int(coord.x), "
        "__g_falcor_OutputBlockITestIncrement_count.load(int(coord.y), int(coord.x)) + value);"
        in tensor_prelude
    )
    count_tensor_nan = module.count_nan.prelude(tensor_prelude).write(
        lambda cursor: output_prelude.bind(cursor, tensor_targets)
    )
    Container.clear(tensor, spec.clear_value)
    count_tensor_nan(spy.uint2(0, 0), spy.float3(0.0))
    count_tensor_nan(spy.uint2(0, 0), spy.float3(float("nan"), 0.0, 0.0))
    count_tensor_nan(spy.uint2(0, 0), spy.float3(float("nan"), 0.0, 0.0))
    assert tensor.to_numpy()[0, 0] == 2


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_generates_noops_for_missing_targets(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = _module(device)

    prelude = OutputPrelude.create(module, "ITestWrite").generate({})

    assert "ParameterBlock<__falcor_OutputBlockITestWrite>" not in prelude
    assert "public RWTexture2D<vector<float,4>> color;" not in prelude
    assert (
        "public static override void color(vector<uint,2> coord, vector<float,4> value)" in prelude
    )
    assert "public static override void depth(vector<uint,2> coord, float value)" in prelude
    assert "public static bool color_enabled()" in prelude
    assert "public static bool depth_enabled()" in prelude
    assert "return false;" in prelude
    assert "export struct TestWrite : ITestWrite = __falcor_OutputImplITestWrite;" in prelude
    assert "export static const bool TestWrite_color = false;" in prelude
    assert "export static const bool TestWrite_depth = false;" in prelude
    assert "indexed_output_" not in prelude
    assert (
        "public static override void generic_data(vector<uint,2> coord, uint id, "
        "vector<float,4> value)" in prelude
    )
    assert "public static bool generic_data_enabled(uint id)" in prelude
    assert "switch (id)" in prelude


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_rejects_unknown_output(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = _module(device)
    target = _texture(device, spy.Format.rgba32_float)
    output_prelude = OutputPrelude.create(module, "ITestWrite")

    with pytest.raises(RuntimeError, match="does not define expected output function"):
        output_prelude.generate(
            {"unknown": target},
        )


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_requires_interface_name_convention(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = _module(device)

    with pytest.raises(ValueError, match="must be in the form 'IBla'"):
        OutputPrelude.create(module, "TestWriteNoPrefix")


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_accepts_explicit_concrete_type_name(
    device_type: spy.DeviceType,
):
    device = helpers.get_device(device_type)
    module = _module(device)
    target = _texture(device, spy.Format.rgba32_float)
    targets = {"color": target}

    output_prelude = OutputPrelude.create(module, "ITestWrite", "CustomTestWrite")
    prelude = output_prelude.generate(targets)

    assert "export struct CustomTestWrite : ITestWrite" in prelude
    assert "public static bool color_enabled()" in prelude
    assert "public static bool depth_enabled()" in prelude
    assert "export static const bool CustomTestWrite_color = true;" in prelude
    assert "export static const bool CustomTestWrite_depth = false;" in prelude
    assert output_prelude.signature(targets) == (
        "ITestWrite\nCustomTestWrite\n" + get_value_signature(targets)
    )


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_bind_uses_generated_block_name(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    target = _texture(device, spy.Format.rgba32_float)
    block: dict[str, object] = {}
    requested_keys: list[str] = []

    class Cursor:
        def __getitem__(self, key: str) -> dict[str, object]:
            requested_keys.append(key)
            return block

    OutputPrelude.create(_module(device), "ITestWrite").bind(Cursor(), {"color": target})

    assert requested_keys == ["__g_falcor_OutputBlockITestWrite"]
    assert block == {"color": target}


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_bind_omits_empty_parameter_block(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)

    class Cursor:
        def __getitem__(self, key: str) -> object:
            raise AssertionError(f"Unexpected parameter block lookup for {key}.")

    OutputPrelude.create(_module(device), "ITestWrite").bind(Cursor(), {})


@pytest.mark.parametrize("device_type", ALL_DEVICE_TYPES)
def test_output_prelude_with_wtensor_compiles(device_type: spy.DeviceType):
    device = helpers.get_device(device_type)
    module = _module(device)
    target = spy.Tensor.empty(device, shape=(2, 3), dtype="float")
    targets = {"depth": target}
    output_prelude = OutputPrelude.create(module, "ITestWrite")
    prelude = output_prelude.generate(targets)

    module.dummy.prelude(prelude).write(lambda cursor: output_prelude.bind(cursor, targets))()
