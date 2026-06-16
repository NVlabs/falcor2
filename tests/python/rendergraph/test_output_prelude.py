# SPDX-License-Identifier: Apache-2.0

import pytest
import slangpy as spy
from slangpy.core.native import get_value_signature

import falcor2.testing.helpers as helpers
from falcor2.rendergraph import OutputPrelude

ALL_DEVICE_TYPES = helpers.DEFAULT_DEVICE_TYPES

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


def _module(device: spy.Device) -> spy.Module:
    return spy.Module(device.load_module_from_source("output_prelude_test", TEST_MODULE))


def _int_module(device: spy.Device) -> spy.Module:
    return spy.Module(device.load_module_from_source("output_prelude_int_test", TEST_INT_MODULE))


def _texture(device: spy.Device, format: spy.Format) -> spy.Texture:
    return device.create_texture(
        type=spy.TextureType.texture_2d,
        format=format,
        width=3,
        height=2,
        mip_count=1,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
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
