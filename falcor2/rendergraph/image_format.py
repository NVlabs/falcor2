# SPDX-License-Identifier: Apache-2.0
"""Helpers for translating between image formats and Slang/Python type descriptions."""

from __future__ import annotations

import slangpy as spy

SCALAR_TYPE_TO_FORMAT = (
    {},
    {
        spy.TypeReflection.ScalarType.float16: spy.Format.r16_float,
        spy.TypeReflection.ScalarType.float32: spy.Format.r32_float,
        spy.TypeReflection.ScalarType.int32: spy.Format.r32_sint,
        spy.TypeReflection.ScalarType.uint32: spy.Format.r32_uint,
    },
    {
        spy.TypeReflection.ScalarType.float16: spy.Format.rg16_float,
        spy.TypeReflection.ScalarType.float32: spy.Format.rg32_float,
        spy.TypeReflection.ScalarType.int32: spy.Format.rg32_sint,
        spy.TypeReflection.ScalarType.uint32: spy.Format.rg32_uint,
    },
    {
        spy.TypeReflection.ScalarType.float32: spy.Format.rgb32_float,
        spy.TypeReflection.ScalarType.int32: spy.Format.rgb32_sint,
        spy.TypeReflection.ScalarType.uint32: spy.Format.rgb32_uint,
    },
    {
        spy.TypeReflection.ScalarType.float16: spy.Format.rgba16_float,
        spy.TypeReflection.ScalarType.float32: spy.Format.rgba32_float,
        spy.TypeReflection.ScalarType.int32: spy.Format.rgba32_sint,
        spy.TypeReflection.ScalarType.uint32: spy.Format.rgba32_uint,
    },
)

UINT_FORMATS = frozenset(
    (
        spy.Format.r8_uint,
        spy.Format.rg8_uint,
        spy.Format.rgba8_uint,
        spy.Format.r16_uint,
        spy.Format.rg16_uint,
        spy.Format.rgba16_uint,
        spy.Format.r32_uint,
        spy.Format.rg32_uint,
        spy.Format.rgb32_uint,
        spy.Format.rgba32_uint,
        spy.Format.r64_uint,
        spy.Format.rgb10a2_uint,
    )
)
SINT_FORMATS = frozenset(
    (
        spy.Format.r8_sint,
        spy.Format.rg8_sint,
        spy.Format.rgba8_sint,
        spy.Format.r16_sint,
        spy.Format.rg16_sint,
        spy.Format.rgba16_sint,
        spy.Format.r32_sint,
        spy.Format.rg32_sint,
        spy.Format.rgb32_sint,
        spy.Format.rgba32_sint,
        spy.Format.r64_sint,
    )
)
HALF_FORMATS = frozenset(
    (
        spy.Format.r16_float,
        spy.Format.rg16_float,
        spy.Format.rgba16_float,
    )
)

FORMAT_TO_TYPENAME = {
    spy.Format.r16_float: "vector<half,1>",
    spy.Format.rg16_float: "vector<half,2>",
    spy.Format.rgba16_float: "vector<half,4>",
    spy.Format.r32_float: "vector<float,1>",
    spy.Format.rg32_float: "vector<float,2>",
    spy.Format.rgb32_float: "vector<float,3>",
    spy.Format.rgba32_float: "vector<float,4>",
    spy.Format.r32_sint: "vector<int,1>",
    spy.Format.rg32_sint: "vector<int,2>",
    spy.Format.rgb32_sint: "vector<int,3>",
    spy.Format.rgba32_sint: "vector<int,4>",
    spy.Format.r32_uint: "vector<uint,1>",
    spy.Format.rg32_uint: "vector<uint,2>",
    spy.Format.rgb32_uint: "vector<uint,3>",
    spy.Format.rgba32_uint: "vector<uint,4>",
}


def slangtype_to_format(dtype: spy.reflection.SlangType) -> spy.Format:
    """Infer an image format from a scalar or vector Slang type."""
    if isinstance(dtype, spy.reflection.ScalarType):
        return SCALAR_TYPE_TO_FORMAT[1][dtype.slang_scalar_type]
    if isinstance(dtype, spy.reflection.VectorType):
        return SCALAR_TYPE_TO_FORMAT[dtype.num_elements][dtype.slang_scalar_type]
    raise ValueError(f"Unsupported Slang type for format inference: {dtype}")


def format_to_typename(format_value: spy.Format) -> str:
    """Return the Slang type name used to allocate a tensor for ``format_value``."""
    return FORMAT_TO_TYPENAME[format_value]


def channel_count(format_value: spy.Format) -> int:
    """Return the channel count for a validated image format."""
    fi = spy.get_format_info(format_value)  # validate format
    return fi.channel_count
