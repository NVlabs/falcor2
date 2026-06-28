// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "property_type_map.h"

#include "falcor2/core/any.h"
#include "falcor2/core/error.h"

#include <array>

namespace falcor {

namespace {

/// Get the Python type object for a C++ type T.
/// For nanobind-bound types, uses nb::type<T>().
/// Specializations below handle Python builtins.
template<typename T>
nb::object get_python_type()
{
    return nb::borrow(nb::type<T>());
}

template<>
nb::object get_python_type<bool>()
{
    return nb::borrow<nb::object>((PyObject*)&PyBool_Type);
}

template<>
nb::object get_python_type<int64_t>()
{
    return nb::borrow<nb::object>((PyObject*)&PyLong_Type);
}

template<>
nb::object get_python_type<double>()
{
    return nb::borrow<nb::object>((PyObject*)&PyFloat_Type);
}

template<>
nb::object get_python_type<std::string>()
{
    return nb::borrow<nb::object>((PyObject*)&PyUnicode_Type);
}

/// Common callbacks shared by both make_entry variants.

template<typename T>
Any to_any_impl(nb::handle obj)
{
    return Any(nb::cast<T>(obj));
}

template<typename T>
nb::object from_any_impl(const Any& a)
{
    const T* p = any_cast<T>(&a);
    FALCOR_CHECK(p != nullptr, "property_type_map: Any does not hold expected type.");
    return nb::cast(*p);
}

template<typename T>
void write_to_props_impl(Properties& props, std::string_view key, nb::handle obj)
{
    props.set<T>(key, nb::cast<T>(obj));
}

template<typename T>
nb::object read_from_props_impl(const Properties& props, std::string_view key)
{
    return nb::cast(props.get<T>(key));
}

/// matches_python: uses isinstance for all types.
template<typename T>
bool matches_python_impl(nb::handle value)
{
    return nb::isinstance<T>(value);
}

/// try_set_list: check first element via isinstance, then cast full sequence.
template<typename T>
bool try_set_list_impl(nb::handle first_elem, nb::handle seq, Properties& props, std::string_view key)
{
    if (nb::isinstance<T>(first_elem)) {
        props.set_list(key, nb::cast<std::vector<T>>(seq));
        return true;
    }
    return false;
}

/// Create a PropertyTypeMapEntry for a scalar/vector type T.
template<typename T>
PropertyTypeMapEntry make_entry(PropertyType pt)
{
    return PropertyTypeMapEntry{
        .type_info = &typeid(T),
        .property_type = pt,
        .python_type = get_python_type<T>(),
        .to_any = &to_any_impl<T>,
        .from_any = &from_any_impl<T>,
        .write_to_props = &write_to_props_impl<T>,
        .read_from_props = &read_from_props_impl<T>,
        .matches_python = &matches_python_impl<T>,
        .try_set_list = &try_set_list_impl<T>,
    };
}

} // anonymous namespace

const std::vector<PropertyTypeMapEntry>& property_type_map()
{
    // Table order is load-bearing for matches_python dispatch:
    // - bool must come before int64_t (Python bool subclasses int)
    static const std::vector<PropertyTypeMapEntry> table = {
        // clang-format off
        make_entry<bool>(PropertyType::bool_),
        make_entry<int64_t>(PropertyType::int_),
        make_entry<double>(PropertyType::float_),
        make_entry<float16_t>(PropertyType::float_),
        make_entry<int2>(PropertyType::int2),
        make_entry<int3>(PropertyType::int3),
        make_entry<int4>(PropertyType::int4),
        make_entry<uint2>(PropertyType::uint2),
        make_entry<uint3>(PropertyType::uint3),
        make_entry<uint4>(PropertyType::uint4),
        make_entry<float2>(PropertyType::float2),
        make_entry<float3>(PropertyType::float3),
        make_entry<float4>(PropertyType::float4),
        make_entry<float16_t2>(PropertyType::float2),
        make_entry<float16_t3>(PropertyType::float3),
        make_entry<float16_t4>(PropertyType::float4),
        make_entry<float3x3>(PropertyType::float3x3),
        make_entry<float4x4>(PropertyType::float4x4),
        make_entry<std::string>(PropertyType::string),
        // clang-format on
    };
    return table;
}

const PropertyTypeMapEntry* property_type_map_find(const std::type_info& ti)
{
    for (const auto& entry : property_type_map())
        if (*entry.type_info == ti)
            return &entry;
    return nullptr;
}

const PropertyTypeMapEntry* property_type_map_find(PropertyType pt)
{
    // Build a static O(1) lookup array indexed by PropertyType enum value.
    static const auto lookup = []
    {
        constexpr int max_pt = static_cast<int>(PropertyType::count_);
        std::array<const PropertyTypeMapEntry*, max_pt> arr{};
        arr.fill(nullptr);
        for (const auto& entry : property_type_map()) {
            const int index = static_cast<int>(entry.property_type);
            if (!arr[index])
                arr[index] = &entry;
        }
        return arr;
    }();

    int idx = static_cast<int>(pt);
    if (idx < 0 || idx >= static_cast<int>(lookup.size()))
        return nullptr;
    return lookup[idx];
}

const PropertyTypeMapEntry* property_type_map_find(nb::handle py_type)
{
    for (const auto& entry : property_type_map())
        if (py_type.is(entry.python_type))
            return &entry;
    return nullptr;
}

} // namespace falcor
