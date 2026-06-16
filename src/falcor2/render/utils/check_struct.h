// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/device/shader.h>

/// Helper macro to check that a C++ struct matches a Slang struct when used in structured buffers.
/// Usage:
/// ```
/// CHECK_STRUCT_BEGIN(slangModule, "SlangStructName", CppStructType);
///     CHECK_STRUCT_FIELD(field1);
///     CHECK_STRUCT_FIELD(field2);
///     ...
/// CHECK_STRUCT_END();
/// ```
/// @param slang_module Slang module containing the struct definition.
/// @param slang_type Name of the Slang struct type.
/// @param host_type C++ struct type.
#define CHECK_STRUCT_BEGIN(slang_module, slang_type, host_type)                                                        \
    do {                                                                                                               \
        FALCOR_ASSERT(slang_module != nullptr);                                                                        \
        FALCOR_ASSERT(slang_module->layout() != nullptr);                                                              \
        auto slang_type_ = slang_module->layout()->find_type_by_name("StructuredBuffer<" slang_type ">");              \
        FALCOR_ASSERT(slang_type_ != nullptr);                                                                         \
        auto slang_type_layout = slang_module->layout()->get_type_layout(slang_type_)->element_type_layout();          \
        FALCOR_ASSERT(slang_type_layout);                                                                              \
        const host_type* host_type_ptr{nullptr};                                                                       \
        {                                                                                                              \
            size_t slang_stride = slang_type_layout->stride();                                                         \
            size_t host_stride = sizeof(host_type);                                                                    \
            FALCOR_ASSERT_EQ(slang_stride, host_stride);                                                               \
        }

#define CHECK_STRUCT_FIELD(name)                                                                                       \
    {                                                                                                                  \
        size_t slang_size = slang_type_layout->find_field_by_name(#name)->type_layout()->size();                       \
        size_t host_size = sizeof(host_type_ptr->name);                                                                \
        FALCOR_ASSERT_EQ(slang_size, host_size);                                                                       \
        size_t slang_offset = slang_type_layout->find_field_by_name(#name)->offset();                                  \
        size_t host_offset = (uintptr_t)&host_type_ptr->name;                                                          \
        FALCOR_ASSERT_EQ(slang_offset, host_offset);                                                                   \
    }

#define CHECK_STRUCT_END()                                                                                             \
    }                                                                                                                  \
    while (0)
