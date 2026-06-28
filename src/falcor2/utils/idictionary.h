// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/object.h"

#include <sgl/device/resource.h>
#include <sgl/device/sampler.h>

#include <string_view>
#include <vector>
#include <span>

namespace falcor {

#define FALCOR_DICTIONARY_TYPES                                                                                        \
    X(int)                                                                                                             \
    X(int1)                                                                                                            \
    X(int2)                                                                                                            \
    X(int3)                                                                                                            \
    X(int4)                                                                                                            \
    X(uint64_t)                                                                                                        \
    X(uint)                                                                                                            \
    X(uint1)                                                                                                           \
    X(uint2)                                                                                                           \
    X(uint3)                                                                                                           \
    X(uint4)                                                                                                           \
    X(float)                                                                                                           \
    X(float1)                                                                                                          \
    X(float2)                                                                                                          \
    X(float3)                                                                                                          \
    X(float4)                                                                                                          \
    X(bool)                                                                                                            \
    X(bool1)                                                                                                           \
    X(bool2)                                                                                                           \
    X(bool3)                                                                                                           \
    X(bool4)                                                                                                           \
    X(float2x2)                                                                                                        \
    X(float2x4)                                                                                                        \
    X(float3x3)                                                                                                        \
    X(float3x4)                                                                                                        \
    X(float4x4)                                                                                                        \
    X(ref<sgl::Buffer>) X(ref<sgl::Sampler>) X(ref<sgl::Texture>) X(sgl::DescriptorHandle)


class IDictionary {
public:
    class Accessor {
    public:
        template<typename T>
        Accessor& operator=(const T& rhs)
        {
            if constexpr (requires(T v, IDictionary& d) { v.to_dictionary(d); }) {
                m_wrapper->push(m_name);
                // This allows `to_dictionary` to be non-const and still permit the usage pattern of:
                // dict["name"] = object_with_to_dictionary;
                const_cast<T&>(rhs).to_dictionary(*m_wrapper);
                m_wrapper->pop();
            } else if constexpr (std::is_enum_v<T>) {
                m_wrapper->set(m_name, std::underlying_type_t<T>(rhs));
            } else {
                m_wrapper->set(m_name, rhs);
            }

            return *this;
        }

    private:
        Accessor(std::string_view name, IDictionary* wrapper)
            : m_name(name)
            , m_wrapper(wrapper)
        {
        }

        std::string_view m_name;
        IDictionary* m_wrapper;

        friend class IDictionary;
    };

public:
    Accessor operator[](std::string_view name) { return Accessor(name, this); }

    virtual void push(std::string_view nested_name) = 0;
    virtual void pop() = 0;

#define X(type_)                                                                                                       \
    virtual void set(std::string_view name, const type_& value) = 0;                                                   \
    virtual void set(std::string_view name, std::span<const type_> value) = 0;
    FALCOR_DICTIONARY_TYPES
#undef X
};

} // namespace falcor
