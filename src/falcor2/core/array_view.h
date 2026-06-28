// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <cstdint>
#include <type_traits>

namespace falcor {

template<typename T>
class ArrayView {
public:
    using Element = T;

public:
    ArrayView() = default;

    ArrayView(Element* data, size_t size, size_t byte_stride = sizeof(Element))
        : m_data(data)
        , m_size(size)
        , m_byte_stride(byte_stride)
    {
    }

    size_t size() const { return m_size; }

    Element& operator[](size_t index) { return *get_ptr(index); };

    const Element& operator[](size_t index) const { return *get_ptr(index); };

    Element* get_ptr(size_t index) { return (Element*)(((uint8_t*)m_data) + index * m_byte_stride); };

    const Element* get_ptr(size_t index) const
    {
        return (const Element*)(((const uint8_t*)m_data) + index * m_byte_stride);
    };

private:
    Element* m_data = nullptr;
    size_t m_size = 0;
    size_t m_byte_stride = sizeof(Element);
};


} // namespace falcor
