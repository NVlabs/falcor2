// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Portions of this file are derived from Mitsuba 3.
// Copyright (c) Mitsuba contributors.
// Licensed under the BSD 3-Clause License.
// See LICENSES/mitsuba3.txt for the full license text.

#pragma once

#include "falcor2/core/macros.h"

#include <cstring>
#include <typeinfo>
#include <utility>

namespace falcor {

/// Type-erased storage for arbitrary objects.
///
/// This class resembles `std::any` but supports advanced customization by exposing the underlying
/// type-erased storage implementation @ref Any::Base.
/// The @ref Properties class uses @ref Any when it needs to store things that aren't part of the
/// supported set of property types.
/// The main reason for using @ref Any (as opposed to the builtin `std::any`) is to enable seamless
/// use in Python bindings, which requires access to @ref Any::Base.
///
/// Instances of this type are copyable with reference semantics.
/// The class is not thread-safe(i.e., it may not be copied concurrently).
class Any {
public:
    // Type-erased storage backing the Any class
    struct Base {
        mutable size_t ref_count = 1;
        virtual ~Base() = default;
        virtual const std::type_info& type() const = 0;
        virtual void* ptr() = 0;

        void inc_ref() const { ++ref_count; }
        bool dec_ref() const { return --ref_count == 0; }
    };

    Any() = default;
    Any(const Any& other)
        : m_storage(other.m_storage)
    {
        if (m_storage)
            m_storage->inc_ref();
    }
    Any(Any&& other) noexcept
        : m_storage(other.m_storage)
    {
        other.m_storage = nullptr;
    }
    template<typename T>
    explicit Any(T&& value)
        : m_storage(new Storage<std::decay_t<T>>(std::forward<T>(value)))
    {
    }
    ~Any() { release(); }

    Any& operator=(const Any& other)
    {
        if (this != &other) {
            release();
            m_storage = other.m_storage;
            if (m_storage)
                m_storage->inc_ref();
        }
        return *this;
    }

    Any& operator=(Any&& other) noexcept
    {
        if (this != &other) {
            release();
            m_storage = other.m_storage;
            other.m_storage = nullptr;
        }
        return *this;
    }

    void* data() { return m_storage ? m_storage->ptr() : nullptr; }
    const void* data() const { return m_storage ? m_storage->ptr() : nullptr; }
    const std::type_info& type() const { return m_storage ? m_storage->type() : typeid(void); }

    bool has_value() const { return m_storage != nullptr; }

    template<typename T>
    friend T* any_cast(Any*);
    template<typename T>
    friend const T* any_cast(const Any*);
    bool operator==(const Any& other) const { return data() == other.data(); }
    bool operator!=(const Any& other) const { return data() != other.data(); }

    // Constructor that accepts a Base pointer. For relatively advanced use cases that repurpose the Any class
    Any(Base* base)
        : m_storage(base)
    {
    }

    /// Access the underlying type-erased storage.
    Base* base_ptr() { return m_storage; }
    const Base* base_ptr() const { return m_storage; }

protected:
    template<typename T>
    struct Storage : Base {
        T value;
        template<typename U>
        Storage(U&& u) noexcept
            : value(std::forward<U>(u))
        {
        }
        const std::type_info& type() const override { return typeid(T); }
        void* ptr() override { return &value; }
    };

    void release()
    {
        if (m_storage && m_storage->dec_ref()) {
            delete m_storage;
            m_storage = nullptr;
        }
    }

    Base* m_storage = nullptr;
};

namespace detail {
inline bool compare_typeid(const std::type_info& a, const std::type_info& b)
{
    return &a == &b || std::strcmp(a.name(), b.name()) == 0;
}
} // namespace detail

template<typename T>
T* any_cast(Any* a)
{
    return (a && a->m_storage && detail::compare_typeid(a->m_storage->type(), typeid(T)))
        ? static_cast<T*>(a->m_storage->ptr())
        : nullptr;
}

template<typename T>
const T* any_cast(const Any* a)
{
    return (a && a->m_storage && detail::compare_typeid(a->m_storage->type(), typeid(T)))
        ? static_cast<const T*>(a->m_storage->ptr())
        : nullptr;
}

} // namespace falcor
