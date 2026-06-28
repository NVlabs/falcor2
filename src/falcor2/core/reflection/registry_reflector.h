// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/reflection/class_descriptor.h"
#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/type_registry.h"

#include <functional>
#include <type_traits>

namespace falcor::reflection {

namespace detail {

// ----------------------------------------------------------------------------
// RegistryClassReflector<T>
// ----------------------------------------------------------------------------

/// Reflector that satisfies the ClassReflector concept and accumulates properties
/// into a ClassDescriptor. The descriptor is registered with the TypeRegistry
/// when the reflector is destroyed (RAII via finalize()).
template<typename T>
class RegistryClassReflector {
public:
    using target_type = T;

    RegistryClassReflector(ClassDescriptor* desc)
        : m_desc(desc)
    {
    }

    ~RegistryClassReflector() { finalize(); }

    // Movable (for return from class_()), but not copyable.
    RegistryClassReflector(const RegistryClassReflector&) = delete;
    RegistryClassReflector& operator=(const RegistryClassReflector&) = delete;

    RegistryClassReflector(RegistryClassReflector&& other) noexcept
        : m_desc(other.m_desc)
    {
        other.m_desc = nullptr;
    }

    RegistryClassReflector& operator=(RegistryClassReflector&& other) noexcept
    {
        if (this != &other) {
            finalize();
            m_desc = other.m_desc;
            other.m_desc = nullptr;
        }
        return *this;
    }

    /// Define a read-write property with getter, setter, and optional metadata.
    /// If OnChange metadata is present, wraps the setter to call the callback after assignment.
    template<typename Getter, typename Setter, typename... Extras>
    RegistryClassReflector& def_prop_rw(const char* name, Getter getter, Setter setter, Extras&&... extras)
    {
        using ReturnType = std::invoke_result_t<Getter, const T&>;
        using V = std::remove_cv_t<std::remove_reference_t<ReturnType>>;

        auto extracted = detail::extract_all<V>(std::forward<Extras>(extras)...);

        // Wrap member function pointers into std::function.
        std::function<V(const T&)> get_fn = [getter](const T& obj) -> V
        {
            return (obj.*getter)();
        };
        std::function<void(T&, const V&)> set_fn = [setter](T& obj, const V& val)
        {
            (obj.*setter)(val);
        };

        // Wrap setter with OnChange callback if present.
        if (extracted.on_change) {
            auto on_change_cb = std::move(extracted.on_change->callback);
            auto inner = std::move(set_fn);
            set_fn = [inner = std::move(inner), on_change_cb = std::move(on_change_cb)](T& obj, const V& val)
            {
                inner(obj, val);
                on_change_cb(&obj);
            };
            extracted.on_change.reset();
        }

        m_desc->add_property(
            new TypedPropertyDescriptor<V, T>(name, std::move(get_fn), std::move(set_fn), std::move(extracted))
        );
        return *this;
    }

    /// Define a read-only property with getter and optional metadata.
    template<typename Getter, typename... Extras>
    RegistryClassReflector& def_prop_ro(const char* name, Getter getter, Extras&&... extras)
    {
        using ReturnType = std::invoke_result_t<Getter, const T&>;
        using V = std::remove_cv_t<std::remove_reference_t<ReturnType>>;

        // Wrap member function pointer into std::function.
        std::function<V(const T&)> get_fn = [getter](const T& obj) -> V
        {
            return (obj.*getter)();
        };

        m_desc->add_property(new TypedPropertyDescriptor<V, T>(
            name,
            std::move(get_fn),
            typename TypedPropertyDescriptor<V, T>::Setter{}, // null setter = read-only
            std::forward<Extras>(extras)...
        ));
        return *this;
    }

    /// Define a read-write property from a member pointer, with optional metadata.
    /// If OnChange metadata is present, wraps the setter to call the callback after assignment.
    template<typename V, typename... Extras>
    RegistryClassReflector& def_rw(const char* name, V T::* member, Extras&&... extras)
    {
        auto extracted = detail::extract_all<V>(std::forward<Extras>(extras)...);

        std::function<V(const T&)> get_fn = [member](const T& obj) -> V
        {
            return obj.*member;
        };
        std::function<void(T&, const V&)> set_fn = [member](T& obj, const V& val)
        {
            obj.*member = val;
        };

        // Wrap setter with OnChange callback if present.
        if (extracted.on_change) {
            auto on_change_cb = std::move(extracted.on_change->callback);
            auto inner = std::move(set_fn);
            set_fn = [inner = std::move(inner), on_change_cb = std::move(on_change_cb)](T& obj, const V& val)
            {
                inner(obj, val);
                on_change_cb(&obj);
            };
            extracted.on_change.reset();
        }

        m_desc->add_property(
            new TypedPropertyDescriptor<V, T>(name, std::move(get_fn), std::move(set_fn), std::move(extracted))
        );
        return *this;
    }

    /// Define a read-only property from a member pointer, with optional metadata.
    template<typename V, typename... Extras>
    RegistryClassReflector& def_ro(const char* name, V T::* member, Extras&&... extras)
    {
        std::function<V(const T&)> get_fn = [member](const T& obj) -> V
        {
            return obj.*member;
        };

        m_desc->add_property(new TypedPropertyDescriptor<V, T>(
            name,
            std::move(get_fn),
            typename TypedPropertyDescriptor<V, T>::Setter{}, // null setter = read-only
            std::forward<Extras>(extras)...
        ));
        return *this;
    }

private:
    /// Register the completed ClassDescriptor with the TypeRegistry.
    /// Called once from the destructor.
    void finalize()
    {
        if (m_desc) {
            TypeRegistry::get().register_class(m_desc);
            m_desc = nullptr;
        }
    }

    ClassDescriptor* m_desc;
};

} // namespace detail

/// Register a ReflectedClass type with the TypeRegistry.
/// Creates a ClassDescriptor using T::reflected_class_name() and T::ReflectedBase,
/// then calls T::reflect() with a RegistryClassReflector to populate properties.
template<typename T>
void register_type()
{
    auto* desc = new ClassDescriptor(T::reflected_class_name(), typeid(T), typeid(typename T::ReflectedBase));
    detail::RegistryClassReflector<T> reflector(desc);
    T::reflect(reflector);
}

} // namespace falcor::reflection
