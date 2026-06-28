// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/object.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflection/metadata.h"
#include "falcor2/core/reflection/property_descriptor.h"

#include <type_traits>

namespace falcor::reflection {

// These reflectors operate on a single class's reflect() definition and do not walk the inheritance chain.
// To serialize the full property set including base classes, use ClassDescriptor::write_to_properties() and
// ClassDescriptor::read_from_properties() via the TypeRegistry instead, which handle base-class traversal
// automatically.

namespace detail {

// ----------------------------------------------------------------------------
// PropertiesSerializer
// ----------------------------------------------------------------------------

/// ClassReflector that serializes an object's properties into a Properties object.
/// @warning This does not handle properties defined in the base class.
template<typename T>
class PropertiesSerializer {
public:
    using target_type = T;

    PropertiesSerializer(const T& obj, Properties& props)
        : m_obj(obj)
        , m_props(props)
    {
    }
    /// Serialize a read-write property: reads via getter and stores into Properties.
    /// Extra arguments (doc strings, metadata) are silently ignored.
    template<typename Getter, typename Setter, typename... Extras>
    PropertiesSerializer& def_prop_rw(const char* name, Getter getter, Setter /*setter*/, Extras&&... /*extras*/)
    {
        using ReturnType = std::invoke_result_t<Getter, const T&>;
        using ValueType = std::remove_cv_t<std::remove_reference_t<ReturnType>>;
        static_assert(PropertyValueType<ValueType>, "Property type must be serializable to Properties.");

        m_props.set<ValueType>(name, (m_obj.*getter)());
        return *this;
    }

    /// Serialize a read-only property: reads via getter and stores into Properties.
    /// Extra arguments (doc strings, metadata) are silently ignored.
    template<typename Getter, typename... Extras>
    PropertiesSerializer& def_prop_ro(const char* name, Getter getter, Extras&&... /*extras*/)
    {
        using ReturnType = std::invoke_result_t<Getter, const T&>;
        using ValueType = std::remove_cv_t<std::remove_reference_t<ReturnType>>;
        static_assert(PropertyValueType<ValueType>, "Property type must be serializable to Properties.");

        m_props.set<ValueType>(name, (m_obj.*getter)());
        return *this;
    }

    /// Serialize a read-write property from a member pointer.
    /// Extra arguments (doc strings, metadata) are silently ignored.
    template<typename V, typename... Extras>
    PropertiesSerializer& def_rw(const char* name, V T::* member, Extras&&... /*extras*/)
    {
        static_assert(PropertyValueType<V>, "Property type must be serializable to Properties.");

        m_props.set<V>(name, m_obj.*member);
        return *this;
    }

    /// Serialize a read-only property from a member pointer.
    /// Extra arguments (doc strings, metadata) are silently ignored.
    template<typename V, typename... Extras>
    PropertiesSerializer& def_ro(const char* name, V T::* member, Extras&&... /*extras*/)
    {
        static_assert(PropertyValueType<V>, "Property type must be serializable to Properties.");

        m_props.set<V>(name, m_obj.*member);
        return *this;
    }

private:
    const T& m_obj;
    Properties& m_props;
};

// ----------------------------------------------------------------------------
// PropertiesDeserializer
// ----------------------------------------------------------------------------

/// ClassReflector that deserializes properties from a Properties object onto an object.
/// @warning This does not handle properties defined in the base class.
template<typename T>
class PropertiesDeserializer {
public:
    using target_type = T;

    PropertiesDeserializer(T& obj, const Properties& props)
        : m_obj(obj)
        , m_props(props)
    {
    }

    /// Deserialize a read-write property: reads from Properties (if present) and applies via setter.
    /// If OnChange metadata is present, fires the callback after assignment.
    template<typename Getter, typename Setter, typename... Extras>
    PropertiesDeserializer& def_prop_rw(const char* name, Getter /*getter*/, Setter setter, Extras&&... extras)
    {
        using ReturnType = std::invoke_result_t<Getter, const T&>;
        using ValueType = std::remove_cv_t<std::remove_reference_t<ReturnType>>;
        static_assert(PropertyValueType<ValueType>, "Property type must be deserializable from Properties.");

        if (m_props.has_property(name)) {
            (m_obj.*setter)(m_props.get<ValueType>(name));

            // Fire OnChange callback if present.
            if constexpr (detail::has_on_change_v<Extras...>) {
                auto extracted = detail::extract_all<ValueType>(std::forward<Extras>(extras)...);
                if (extracted.on_change) {
                    extracted.on_change->callback(&m_obj);
                }
            }
        }
        ((void)extras, ...);
        return *this;
    }

    /// Read-only properties are skipped during deserialization.
    /// Extra arguments (doc strings, metadata) are silently ignored.
    template<typename Getter, typename... Extras>
    PropertiesDeserializer& def_prop_ro(const char* /*name*/, Getter /*getter*/, Extras&&... /*extras*/)
    {
        return *this;
    }

    /// Deserialize a read-write property from a member pointer.
    /// If OnChange metadata is present, fires the callback after assignment.
    template<typename V, typename... Extras>
    PropertiesDeserializer& def_rw(const char* name, V T::* member, Extras&&... extras)
    {
        static_assert(PropertyValueType<V>, "Property type must be deserializable from Properties.");

        if (m_props.has_property(name)) {
            m_obj.*member = m_props.get<V>(name);

            // Fire OnChange callback if present.
            if constexpr (detail::has_on_change_v<Extras...>) {
                auto extracted = detail::extract_all<V>(std::forward<Extras>(extras)...);
                if (extracted.on_change) {
                    extracted.on_change->callback(&m_obj);
                }
            }
        }
        return *this;
    }

    /// Read-only properties from member pointers are skipped during deserialization.
    template<typename V, typename... Extras>
    PropertiesDeserializer& def_ro(const char* /*name*/, V T::* /*member*/, Extras&&... /*extras*/)
    {
        return *this;
    }

private:
    T& m_obj;
    const Properties& m_props;
};

} // namespace detail

/// Serialize the properties of a reflected object into a Properties object.
/// @warning This does not handle properties defined in the base class.
template<typename T>
void serialize_properties(const T& obj, Properties& props)
{
    detail::PropertiesSerializer<T> reflector(obj, props);
    T::reflect(reflector);
}

/// Deserialize the properties of a reflected object from a Properties object.
/// @warning This does not handle properties defined in the base class.
template<typename T>
void deserialize_properties(T& obj, const Properties& props)
{
    detail::PropertiesDeserializer<T> reflector(obj, props);
    T::reflect(reflector);
}

} // namespace falcor::reflection
