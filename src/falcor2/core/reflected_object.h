// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/fwd.h"
#include "falcor2/core/any.h"
#include "falcor2/core/reflection/property_descriptor.h"

#include <sgl/core/object.h>

namespace falcor {

/// Base class for objects with reflection-based property serialization.
///
/// Serialization is driven by the TypeRegistry: properties() and set_properties()
/// look up the ClassDescriptor for the concrete type and delegate to
/// ClassDescriptor::write_to_properties / read_from_properties, which walk the inheritance chain
/// automatically. Dynamic per-instance properties are handled separately
/// via the dynamic_properties() accessor.
class FALCOR_API ReflectedObject : public Object {
    FALCOR_OBJECT(ReflectedObject)
public:
    virtual ~ReflectedObject() = default;

    virtual const reflection::ClassDescriptor& class_descriptor() const = 0;

    virtual Properties properties() const;
    virtual void set_properties(const Properties& props);

    /// Access per-instance dynamic properties, or nullptr if this object has none.
    /// Objects that own dynamic properties (e.g. MDL materials) override this.
    virtual reflection::DynamicPropertySet* dynamic_properties() { return nullptr; }
    virtual const reflection::DynamicPropertySet* dynamic_properties() const { return nullptr; }

    /// Find a property descriptor by name.
    /// Searches static properties (via ClassDescriptor) first, then dynamic properties.
    /// Returns nullptr if no property with the given name exists.
    const reflection::PropertyDescriptor* find_property(std::string_view name) const;

    /// Check if a property with the given name exists.
    bool has_property(std::string_view name) const;

    /// Type-erased get: returns the current value of the named property wrapped in Any.
    /// Throws if the property does not exist.
    Any get_property(std::string_view name) const;

    /// Type-erased set: sets the named property from an Any value.
    /// Throws if the property does not exist, is read-only, or if the Any holds the wrong type.
    void set_property(std::string_view name, const Any& value);

    /// Typed get: returns the current value of the named property.
    /// Throws if the property does not exist or holds a different type.
    template<typename T>
    T get_property(std::string_view name) const
    {
        const reflection::PropertyDescriptor* prop = find_property(name);
        FALCOR_CHECK(prop != nullptr, "Property \"{}\" not found.", name);
        return prop->get<T>(this);
    }

    /// Typed set: sets the named property value directly.
    /// Throws if the property does not exist, is read-only, or holds a different type.
    template<typename T>
    void set_property(std::string_view name, const T& value)
    {
        const reflection::PropertyDescriptor* prop = find_property(name);
        FALCOR_CHECK(prop != nullptr, "Property \"{}\" not found.", name);
        prop->set<T>(this, value);
    }
};

namespace reflection {
FALCOR_API const reflection::ClassDescriptor& get_class_descriptor(const std::type_info& type);
}

/// Macro to declare a reflected object class.
/// The type must be registered with the TypeRegistry (via register_type<T>())
/// for properties() / set_properties() to work.
/// @param name The class name.
/// @param base The reflected base class (use void for root reflected types).
#define FALCOR_REFLECTED_OBJECT(name, base)                                                                            \
    FALCOR_OBJECT(name)                                                                                                \
public:                                                                                                                \
    using ReflectedBase = base;                                                                                        \
    static constexpr const char* reflected_class_name()                                                                \
    {                                                                                                                  \
        return #name;                                                                                                  \
    }                                                                                                                  \
    static const reflection::ClassDescriptor& static_class_descriptor()                                                \
    {                                                                                                                  \
        static const reflection::ClassDescriptor& desc = ::falcor::reflection::get_class_descriptor(typeid(name));     \
        return desc;                                                                                                   \
    }                                                                                                                  \
    const reflection::ClassDescriptor& class_descriptor() const override                                               \
    {                                                                                                                  \
        return static_class_descriptor();                                                                              \
    }

} // namespace falcor
