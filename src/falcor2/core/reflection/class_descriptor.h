// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/property_range.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/any.h"

#include <atomic>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace falcor::reflection {

class TypeRegistry;

// ----------------------------------------------------------------------------
// ClassDescriptor
// ----------------------------------------------------------------------------

/// Describes a reflected class: its name, type, base class, and properties.
///
/// ClassDescriptor owns its PropertyDescriptor instances (raw pointer ownership
/// so that properties() can return std::span directly over the storage).
///
/// The base() pointer is resolved lazily on first access via TypeRegistry::get().find()
/// and cached. Returns null when the base class is not registered.
class FALCOR_API ClassDescriptor {
public:
    /// Construct a ClassDescriptor.
    /// @param name         Human-readable class name (from class_<T, Base>("Name")).
    /// @param type_info    typeid(T) of the described class.
    /// @param base_type_info typeid(Base) of the base class, or typeid(void) if none.
    ClassDescriptor(std::string name, const std::type_info& type_info, const std::type_info& base_type_info);

    ~ClassDescriptor();

    /// Class name as provided to class_<T, Base>("Name").
    std::string_view name() const { return m_name; }

    /// typeid of the described class.
    const std::type_info& type() const { return m_type_info; }

    /// typeid of the base class (typeid(void) if none).
    const std::type_info& base_type() const { return m_base_type_info; }

    /// Pointer to the base class's ClassDescriptor, or null if unregistered.
    /// Resolved lazily on first access via TypeRegistry.
    const ClassDescriptor* base() const;

    /// Own properties only (not including base class properties).
    std::span<const PropertyDescriptor* const> properties() const { return {m_properties.data(), m_properties.size()}; }

    /// Range over just own properties.
    PropertyRange property_range() const { return PropertyRange::own(*this); }

    /// Range over all properties in the registered inheritance chain (base first).
    PropertyRange all_property_range() const { return PropertyRange::all(*this); }

    /// Find a property by name across own + registered base chain.
    const PropertyDescriptor* find_property(std::string_view name) const;

    /// Type-erased get: look up property by name and return its value.
    Any get_any(const void* instance, std::string_view name) const;

    /// Type-erased set: look up property by name and set its value.
    void set_any(void* instance, std::string_view name, const Any& value) const;

    /// Write all properties (base first) into a Properties object.
    void write_to_properties(const void* instance, Properties& props) const;

    /// Read all properties (base first) from a Properties object.
    void read_from_properties(void* instance, const Properties& props) const;

    /// Reset all properties that have defaults (base first).
    void reset(void* instance) const;

    /// Append a property descriptor. Takes ownership.
    /// Called by RegistryClassReflector during construction.
    void add_property(PropertyDescriptor* prop);

private:
    /// Resolve the base pointer (called once, then cached).
    void resolve_base() const;

    std::string m_name;
    const std::type_info& m_type_info;
    const std::type_info& m_base_type_info;

    /// Owned property descriptors.
    std::vector<PropertyDescriptor*> m_properties;
    /// Name -> index into m_properties.
    /// @warning Key points into PropertyDescriptor::m_name and expects them not to be copied/moved.
    std::unordered_map<std::string_view, size_t> m_property_index;

    /// Lazily resolved base class descriptor.
    mutable std::atomic<const ClassDescriptor*> m_base{nullptr};
    mutable std::atomic<bool> m_base_resolved{false};

    FALCOR_NON_COPYABLE_AND_MOVABLE(ClassDescriptor);
};

} // namespace falcor::reflection
