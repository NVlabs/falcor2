// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/any.h"
#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/class_descriptor.h"

#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace falcor::reflection {

// ----------------------------------------------------------------------------
// DynamicPropertySet
// ----------------------------------------------------------------------------

/// Per-instance storage for dynamic properties, using the same
/// PropertyDescriptor interface as static (TypedPropertyDescriptor) properties.
///
/// Internally stores StoredPropertyDescriptor<T> instances via the
/// PropertyDescriptor* base pointer. Consumer code (UI, serialization)
/// treats all descriptors uniformly.
class DynamicPropertySet {
public:
    DynamicPropertySet() = default;

    ~DynamicPropertySet()
    {
        for (auto* p : m_properties)
            delete p;
    }

    /// Add a dynamic property with an initial value and optional metadata.
    /// Throws if a property with the same name already exists in this set.
    template<typename T, typename... Extras>
    void add_property(std::string_view name, T initial_value, bool read_only = false, Extras&&... extras)
    {
        FALCOR_CHECK(
            m_property_index.find(name) == m_property_index.end(),
            "DynamicPropertySet: property \"{}\" already exists.",
            name
        );

        auto* desc = new StoredPropertyDescriptor<T>(
            std::string(name),
            std::move(initial_value),
            read_only,
            std::forward<Extras>(extras)...
        );
        m_property_index[desc->name()] = m_properties.size();
        m_properties.push_back(desc);
        ++m_generation;
    }

    /// Add a dynamic property, also checking for name collisions with
    /// the owning class's static ClassDescriptor properties.
    /// Throws if the name collides with either set.
    template<typename T, typename... Extras>
    void add_property_checked(
        const ClassDescriptor& class_desc,
        std::string_view name,
        T initial_value,
        bool read_only = false,
        Extras&&... extras
    )
    {
        FALCOR_CHECK(
            class_desc.find_property(name) == nullptr,
            "DynamicPropertySet: property \"{}\" collides with a static property on \"{}\".",
            name,
            class_desc.name()
        );
        add_property<T>(std::string(name), std::move(initial_value), read_only, std::forward<Extras>(extras)...);
    }

    /// Remove a dynamic property by name.
    /// No-op if the property does not exist.
    /// Preserves insertion order of remaining properties.
    void remove_property(std::string_view name)
    {
        auto it = m_property_index.find(name);
        if (it == m_property_index.end())
            return;

        size_t index = it->second;
        delete m_properties[index];

        // Erase from vector (preserves order, O(n) but sets are typically small).
        m_properties.erase(m_properties.begin() + index);
        m_property_index.erase(it);

        // Re-index entries after the removed slot.
        for (size_t i = index; i < m_properties.size(); ++i)
            m_property_index[m_properties[i]->name()] = i;
        ++m_generation;
    }

    /// Find a property by name. Returns null if not found.
    const PropertyDescriptor* find_property(std::string_view name) const
    {
        auto it = m_property_index.find(name);
        if (it == m_property_index.end())
            return nullptr;
        return m_properties[it->second];
    }

    /// Get a property by name. Throws if not found.
    const PropertyDescriptor* get_property(std::string_view name) const
    {
        const PropertyDescriptor* prop = find_property(name);
        FALCOR_CHECK(prop != nullptr, "DynamicPropertySet: property \"{}\" not found.", name);
        return prop;
    }

    /// Get a mutable property by name. Throws if not found.
    PropertyDescriptor* get_property(std::string_view name)
    {
        auto it = m_property_index.find(name);
        FALCOR_CHECK(it != m_property_index.end(), "DynamicPropertySet: property \"{}\" not found.", name);
        return m_properties[it->second];
    }

    /// Check if a property with the given name exists.
    bool has_property(std::string_view name) const { return find_property(name) != nullptr; }

    /// Type-erased get by name.
    Any get_any(std::string_view name) const { return get_property(name)->get_any(m_instance); }

    /// Typed get by name. Returns a copy of the stored value.
    /// Throws if the property does not exist or has a different type.
    template<typename T>
    T get(std::string_view name) const
    {
        return get_property(name)->get<T>(m_instance);
    }

    /// Typed get by name returning a const reference to the stored value.
    /// Only valid for DynamicPropertySet (StoredPropertyDescriptor) properties.
    /// Throws if the property does not exist or has a different type.
    template<typename T>
    const T& get_ref(std::string_view name) const
    {
        auto* stored = dynamic_cast<const StoredPropertyDescriptor<T>*>(get_property(name));
        FALCOR_CHECK(stored != nullptr, "DynamicPropertySet: property \"{}\" type mismatch.", name);
        return stored->value();
    }

    /// Type-erased set by name.
    void set_any(std::string_view name, const Any& value) { get_property(name)->set_any(m_instance, value); }

    /// Typed set by name.
    template<typename T>
    void set(std::string_view name, const T& value)
    {
        get_property(name)->set<T>(m_instance, value);
    }

    /// Set the owner instance pointer, forwarded to OnChange and UIEnableIf callbacks.
    void set_instance(void* instance) { m_instance = instance; }

    /// All dynamic property descriptors.
    std::span<const PropertyDescriptor* const> descriptors() const
    {
        return {m_properties.data(), m_properties.size()};
    }

    /// Number of dynamic properties.
    size_t size() const { return m_properties.size(); }

    /// True if no dynamic properties.
    bool empty() const { return m_properties.empty(); }

    /// Generation counter, incremented on every structural change
    /// (add_property, remove_property, clear). Can be used to invalidate caches.
    uint64_t generation() const { return m_generation; }

    /// Write all dynamic properties into a Properties object.
    void write_to_properties(Properties& props) const
    {
        for (auto* p : m_properties) {
            if (p->is_serializable_to_properties())
                p->write_to_properties(m_instance, props);
        }
    }

    /// Read dynamic properties from a Properties object.
    void read_from_properties(const Properties& props)
    {
        for (auto* p : m_properties) {
            if (p->is_serializable_to_properties() && !p->is_read_only())
                p->read_from_properties(m_instance, props);
        }
    }

    /// Reset all dynamic properties to their default values.
    void reset()
    {
        for (auto* p : m_properties)
            p->reset(m_instance);
    }

    /// Reset a single property to its default value by name.
    /// Throws if the property does not exist.
    void reset(std::string_view name) { get_property(name)->reset(m_instance); }

    /// Remove all dynamic properties, freeing all descriptors.
    void clear()
    {
        for (auto* p : m_properties)
            delete p;
        m_properties.clear();
        m_property_index.clear();
        ++m_generation;
    }

    /// Check if the named property's current value equals the same-named
    /// property in another Properties object.
    /// Returns false if the property does not exist in this set or in other.
    bool is_equal_property(std::string_view name, const Properties& other) const
    {
        const PropertyDescriptor* prop = find_property(name);
        if (!prop || !prop->is_serializable_to_properties())
            return false;
        Properties tmp;
        prop->write_to_properties(m_instance, tmp);
        return tmp.is_equal_property(name, other);
    }

    /// Check if a property is at its default value.
    /// Returns false if the property does not exist.
    bool is_default(std::string_view name) const
    {
        const PropertyDescriptor* prop = find_property(name);
        if (!prop)
            return false;
        return prop->is_default(m_instance);
    }

private:
    /// Owned property descriptors (StoredPropertyDescriptor<T> instances).
    std::vector<PropertyDescriptor*> m_properties;
    /// Name -> index into m_properties.
    /// @warning Key points into PropertyDescriptor::m_name and expects them not to be copied/moved.
    std::unordered_map<std::string_view, size_t> m_property_index;
    /// Owner instance pointer forwarded to OnChange and UIEnableIf callbacks.
    void* m_instance{nullptr};
    /// Generation counter incremented on structural changes.
    uint64_t m_generation{0};

    FALCOR_NON_COPYABLE_AND_MOVABLE(DynamicPropertySet);
};

} // namespace falcor::reflection
