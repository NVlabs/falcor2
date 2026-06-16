// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/any.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/reflection/metadata.h"

#include <concepts>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <typeinfo>
#include <vector>

namespace falcor::reflection {

// ----------------------------------------------------------------------------
// PropertyDescriptor (abstract base)
// ----------------------------------------------------------------------------

/// Abstract base class for property descriptors.
///
/// Provides a type-erased interface for accessing property values on instances,
/// querying metadata (default value, value range, doc string), and
/// serializing/deserializing via the Properties system.
///
/// Two concrete subclasses exist:
/// - TypedPropertyDescriptor<T, OwnerT>: for static C++ properties (member-function-pointer based)
/// - StoredPropertyDescriptor<T>: for dynamic per-instance properties (value-storing)
class PropertyDescriptor {
public:
    virtual ~PropertyDescriptor() = default;

    /// Property name.
    std::string_view name() const { return m_name; }

    /// C++ typeid of the property value type.
    virtual const std::type_info& type() const = 0;

    /// True if the property has no setter.
    bool is_read_only() const { return m_read_only; }

    /// True if the property has a default value and the current value equals it.
    /// Returns false if no default is set.
    virtual bool is_default(const void* instance) const = 0;

    /// Type-erased read: returns the current value wrapped in Any.
    virtual Any get_any(const void* instance) const = 0;

    /// Type-erased write: sets the property value from an Any.
    /// Throws if read-only or if Any holds wrong type.
    virtual void set_any(void* instance, const Any& value) const = 0;

    /// Typed read: returns the current value, bypassing Any.
    /// Throws if the property holds a different type.
    template<typename T>
    T get(const void* instance) const
    {
        FALCOR_CHECK(type() == typeid(T), "Property \"{}\": type mismatch in get<T>().", m_name);
        T result{};
        get_value(instance, &result);
        return result;
    }

    /// Typed write: sets the property value directly, bypassing Any.
    /// Throws if read-only or if the property holds a different type.
    template<typename T>
    void set(void* instance, const T& value) const
    {
        FALCOR_CHECK(type() == typeid(T), "Property \"{}\": type mismatch in set<T>().", m_name);
        set_value(instance, &value);
    }

    /// Read an enum property as int64_t.
    /// Throws if the property is not an enum.
    virtual int64_t get_enum_as_int64(const void* instance) const = 0;

    /// Set an enum property from int64_t.
    /// Throws if read-only or if the property is not an enum.
    virtual void set_enum_from_int64(void* instance, int64_t value) const = 0;

    /// True if the property's value type can be written to / read from Properties.
    virtual bool is_serializable_to_properties() const = 0;

    /// Write the property value into a Properties object using the property's name as key.
    /// Throws if the property type is not serializable.
    virtual void write_to_properties(const void* instance, Properties& props) const = 0;

    /// Read the property value from a Properties object if the key exists.
    /// Returns true if the property was found and applied.
    /// Returns false if the property type is not serializable or the key is absent.
    virtual bool read_from_properties(void* instance, const Properties& props) const = 0;

    /// Reset the property to its default value if one was provided.
    /// No-op if no default or if read-only.
    virtual void reset(void* instance) const = 0;

    /// True if the property has a default value.
    virtual bool has_default_value() const = 0;

    /// Search the metadata list for an entry whose typeid matches T.
    /// Returns null if not found.
    /// This is the generic extensible metadata accessor -- new metadata kinds
    /// can be added without modifying PropertyDescriptor.
    template<typename T>
    const T* find_metadata() const
    {
        for (const auto& entry : m_metadata) {
            if (const T* ptr = any_cast<T>(&entry))
                return ptr;
        }
        return nullptr;
    }

    /// Convenience: returns the doc string, or empty view if absent.
    std::string_view doc() const
    {
        const std::string* s = find_metadata<std::string>();
        return s ? std::string_view(*s) : std::string_view{};
    }

    /// Convenience: returns the ValueRange metadata, or nullptr.
    const ValueRange* value_range() const { return find_metadata<ValueRange>(); }

    /// Convenience: returns the EnumDescriptor metadata, or nullptr.
    const EnumDescriptor* enum_descriptor() const { return find_metadata<EnumDescriptor>(); }

    /// Convenience: returns combined UIFlags, or UIFlags::none if absent.
    UIFlags ui_flags() const
    {
        const UIFlags* f = find_metadata<UIFlags>();
        return f ? *f : UIFlags::none;
    }

    /// Convenience: returns the UILabel metadata, or nullptr.
    const UILabel* ui_label() const { return find_metadata<UILabel>(); }

    /// Convenience: returns the UIGroup metadata, or nullptr.
    const UIGroup* ui_group() const { return find_metadata<UIGroup>(); }

    /// Convenience: returns the UIDragSpeed metadata, or nullptr.
    const UIDragSpeed* ui_drag_speed() const { return find_metadata<UIDragSpeed>(); }

    /// Convenience: returns the UIEnableIf metadata, or nullptr.
    const UIEnableIf* ui_enable_if() const { return find_metadata<UIEnableIf>(); }

    /// Convenience: returns the UIEditor metadata, or nullptr.
    const UIEditor* ui_editor() const { return find_metadata<UIEditor>(); }

protected:
    PropertyDescriptor(std::string name, bool read_only, std::vector<Any> metadata)
        : m_name(std::move(name))
        , m_read_only(read_only)
        , m_metadata(std::move(metadata))
    {
    }

    /// Copy the current value into caller-provided storage.
    /// The caller must ensure out points to valid memory for the property's type.
    virtual void get_value(const void* instance, void* out) const = 0;

    /// Set the property value from caller-provided storage.
    /// The caller must ensure value points to valid memory for the property's type.
    /// Throws if read-only.
    virtual void set_value(void* instance, const void* value) const = 0;

    std::string m_name;
    bool m_read_only;
    std::vector<Any> m_metadata;
};

// ----------------------------------------------------------------------------
// extract_metadata helper
// ----------------------------------------------------------------------------

namespace detail {

/// Trait to detect DefaultValue<T> specializations.
template<typename T>
struct is_default_value : std::false_type { };
template<typename T>
struct is_default_value<DefaultValue<T>> : std::true_type { };

/// OR-accumulate a flags value into the metadata bag.
/// If an entry of the same flags type already exists, OR it in; otherwise push a new entry.
template<typename F>
void accumulate_flags(std::vector<Any>& metadata, F value)
{
    for (auto& entry : metadata) {
        if (F* existing = any_cast<F>(&entry)) {
            *existing = *existing | value;
            return;
        }
    }
    metadata.push_back(Any(value));
}

/// Result of extract_all: both the metadata bag and the optional default value,
/// extracted from the variadic extras pack in a single pass.
template<typename V>
struct ExtractedMetadata {
    std::vector<Any> metadata;
    std::optional<V> default_value;
    std::optional<OnChange> on_change;
};

/// Walk the variadic metadata pack once, extracting both the metadata bag and
/// the optional default value.
///
/// - DefaultValue<V> entries populate default_value.
/// - Doc strings (const char*) are stored as std::string in metadata.
/// - Flags types (is_reflection_flags_v) are OR-accumulated into a single entry per flags type.
/// - All other types satisfying is_reflection_metadata_v are stored as-is.
/// - Unrecognized non-metadata types cause a compile-time error.
///
/// Each non-flags metadata type (ValueRange, UILabel, UIGroup, UIDragSpeed, UIEnableIf, etc.)
/// should appear at most once per property. If duplicated, only the first will be
/// found by find_metadata<T>().
template<typename V, typename... Extras>
ExtractedMetadata<V> extract_all(Extras&&... extras)
{
    ExtractedMetadata<V> result;
    result.metadata.reserve(sizeof...(Extras));

    auto process = [&](auto&& extra)
    {
        using E = std::decay_t<decltype(extra)>;

        if constexpr (is_default_value<E>::value) {
            result.default_value.emplace(V(extra.value));
        } else if constexpr (std::is_same_v<E, OnChange>) {
            result.on_change.emplace(std::forward<decltype(extra)>(extra));
        } else if constexpr (is_reflection_flags_v<E>) {
            // Flags types: OR-accumulate into a single entry per flags type.
            accumulate_flags<E>(result.metadata, extra);
        } else if constexpr (std::is_convertible_v<E, const char*> && !is_reflection_metadata_v<E>) {
            // Doc string
            result.metadata.push_back(Any(std::string(extra)));
        } else if constexpr (is_reflection_metadata_v<E>) {
            // Known metadata -- store as-is.
            result.metadata.push_back(Any(std::forward<decltype(extra)>(extra)));
        } else {
            // Fail at compile time for types that are not recognized metadata.
            static_assert(
                is_reflection_metadata_v<E> || std::is_convertible_v<E, const char*>,
                "Unknown metadata type passed to property definition. "
                "Use DefaultValue<T>, ValueRange, a doc string (const char*), "
                "or a type with an is_reflection_metadata specialization."
            );
        }
    };

    (process(std::forward<Extras>(extras)), ...);

    // Auto-inject EnumDescriptor for enum types that have SGL enum info,
    // unless one was already provided explicitly in the extras.
    if constexpr (sgl::has_enum_info<V>) {
        bool has_enum_desc = false;
        for (const auto& entry : result.metadata) {
            if (any_cast<EnumDescriptor>(&entry)) {
                has_enum_desc = true;
                break;
            }
        }
        if (!has_enum_desc)
            result.metadata.push_back(Any(detail::enum_descriptor_from_sgl_enum<V>()));
    }

    return result;
}

} // namespace detail

// ----------------------------------------------------------------------------
// TypedPropertyDescriptor<T, OwnerT>
// ----------------------------------------------------------------------------

/// Concrete property descriptor for static C++ properties.
///
/// Stores getter/setter as std::function, invoked on the owner instance
/// (cast from void* to OwnerT&). Metadata is stored in the base class's
/// metadata bag via extract_all.
template<typename T, typename OwnerT>
class TypedPropertyDescriptor : public PropertyDescriptor {
public:
    using Getter = std::function<T(const OwnerT&)>;
    using Setter = std::function<void(OwnerT&, const T&)>;

    /// Construct from getter, optional setter, and metadata.
    template<typename... Extras>
    TypedPropertyDescriptor(std::string name, Getter getter, Setter setter, Extras&&... extras)
        : TypedPropertyDescriptor(
              std::move(name),
              std::move(getter),
              std::move(setter),
              detail::extract_all<T>(std::forward<Extras>(extras)...)
          )
    {
    }

    /// Construct from getter, optional setter, and pre-extracted metadata.
    TypedPropertyDescriptor(std::string name, Getter getter, Setter setter, detail::ExtractedMetadata<T> extracted)
        : PropertyDescriptor(std::move(name), /*read_only=*/!setter, std::move(extracted.metadata))
        , m_getter(std::move(getter))
        , m_setter(std::move(setter))
        , m_default(std::move(extracted.default_value))
    {
    }

public:
    const std::type_info& type() const override { return typeid(T); }

    bool has_default_value() const override { return m_default.has_value(); }

    bool is_default(const void* instance) const override
    {
        if (!m_default)
            return false;
        if constexpr (std::equality_comparable<T>) {
            return m_getter(cast_instance(instance)) == *m_default;
        } else {
            return false;
        }
    }

    Any get_any(const void* instance) const override { return Any(m_getter(cast_instance(instance))); }

    void set_any(void* instance, const Any& value) const override
    {
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        const T* val = any_cast<T>(&value);
        FALCOR_CHECK(val != nullptr, "Property \"{}\": type mismatch in set().", m_name);
        m_setter(cast_mutable_instance(instance), *val);
    }

    int64_t get_enum_as_int64(const void* instance) const override
    {
        if constexpr (std::is_enum_v<T>) {
            return static_cast<int64_t>(m_getter(cast_instance(instance)));
        } else {
            FALCOR_THROW("Property \"{}\": type is not an enum.", m_name);
        }
    }

    void set_enum_from_int64(void* instance, int64_t value) const override
    {
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        if constexpr (std::is_enum_v<T>) {
            m_setter(cast_mutable_instance(instance), static_cast<T>(value));
        } else {
            FALCOR_THROW("Property \"{}\": type is not an enum.", m_name);
        }
    }

    bool is_serializable_to_properties() const override { return PropertyValueType<T>; }

    void write_to_properties(const void* instance, Properties& props) const override
    {
        if constexpr (PropertyValueType<T>) {
            props.set<T>(m_name, m_getter(cast_instance(instance)));
        } else {
            FALCOR_THROW("Property \"{}\": type is not serializable to Properties.", m_name);
        }
    }

    bool read_from_properties(void* instance, const Properties& props) const override
    {
        if constexpr (PropertyValueType<T>) {
            if (m_read_only)
                return false;
            if (!props.has_property(m_name))
                return false;
            if constexpr (std::is_enum_v<T>) {
                if (props.type(m_name) == PropertyType::int_) {
                    m_setter(cast_mutable_instance(instance), static_cast<T>(props.get<int64_t>(m_name)));
                    return true;
                }
            }
            m_setter(cast_mutable_instance(instance), props.get<T>(m_name));
            return true;
        } else {
            return false;
        }
    }

    void reset(void* instance) const override
    {
        if (m_read_only)
            return;
        if (m_default)
            m_setter(cast_mutable_instance(instance), *m_default);
    }

protected:
    void get_value(const void* instance, void* out) const override
    {
        *static_cast<T*>(out) = m_getter(cast_instance(instance));
    }

    void set_value(void* instance, const void* value) const override
    {
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        m_setter(cast_mutable_instance(instance), *static_cast<const T*>(value));
    }

private:
    static const OwnerT& cast_instance(const void* instance) { return *static_cast<const OwnerT*>(instance); }

    static OwnerT& cast_mutable_instance(void* instance) { return *static_cast<OwnerT*>(instance); }

    Getter m_getter;
    Setter m_setter;
    std::optional<T> m_default;

    FALCOR_NON_COPYABLE_AND_MOVABLE(TypedPropertyDescriptor);
};

// ----------------------------------------------------------------------------
// StoredPropertyDescriptor<T>
// ----------------------------------------------------------------------------

/// Concrete property descriptor for dynamic per-instance properties.
///
/// Stores the property value directly as a T member. The instance pointer
/// passed to get/set is ignored (asserted to be nullptr).
/// This allows DynamicPropertySet to use the same PropertyDescriptor*
/// interface as static properties.
template<typename T>
class StoredPropertyDescriptor : public PropertyDescriptor {
public:
    /// Construct with a name, initial value, and metadata.
    template<typename... Extras>
    StoredPropertyDescriptor(std::string name, T initial_value, bool read_only, Extras&&... extras)
        : StoredPropertyDescriptor(
              std::move(name),
              std::move(initial_value),
              read_only,
              detail::extract_all<T>(std::forward<Extras>(extras)...)
          )
    {
    }

private:
    /// Delegating constructor: receives pre-extracted metadata and default value.
    StoredPropertyDescriptor(std::string name, T initial_value, bool read_only, detail::ExtractedMetadata<T> extracted)
        : PropertyDescriptor(std::move(name), read_only, std::move(extracted.metadata))
        , m_value(std::move(initial_value))
        , m_default(std::move(extracted.default_value))
        , m_on_change(std::move(extracted.on_change))
    {
    }

public:
    const std::type_info& type() const override { return typeid(T); }

    bool has_default_value() const override { return m_default.has_value(); }

    bool is_default(const void* instance) const override
    {
        FALCOR_UNUSED(instance);
        if (!m_default)
            return false;
        if constexpr (std::equality_comparable<T>) {
            return m_value == *m_default;
        } else {
            return false;
        }
    }

    Any get_any(const void* instance) const override
    {
        FALCOR_UNUSED(instance);
        return Any(m_value);
    }

    void set_any(void* instance, const Any& value) const override
    {
        FALCOR_UNUSED(instance);
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        const T* val = any_cast<T>(&value);
        FALCOR_CHECK(val != nullptr, "Property \"{}\": type mismatch in set().", m_name);
        m_value = *val;
        if (m_on_change)
            m_on_change->callback(instance);
    }

    int64_t get_enum_as_int64(const void* instance) const override
    {
        FALCOR_UNUSED(instance);
        if constexpr (std::is_enum_v<T>) {
            return static_cast<int64_t>(m_value);
        } else {
            FALCOR_THROW("Property \"{}\": type is not an enum.", m_name);
        }
    }

    void set_enum_from_int64(void* instance, int64_t value) const override
    {
        FALCOR_UNUSED(instance);
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        if constexpr (std::is_enum_v<T>) {
            m_value = static_cast<T>(value);
            if (m_on_change)
                m_on_change->callback(instance);
        } else {
            FALCOR_THROW("Property \"{}\": type is not an enum.", m_name);
        }
    }

    bool is_serializable_to_properties() const override { return PropertyValueType<T>; }

    void write_to_properties(const void* instance, Properties& props) const override
    {
        FALCOR_UNUSED(instance);
        if constexpr (PropertyValueType<T>) {
            props.set<T>(m_name, m_value);
        } else {
            FALCOR_THROW("Property \"{}\": type is not serializable to Properties.", m_name);
        }
    }

    bool read_from_properties(void* instance, const Properties& props) const override
    {
        FALCOR_UNUSED(instance);
        if constexpr (PropertyValueType<T>) {
            if (m_read_only)
                return false;
            if (!props.has_property(m_name))
                return false;
            if constexpr (std::is_enum_v<T>) {
                if (props.type(m_name) == PropertyType::int_) {
                    m_value = static_cast<T>(props.get<int64_t>(m_name));
                    if (m_on_change)
                        m_on_change->callback(instance);
                    return true;
                }
            }
            m_value = props.get<T>(m_name);
            if (m_on_change)
                m_on_change->callback(instance);
            return true;
        } else {
            return false;
        }
    }

    void reset(void* instance) const override
    {
        FALCOR_UNUSED(instance);
        if (m_read_only)
            return;
        if (m_default) {
            m_value = *m_default;
            if (m_on_change)
                m_on_change->callback(instance);
        }
    }

    /// Direct access to the stored value (for DynamicPropertySet internals).
    const T& value() const { return m_value; }

protected:
    void get_value(const void* instance, void* out) const override
    {
        FALCOR_UNUSED(instance);
        *static_cast<T*>(out) = m_value;
    }

    void set_value(void* instance, const void* value) const override
    {
        FALCOR_UNUSED(instance);
        FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
        m_value = *static_cast<const T*>(value);
        if (m_on_change)
            m_on_change->callback(instance);
    }

private:
    /// Mutable because StoredPropertyDescriptor owns the value and
    /// set()/read_from_properties()/reset() need to mutate it through the const
    /// virtual interface inherited from PropertyDescriptor.
    mutable T m_value;
    std::optional<T> m_default;
    std::optional<OnChange> m_on_change;

    FALCOR_NON_COPYABLE_AND_MOVABLE(StoredPropertyDescriptor);
};

} // namespace falcor::reflection
