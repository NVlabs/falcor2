// SPDX-License-Identifier: Apache-2.0

// Portions of this file are derived from Mitsuba 3.
// Copyright (c) Mitsuba contributors.
// Licensed under the BSD 3-Clause License.
// See LICENSES/mitsuba3.txt for the full license text.

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/enum.h"
#include "falcor2/core/object.h"
#include "falcor2/core/any.h"
#include "falcor2/core/error.h"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string_view>
#include <string>
#include <type_traits>
#include <vector>

namespace falcor {

/// Enumeration of property value types.
/// The order of entries must exactly match the order of types in the
/// PropertyValue variant defined in properties.cpp.
enum class PropertyType {
    invalid,  ///< Unknown/deleted property (used for tombstones).
    bool_,    ///< Boolean value.
    int_,     ///< 64-bit signed integer (all integer types are widened to this).
    float_,   ///< Double-precision float (float and double are both stored as double).
    int2,     ///< 2-component signed integer vector.
    int3,     ///< 3-component signed integer vector.
    int4,     ///< 4-component signed integer vector.
    uint2,    ///< 2-component unsigned integer vector.
    uint3,    ///< 3-component unsigned integer vector.
    uint4,    ///< 4-component unsigned integer vector.
    float2,   ///< 2-component float vector.
    float3,   ///< 3-component float vector.
    float4,   ///< 4-component float vector.
    float3x3, ///< 3x3 float matrix.
    float4x4, ///< 4x4 float matrix.
    string,   ///< String value (std::string). Also used for filesystem paths.
    enum_,    ///< Enum value with preserved type identity (see detail::PropertyEnumValue).
    object,   ///< Reference-counted Object (ref<Object>).
    any,      ///< Type-erased storage for arbitrary data (see Any).
    list,     ///< Homogeneous typed list (see detail::PropertyList).
    count_,   ///< Number of property types (not a valid type).
};

SGL_ENUM_INFO(
    PropertyType,
    {
        {PropertyType::invalid, "invalid"}, {PropertyType::bool_, "bool"},        {PropertyType::int_, "int"},
        {PropertyType::float_, "float"},    {PropertyType::int2, "int2"},         {PropertyType::int3, "int3"},
        {PropertyType::int4, "int4"},       {PropertyType::uint2, "uint2"},       {PropertyType::uint3, "uint3"},
        {PropertyType::uint4, "uint4"},     {PropertyType::float2, "float2"},     {PropertyType::float3, "float3"},
        {PropertyType::float4, "float4"},   {PropertyType::float3x3, "float3x3"}, {PropertyType::float4x4, "float4x4"},
        {PropertyType::string, "string"},   {PropertyType::enum_, "enum"},        {PropertyType::object, "object"},
        {PropertyType::any, "any"},         {PropertyType::list, "list"},
    }
);
SGL_ENUM_REGISTER(PropertyType);

namespace detail {

/// Type-erased enum value that preserves the original enum's type identity.
/// Stored as part of the PropertyValue variant for enum types.
struct PropertyEnumValue {
    const std::type_info* type{nullptr};
    int64_t value{0};

    bool operator==(const PropertyEnumValue& other) const
    {
        return (type == other.type || (type && other.type && *type == *other.type)) ? value == other.value : false;
    }
};

/// Homogeneous typed list that can be stored as a property value.
///
/// Wraps a std::vector<T> together with a PropertyType tag describing the
/// element type. The element type must be one of the scalar/vector property
/// types (bool through float4), std::string, or ref<Object>. Any elements
/// are not supported.
class FALCOR_API PropertyList {
public:
    PropertyList() = default;

    /// Create a PropertyList from a typed vector.
    template<typename T>
    static PropertyList create(std::vector<T> data);

    /// Element type stored in this list.
    PropertyType element_type() const { return m_element_type; }

    /// Number of elements.
    size_t size() const;

    /// True if the list contains no elements.
    bool empty() const { return size() == 0; }

    /// Retrieve the stored vector, throwing on element type mismatch.
    template<typename T>
    const std::vector<T>& as_vector() const;

    /// Dispatch a callable on the typed vector.
    /// The callable receives a const reference to std::vector<T> where T is the
    /// element type. Returns the result of the callable.
    template<typename Func>
    auto dispatch(Func&& func) const;

    /// Human-readable string representation.
    std::string to_string() const;

    /// Compute a hash of the list contents.
    size_t hash() const;

    bool operator==(const PropertyList& other) const;
    bool operator!=(const PropertyList& other) const { return !operator==(other); }

private:
    PropertyType m_element_type{PropertyType::invalid};
    Any m_data; // wraps std::vector<T>
};

/// PropertyValueTraits template. Used to map types to PropertyType enum.
template<typename T>
struct PropertyValueTraits { };

// clang-format off
template<> struct PropertyValueTraits<bool> { static constexpr PropertyType type = PropertyType::bool_; };
template<> struct PropertyValueTraits<int64_t> { static constexpr PropertyType type = PropertyType::int_; };
template<> struct PropertyValueTraits<double> { static constexpr PropertyType type = PropertyType::float_; };
template<> struct PropertyValueTraits<int2> { static constexpr PropertyType type = PropertyType::int2; };
template<> struct PropertyValueTraits<int3> { static constexpr PropertyType type = PropertyType::int3; };
template<> struct PropertyValueTraits<int4> { static constexpr PropertyType type = PropertyType::int4; };
template<> struct PropertyValueTraits<uint2> { static constexpr PropertyType type = PropertyType::uint2; };
template<> struct PropertyValueTraits<uint3> { static constexpr PropertyType type = PropertyType::uint3; };
template<> struct PropertyValueTraits<uint4> { static constexpr PropertyType type = PropertyType::uint4; };
template<> struct PropertyValueTraits<float2> { static constexpr PropertyType type = PropertyType::float2; };
template<> struct PropertyValueTraits<float3> { static constexpr PropertyType type = PropertyType::float3; };
template<> struct PropertyValueTraits<float4> { static constexpr PropertyType type = PropertyType::float4; };
template<> struct PropertyValueTraits<float3x3> { static constexpr PropertyType type = PropertyType::float3x3; };
template<> struct PropertyValueTraits<float4x4> { static constexpr PropertyType type = PropertyType::float4x4; };
template<> struct PropertyValueTraits<std::string> { static constexpr PropertyType type = PropertyType::string; };
template<> struct PropertyValueTraits<PropertyEnumValue> { static constexpr PropertyType type = PropertyType::enum_; };
template<> struct PropertyValueTraits<ref<Object>> { static constexpr PropertyType type = PropertyType::object; };
template<> struct PropertyValueTraits<Any> { static constexpr PropertyType type = PropertyType::any; };
template<> struct PropertyValueTraits<PropertyList> { static constexpr PropertyType type = PropertyType::list; };
// clang-format on

template<typename T>
PropertyList PropertyList::create(std::vector<T> data)
{
    PropertyList list;
    list.m_element_type = PropertyValueTraits<T>::type;
    list.m_data = Any(std::move(data));
    return list;
}

template<typename T>
const std::vector<T>& PropertyList::as_vector() const
{
    PropertyType expected = PropertyValueTraits<T>::type;
    if (m_element_type != expected)
        FALCOR_THROW("PropertyList element type mismatch (expected {}, got {})", expected, m_element_type);
    return *any_cast<std::vector<T>>(&m_data);
}

template<typename Func>
auto PropertyList::dispatch(Func&& func) const
{
    // clang-format off
    switch (m_element_type) {
    case PropertyType::bool_: return func(*any_cast<std::vector<bool>>(&m_data));
    case PropertyType::int_: return func(*any_cast<std::vector<int64_t>>(&m_data));
    case PropertyType::float_: return func(*any_cast<std::vector<double>>(&m_data));
    case PropertyType::int2: return func(*any_cast<std::vector<int2>>(&m_data));
    case PropertyType::int3: return func(*any_cast<std::vector<int3>>(&m_data));
    case PropertyType::int4: return func(*any_cast<std::vector<int4>>(&m_data));
    case PropertyType::uint2: return func(*any_cast<std::vector<uint2>>(&m_data));
    case PropertyType::uint3: return func(*any_cast<std::vector<uint3>>(&m_data));
    case PropertyType::uint4: return func(*any_cast<std::vector<uint4>>(&m_data));
    case PropertyType::float2: return func(*any_cast<std::vector<float2>>(&m_data));
    case PropertyType::float3: return func(*any_cast<std::vector<float3>>(&m_data));
    case PropertyType::float4: return func(*any_cast<std::vector<float4>>(&m_data));
    case PropertyType::float3x3: return func(*any_cast<std::vector<float3x3>>(&m_data));
    case PropertyType::float4x4: return func(*any_cast<std::vector<float4x4>>(&m_data));
    case PropertyType::string: return func(*any_cast<std::vector<std::string>>(&m_data));
    case PropertyType::enum_: return func(*any_cast<std::vector<PropertyEnumValue>>(&m_data));
    case PropertyType::object: return func(*any_cast<std::vector<ref<Object>>>(&m_data));
    // invalid and unsupported types (explicit to enable warnings for missing cases)
    case PropertyType::invalid:
    case PropertyType::any:
    case PropertyType::list:
    case PropertyType::count_:
        break;
    }
    // clang-format on
    FALCOR_THROW("PropertyList: unsupported element type {}", m_element_type);
}

/// Type mapping trait for the Properties container.
///
/// Maps user-facing C++ types to the internal storage types used by the
/// PropertyValue variant. The default template maps to void, which makes
/// the type ineligible for storage (rejected by the PropertyValueType concept).
template<typename T, typename Enable = void>
struct prop_map {
    using type = void;
};

/// Convenience alias: maps a (decayed) type to its internal property storage type.
template<typename T>
using prop_map_t = typename prop_map<std::decay_t<T>>::type;

// clang-format off
template<> struct prop_map<bool> { using type = bool; };
template<typename T> struct prop_map<T, std::enable_if_t<std::is_integral_v<T> && !std::is_same_v<T, bool>>> { using type = int64_t; };
template<typename T> struct prop_map<T, std::enable_if_t<std::is_floating_point_v<T>>> { using type = double; };
template<typename T> struct prop_map<T, std::enable_if_t<std::is_enum_v<T>>> { using type = PropertyEnumValue; };
template<> struct prop_map<int2> { using type = int2; };
template<> struct prop_map<int3> { using type = int3; };
template<> struct prop_map<int4> { using type = int4; };
template<> struct prop_map<uint2> { using type = uint2; };
template<> struct prop_map<uint3> { using type = uint3; };
template<> struct prop_map<uint4> { using type = uint4; };
template<> struct prop_map<float2> { using type = float2; };
template<> struct prop_map<float3> { using type = float3; };
template<> struct prop_map<float4> { using type = float4; };
template<> struct prop_map<float3x3> { using type = float3x3; };
template<> struct prop_map<float4x4> { using type = float4x4; };
template<> struct prop_map<std::string> { using type = std::string; };
template<> struct prop_map<std::string_view> { using type = std::string; };
template<> struct prop_map<const char*> { using type = std::string; };
template<> struct prop_map<char*> { using type = std::string; };
template<size_t N> struct prop_map<char[N]> { using type = std::string; };
template<size_t N> struct prop_map<const char[N]> { using type = std::string; };
template<> struct prop_map<std::filesystem::path> { using type = std::string; };
template<> struct prop_map<PropertyEnumValue> { using type = PropertyEnumValue; };
template<typename T> struct prop_map<ref<T>, std::enable_if_t<std::is_base_of_v<Object, T>>> { using type = ref<Object>; };
template<typename T> struct prop_map<T*, std::enable_if_t<std::is_base_of_v<Object, T>>> { using type = ref<Object>; };
template<> struct prop_map<Any> { using type = Any; };
template<> struct prop_map<PropertyList> { using type = PropertyList; };
// clang-format on

/// Extracts the underlying Object-derived type from T or ref<T>.
/// Used internally by get() and try_get() for dynamic_cast on Object properties.
template<typename T>
struct object_type {
    using type = T;
};

template<typename T>
struct object_type<ref<T>> {
    using type = T;
};

template<typename T>
using object_type_t = typename object_type<std::decay_t<T>>::type;

/// Convert a single value to its property storage type.
/// Handles path->string, arithmetic widening, and identity forwarding.
template<typename T>
decltype(auto) to_storage_value(T&& value)
{
    using Decayed = std::decay_t<T>;
    using T2 = prop_map_t<Decayed>;
    static_assert(!std::is_same_v<T2, void>, "Invalid property type");

    if constexpr (std::is_same_v<Decayed, std::filesystem::path>)
        return value.string();
    else if constexpr (std::is_enum_v<Decayed>)
        return PropertyEnumValue{&typeid(Decayed), static_cast<int64_t>(value)};
    else if constexpr (!std::is_same_v<Decayed, T2>)
        return static_cast<T2>(std::forward<T>(value));
    else
        return std::forward<T>(value);
}

/// Convert a vector of values to property storage type.
/// Delegates per-element conversion to to_storage_value for a single source of truth.
/// When types match, the input is moved through. Otherwise elements are converted individually.
template<typename T>
std::vector<prop_map_t<T>> to_storage_vector(std::vector<T> data)
{
    using T2 = prop_map_t<T>;
    static_assert(!std::is_same_v<T2, void>, "Invalid property element type");

    if constexpr (std::is_same_v<T, T2>)
        return data;
    else {
        std::vector<T2> result;
        result.reserve(data.size());
        for (auto&& elem : data)
            result.push_back(to_storage_value(std::move(elem)));
        return result;
    }
}

/// Throw an exception when an enum property has an incompatible type.
[[noreturn]] FALCOR_API void
throw_enum_type_error(std::string_view name, const std::type_info& expected_type, const std::type_info* actual_type);

/// Convert a storage value back to the user-facing type.
/// Handles range checking for integers and static_cast for narrowing.
/// Object types are not handled here (they need dynamic_cast).
template<typename T>
T from_storage_value(const prop_map_t<T>& value, std::string_view name)
{
    using T2 = prop_map_t<T>;

    if constexpr (std::is_enum_v<T>) {
        if (!value.type || *value.type != typeid(T))
            throw_enum_type_error(name, typeid(T), value.type);
        return static_cast<T>(value.value);
    } else if constexpr (std::is_integral_v<T2> && !std::is_same_v<T2, bool>) {
        constexpr T min = std::numeric_limits<T>::min(), max = std::numeric_limits<T>::max();

        if constexpr (std::is_unsigned_v<T>) {
            bool out_of_bounds = value < 0;
            if constexpr (max <= (T)std::numeric_limits<T2>::max())
                out_of_bounds |= value > (T2)max;

            if (out_of_bounds)
                FALCOR_THROW(
                    "Property \"{}\": value {} is out of bounds, must be in the range [{}, {}]",
                    name,
                    value,
                    min,
                    max
                );
        } else {
            if (value < (T2)min || value > (T2)max)
                FALCOR_THROW(
                    "Property \"{}\": value {} is out of bounds, must be in the range [{}, {}]",
                    name,
                    value,
                    min,
                    max
                );
        }
        return static_cast<T>(value);
    } else {
        return static_cast<T>(value);
    }
}

/// Convert a vector of storage values back to user-facing type.
/// Delegates per-element conversion to from_storage_value for a single source of truth.
template<typename T>
std::vector<T> from_storage_vector(const std::vector<prop_map_t<T>>& src, std::string_view name)
{
    using T2 = prop_map_t<T>;

    if constexpr (std::is_same_v<T, T2>)
        return src;
    else {
        std::vector<T> result;
        result.reserve(src.size());
        for (const auto& elem : src)
            result.push_back(from_storage_value<T>(elem, name));
        return result;
    }
}

} // namespace detail

/// Concept that accepts any type with a valid prop_map mapping.
/// A type satisfying this concept can be stored in and retrieved from a Properties container.
template<typename T>
concept PropertyValueType = !std::is_same_v<detail::prop_map_t<T>, void>;

/// Associative container for passing typed configuration parameters.
///
/// Properties maps string keys to values of various types: booleans, integers,
/// floats, vectors, strings, Object references, and type-erased Any values.
///
/// Key features:
///   - O(1) insertion and lookup by property name.
///   - Traversal of properties in the original insertion order.
///   - Type-safe storage with automatic widening (e.g. int -> int64_t, float -> double).
///   - Range-checked retrieval for integer types.
///   - Type-safe dynamic_cast for Object-derived property values.
///
/// Deleting properties leaves tombstone entries that are skipped during
/// iteration but occupy memory until the container is cleared.
class FALCOR_API Properties {
public:
    /// Sentinel value returned by key_index() when a property is not found.
    static constexpr size_t npos = size_t(-1);

    /// Default constructor.
    Properties();

    /// Copy constructor.
    Properties(const Properties& other);

    /// Move constructor.
    Properties(Properties&& other) noexcept;

    /// Destructor.
    ~Properties();

    /// Assignment operator.
    Properties& operator=(const Properties& props);

    /// Move assignment operator.
    Properties& operator=(Properties&& props) noexcept;

    /// Comparison operator.
    bool operator==(const Properties& other) const;
    bool operator!=(const Properties& other) const { return !operator==(other); }

    /// Clear all properties.
    void clear();

    /// Number of properties in the container.
    size_t size() const;

    /// True if the container is empty.
    bool empty() const { return size() == 0; }

    /// Merge properties from another Properties object.
    /// Existing properties will be overridden.
    void merge(const Properties& other);

    /// Check if a property exists.
    bool has_property(std::string_view name) const;

    /// Remove a property.
    /// Returns true if the property was found and removed, false otherwise.
    bool remove_property(std::string_view name);

    /// Get all property keys (in insertion order).
    std::vector<std::string_view> keys() const;

    /// Rename a property.
    /// Returns true if the property was renamed, false if old_name doesn't exist
    /// or new_name already exists.
    bool rename_property(std::string_view old_name, std::string_view new_name);

    /// Get the type of a stored property.
    PropertyType type(std::string_view name) const;

    /// Return a human-readable string representation of a property value.
    std::string as_string(std::string_view name) const;

    /// Set a property value.
    /// @param name Property name.
    /// @param value Value to set.
    /// @param throw_if_exist Throw an exception if a property with the same name already exists.
    template<PropertyValueType T>
    void set(std::string_view name, const T& value, bool throw_if_exists = false);

    /// @overload
    template<PropertyValueType T>
    void set(std::string_view name, T&& value, bool throw_if_exists = false);

    /// Get a property value.
    /// Throws if the property does not exist or has an incompatible type.
    /// @param name Property name.
    /// @return The value.
    template<PropertyValueType T>
    T get(std::string_view name) const
    {
        return get<T>(require_key_index(name));
    }

    /// Get a property value, returning std::nullopt if the property does not exist or has an incompatible type.
    /// @param name Property name.
    /// @param throw_if_incompatible If true, throw on type mismatch.
    /// @return The value or std::nullopt if the property does not exist or has an incompatible type.
    template<PropertyValueType T>
    std::optional<T> get_optional(std::string_view name, bool throw_if_incompatible = false) const
    {
        size_t index = key_index(name);
        if (index == npos)
            return std::nullopt;
        if (throw_if_incompatible)
            return get<T>(index);

        using T2 = detail::prop_map_t<T>;
        const T2* value = get_impl<T2>(index, false);
        if (!value)
            return std::nullopt;

        if constexpr (std::is_same_v<T2, ref<Object>>) {
            using TargetType = detail::object_type_t<T>;
            TargetType* ptr = dynamic_cast<TargetType*>(const_cast<Object*>(value->get()));
            if (value && !ptr)
                return std::nullopt;
            return T(ptr);
        } else {
            return detail::from_storage_value<T>(*value, entry_name(index));
        }
    }


    /// Get a property value, returning a default value if the property does not exist.
    /// Throws on type mismatch if the property exists but has an incompatible type.
    /// @param name Property name.
    /// @return The value or the default value if the property does not exist.
    template<PropertyValueType T>
    T get(std::string_view name, const T& default_value) const
    {
        size_t index = key_index(name);
        return index != npos ? get<T>(index) : default_value;
    }

    /// Try to retrieve a property value without implicit conversions.
    /// Returns a pointer to the stored value, or nullptr if the property
    /// doesn't exist or has a different type.
    /// @param name Property name.
    /// @return The value or nullptr if the property does not exist.
    template<PropertyValueType T>
    T* try_get(std::string_view name) const
    {
        size_t index = key_index(name);
        if (index == npos)
            return nullptr;
        return try_get<T>(index);
    }

    /// Store an arbitrary value wrapped in a type-erased Any container.
    ///
    /// This enables storing types that are not natively supported by the
    /// Properties system. The value can be retrieved later using get_any<T>().
    /// @see Any
    /// @param name Property name.
    /// @param value Value to set.
    template<typename T>
    void set_any(std::string_view name, const T& value)
    {
        set(name, Any(value));
    }

    /// @overload
    template<typename T>
    void set_any(std::string_view name, T&& value)
    {
        set(name, Any(std::forward<T>(value)));
    }

    /// Retrieve a value previously stored using set_any().
    /// Throws if the property does not exist or cannot be cast to the requested type.
    /// @param name Property name.
    /// @return The value.
    template<typename T>
    const T& get_any(std::string_view name) const
    {
        size_t index = require_key_index(name);
        const T* value = any_cast<T>(get_impl<Any>(index));
        if (!value)
            throw_any_type_error(index, typeid(T));
        return *value;
    }

    /// Set a list property from a typed vector.
    /// @param name Property name.
    /// @param data The vector to store.
    /// @param throw_if_exists Throw if a property with the same name already exists.
    template<typename T>
    void set_list(std::string_view name, std::vector<T> data, bool throw_if_exists = false)
    {
        set(name, detail::PropertyList::create(detail::to_storage_vector(std::move(data))), throw_if_exists);
    }

    /// Get a list property as a typed vector.
    /// Throws if the property does not exist or has an incompatible element type.
    /// @param name Property name.
    /// @return The vector.
    template<typename T>
    std::vector<T> get_list(std::string_view name) const
    {
        using T2 = detail::prop_map_t<T>;
        return detail::from_storage_vector<T>(get<detail::PropertyList>(name).as_vector<T2>(), name);
    }

    /// Get a list property as a typed vector, returning std::nullopt if the
    /// property does not exist or has an incompatible type.
    /// @param name Property name.
    /// @param throw_if_incompatible If true, throw on type/element mismatch.
    /// @return The vector or std::nullopt.
    template<typename T>
    std::optional<std::vector<T>> get_list_optional(std::string_view name, bool throw_if_incompatible = false) const
    {
        if (throw_if_incompatible) {
            auto opt = get_optional<detail::PropertyList>(name, true);
            if (!opt)
                return std::nullopt;
            using T2 = detail::prop_map_t<T>;
            return detail::from_storage_vector<T>(opt->as_vector<T2>(), name);
        } else {
            const detail::PropertyList* pl = try_get<detail::PropertyList>(name);
            if (!pl)
                return std::nullopt;
            using T2 = detail::prop_map_t<T>;
            if (pl->element_type() != detail::PropertyValueTraits<T2>::type)
                return std::nullopt;
            return detail::from_storage_vector<T>(pl->as_vector<T2>(), name);
        }
    }

    /// Try to retrieve a list property without implicit conversions.
    /// Returns a pointer to the stored vector, or nullptr if the property
    /// doesn't exist or has a different element type.
    /// @param name Property name.
    /// @return Pointer to the stored vector, or nullptr.
    template<typename T>
    const std::vector<T>* try_get_list(std::string_view name) const
    {
        using T2 = detail::prop_map_t<T>;
        static_assert(
            std::is_same_v<T, T2>,
            "try_get_list does not perform implicit conversions; T must match storage type"
        );
        const detail::PropertyList* pl = try_get<detail::PropertyList>(name);
        if (!pl || pl->element_type() != detail::PropertyValueTraits<T>::type)
            return nullptr;
        return &pl->as_vector<T>();
    }

    /// Compute a hash of the object.
    /// The hash is order-independent (sorted by key).
    size_t hash() const;

    class iterator;

    /// Iterator to the first property.
    iterator begin() const;

    /// Past-the-end iterator.
    iterator end() const;

    /// Returns iterator to a property of the given name, or end().
    iterator find(std::string_view name) const;

    /// String representation of all properties.
    std::string to_string() const;

    /// Copy a property pointed to by an iterator of another Properties object.
    void copy_property(const iterator& source);

    /// Returns true if either both this and other Properties miss the property of a given name,
    /// or if the property is present and equal in both.
    bool is_equal_property(std::string_view name, const Properties& other) const;

    /// Returns PropertyType for the corresponding type T.
    template<PropertyValueType T>
    static PropertyType check_type();

private:
    /// Append a new property.
    /// @param name Property name.
    /// @param throw_if_exists Throw if a property with the same name exists.
    /// @return The index of the property.
    size_t append(std::string_view name, bool throw_if_exists);

    /// Get the index of a property by name.
    /// @param name Property name.
    /// @return The index of the property, or `npos` if not found.
    size_t key_index(std::string_view name) const noexcept;

    /// Get the index of a property by name or throw an exception if the property was not found.
    /// @param name Property name.
    /// @return The index of the property.
    size_t require_key_index(std::string_view name) const;

    /// Get the name of a property.
    /// @param index Index of the property.
    /// @return The name of the property.
    std::string_view entry_name(size_t index) const;

    /// Set a property value.
    /// @param index Index of the property.
    /// @param value Property value.
    template<typename T>
    void set_impl(size_t index, const T& value);

    /// Set a property value.
    /// @param index Index of the property.
    /// @param value Property value.
    template<typename T>
    void set_impl(size_t index, T&& value);

    /// Retrieve a property value by index.
    /// @param index Index of the property.
    /// @return The value of the property.
    template<typename T>
    T get(size_t index) const;

    /// Low-level access to a property's stored value by index.
    /// @param index Index of the property.
    /// @param throw_if_incompatible If true, throw on type mismatch, otherwise return nullptr.
    /// @return A pointer to the stored value, or nullptr on type mismatch when not throwing.
    template<typename T>
    const T* get_impl(size_t index, bool throw_if_incompatible = true) const;

    /// Try to get a property value by index without implicit conversions.
    /// @param index Index of the property.
    /// @return A pointer to the stored value, or nullptr on type mismatch.
    template<PropertyValueType T>
    T* try_get(size_t index) const;

    /// Throw an exception when an object has incompatible type.
    [[noreturn]] void
    throw_object_type_error(size_t index, const std::type_info& expected_type, const ref<Object>& value) const;

    /// Throw an exception when get_any() encounters an incompatible type.
    [[noreturn]] void throw_any_type_error(size_t index, const std::type_info& requested_type) const;

    /// Convert a storage type to a PropertyType enum value.
    template<typename T>
    static PropertyType check_type_impl();

    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

template<PropertyValueType T>
void Properties::set(std::string_view name, const T& value, bool throw_if_exists)
{
    using T2 = detail::prop_map_t<T>;
    size_t index = append(name, throw_if_exists);
    set_impl<T2>(index, detail::to_storage_value(value));
}

template<PropertyValueType T>
void Properties::set(std::string_view name, T&& value, bool throw_if_exists)
{
    using T2 = detail::prop_map_t<T>;
    size_t index = append(name, throw_if_exists);
    set_impl<T2>(index, detail::to_storage_value(std::forward<T>(value)));
}

template<typename T>
T Properties::get(size_t index) const
{
    using T2 = detail::prop_map_t<T>;
    static_assert(!std::is_same_v<T2, void>, "Invalid type");

    const T2& value = *get_impl<T2>(index, true);

    if constexpr (std::is_same_v<T2, ref<Object>>) {
        using TargetType = detail::object_type_t<T>;
        TargetType* ptr = dynamic_cast<TargetType*>(const_cast<Object*>(value.get()));
        if (value && !ptr)
            throw_object_type_error(index, typeid(TargetType), value);
        return T(ptr);
    } else {
        return detail::from_storage_value<T>(value, entry_name(index));
    }
}

template<PropertyValueType T>
T* Properties::try_get(size_t index) const
{
    using T2 = detail::prop_map_t<T>;
    static_assert(!std::is_same_v<T2, void>, "Invalid type for try_get");

    if constexpr (std::is_base_of_v<Object, detail::object_type_t<T>>) {
        // For Object-derived types, get the ref<Object> without throwing
        const ref<Object>* obj = get_impl<ref<Object>>(index, false);
        if (!obj)
            return nullptr;
        // dynamic_cast to the target type
        using TargetType = detail::object_type_t<T>;
        return dynamic_cast<TargetType*>(const_cast<Object*>(obj->get()));
    } else {
        static_assert(
            std::is_same_v<std::remove_cv_t<T>, std::remove_cv_t<T2>>,
            "try_get does not perform implicit conversions; T must match storage type"
        );
        // For non-Object types, directly get the value without conversions
        const T2* value = get_impl<T2>(index, false);
        if (!value)
            return nullptr;
        return const_cast<T2*>(value);
    }
}

template<PropertyValueType T>
PropertyType Properties::check_type()
{
    using T2 = detail::prop_map_t<T>;
    static_assert(!std::is_same_v<T2, void>, "Invalid type for check_type");

    return check_type_impl<T2>();
}


/// Forward iterator over the entries of a Properties container.
///
/// The iterator dereferences to itself, exposing name(), type(), and get<T>()
/// accessors on the current property. Tombstone entries (deleted properties)
/// are automatically skipped.
///
/// It is legal to mutate the container while iterating (e.g. removing entries),
/// but newly added entries may or may not be visited.
class FALCOR_API Properties::iterator {
public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = iterator;
    using difference_type = std::ptrdiff_t;
    using pointer = const iterator*;
    using reference = const iterator&;

    iterator() = default;

    /// Name of the current property.
    std::string_view name() const { return m_name; }

    /// Type of the current property.
    PropertyType type() const { return m_type; }

    /// Internal index of the current property.
    size_t index() const { return m_index; }

    /// Retrieve the current property's value, cast to the requested type.
    template<typename T>
    T get() const
    {
        return m_props->get<T>(m_index);
    }

    /// String representation of the current property entry.
    std::string to_string() const;

    reference operator*() const { return *this; }
    pointer operator->() const { return this; }

    iterator& operator++();
    iterator operator++(int);

    bool operator==(const iterator& other) const { return m_index == other.m_index; }
    bool operator!=(const iterator& other) const { return m_index != other.m_index; }

private:
    friend class Properties;
    iterator(const Properties* props, size_t index);
    void skip_invalid_entries();

    const Properties* m_props{nullptr};
    size_t m_index{0};
    std::string_view m_name;
    PropertyType m_type;
};

/// Explicit template instantiation declarations/definitions for property accessor methods.
/// These ensure that set_impl and get_impl are instantiated for every supported storage type.
#define FALCOR_PROPERTY_ACCESSOR(mode, T)                                                                              \
    mode template FALCOR_API void Properties::set_impl<T>(size_t, const T&);                                           \
    mode template FALCOR_API void Properties::set_impl<T>(size_t, T&&);                                                \
    mode template FALCOR_API const T* Properties::get_impl<T>(size_t, bool) const;                                     \
    mode template FALCOR_API PropertyType Properties::check_type_impl<T>();

#define FALCOR_PROPERTY_ACCESSORS(mode)                                                                                \
    FALCOR_PROPERTY_ACCESSOR(mode, bool)                                                                               \
    FALCOR_PROPERTY_ACCESSOR(mode, int64_t)                                                                            \
    FALCOR_PROPERTY_ACCESSOR(mode, double)                                                                             \
    FALCOR_PROPERTY_ACCESSOR(mode, int2)                                                                               \
    FALCOR_PROPERTY_ACCESSOR(mode, int3)                                                                               \
    FALCOR_PROPERTY_ACCESSOR(mode, int4)                                                                               \
    FALCOR_PROPERTY_ACCESSOR(mode, uint2)                                                                              \
    FALCOR_PROPERTY_ACCESSOR(mode, uint3)                                                                              \
    FALCOR_PROPERTY_ACCESSOR(mode, uint4)                                                                              \
    FALCOR_PROPERTY_ACCESSOR(mode, float2)                                                                             \
    FALCOR_PROPERTY_ACCESSOR(mode, float3)                                                                             \
    FALCOR_PROPERTY_ACCESSOR(mode, float4)                                                                             \
    FALCOR_PROPERTY_ACCESSOR(mode, float3x3)                                                                           \
    FALCOR_PROPERTY_ACCESSOR(mode, float4x4)                                                                           \
    FALCOR_PROPERTY_ACCESSOR(mode, std::string)                                                                        \
    FALCOR_PROPERTY_ACCESSOR(mode, detail::PropertyEnumValue)                                                          \
    FALCOR_PROPERTY_ACCESSOR(mode, ref<Object>)                                                                        \
    FALCOR_PROPERTY_ACCESSOR(mode, Any)                                                                                \
    FALCOR_PROPERTY_ACCESSOR(mode, detail::PropertyList)

FALCOR_PROPERTY_ACCESSORS(extern)

} // namespace falcor
