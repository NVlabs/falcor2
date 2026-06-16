// SPDX-License-Identifier: Apache-2.0

// Portions of this file are derived from Mitsuba 3.
// Copyright (c) Mitsuba contributors.
// Licensed under the BSD 3-Clause License.
// See LICENSES/mitsuba3.txt for the full license text.

#include "properties.h"

#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"

#include <sgl/core/hash.h>

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

namespace falcor {

/// Minimal heap-allocated string for efficient storage with string_view compatibility
/// Unlike std::string which uses small string optimization, heap_string is ALWAYS
/// heap-allocated, ensuring the string data remains stable when the object is moved
/// around (e.g., when std::vector<Entry> reallocates). This stability allows us to
/// safely store string_views pointing to this data in key_to_index.
struct heap_string {
    std::unique_ptr<char[]> data;
    size_t length = 0;

    // Initialization from std::string_view
    heap_string(std::string_view str)
        : length(str.length())
    {
        if (length > 0) {
            data = std::make_unique<char[]>(length + 1);
            std::memcpy(data.get(), str.data(), length);
            data[length] = '\0';
        }
    }

    heap_string() = default;
    heap_string(heap_string&&) = default;
    ~heap_string() = default;

    heap_string(const heap_string& other)
        : length(other.length)
    {
        if (length > 0) {
            data = std::make_unique<char[]>(length + 1);
            std::memcpy(data.get(), other.data.get(), length);
            data[length] = '\0';
        }
    }

    heap_string& operator=(const heap_string& other)
    {
        if (this != &other)
            *this = heap_string(other);
        return *this;
    }

    heap_string& operator=(heap_string&&) = default;

    operator std::string_view() const { return length > 0 ? std::string_view(data.get(), length) : std::string_view(); }

    bool operator==(const heap_string& other) const
    {
        return static_cast<std::string_view>(*this) == static_cast<std::string_view>(other);
    }

    bool operator!=(const heap_string& other) const
    {
        return static_cast<std::string_view>(*this) != static_cast<std::string_view>(other);
    }
};

} // namespace falcor

template<>
struct fmt::formatter<falcor::heap_string> : formatter<std::string_view> {
    template<typename FormatContext>
    auto format(falcor::heap_string heap_string, FormatContext& ctx) const
    {
        return formatter<std::string_view>::format(std::string_view(heap_string), ctx);
    }
};

namespace falcor {

/// PropertyValue variant. Used to store property values.
using PropertyValue = std::variant<
    std::monostate,            // PropertyType::invalid
    bool,                      // PropertyType::bool_
    int64_t,                   // PropertyType::int_
    double,                    // PropertyType::float_
    int2,                      // PropertyType::int2
    int3,                      // PropertyType::int3
    int4,                      // PropertyType::int4
    uint2,                     // PropertyType::uint2
    uint3,                     // PropertyType::uint3
    uint4,                     // PropertyType::uint4
    float2,                    // PropertyType::float2
    float3,                    // PropertyType::float3
    float4,                    // PropertyType::float4
    float3x3,                  // PropertyType::float3x3
    float4x4,                  // PropertyType::float4x4
    std::string,               // PropertyType::string
    detail::PropertyEnumValue, // PropertyType::enum_
    ref<Object>,               // PropertyType::object
    Any,                       // PropertyType::any
    detail::PropertyList       // PropertyType::list
    >;

// ----------------------------------------------------------------------------
// FormatVisitor
// ----------------------------------------------------------------------------

struct FormatVisitor {
    std::string& str;
    void operator()(std::monostate) { str += "<invalid>"; }
    void operator()(const bool& value) { str += value ? "true" : "false"; }
    void operator()(const std::string& value) { str += '"' + value + '"'; }
    void operator()(const detail::PropertyEnumValue& value)
    {
        str += fmt::format(
            "{}({})",
            value.type ? falcor::platform::demangle_type_name(*value.type) : "<enum>",
            value.value
        );
    }
    void operator()(const ref<Object>& value) { str += value ? "<object>" : "<null>"; }
    void operator()(const Any& value) { str += value.has_value() ? "<any>" : "<empty>"; }
    void operator()(const detail::PropertyList& value) { str += value.to_string(); }
    template<typename T>
    void operator()(const T& value)
    {
        str += fmt::format("{}", value);
    }
};

// ----------------------------------------------------------------------------
// CompareVisitor
// ----------------------------------------------------------------------------

struct CompareVisitor {
    const PropertyValue& other;
    bool operator()(std::monostate) const { return std::holds_alternative<std::monostate>(other); }
    template<typename T>
    bool operator()(const T& value) const
    {
        return value == std::get<T>(other);
    }
};

// ----------------------------------------------------------------------------
// HashVisitor
// ----------------------------------------------------------------------------

struct HashVisitor {
    size_t operator()(std::monostate) const { return 0; }
    size_t operator()(const detail::PropertyEnumValue& v) const
    {
        return sgl::hash(v.type ? v.type->hash_code() : 0, v.value);
    }
    size_t operator()(const ref<Object>& v) const { return sgl::hash(v.get()); }
    size_t operator()(const Any&) const { return 0; }
    size_t operator()(const detail::PropertyList& v) const { return v.hash(); }
    template<typename T>
    size_t operator()(const T& v) const
    {
        return sgl::hash(v);
    }
};

// ----------------------------------------------------------------------------
// Properties::Impl
// ----------------------------------------------------------------------------

struct Entry {
    heap_string key;
    PropertyValue value;

    PropertyType type() const { return PropertyType(value.index()); }

    bool is_valid() const { return type() != PropertyType::invalid; }
};

std::string format_property_entry(const Entry& entry, std::string_view prefix)
{
    std::string value_str;
    FormatVisitor visitor{value_str};
    std::visit(visitor, entry.value);
    return fmt::format("{}{}: {} = {}", prefix, entry.key, entry.type(), value_str);
}

struct Properties::Impl {
    std::vector<Entry> entries;
    std::unordered_map<std::string_view, size_t, std::hash<std::string_view>, std::equal_to<>> key_to_index;

    void rebuild_index()
    {
        key_to_index.clear();
        for (size_t i = 0; i < entries.size(); ++i) {
            if (entries[i].is_valid())
                key_to_index[entries[i].key] = i;
        }
    }
};

// ----------------------------------------------------------------------------
// PropertyList
// ----------------------------------------------------------------------------

namespace detail {

size_t PropertyList::size() const
{
    if (m_element_type == PropertyType::invalid)
        return 0;
    return dispatch(
        [](const auto& vec) -> size_t
        {
            return vec.size();
        }
    );
}

std::string PropertyList::to_string() const
{
    if (m_element_type == PropertyType::invalid)
        return "[]";

    return dispatch(
        [](const auto& vec) -> std::string
        {
            std::string result = "[";
            FormatVisitor visitor{result};
            for (size_t i = 0; i < vec.size(); ++i) {
                if (i > 0)
                    result += ", ";
                visitor(vec[i]);
            }
            result += "]";
            return result;
        }
    );
}

size_t PropertyList::hash() const
{
    if (m_element_type == PropertyType::invalid)
        return 0;

    size_t h = sgl::hash(m_element_type);

    dispatch(
        [&](const auto& vec)
        {
            HashVisitor hash_visitor;
            for (const auto& elem : vec)
                h = sgl::hash_combine(h, hash_visitor(elem));
        }
    );

    return h;
}

bool PropertyList::operator==(const PropertyList& other) const
{
    if (m_element_type != other.m_element_type)
        return false;
    if (m_element_type == PropertyType::invalid)
        return true;

    return dispatch(
        [&](const auto& vec) -> bool
        {
            using VecT = std::decay_t<decltype(vec)>;
            const auto* other_vec = any_cast<VecT>(&other.m_data);
            return other_vec && vec == *other_vec;
        }
    );
}

[[noreturn]] FALCOR_API void
throw_enum_type_error(std::string_view name, const std::type_info& expected_type, const std::type_info* actual_type)
{
    std::string expected_type_name = falcor::platform::demangle_type_name(expected_type);
    std::string actual_type_name = actual_type ? falcor::platform::demangle_type_name(*actual_type) : "unknown";
    FALCOR_THROW(
        "Property \"{}\": enum type mismatch (expected {}, got {})",
        name,
        expected_type_name,
        actual_type_name
    );
}

} // namespace detail

// ----------------------------------------------------------------------------
// Properties
// ----------------------------------------------------------------------------

Properties::Properties()
    : m_impl(std::make_unique<Impl>())
{
}

Properties::Properties(const Properties& other)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->entries = other.m_impl->entries;
    m_impl->rebuild_index();
}

Properties::Properties(Properties&& other) noexcept = default;

Properties::~Properties() = default;

Properties& Properties::operator=(const Properties& props)
{
    if (this != &props)
        *this = Properties(props);
    return *this;
}

Properties& Properties::operator=(Properties&& props) noexcept = default;

bool Properties::operator==(const Properties& other) const
{
    if (m_impl == other.m_impl)
        return true;

    if (m_impl->key_to_index.size() != other.m_impl->key_to_index.size())
        return false;

    for (const auto& [key, index] : m_impl->key_to_index) {
        auto it = other.m_impl->key_to_index.find(key);
        if (it == other.m_impl->key_to_index.end())
            return false;

        const Entry& entry1 = m_impl->entries[index];
        const Entry& entry2 = other.m_impl->entries[it->second];
        if (entry1.value.index() != entry2.value.index())
            return false;
        if (!std::visit(CompareVisitor{entry2.value}, entry1.value))
            return false;
    }

    return true;
}

void Properties::clear()
{
    m_impl->entries.clear();
    m_impl->key_to_index.clear();
}

size_t Properties::size() const
{
    return m_impl->key_to_index.size();
}

void Properties::merge(const Properties& other)
{
    for (const auto& entry : other.m_impl->entries) {
        if (!entry.is_valid())
            continue;
        auto it = m_impl->key_to_index.find(entry.key);
        if (it != m_impl->key_to_index.end()) {
            m_impl->entries[it->second].value = entry.value;
        } else {
            size_t idx = m_impl->entries.size();
            m_impl->entries.push_back(entry);
            m_impl->key_to_index[m_impl->entries[idx].key] = idx;
        }
    }
}

bool Properties::has_property(std::string_view name) const
{
    return m_impl->key_to_index.contains(name);
}

bool Properties::remove_property(std::string_view name)
{
    auto it = m_impl->key_to_index.find(name);
    if (it == m_impl->key_to_index.end())
        return false;
    m_impl->entries[it->second].value = {};
    m_impl->key_to_index.erase(it);
    return true;
}

bool Properties::rename_property(std::string_view old_name, std::string_view new_name)
{
    auto it = m_impl->key_to_index.find(old_name);
    if (it == m_impl->key_to_index.end())
        return false;

    // Check if new name already exists
    if (m_impl->key_to_index.find(new_name) != m_impl->key_to_index.end())
        return false;

    size_t index = it->second;
    Entry& entry = m_impl->entries[index];

    // Remove old mapping
    m_impl->key_to_index.erase(it);

    // Update the key in the entry
    entry.key = heap_string(new_name);

    // Add new mapping using the updated key
    m_impl->key_to_index[entry.key] = index;
    return true;
}

std::vector<std::string_view> Properties::keys() const
{
    std::vector<std::string_view> result;
    result.reserve(m_impl->key_to_index.size());
    for (const auto& entry : m_impl->entries) {
        if (entry.is_valid())
            result.push_back(entry.key);
    }
    return result;
}

PropertyType Properties::type(std::string_view name) const
{
    auto it = m_impl->key_to_index.find(name);
    if (it == m_impl->key_to_index.end())
        FALCOR_THROW("Property \"{}\" not found", name);
    return m_impl->entries[it->second].type();
}

std::string Properties::as_string(std::string_view name) const
{
    auto it = m_impl->key_to_index.find(name);
    if (it == m_impl->key_to_index.end())
        FALCOR_THROW("Property \"{}\" not found", name);
    std::string result;
    FormatVisitor visitor{result};
    std::visit(visitor, m_impl->entries[it->second].value);
    return result;
}

size_t Properties::hash() const
{
    // Collect and sort properties by name for order-independent hashing
    std::vector<std::pair<std::string_view, size_t>> sorted_props;
    sorted_props.reserve(m_impl->key_to_index.size());
    for (const auto& [key, index] : m_impl->key_to_index) {
        sorted_props.emplace_back(key, index);
    }
    std::sort(sorted_props.begin(), sorted_props.end());

    size_t h = 0;

    for (const auto& [key, index] : sorted_props) {
        h = sgl::hash_combine(h, sgl::hash(key));

        const auto& entry = m_impl->entries[index];
        // Hash the type
        h = sgl::hash_combine(h, sgl::hash(entry.value.index()));

        h = sgl::hash_combine(h, std::visit(HashVisitor{}, entry.value));
    }

    return h;
}

Properties::iterator Properties::begin() const
{
    return iterator(this, 0);
}

Properties::iterator Properties::end() const
{
    return iterator(this, npos);
}

Properties::iterator Properties::find(std::string_view name) const
{
    return iterator(this, key_index(name));
}

std::string Properties::to_string() const
{
    std::string result;
    if (empty())
        return "Properties({})";
    result += "Properties({\n";
    for (const auto& entry : m_impl->entries) {
        if (!entry.is_valid())
            continue;
        result += format_property_entry(entry, "  ");
        result += ",\n";
    }
    result += "})";
    return result;
}

size_t Properties::append(std::string_view name, bool throw_if_exists)
{
    auto it = m_impl->key_to_index.find(name);
    if (it != m_impl->key_to_index.end()) {
        if (throw_if_exists)
            FALCOR_THROW("Property \"{}\" already exists", name);
        return it->second;
    }
    size_t index = m_impl->entries.size();
    m_impl->entries.push_back({});
    Entry& entry = m_impl->entries.back();
    entry.key = heap_string(name);
    m_impl->key_to_index.emplace(entry.key, index);
    return index;
}

size_t Properties::key_index(std::string_view name) const noexcept
{
    auto it = m_impl->key_to_index.find(name);
    if (it == m_impl->key_to_index.end())
        return npos;
    return it->second;
}

size_t Properties::require_key_index(std::string_view name) const
{
    auto it = m_impl->key_to_index.find(name);
    if (it == m_impl->key_to_index.end())
        FALCOR_THROW("Property \"{}\" not found", name);
    return it->second;
}

std::string_view Properties::entry_name(size_t index) const
{
    FALCOR_ASSERT_LT(index, m_impl->entries.size());
    return m_impl->entries[index].key;
}

template<typename T>
void Properties::set_impl(size_t index, const T& value)
{
    FALCOR_ASSERT_LT(index, m_impl->entries.size());
    m_impl->entries[index].value = value;
}

template<typename T>
void Properties::set_impl(size_t index, T&& value)
{
    FALCOR_ASSERT_LT(index, m_impl->entries.size());
    m_impl->entries[index].value = std::forward<T>(value);
}

template<typename T>
const T* Properties::get_impl(size_t index, bool throw_if_incompatible) const
{
    FALCOR_ASSERT_LT(index, m_impl->entries.size());
    const Entry& entry = m_impl->entries[index];
    FALCOR_ASSERT(entry.is_valid());
    PropertyType expected_type = detail::PropertyValueTraits<T>::type;
    if (entry.type() != expected_type) {
        if (throw_if_incompatible)
            FALCOR_THROW("Property type mismatch (expected {}, got {})", expected_type, entry.type());
        else
            return nullptr;
    }
    return &std::get<T>(m_impl->entries[index].value);
}

[[noreturn]] void
Properties::throw_object_type_error(size_t index, const std::type_info& expected_type, const ref<Object>& value) const
{
    const Entry& entry = m_impl->entries[index];
    std::string expected_type_name = falcor::platform::demangle_type_name(expected_type);
    std::string actual_type_name = value ? value->class_name() : "null";
    FALCOR_THROW(
        "Property \"{}\" has an incompatible object type (expected {}, got {})",
        entry.key,
        expected_type_name,
        actual_type_name
    );
}

[[noreturn]] void Properties::throw_any_type_error(size_t index, const std::type_info& requested_type) const
{
    // Get the Any value to determine its actual type.
    const Any* any_value = get_impl<Any>(index, false);
    const Entry& entry = m_impl->entries[index];
    if (!any_value)
        FALCOR_THROW("Property \"{}\" has not been specified!", entry.key);

    std::string requested_type_name = falcor::platform::demangle_type_name(requested_type);
    std::string actual_type_name = falcor::platform::demangle_type_name(any_value->type());

    FALCOR_THROW(
        "Property \"{}\" cannot be cast to the requested type! (requested: {}, actual: {})",
        entry.key,
        requested_type_name,
        actual_type_name
    );
}

void Properties::copy_property(const iterator& source)
{
    FALCOR_CHECK(source.m_index != npos, "Cannot copy from invalid iterator");
    size_t dst_index = append(source.name(), false /* throw_if_exists */);
    m_impl->entries[dst_index].value = source.m_props->m_impl->entries[source.m_index].value;
}

bool Properties::is_equal_property(std::string_view name, const Properties& other) const
{
    size_t index = key_index(name);
    size_t other_index = other.key_index(name);
    if (index == npos || other_index == npos) {
        return (index == npos && other_index == npos);
    }

    const Entry& entry1 = m_impl->entries[index];
    const Entry& entry2 = other.m_impl->entries[other_index];
    if (entry1.value.index() != entry2.value.index())
        return false;
    return std::visit(CompareVisitor{entry2.value}, entry1.value);
}

template<typename T>
PropertyType Properties::check_type_impl()
{
    return detail::PropertyValueTraits<T>::type;
}

// ----------------------------------------------------------------------------
// Properties::iterator
// ----------------------------------------------------------------------------

Properties::iterator::iterator(const Properties* props, size_t index)
    : m_props(props)
    , m_index(index)
{
    skip_invalid_entries();
}

void Properties::iterator::skip_invalid_entries()
{
    if (m_index == npos)
        return;

    while (m_index < m_props->m_impl->entries.size()) {
        const Entry& entry = m_props->m_impl->entries[m_index];
        PropertyType type = entry.type();
        if (type != PropertyType::invalid) {
            m_name = entry.key;
            m_type = type;
            return;
        }
        ++m_index;
    }

    m_index = npos;
}

Properties::iterator& Properties::iterator::operator++()
{
    ++m_index;
    skip_invalid_entries();
    return *this;
}

Properties::iterator Properties::iterator::operator++(int)
{
    iterator copy = *this;
    ++(*this);
    return copy;
}

std::string Properties::iterator::to_string() const
{
    FALCOR_CHECK(m_props && m_index != size_t(-1), "Cannot stringify invalid iterator");

    const Entry& entry = m_props->m_impl->entries[m_index];
    FALCOR_CHECK(entry.is_valid(), "Cannot stringify invalid iterator");
    return format_property_entry(entry, "");
}

SGL_DIAGNOSTIC_PUSH
SGL_DISABLE_MSVC_WARNING(4003)
FALCOR_PROPERTY_ACCESSORS()
SGL_DIAGNOSTIC_POP

} // namespace falcor
