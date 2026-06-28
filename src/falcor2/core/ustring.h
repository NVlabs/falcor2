// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <compare>
#include <string>
#include <string_view>
#include <limits>

namespace falcor {

/// Class representing an unique (interned) string,
/// see https://en.wikipedia.org/wiki/String_interning for details.
///
/// When a new ustring is created, the string is compared to all
/// previously created strings stored in a global registry.
/// If found, the new ustring points to the same existing global string.
/// If not found, a new string is inserted into the registry.
/// As a result, all ustrings containing the same string point to the same memory,
/// turning ustring comparison into pointer comparison.
class FALCOR_API ustring {
private:
    // Header structure stored in front of m_unique.
    struct Header {
        size_t hash;
        size_t length;
    };

public:
    using value_type = char;
    using pointer = value_type*;
    using reference = value_type&;
    using const_reference = const value_type;
    using size_type = size_t;
    static constexpr size_type npos = std::numeric_limits<size_type>::max();
    using const_iterator = std::string::const_iterator;
    using const_reverse_iterator = std::string::const_reverse_iterator;

    ustring() { m_unique = make_unique({}); }
    explicit ustring(std::string_view str) { m_unique = make_unique(str); }
    explicit ustring(const char* str) { m_unique = make_unique(str ? std::string_view{str} : std::string_view{}); }
    explicit ustring(const char* str, size_t length)
    {
        m_unique = make_unique(str ? std::string_view{str, length} : std::string_view{});
    }

    ustring(const ustring& other) = default;
    ustring(ustring&& other) noexcept = default;
    ustring& operator=(const ustring& other) = default;
    ustring& operator=(ustring&& other) noexcept = default;

    constexpr const char* c_str() const noexcept { return m_unique; }
    constexpr const char* data() const noexcept { return m_unique; }
    std::string string() const { return std::string(*this); }
    bool empty() const noexcept { return length() == 0; }
    size_t size() const noexcept { return length(); }
    size_t length() const noexcept { return header().length; }
    size_t hash() const noexcept { return header().hash; }

    void reset(std::string_view s) { m_unique = make_unique(s); }
    void reset(const char* s) { reset(s ? std::string_view(s) : std::string_view{}); }
    void reset(const char* s, size_t length) { reset(s ? std::string_view(s, length) : std::string_view{}); }

    /// Conversion to std::string_view.
    operator std::string_view() const noexcept { return std::string_view(c_str(), length()); }

    /// Conversion to std::string.
    explicit operator std::string() const noexcept { return std::string(c_str(), length()); }

    constexpr auto operator<=>(const ustring& rhs) const
    {
        if (m_unique == rhs.m_unique)
            return 0;
        return (m_unique < rhs.m_unique) ? -1 : 1;
    }

    bool operator==(const ustring& rhs) const { return m_unique == rhs.m_unique; }
    bool operator!=(const ustring& rhs) const { return m_unique != rhs.m_unique; }

    bool operator==(std::string_view rhs) const { return std::string_view(*this) == rhs; }
    friend bool operator==(std::string_view lhs, const ustring& rhs) { return lhs == std::string_view(rhs); }

    bool operator!=(std::string_view rhs) const { return !(*this == rhs); }
    friend bool operator!=(std::string_view lhs, const ustring& rhs) { return !(lhs == rhs); }

private:
    const Header& header() const { return *(reinterpret_cast<const Header*>(m_unique) - 1); }

    static const char* make_unique(std::string_view s);

    const char* m_unique;

    class Registry;
};

} // namespace falcor
