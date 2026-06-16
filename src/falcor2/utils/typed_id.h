// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>

#include <cstdint>

namespace falcor {

/// Strongly typed ID.
template<typename Tag, typename Index = uint32_t>
class TypedID {
public:
    static constexpr Index INVALID_INDEX = std::numeric_limits<Index>::max();

    TypedID() = default;
    explicit TypedID(Index index)
        : m_index(index)
    {
    }
    TypedID(const TypedID&) = default;
    TypedID(TypedID&&) = default;
    TypedID& operator=(const TypedID&) = default;
    TypedID& operator=(TypedID&&) = default;
    ~TypedID() = default;

    Index index() const { return m_index; }

    bool operator==(const TypedID& other) const { return m_index == other.m_index; }
    bool operator!=(const TypedID& other) const { return m_index != other.m_index; }

private:
    Index m_index{INVALID_INDEX};
};

} // namespace falcor
