// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <span>

namespace falcor {

/// Turns array of values into a set of values, and an array of indices to them.
/// Effectively deduplicating the values.
template<typename T, typename TIndex, typename THash = std::hash<T>, typename TEqual = std::equal_to<T>>
class IndexedVector {
public:
    /// Appends value to the vector, returns the index at which it is positioned,
    /// and returns true if the value already existed.
    bool append(const T& v, uint32_t& out_index)
    {
        auto it = m_index_map.find(v);
        if (it == m_index_map.end()) {
            m_index_map.insert(std::make_pair(v, TIndex(m_values.size())));
            out_index = (uint32_t)m_values.size();
            m_values.push_back(v);
            m_indices.push_back(out_index);
            return false;
        }

        m_indices.push_back(it->second);
        return true;
    }

    bool append(const T& v)
    {
        uint32_t idx;
        return append(v, idx);
    }

    std::span<const T> get_values() const { return m_values; }
    std::span<const TIndex> get_indices() const { return m_indices; }

private:
    std::unordered_map<T, TIndex, THash, TEqual> m_index_map;
    std::vector<T> m_values;
    std::vector<TIndex> m_indices;
};

} // namespace falcor
