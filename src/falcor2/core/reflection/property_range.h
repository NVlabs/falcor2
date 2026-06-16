// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"

#include <cstddef>
#include <iterator>
#include <span>
#include <vector>

namespace falcor::reflection {

class ClassDescriptor;
class PropertyDescriptor;

// ----------------------------------------------------------------------------
// PropertyRange
// ----------------------------------------------------------------------------

/// A lightweight, non-owning range that iterates over PropertyDescriptor
/// pointers from one or more ClassDescriptors in the inheritance chain.
///
/// When constructed with `include_base = true`, properties are yielded in
/// base-first order (root of the registered chain first, most-derived last).
///
/// This is a value type -- no shared mutable state, no mutex, no caching.
/// Each construction walks the base() chain to collect ClassDescriptor
/// pointers (typically 1-3 levels deep).
class FALCOR_API PropertyRange {
public:
    /// Forward iterator over const PropertyDescriptor pointers.
    class Iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const PropertyDescriptor*;
        using difference_type = std::ptrdiff_t;
        using pointer = const PropertyDescriptor* const*;
        using reference = const PropertyDescriptor*;

        Iterator() = default;

        reference operator*() const { return m_range->m_spans[m_span_idx][m_prop_idx]; }

        Iterator& operator++()
        {
            ++m_prop_idx;
            advance_to_valid();
            return *this;
        }

        Iterator operator++(int)
        {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator==(const Iterator& other) const
        {
            FALCOR_ASSERT(m_range == other.m_range);
            return m_span_idx == other.m_span_idx && m_prop_idx == other.m_prop_idx;
        }

        bool operator!=(const Iterator& other) const { return !(*this == other); }

    private:
        friend class PropertyRange;

        Iterator(const PropertyRange* range, size_t span_idx, size_t prop_idx)
            : m_range(range)
            , m_span_idx(span_idx)
            , m_prop_idx(prop_idx)
        {
        }

        /// Advance past any empty spans to the next valid property.
        void advance_to_valid()
        {
            while (m_span_idx < m_range->m_spans.size() && m_prop_idx >= m_range->m_spans[m_span_idx].size()) {
                ++m_span_idx;
                m_prop_idx = 0;
            }
        }

        const PropertyRange* m_range{nullptr};
        size_t m_span_idx{0};
        size_t m_prop_idx{0};
    };

    using iterator = Iterator;
    using const_iterator = Iterator;

    /// Construct a range over just the own properties of a single ClassDescriptor.
    static PropertyRange own(const ClassDescriptor& desc);

    /// Construct a range over all properties in the inheritance chain (base first).
    static PropertyRange all(const ClassDescriptor& desc);

    Iterator begin() const
    {
        Iterator it(this, 0, 0);
        it.advance_to_valid();
        return it;
    }

    Iterator end() const { return Iterator(this, m_spans.size(), 0); }

    /// Returns true if the range contains no properties.
    bool empty() const { return begin() == end(); }

    /// Count the number of properties in the range.
    /// Note: this is O(n) where n is the number of spans (i.e. depth of inheritance chain).
    size_t size() const
    {
        size_t count = 0;
        for (const auto& span : m_spans)
            count += span.size();
        return count;
    }

private:
    /// Spans of property pointers, one per ClassDescriptor in the chain (base first).
    /// For own-only ranges, this contains a single span.
    std::vector<std::span<const PropertyDescriptor* const>> m_spans;
};

} // namespace falcor::reflection
