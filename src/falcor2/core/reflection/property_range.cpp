// SPDX-License-Identifier: Apache-2.0

#include "property_range.h"

#include "falcor2/core/reflection/class_descriptor.h"

#include <algorithm>

namespace falcor::reflection {

PropertyRange PropertyRange::own(const ClassDescriptor& desc)
{
    PropertyRange range;
    range.m_spans.push_back(desc.properties());
    return range;
}

PropertyRange PropertyRange::all(const ClassDescriptor& desc)
{
    PropertyRange range;

    // Walk the base chain and collect property spans.
    const ClassDescriptor* cur = &desc;
    while (cur) {
        range.m_spans.push_back(cur->properties());
        cur = cur->base();
    }

    // Reverse so that base properties come first.
    std::reverse(range.m_spans.begin(), range.m_spans.end());
    return range;
}

} // namespace falcor::reflection
