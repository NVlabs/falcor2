// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <cstddef>
#include <optional>
#include <string_view>

namespace falcor {

class ReflectedObject;

namespace reflection {

/// Creates concrete ReflectedObject instances for an object-valued property.
///
/// A factory exposes the labeled construction entries available to one property.
/// The opaque owner pointer follows the same convention as the property's
/// PropertyDescriptor instance pointer.
class FALCOR_API ReflectedObjectFactory : public Object {
    FALCOR_OBJECT(ReflectedObjectFactory)
public:
    virtual size_t count() const = 0;
    virtual std::string_view label(size_t index) const = 0;
    /// Find the entry matching an object, including the null entry when available.
    virtual std::optional<size_t> find_index(const ReflectedObject* object) const = 0;
    virtual ref<ReflectedObject> create(size_t index, void* owner) const = 0;
};

} // namespace reflection
} // namespace falcor
