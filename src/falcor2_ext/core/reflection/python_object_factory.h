// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"

#include "falcor2/core/reflection/reflected_object_factory.h"

#include <string>
#include <vector>

namespace falcor::reflection {

/// Python-backed factory for a reflected-object property.
class PythonObjectFactory final : public ReflectedObjectFactory {
    FALCOR_OBJECT(PythonObjectFactory)
public:
    explicit PythonObjectFactory(nb::handle factories);

    size_t count() const override { return m_entries.size(); }
    std::string_view label(size_t index) const override;
    std::optional<size_t> find_index(const ReflectedObject* object) const override;
    ref<ReflectedObject> create(size_t index, void* owner) const override;

private:
    struct Entry {
        nb::object object_type;
        nb::object factory;
        std::string label;
    };

    std::vector<Entry> m_entries;
};

} // namespace falcor::reflection
