// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "type_registry.h"

#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"

namespace falcor::reflection {

TypeRegistry& TypeRegistry::get()
{
    static TypeRegistry instance;
    return instance;
}

TypeRegistry::~TypeRegistry()
{
    for (const ClassDescriptor* desc : m_descs)
        delete desc;
}

void TypeRegistry::register_class(const ClassDescriptor* desc)
{
    std::type_index idx(desc->type());

    FALCOR_CHECK(
        m_by_type.find(idx) == m_by_type.end(),
        "TypeRegistry: class \"{}\" (type \"{}\") is already registered.",
        desc->name(),
        platform::demangle_type_name(desc->type())
    );

    auto name_it = m_by_name.find(desc->name());
    FALCOR_CHECK(
        name_it == m_by_name.end(),
        "TypeRegistry: a class with name \"{}\" is already registered (existing type \"{}\", new type \"{}\").",
        desc->name(),
        platform::demangle_type_name(name_it->second->type()),
        platform::demangle_type_name(desc->type())
    );

    m_descs.push_back(desc);
    m_by_type[idx] = desc;
    m_by_name[desc->name()] = desc;
}

const ClassDescriptor* TypeRegistry::find(const std::type_info& type) const
{
    auto it = m_by_type.find(std::type_index(type));
    return it != m_by_type.end() ? it->second : nullptr;
}

const ClassDescriptor* TypeRegistry::find(std::string_view name) const
{
    auto it = m_by_name.find(name);
    return it != m_by_name.end() ? it->second : nullptr;
}

const ClassDescriptor& get_class_descriptor(const std::type_info& type)
{
    const ClassDescriptor* desc = TypeRegistry::get().find(type);
    if (!desc)
        FALCOR_THROW("TypeRegistry: no class registered for type \"{}\".", platform::demangle_type_name(type));
    return *desc;
}

} // namespace falcor::reflection
