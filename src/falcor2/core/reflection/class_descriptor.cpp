// SPDX-License-Identifier: Apache-2.0

#include "class_descriptor.h"

#include "falcor2/core/reflection/type_registry.h"

namespace falcor::reflection {

ClassDescriptor::ClassDescriptor(
    std::string name,
    const std::type_info& type_info,
    const std::type_info& base_type_info
)
    : m_name(std::move(name))
    , m_type_info(type_info)
    , m_base_type_info(base_type_info)
{
}

ClassDescriptor::~ClassDescriptor()
{
    for (auto* p : m_properties)
        delete p;
}

void ClassDescriptor::resolve_base() const
{
    if (m_base_resolved.load(std::memory_order_acquire))
        return;

    // No mutex needed: resolving twice is harmless (idempotent read-only lookup).
    // typeid(void) means no base was declared.
    if (m_base_type_info != typeid(void))
        m_base.store(TypeRegistry::get().find(m_base_type_info), std::memory_order_release);

    m_base_resolved.store(true, std::memory_order_release);
}

const ClassDescriptor* ClassDescriptor::base() const
{
    resolve_base();
    return m_base.load(std::memory_order_acquire);
}

const PropertyDescriptor* ClassDescriptor::find_property(std::string_view name) const
{
    // Search own properties first.
    auto it = m_property_index.find(name);
    if (it != m_property_index.end())
        return m_properties[it->second];

    // Walk up the registered base chain.
    if (const ClassDescriptor* b = base())
        return b->find_property(name);

    return nullptr;
}

Any ClassDescriptor::get_any(const void* instance, std::string_view name) const
{
    const PropertyDescriptor* prop = find_property(name);
    FALCOR_CHECK(prop != nullptr, "ClassDescriptor \"{}\": property \"{}\" not found.", m_name, name);
    return prop->get_any(instance);
}

void ClassDescriptor::set_any(void* instance, std::string_view name, const Any& value) const
{
    const PropertyDescriptor* prop = find_property(name);
    FALCOR_CHECK(prop != nullptr, "ClassDescriptor \"{}\": property \"{}\" not found.", m_name, name);
    prop->set_any(instance, value);
}

void ClassDescriptor::write_to_properties(const void* instance, Properties& props) const
{
    for (auto* p : all_property_range()) {
        if (p->is_serializable_to_properties())
            p->write_to_properties(instance, props);
    }
}

void ClassDescriptor::read_from_properties(void* instance, const Properties& props) const
{
    for (auto* p : all_property_range()) {
        if (p->is_serializable_to_properties() && !p->is_read_only())
            p->read_from_properties(instance, props);
    }
}

void ClassDescriptor::reset(void* instance) const
{
    for (auto* p : all_property_range())
        p->reset(instance);
}

void ClassDescriptor::add_property(PropertyDescriptor* prop)
{
    m_property_index[prop->name()] = m_properties.size();
    m_properties.push_back(prop);
}

} // namespace falcor::reflection
