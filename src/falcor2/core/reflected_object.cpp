// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "reflected_object.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/reflection/class_descriptor.h"
#include "falcor2/core/reflection/dynamic_properties.h"
#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/type_registry.h"

namespace falcor {

Properties ReflectedObject::properties() const
{
    Properties props;

    // Serialize static properties.
    class_descriptor().write_to_properties(this, props);

    // Serialize dynamic properties (if any).
    if (const auto* dyn = dynamic_properties())
        dyn->write_to_properties(props);

    return props;
}

void ReflectedObject::set_properties(const Properties& props)
{
    // Deserialize static properties.
    class_descriptor().read_from_properties(this, props);

    // Deserialize dynamic properties (if any).
    if (auto* dyn = dynamic_properties())
        dyn->read_from_properties(props);
}

const reflection::PropertyDescriptor* ReflectedObject::find_property(std::string_view name) const
{
    // Search static properties first.
    if (const auto* prop = class_descriptor().find_property(name))
        return prop;

    // Then dynamic properties.
    if (const auto* dyn = dynamic_properties())
        return dyn->find_property(name);

    return nullptr;
}

bool ReflectedObject::has_property(std::string_view name) const
{
    return find_property(name) != nullptr;
}

Any ReflectedObject::get_property(std::string_view name) const
{
    const reflection::PropertyDescriptor* prop = find_property(name);
    FALCOR_CHECK(prop != nullptr, "Property \"{}\" not found.", name);
    return prop->get_any(this);
}

void ReflectedObject::set_property(std::string_view name, const Any& value)
{
    const reflection::PropertyDescriptor* prop = find_property(name);
    FALCOR_CHECK(prop != nullptr, "Property \"{}\" not found.", name);
    prop->set_any(this, value);
}

} // namespace falcor
