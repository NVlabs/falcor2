// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/core/reflected_object.h"
#include "falcor2/core/properties.h"

namespace falcor {

// Defined in properties.cpp
nb::object properties_getitem(const Properties& props, std::string_view key);
bool properties_setitem(Properties& props, std::string_view key, const nb::handle& value);

} // namespace falcor

FALCOR_PY_EXPORT(core_reflected_object)
{
    using namespace falcor;

    nb::class_<ReflectedObject, Object>(m, "ReflectedObject")
        .DEF_PROP_RO(ReflectedObject, properties)
        .def("set_properties", &ReflectedObject::set_properties, D(ReflectedObject, set_properties))
        .def(
            "__setitem__",
            [](ReflectedObject& self, std::string_view key, nb::handle value)
            {
                const auto* desc = self.find_property(key);
                if (desc) {
                    // TODO: Optimize by writing directly to the property.
                    Properties props;
                    properties_setitem(props, key, value);
                    desc->read_from_properties(&self, props);
                } else {
                    // Fall back to full set_properties for unknown properties
                    // (e.g. if a subclass overrides set_properties with custom logic).
                    Properties props;
                    properties_setitem(props, key, value);
                    self.set_properties(props);
                }
            }
        )
        .def(
            "__getitem__",
            [](const ReflectedObject& self, std::string_view key)
            {
                const auto* desc = self.find_property(key);
                if (desc) {
                    // TODO: Optimize by reading directly from the property.
                    Properties props;
                    desc->write_to_properties(&self, props);
                    return properties_getitem(props, key);
                }
                // Fall back to full properties() for unknown properties.
                return properties_getitem(self.properties(), key);
            }
        )
        .def(
            "__contains__",
            [](const ReflectedObject& self, std::string_view key)
            {
                return self.has_property(key);
            }
        );
}
