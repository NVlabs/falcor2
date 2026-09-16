// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "core/reflection/python_object_factory.h"
#include "core/reflection/python_property_descriptor.h"

#include "falcor2/core/reflected_object.h"

#include <string>
#include <utility>

namespace {

falcor::ref<falcor::reflection::PythonObjectFactory> make_python_object_factory(nb::handle info)
{
    return falcor::make_ref<falcor::reflection::PythonObjectFactory>(info.attr("object_factories"));
}

} // namespace

FALCOR_PY_EXPORT(testing_reflection)
{
    nb::module_ native = nb::module_::import_("falcor2.testing._native");

    native.def(
        "_property_enum_is_flags",
        [](nb::object info)
        {
            falcor::reflection::PythonPropertyDescriptor descriptor(std::move(info));
            const auto* enum_descriptor = descriptor.enum_descriptor();
            return enum_descriptor && enum_descriptor->is_flags;
        }
    );

    native.def(
        "_object_factory_labels",
        [](nb::object info)
        {
            auto factory = make_python_object_factory(info);
            nb::list labels;
            for (size_t i = 0; i < factory->count(); ++i)
                labels.append(nb::cast(std::string(factory->label(i))));
            return labels;
        }
    );
    native.def(
        "_object_factory_find_index",
        [](nb::object info, nb::object object) -> nb::object
        {
            auto factory = make_python_object_factory(info);
            if (object.is_none()) {
                auto index = factory->find_index(nullptr);
                return index ? nb::cast(*index) : nb::none();
            }
            auto native_object = nb::cast<falcor::ref<falcor::ReflectedObject>>(object);
            auto index = factory->find_index(native_object.get());
            return index ? nb::cast(*index) : nb::none();
        },
        nb::arg("info"),
        nb::arg("object").none()
    );
    native.def(
        "_object_factory_create",
        [](nb::object info, nb::object owner, size_t index)
        {
            auto factory = make_python_object_factory(info);
            nb::handle owner_handle = owner;
            auto object = factory->create(index, &owner_handle);
            return object ? nb::cast(object) : nb::none();
        }
    );
}
