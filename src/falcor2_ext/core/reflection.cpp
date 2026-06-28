// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "core/reflection/python_class_reflection.h"
#include "core/reflection/python_property_descriptor.h"

#include "falcor2/core/reflection/property_descriptor.h"
#include "falcor2/core/reflection/metadata.h"
#include "falcor2/ui/property_editor.h"

FALCOR_PY_EXPORT(core_reflection)
{
    using namespace falcor;
    using namespace falcor::reflection;

    nb::module_ reflection = nb::module_::import_("falcor2.reflection");

    nb::sgl_enum_flags<UIFlags>(reflection, "UIFlags");

    nb::class_<PythonPropertyDescriptor>(reflection, "PythonPropertyDescriptor")
        .def(nb::init<nb::object>(), "info"_a)
        .def_prop_ro("name", &PythonPropertyDescriptor::name)
        .def_prop_ro("is_read_only", &PythonPropertyDescriptor::is_read_only)
        .def_prop_ro("has_default_value", &PythonPropertyDescriptor::has_default_value)
        .def_prop_ro("is_serializable_to_properties", &PythonPropertyDescriptor::is_serializable_to_properties)
        .def_prop_ro(
            "doc",
            [](const PythonPropertyDescriptor& self) -> nb::object
            {
                auto sv = self.doc();
                if (sv.empty())
                    return nb::none();
                return nb::cast(std::string(sv));
            },
            nb::sig("def doc(self) -> str | None")
        )
        .def_prop_ro(
            "value_range",
            [](const PythonPropertyDescriptor& self) -> nb::object
            {
                const ValueRange* vr = self.value_range();
                if (!vr)
                    return nb::none();
                return nb::make_tuple(vr->min, vr->max);
            },
            nb::sig("def value_range(self) -> tuple[float, float] | None")
        )
        .def_prop_ro(
            "ui_group",
            [](const PythonPropertyDescriptor& self) -> nb::object
            {
                const UIGroup* g = self.ui_group();
                if (!g)
                    return nb::none();
                return nb::cast(g->path);
            },
            nb::sig("def ui_group(self) -> str | None")
        )
        .def_prop_ro(
            "ui_label",
            [](const PythonPropertyDescriptor& self) -> nb::object
            {
                const UILabel* l = self.ui_label();
                if (!l)
                    return nb::none();
                return nb::cast(l->text);
            },
            nb::sig("def ui_label(self) -> str | None")
        )
        .def_prop_ro(
            "ui_drag_speed",
            [](const PythonPropertyDescriptor& self) -> nb::object
            {
                const UIDragSpeed* ds = self.ui_drag_speed();
                if (!ds)
                    return nb::none();
                return nb::cast(ds->speed);
            },
            nb::sig("def ui_drag_speed(self) -> float | None")
        )
        .def_prop_ro("ui_flags", &PropertyDescriptor::ui_flags)
        .def_prop_ro(
            "has_ui_enable_if",
            [](const PythonPropertyDescriptor& self) -> bool
            {
                return self.ui_enable_if() != nullptr;
            }
        )
        .def(
            "get_any_value",
            [](const PythonPropertyDescriptor& self, nb::object instance) -> nb::object
            {
                nb::handle handle = instance;
                Any a = self.get_any(static_cast<const void*>(&handle));
                if (!a.has_value())
                    return nb::none();
                // Use the type map to convert back.
                const PropertyTypeMapEntry* entry = property_type_map_find(a.type());
                if (entry)
                    return entry->from_any(a);
                // Fallback for enum.
                const int64_t* iv = any_cast<int64_t>(&a);
                if (iv)
                    return nb::cast(*iv);
                return nb::none();
            },
            "instance"_a
        )
        .def(
            "set_any_value",
            [](PythonPropertyDescriptor& self, nb::object instance, nb::object value)
            {
                nb::handle handle = instance;
                // Use the type map to convert to Any.
                const PropertyTypeMapEntry* entry = property_type_map_find(self.type());
                if (entry) {
                    Any a = entry->to_any(value);
                    self.set_any(static_cast<void*>(&handle), a);
                    return;
                }
                // Fallback for enum.
                if (nb::isinstance<nb::int_>(value)) {
                    Any a(nb::cast<int64_t>(value));
                    self.set_any(static_cast<void*>(&handle), a);
                    return;
                }
                FALCOR_THROW("PythonPropertyDescriptor: cannot convert value for set_any_value.");
            },
            "instance"_a,
            "value"_a
        )
        .def(
            "is_default",
            [](const PythonPropertyDescriptor& self, nb::object instance) -> bool
            {
                nb::handle handle = instance;
                return self.is_default(static_cast<const void*>(&handle));
            },
            "instance"_a
        )
        .def(
            "reset",
            [](PythonPropertyDescriptor& self, nb::object instance)
            {
                nb::handle handle = instance;
                self.reset(static_cast<void*>(&handle));
            },
            "instance"_a
        )
        .def(
            "get_enum_as_int64",
            [](const PythonPropertyDescriptor& self, nb::object instance) -> int64_t
            {
                nb::handle handle = instance;
                return self.get_enum_as_int64(static_cast<const void*>(&handle));
            },
            "instance"_a
        )
        .def(
            "set_enum_from_int64",
            [](PythonPropertyDescriptor& self, nb::object instance, int64_t value)
            {
                nb::handle handle = instance;
                self.set_enum_from_int64(static_cast<void*>(&handle), value);
            },
            "instance"_a,
            "value"_a
        )
        .def(
            "write_to_properties",
            [](const PythonPropertyDescriptor& self, nb::object instance, Properties& props)
            {
                nb::handle handle = instance;
                self.write_to_properties(static_cast<const void*>(&handle), props);
            },
            "instance"_a,
            "props"_a
        )
        .def(
            "read_from_properties",
            [](PythonPropertyDescriptor& self, nb::object instance, const Properties& props) -> bool
            {
                nb::handle handle = instance;
                return self.read_from_properties(static_cast<void*>(&handle), props);
            },
            "instance"_a,
            "props"_a
        );

    nb::class_<PythonClassReflection>(reflection, "PythonClassReflection")
        .def(nb::init<nb::object>(), "py_class"_a)
        .def_prop_ro("class_name", &PythonClassReflection::class_name)
        .def_prop_ro("generation", &PythonClassReflection::generation)
        .def("__len__", &PythonClassReflection::size)
        .def(
            "__getitem__",
            [](const PythonClassReflection& self, size_t index) -> const PythonPropertyDescriptor*
            {
                FALCOR_CHECK(index < self.size(), "Index {} out of range.", index);
                return static_cast<const PythonPropertyDescriptor*>(self[index]);
            },
            nb::rv_policy::reference_internal,
            "index"_a
        )
        .def(
            "find_property",
            [](const PythonClassReflection& self, std::string_view name) -> const PythonPropertyDescriptor*
            {
                return static_cast<const PythonPropertyDescriptor*>(self.find_property(name));
            },
            nb::rv_policy::reference_internal,
            "name"_a
        )
        .def("rebuild", &PythonClassReflection::rebuild)
        .def("ensure_up_to_date", &PythonClassReflection::ensure_up_to_date)
        .def_static("clear_cache", &PythonClassReflection::clear_cache);
}
