// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "python_property_descriptor.h"

#include "core/any.h"

#include "falcor2/core/error.h"

namespace falcor::reflection {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Build the metadata vector from a Python PythonPropertyInfo.
std::vector<Any> build_metadata(nb::handle info)
{
    std::vector<Any> metadata;

    // doc string
    nb::object doc = info.attr("doc");
    if (!doc.is_none())
        metadata.push_back(Any(nb::cast<std::string>(doc)));

    // value_range
    nb::object value_range = info.attr("value_range");
    if (!value_range.is_none()) {
        nb::tuple t = nb::borrow<nb::tuple>(value_range);
        metadata.push_back(Any(ValueRange{nb::cast<double>(t[0]), nb::cast<double>(t[1])}));
    }

    // enum_type -> EnumDescriptor
    nb::object enum_type = info.attr("enum_type");
    if (!enum_type.is_none()) {
        EnumDescriptor desc;
        nb::object members = enum_type.attr("__members__");
        for (auto item : members.attr("items")()) {
            nb::tuple kv = nb::borrow<nb::tuple>(item);
            std::string name = nb::cast<std::string>(kv[0]);
            int64_t value = nb::cast<int64_t>(kv[1].attr("value"));
            desc.items.push_back(EnumItem{value, std::move(name)});
        }
        metadata.push_back(Any(std::move(desc)));
    }

    // ui_flags
    nb::object ui_flags = info.attr("ui_flags");
    if (!ui_flags.is_none())
        metadata.push_back(Any(nb::cast<UIFlags>(ui_flags)));

    // ui_label
    nb::object ui_label = info.attr("ui_label");
    if (!ui_label.is_none())
        metadata.push_back(Any(UILabel{nb::cast<std::string>(ui_label)}));

    // ui_group
    nb::object ui_group = info.attr("ui_group");
    if (!ui_group.is_none())
        metadata.push_back(Any(UIGroup{nb::cast<std::string>(ui_group)}));

    // ui_drag_speed
    nb::object ui_drag_speed = info.attr("ui_drag_speed");
    if (!ui_drag_speed.is_none())
        metadata.push_back(Any(UIDragSpeed{nb::cast<float>(ui_drag_speed)}));

    // ui_enable_if -- wraps a Python predicate as a C++ UIEnableIf.
    // The void* instance parameter is a nb::handle* pointing to the Python object.
    nb::object ui_enable_if = info.attr("ui_enable_if");
    if (!ui_enable_if.is_none()) {
        nb::object py_pred = nb::borrow(ui_enable_if);
        metadata.push_back(
            Any(UIEnableIf{
                [py_pred](const void* instance) -> bool
                {
                    nb::gil_scoped_acquire gil;
                    const nb::handle* obj = static_cast<const nb::handle*>(instance);
                    return nb::cast<bool>(py_pred(*obj));
                }
            })
        );
    }

    // on_change -- wraps a Python callback as a C++ OnChange.
    // The void* instance parameter is a nb::handle* pointing to the Python object.
    nb::object on_change = info.attr("on_change");
    if (!on_change.is_none()) {
        nb::object py_callback = nb::borrow(on_change);
        metadata.push_back(
            Any(OnChange{[py_callback](void* instance)
                         {
                             nb::gil_scoped_acquire gil;
                             nb::handle* obj = static_cast<nb::handle*>(instance);
                             py_callback(*obj);
                         }})
        );
    }

    return metadata;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// PythonPropertyDescriptor
// ---------------------------------------------------------------------------

PythonPropertyDescriptor::PythonPropertyDescriptor(nb::object info)
    : PropertyDescriptor(nb::cast<std::string>(info.attr("name")), info.attr("setter").is_none(), build_metadata(info))
    , m_info(std::move(info))
    , m_getter(nb::borrow(m_info.attr("getter")))
    , m_setter(nb::borrow(m_info.attr("setter")))
    , m_default_value(nb::borrow(m_info.attr("default_value")))
    , m_is_enum(false)
{
    // Resolve the type map entry by Python type object (not by string).
    nb::object value_type = m_info.attr("value_type");
    if (!value_type.is_none())
        m_type_entry = property_type_map_find(value_type);

    // Resolve enum type.
    nb::object et = m_info.attr("enum_type");
    if (!et.is_none()) {
        m_enum_type = et;
        m_is_enum = true;
    } else if (!value_type.is_none() && !m_type_entry) {
        // If value_type is set but not found in the type map, check if it's an IntEnum subclass.
        nb::object int_enum = nb::module_::import_("enum").attr("IntEnum");
        m_is_enum = PyObject_IsSubclass(value_type.ptr(), int_enum.ptr()) == 1;
    }
}

const std::type_info& PythonPropertyDescriptor::type() const
{
    if (m_type_entry)
        return *m_type_entry->type_info;
    if (m_is_enum)
        return typeid(int64_t);
    return typeid(nb::object);
}

bool PythonPropertyDescriptor::has_default_value() const
{
    return !m_default_value.is_none();
}

bool PythonPropertyDescriptor::is_default(const void* instance) const
{
    if (m_default_value.is_none())
        return false;
    nb::gil_scoped_acquire gil;
    nb::object current = py_get(instance);
    return current.equal(m_default_value);
}

Any PythonPropertyDescriptor::get_any(const void* instance) const
{
    nb::gil_scoped_acquire gil;
    nb::object val = py_get(instance);

    if (m_type_entry)
        return m_type_entry->to_any(val);
    if (m_is_enum)
        return Any(nb::cast<int64_t>(val.attr("value")));

    // Fallback: wrap as Python object via any_wrap.
    return any_wrap(val);
}

void PythonPropertyDescriptor::set_any(void* instance, const Any& value) const
{
    FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
    nb::gil_scoped_acquire gil;

    if (m_type_entry) {
        nb::object py_val = m_type_entry->from_any(value);
        py_set(instance, py_val);
        return;
    }

    if (m_is_enum) {
        const int64_t* iv = any_cast<int64_t>(&value);
        FALCOR_CHECK(iv != nullptr, "Property \"{}\": expected int64_t Any for enum.", m_name);
        if (m_enum_type.is_valid()) {
            nb::object py_val = m_enum_type(*iv);
            py_set(instance, py_val);
        } else {
            py_set(instance, nb::cast(*iv));
        }
        return;
    }

    // Fallback: try to unwrap a Python-wrapped Any first, then fall back to nanobind type lookup.
    auto py_handle = any_try_unwrap(value);
    if (py_handle) {
        py_set(instance, py_handle);
        return;
    }
    nb::handle py_obj = nb_type_to_python(value.type(), const_cast<void*>(value.data()));
    FALCOR_CHECK(py_obj.is_valid(), "Property \"{}\": cannot convert Any to Python object.", m_name);
    py_set(instance, py_obj);
}

int64_t PythonPropertyDescriptor::get_enum_as_int64(const void* instance) const
{
    FALCOR_CHECK(m_is_enum, "Property \"{}\": type is not an enum.", m_name);
    nb::gil_scoped_acquire gil;
    nb::object val = py_get(instance);
    return nb::cast<int64_t>(val.attr("value"));
}

void PythonPropertyDescriptor::set_enum_from_int64(void* instance, int64_t value) const
{
    FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
    FALCOR_CHECK(m_is_enum, "Property \"{}\": type is not an enum.", m_name);
    nb::gil_scoped_acquire gil;
    if (m_enum_type.is_valid()) {
        nb::object py_val = m_enum_type(value);
        py_set(instance, py_val);
    } else {
        py_set(instance, nb::cast(value));
    }
}

bool PythonPropertyDescriptor::is_serializable_to_properties() const
{
    // Supported if we have a type map entry (scalar/vector types) or if it's an enum.
    return m_type_entry != nullptr || m_is_enum;
}

void PythonPropertyDescriptor::write_to_properties(const void* instance, Properties& props) const
{
    nb::gil_scoped_acquire gil;
    nb::object val = py_get(instance);

    if (m_type_entry) {
        m_type_entry->write_to_props(props, m_name, val);
        return;
    }

    if (m_is_enum) {
        int64_t iv = nb::cast<int64_t>(val.attr("value"));
        props.set(m_name, falcor::detail::PropertyEnumValue{nullptr, iv});
        return;
    }

    FALCOR_THROW("Property \"{}\": type is not serializable to Properties.", m_name);
}

bool PythonPropertyDescriptor::read_from_properties(void* instance, const Properties& props) const
{
    if (m_read_only)
        return false;
    if (!props.has_property(m_name))
        return false;

    nb::gil_scoped_acquire gil;

    if (m_type_entry) {
        nb::object py_val = m_type_entry->read_from_props(props, m_name);
        if (py_val.is_none())
            return false;
        py_set(instance, py_val);
        return true;
    }

    if (m_is_enum) {
        auto ev = props.get<falcor::detail::PropertyEnumValue>(m_name);
        if (m_enum_type.is_valid()) {
            nb::object py_val = m_enum_type(ev.value);
            py_set(instance, py_val);
        } else {
            py_set(instance, nb::cast(ev.value));
        }
        return true;
    }

    return false;
}

void PythonPropertyDescriptor::reset(void* instance) const
{
    if (m_read_only || m_default_value.is_none())
        return;
    nb::gil_scoped_acquire gil;
    py_set(instance, m_default_value);
}

void PythonPropertyDescriptor::get_value(const void* instance, void* out) const
{
    FALCOR_UNUSED(instance);
    FALCOR_UNUSED(out);
    // get_value is the low-level typed path which requires compile-time type knowledge.
    // Python properties should use get_any() instead.
    FALCOR_THROW("Property \"{}\": get_value() not supported for Python properties. Use get_any().", m_name);
}

void PythonPropertyDescriptor::set_value(void* instance, const void* value) const
{
    FALCOR_UNUSED(instance);
    FALCOR_UNUSED(value);
    // set_value is the low-level typed path which requires compile-time type knowledge.
    // Python properties should use set_any() instead.
    FALCOR_THROW("Property \"{}\": set_value() not supported for Python properties. Use set_any().", m_name);
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

nb::object PythonPropertyDescriptor::py_get(const void* instance) const
{
    const nb::handle* obj = static_cast<const nb::handle*>(instance);
    return m_getter(*obj);
}

void PythonPropertyDescriptor::py_set(void* instance, nb::handle value) const
{
    nb::handle* obj = static_cast<nb::handle*>(instance);
    m_setter(*obj, value);
}

} // namespace falcor::reflection
