// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "python_property_descriptor.h"
#include "python_object_factory.h"

#include "core/any.h"

#include "falcor2/core/error.h"
#include "falcor2/core/object.h"
#include "falcor2/core/reflected_object.h"

namespace falcor::reflection {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {
bool is_python_subclass(nb::handle type, nb::handle base)
{
    if (type.is_none() || !PyType_Check(type.ptr()))
        return false;

    int result = PyObject_IsSubclass(type.ptr(), base.ptr());
    if (result < 0)
        throw nb::python_error();
    return result == 1;
}

bool is_enum_type(nb::handle type)
{
    nb::object enum_type = nb::module_::import_("enum").attr("Enum");
    return is_python_subclass(type, enum_type);
}

bool is_enum_flags_type(nb::handle type)
{
    nb::object flag_type = nb::module_::import_("enum").attr("Flag");
    return is_python_subclass(type, flag_type);
}

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

    // Integer-valued Enum type -> EnumDescriptor
    nb::object value_type = info.attr("value_type");
    if (is_enum_type(value_type)) {
        EnumDescriptor desc;
        desc.is_flags = is_enum_flags_type(value_type);
        nb::object members = value_type.attr("__members__");
        for (auto item : members.attr("items")()) {
            nb::tuple kv = nb::borrow<nb::tuple>(item);
            std::string name = nb::cast<std::string>(kv[0]);
            int64_t value;
            FALCOR_CHECK(
                nb::try_cast<int64_t>(kv[1].attr("value"), value),
                "Property \"{}\": enum {} member {} must have an integer value representable as int64.",
                nb::cast<std::string>(info.attr("name")),
                nb::cast<std::string>(value_type.attr("__name__")),
                name
            );
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
    , m_on_change(nb::borrow(m_info.attr("on_change")))
    , m_default_value(nb::borrow(m_info.attr("default_value")))
    , m_has_default_value(nb::cast<bool>(m_info.attr("has_default_value")))
    , m_is_enum(false)
{
    // Classify the canonical Python value type.
    m_value_type = nb::borrow(m_info.attr("value_type"));
    m_is_reflected_object = is_python_subclass(m_value_type, nb::type<ReflectedObject>());
    m_is_enum = is_enum_type(m_value_type);
    if (!m_is_reflected_object && !m_is_enum)
        m_type_entry = property_type_map_find(m_value_type);

    nb::object object_factories = m_info.attr("object_factories");
    if (nb::len(object_factories) > 0) {
        FALCOR_CHECK(
            m_is_reflected_object,
            "Property \"{}\": object_factories requires a ReflectedObject value_type.",
            m_name
        );
        set_object_factory(make_ref<PythonObjectFactory>(object_factories));
    }
}

const std::type_info& PythonPropertyDescriptor::type() const
{
    if (m_type_entry)
        return *m_type_entry->type_info;
    if (m_is_reflected_object)
        return typeid(ref<ReflectedObject>);
    if (m_is_enum)
        return typeid(int64_t);
    return typeid(nb::object);
}

bool PythonPropertyDescriptor::has_default_value() const
{
    return m_has_default_value;
}

bool PythonPropertyDescriptor::is_default(const void* instance) const
{
    if (!m_has_default_value)
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
    if (m_is_reflected_object) {
        validate_reflected_object(val);
        return Any(val.is_none() ? ref<ReflectedObject>{} : nb::cast<ref<ReflectedObject>>(val));
    }
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

    if (m_is_reflected_object) {
        const ref<ReflectedObject>* object = any_cast<ref<ReflectedObject>>(&value);
        FALCOR_CHECK(object != nullptr, "Property \"{}\": expected ref<ReflectedObject> Any.", m_name);
        nb::object py_object = *object ? nb::cast(*object) : nb::none();
        validate_reflected_object(py_object);
        py_set(instance, py_object);
        return;
    }

    if (m_is_enum) {
        const int64_t* iv = any_cast<int64_t>(&value);
        FALCOR_CHECK(iv != nullptr, "Property \"{}\": expected int64_t Any for enum.", m_name);
        py_set(instance, m_value_type(*iv));
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
    py_set(instance, m_value_type(value));
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
        py_set(instance, m_value_type(ev.value));
        return true;
    }

    return false;
}

void PythonPropertyDescriptor::reset(void* instance) const
{
    if (m_read_only || !m_has_default_value)
        return;
    nb::gil_scoped_acquire gil;
    py_set(instance, m_default_value);
}

void PythonPropertyDescriptor::get_value(const void* instance, void* out) const
{
    nb::gil_scoped_acquire gil;
    nb::object value = py_get(instance);
    if (m_type_entry) {
        m_type_entry->copy_from_python(value, out);
        return;
    }
    if (m_is_reflected_object) {
        validate_reflected_object(value);
        *static_cast<ref<ReflectedObject>*>(out)
            = value.is_none() ? ref<ReflectedObject>{} : nb::cast<ref<ReflectedObject>>(value);
        return;
    }
    if (m_is_enum) {
        *static_cast<int64_t*>(out) = nb::cast<int64_t>(value.attr("value"));
        return;
    }
    *static_cast<nb::object*>(out) = std::move(value);
}

void PythonPropertyDescriptor::set_value(void* instance, const void* value) const
{
    FALCOR_CHECK(!m_read_only, "Property \"{}\" is read-only.", m_name);
    nb::gil_scoped_acquire gil;
    if (m_type_entry) {
        py_set(instance, m_type_entry->copy_to_python(value));
        return;
    }
    if (m_is_reflected_object) {
        const ref<ReflectedObject>& object = *static_cast<const ref<ReflectedObject>*>(value);
        nb::object py_object = object ? nb::cast(object) : nb::none();
        validate_reflected_object(py_object);
        py_set(instance, py_object);
        return;
    }
    if (m_is_enum) {
        int64_t enum_value = *static_cast<const int64_t*>(value);
        py_set(instance, m_value_type(enum_value));
        return;
    }
    py_set(instance, *static_cast<const nb::object*>(value));
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
    if (!m_on_change.is_none())
        m_on_change(*obj);
}

void PythonPropertyDescriptor::validate_reflected_object(nb::handle value) const
{
    if (value.is_none()) {
        const ReflectedObjectFactory* factory = object_factory();
        FALCOR_CHECK(
            factory && factory->find_index(nullptr).has_value(),
            "Property \"{}\" does not allow None.",
            m_name
        );
        return;
    }

    int is_instance = PyObject_IsInstance(value.ptr(), m_value_type.ptr());
    if (is_instance < 0)
        throw nb::python_error();
    FALCOR_CHECK(
        is_instance == 1,
        "Property \"{}\": value must be an instance of {}.",
        m_name,
        nb::cast<std::string>(m_value_type.attr("__name__"))
    );
}

} // namespace falcor::reflection
