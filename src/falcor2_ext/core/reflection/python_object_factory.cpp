// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "python_object_factory.h"

#include "falcor2/core/error.h"
#include "falcor2/core/reflected_object.h"

namespace falcor::reflection {

PythonObjectFactory::PythonObjectFactory(nb::handle factories)
{
    nb::tuple entries = nb::borrow<nb::tuple>(factories);
    m_entries.reserve(entries.size());
    for (nb::handle entry : entries) {
        if (entry.is_none()) {
            m_entries.push_back(Entry{nb::none(), nb::none(), "None"});
            continue;
        }
        m_entries.push_back(
            Entry{
                nb::borrow<nb::object>(entry.attr("object_type")),
                nb::borrow<nb::object>(entry.attr("factory")),
                nb::cast<std::string>(entry.attr("label")),
            }
        );
    }
}

std::string_view PythonObjectFactory::label(size_t index) const
{
    FALCOR_CHECK(index < m_entries.size(), "Object factory index {} is out of range.", index);
    return m_entries[index].label;
}

std::optional<size_t> PythonObjectFactory::find_index(const ReflectedObject* object) const
{
    nb::gil_scoped_acquire gil;
    if (!object) {
        for (size_t i = 0; i < m_entries.size(); ++i) {
            if (m_entries[i].object_type.is_none())
                return i;
        }
        return std::nullopt;
    }

    nb::object py_object = nb::cast(ref<ReflectedObject>(const_cast<ReflectedObject*>(object)));
    PyObject* py_type = reinterpret_cast<PyObject*>(Py_TYPE(py_object.ptr()));
    for (size_t i = 0; i < m_entries.size(); ++i) {
        if (m_entries[i].object_type.ptr() == py_type)
            return i;
    }
    return std::nullopt;
}

ref<ReflectedObject> PythonObjectFactory::create(size_t index, void* owner) const
{
    FALCOR_CHECK(index < m_entries.size(), "Object factory index {} is out of range.", index);

    nb::gil_scoped_acquire gil;
    if (m_entries[index].object_type.is_none())
        return {};

    nb::handle* py_owner = static_cast<nb::handle*>(owner);
    nb::object result = m_entries[index].factory(*py_owner);
    FALCOR_CHECK(!result.is_none(), "Object factory \"{}\" returned None.", m_entries[index].label);

    FALCOR_CHECK(
        reinterpret_cast<PyObject*>(Py_TYPE(result.ptr())) == m_entries[index].object_type.ptr(),
        "Object factory \"{}\" did not return its declared concrete type.",
        m_entries[index].label
    );
    return nb::cast<ref<ReflectedObject>>(result);
}

} // namespace falcor::reflection
