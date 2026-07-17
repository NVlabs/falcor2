// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"

#include "falcor2/render/scene.h"

#include <optional>

namespace falcor {

/// Retains a Python object and releases it while holding the GIL.
class PythonObjectRef {
public:
    explicit PythonObjectRef(nb::handle object)
        : m_object(object)
    {
        nb::gil_scoped_acquire gil;
        m_object.inc_ref();
    }

    ~PythonObjectRef()
    {
        if (m_object.is_valid()) {
            nb::gil_scoped_acquire gil;
            m_object.dec_ref();
        }
    }

    PythonObjectRef(const PythonObjectRef&) = delete;
    PythonObjectRef& operator=(const PythonObjectRef&) = delete;

    nb::handle get() const { return m_object; }

private:
    nb::handle m_object;
};

/// Check if any of the bases of a Python class inherits from T.
template<typename T>
bool python_class_inherits_from(nb::handle cls)
{
    if (cls.is_none() || !cls.is_type()) {
        return false;
    }
    if (cls.ptr() == nb::type<T>().ptr()) {
        return true;
    }
    for (const auto& base : cls.attr("__bases__")) {
        if (python_class_inherits_from<T>(base)) {
            return true;
        }
    }
    return false;
}

/// Create an instance of a Python scene object class and add it to the scene.
template<typename T>
nb::object create_python_scene_object(Scene* scene, nb::type_object cls, std::optional<Properties> props = {})
{
    nb::object object = cls.attr("__new__")(cls);
    cls.attr("__init__")(object);
    T* scene_object = nb::cast<T*>(object);
    scene->_add_object<T>(ref<T>(scene_object));
    if (props.has_value()) {
        scene_object->set_properties(props.value());
    }
    return object;
}

} // namespace falcor
