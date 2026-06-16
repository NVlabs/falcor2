// SPDX-License-Identifier: Apache-2.0

// Portions of this file are derived from Mitsuba 3.
// Copyright (c) Mitsuba contributors.
// Licensed under the BSD 3-Clause License.
// See LICENSES/mitsuba3.txt for the full license text.

#include "any.h"

namespace falcor {

/// Storage helper for arbitrary Python objects in Properties.
class PythonObjectStorage : public Any::Base {
    nb::handle m_obj;
    const std::type_info* m_cpp_type_info;
    void* m_cpp_ptr;

public:
    explicit PythonObjectStorage(nb::handle obj)
    {
        nb::gil_scoped_acquire guard;
        m_obj = obj;
        m_cpp_type_info = &nb::type_info(obj.type());
        m_cpp_ptr = nb::inst_ptr<void>(obj);
        m_obj.inc_ref();
    }

    ~PythonObjectStorage()
    {
        nb::gil_scoped_acquire gil;
        m_obj.dec_ref();
    }

    const std::type_info& type() const override { return *m_cpp_type_info; }
    void* ptr() override { return m_cpp_ptr; }

    /// Access the stored Python handle.
    nb::handle python_handle() const { return m_obj; }
};

Any any_wrap(nb::handle obj)
{
    if (!nb::inst_check(obj))
        throw std::bad_cast();

    Any::Base* storage = new PythonObjectStorage(obj);
    return Any(storage);
}

nb::handle any_try_unwrap(const Any& a)
{
    const auto* storage = dynamic_cast<const PythonObjectStorage*>(a.base_ptr());
    return storage ? storage->python_handle() : nb::handle();
}

nb::handle nb_type_to_python(const std::type_info& type, void* data)
{
    // This uses nanobind's internal API to look up the Python type wrapper
    // for a C++ type_info and wrap the data pointer as a Python object.
    // There is no public nanobind API for this runtime type_info -> PyObject*
    // conversion; nb::cast<T>() requires compile-time type knowledge.
    return nanobind::detail::nb_type_put(&type, data, nanobind::rv_policy::none, nullptr);
}

} // namespace falcor
