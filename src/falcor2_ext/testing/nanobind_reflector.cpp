// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind_reflector.h"

#include "falcor2/core/reflection/reflector.h"

namespace {

class OnChangeTest {
public:
    using ReflectedBase = void;

    static constexpr const char* reflected_class_name() { return "OnChangeTest"; }

    template<falcor::reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r.def_rw("value", &OnChangeTest::m_value, falcor::reflection::on_change(&OnChangeTest::mark_changed))
            .def_ro("on_change_count", &OnChangeTest::m_on_change_count);
    }

private:
    void mark_changed() { ++m_on_change_count; }

    int m_value{0};
    int m_on_change_count{0};
};

} // namespace

FALCOR_PY_EXPORT(testing_nanobind_reflector)
{
    nb::module_ testing = nb::module_::import_("falcor2.testing");
    nb::module_ native = testing.def_submodule("_native");

    auto cls = nb::class_<OnChangeTest>(native, "OnChangeTest").def(nb::init<>());
    falcor::reflection::bind(cls);
}
