// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/ustring.h"

#include <nanobind/stl/string_view.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template<>
struct type_caster<falcor::ustring> {
    using StringCaster = make_caster<std::string_view>;
    NB_TYPE_CASTER(falcor::ustring, StringCaster::Name);

    bool from_python(handle src, uint8_t flags, cleanup_list* cleanup) noexcept
    {
        StringCaster caster;
        if (!caster.from_python(src, flags_for_local_caster<std::string_view>(flags), cleanup))
            return false;
        try {
            // Intern the contents while the Python string still owns the UTF-8 view.
            value.reset(caster.operator cast_t<std::string_view>());
            return true;
        } catch (...) {
            // Failed argument conversions must not throw through nanobind's dispatch.
            return false;
        }
    }

    static handle from_cpp(const falcor::ustring& value, rv_policy policy, cleanup_list* cleanup) noexcept
    {
        return StringCaster::from_cpp(std::string_view(value), policy, cleanup);
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
