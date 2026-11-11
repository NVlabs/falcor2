// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

FALCOR_PY_EXPORT(testing_ustring)
{
    nb::module_ native = nb::module_::import_("falcor2.testing._native");
    native.def(
        "_ustring_roundtrip",
        [](falcor::ustring value)
        {
            return value;
        },
        "value"_a
    );
    native.def(
        "_ustring_shares_storage",
        [](falcor::ustring a, falcor::ustring b)
        {
            return a.data() == b.data();
        }
    );
    native.def(
        "_ustring_invalid_utf8",
        []()
        {
            return falcor::ustring("\xff", 1);
        }
    );
}
