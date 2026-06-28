// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"
import falcor2.utils;

enum class TestFlags : uint {
    none = 0,
    a = 1,
    b = 2,
    c = 4,
};

TEST_CASE(enum_helpers_is_set_and_flip_bit)
{
    TestFlags flags = TestFlags::a | TestFlags::c;

    CHECK(is_set(flags, TestFlags::a));
    CHECK_FALSE(is_set(flags, TestFlags::b));

    flip_bit(flags, TestFlags::b);
    CHECK(is_set(flags, TestFlags::b));

    flip_bit(flags, TestFlags::a);
    CHECK_FALSE(is_set(flags, TestFlags::a));
    CHECK(is_set(flags, TestFlags::c));
}
