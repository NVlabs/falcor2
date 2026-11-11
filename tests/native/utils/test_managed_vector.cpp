// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/utils/managed_vector.h"
#include "testing.h"

using namespace falcor;

TEST_CASE("ManagedVector host reservation")
{
    ManagedVector<uint32_t> values({.max_buffer_size = 16 * sizeof(uint32_t)});
    values.reserve(4);
    CHECK(values.capacity() >= 4);
    CHECK(values.size() == 0);
    CHECK_FALSE(values.is_dirty());
    CHECK_FALSE(values.buffer());

    values.resize(2);
    values.data()[0] = 17;
    values.reserve(8);
    CHECK(values.capacity() >= 8);
    CHECK(values.size() == 2);
    CHECK(values.data()[0] == 17);
    CHECK_FALSE(values.is_dirty());

    values.mark_dirty(0, 1);
    values.reserve(16);
    CHECK(values.capacity() >= 16);
    CHECK(values.size() == 2);
    CHECK(values.data()[0] == 17);
    CHECK(values.is_dirty());
    CHECK_FALSE(values.buffer());
    CHECK_THROWS(values.reserve(17));
}
