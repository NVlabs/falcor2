// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/utils/math/packing.h"

#include <array>
#include <cmath>
#include <limits>

using namespace falcor;

TEST_CASE("unsigned E8M8 packing rounds upward")
{
    CHECK_EQ(pack_ufloat_e8m8_round_up(0.f), 0u);
    CHECK_EQ(unpack_ufloat_e8m8(0u), 0.f);

    CHECK_EQ(pack_ufloat_e8m8_round_up(1.f), 0x7f00u);
    CHECK_EQ(unpack_ufloat_e8m8(0x7f00u), 1.f);
    CHECK_EQ(unpack_ufloat_e8m8(0xabcd7f00u), 1.f);
    CHECK_EQ(unpack_ufloat_e8m8(0xabcd0000u), 0.f);

    const float below_one = std::nextafter(1.f, 0.f);
    const float above_one = std::nextafter(1.f, 2.f);
    CHECK_EQ(pack_ufloat_e8m8_round_up(below_one), 0x7f00u);
    CHECK_EQ(pack_ufloat_e8m8_round_up(above_one), 0x7f01u);

    for (const float value : std::array{
             std::numeric_limits<float>::denorm_min(),
             std::numeric_limits<float>::min(),
             0.1f,
             below_one,
             1.f,
             above_one,
             1e20f,
             std::numeric_limits<float>::max(),
         }) {
        const uint32_t packed = pack_ufloat_e8m8_round_up(value);
        const float unpacked = unpack_ufloat_e8m8(packed);
        CHECK(std::isfinite(unpacked));
        CHECK_GE(unpacked, value);
        CHECK_EQ(pack_ufloat_e8m8_round_up(unpacked), packed);
    }

    for (uint32_t packed = 0; packed <= 0xfeff; ++packed)
        CHECK_EQ(pack_ufloat_e8m8_round_up(unpack_ufloat_e8m8(packed)), packed);
}

TEST_CASE("unsigned E8M8 reserved encodings saturate")
{
    constexpr float max_float = std::numeric_limits<float>::max();
    CHECK_EQ(pack_ufloat_e8m8_round_up(max_float), 0xffffu);
    CHECK_EQ(unpack_ufloat_e8m8(0xff00u), max_float);
    CHECK_EQ(unpack_ufloat_e8m8(0xfffeu), max_float);
    CHECK_EQ(unpack_ufloat_e8m8(0xffffu), max_float);
    CHECK_EQ(unpack_ufloat_e8m8(0xabcdffffu), max_float);
}
