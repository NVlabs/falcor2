// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/utils/color.h"

#include <array>
#include <cmath>

using namespace falcor;

TEST_CASE("color temperature to RGB")
{
    const float3 white = color_temperature_to_rgb(6500.f);
    CHECK_EQ(white.x, doctest::Approx(1.f));
    CHECK_EQ(white.y, doctest::Approx(1.f));
    CHECK_EQ(white.z, doctest::Approx(1.f));

    const float3 warm = color_temperature_to_rgb(3000.f);
    CHECK_GT(warm.x, warm.y);
    CHECK_GT(warm.y, warm.z);

    const float3 cool = color_temperature_to_rgb(10000.f);
    CHECK_GT(cool.z, cool.y);
    CHECK_GT(cool.y, cool.x);

    for (const float temperature : std::array{1000.f, 3000.f, 6500.f, 10000.f}) {
        const float3 rgb = color_temperature_to_rgb(temperature);
        CHECK(std::isfinite(rgb.x));
        CHECK(std::isfinite(rgb.y));
        CHECK(std::isfinite(rgb.z));
        CHECK_GE(rgb.x, 0.f);
        CHECK_GE(rgb.y, 0.f);
        CHECK_GE(rgb.z, 0.f);
        CHECK_EQ(0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z, doctest::Approx(1.f).epsilon(1e-4));
    }

    CHECK_EQ(color_temperature_to_rgb(0.f), color_temperature_to_rgb(MIN_COLOR_TEMPERATURE));
    CHECK_EQ(color_temperature_to_rgb(20000.f), color_temperature_to_rgb(MAX_COLOR_TEMPERATURE));
}
