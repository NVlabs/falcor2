// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/ui/icon_library.h"

#include <sgl/device/resource.h>

using namespace falcor;
using namespace falcor::ui;

TEST_SUITE_BEGIN("IconLibrary");

TEST_CASE_GPU("loads embedded atlas and exposes icon metadata")
{
    auto icons = make_ref<IconLibrary>(ref(ctx.device));
    REQUIRE(icons->texture());
    CHECK(icons->texture()->width() > 0);
    CHECK(icons->texture()->height() > 0);

    const IconImage select = icons->icon(Icon::select);
    CHECK(select.texture == icons->texture());
    CHECK_EQ(select.preferred_size, 20.f);
    CHECK(select.uv_min.x > 0.f);
    CHECK(select.uv_min.y > 0.f);
    CHECK(select.uv_max.x > select.uv_min.x);
    CHECK(select.uv_max.y > select.uv_min.y);

    const IconImage light = icons->icon(Icon::light_point);
    CHECK(light.texture == icons->texture());
    CHECK_EQ(light.preferred_size, 32.f);
    const bool icons_do_not_overlap = light.uv_max.x <= select.uv_min.x || light.uv_min.x >= select.uv_max.x
        || light.uv_max.y <= select.uv_min.y || light.uv_min.y >= select.uv_max.y;
    CHECK(icons_do_not_overlap);
}

TEST_SUITE_END();
