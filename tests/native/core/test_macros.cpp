// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/macros.h"

using namespace falcor;

TEST_SUITE_BEGIN("macros");

static int s_once_count_1;
static int s_once_count_2;

FALCOR_STATIC_ONCE({ s_once_count_1++; });
FALCOR_STATIC_ONCE({ s_once_count_2++; });

TEST_CASE("FALCOR_STATIC_ONCE")
{
    CHECK(s_once_count_1 == 1);
    CHECK(s_once_count_2 == 1);
}

TEST_SUITE_END();
