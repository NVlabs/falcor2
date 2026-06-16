// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/shader.h"

using namespace sgl;

TEST_SUITE_BEGIN("device");

TEST_CASE("enumerate_adapters")
{
    std::vector<AdapterInfo> adapters = Device::enumerate_adapters();
    CHECK(!adapters.empty());
}

TEST_CASE_GPU("init")
{
    CHECK(ctx.device);
}

TEST_SUITE_END();
