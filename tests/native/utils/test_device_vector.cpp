// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/utils/device_vector.h"
#include "testing.h"

#include <array>

using namespace falcor;

TEST_CASE_GPU("DeviceVector growth preserves contents")
{
    const std::array<uint32_t, 4> expected{11, 23, 37, 53};
    auto readback = ctx.device->create_buffer({
        .size = sizeof(expected),
        .memory_type = sgl::MemoryType::read_back,
        .usage = sgl::BufferUsage::copy_destination,
    });
    auto encoder = ctx.device->create_command_encoder();
    {
        DeviceVector<uint32_t> values;
        values.reserve(encoder, 8);
        CHECK(values.size() == 0);
        CHECK(values.capacity() >= 8);
        values.resize(encoder, expected.size());
        encoder->upload_buffer_data(values.buffer(), 0, sizeof(expected), expected.data());

        auto* original = values.buffer().get();
        values.resize(encoder, 6);
        values.ensure_size(encoder, 3);
        CHECK(values.size() == 6);
        CHECK(values.buffer().get() == original);
        values.resize(encoder, expected.size());

        values.reserve(encoder, 16);
        CHECK(values.size() == expected.size());
        CHECK(values.capacity() >= 16);
        values.ensure_size(encoder, 19);
        CHECK(values.size() == 19);
        CHECK(values.capacity() >= 19);
        encoder->copy_buffer(readback, 0, values.buffer(), 0, sizeof(expected));
    }
    // The recorded copies own their resources after the vector is destroyed.
    ctx.device->submit_command_buffer(encoder->finish());
    encoder = nullptr;
    ctx.device->wait();
    const auto result = readback->get_elements<uint32_t>();
    REQUIRE(result.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        CHECK(result[i] == expected[i]);
}
