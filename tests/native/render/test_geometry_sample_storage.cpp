// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/render/geometry/geometry_sample_storage.h"

using namespace falcor;

TEST_SUITE_BEGIN("GeometrySampleStorage");

TEST_CASE_GPU("resident_sample_buffers")
{
    GeometrySampleStorage<uint32_t> storage;
    const std::vector<uint32_t> samples{10, 11, 20, 21, 30, 31, 40, 41, 50, 51};
    // Force boundaries with tiny buffers, including unused space and a partial final buffer.
    for (size_t max_bytes : {size_t(20), size_t(8)}) {
        storage.reset(ctx.device, samples, 2, max_bytes);
        auto encoder = ctx.device->create_command_encoder();
        const auto first = storage.prepare(encoder, 0, 1);
        CHECK((first[0].buffer == first[1].buffer) == (max_bytes == 20));
        const auto boundary = storage.prepare(encoder, 1, 2);
        CHECK(boundary[0].buffer != boundary[1].buffer);
        CHECK(boundary[0].element_offset == (max_bytes == 20 ? 2 : 0));
        CHECK(boundary[1].element_offset == 0);
        const auto last = storage.prepare(encoder, 4, 4);
        CHECK(last[0].buffer == last[1].buffer);
        ctx.device->submit_command_buffer(encoder->finish());
        ctx.device->wait();
        for (size_t i = 0; i < boundary.size(); ++i) {
            const auto values = boundary[i].buffer->get_elements<uint32_t>();
            CHECK(values[boundary[i].element_offset] == samples[(i + 1) * 2]);
            CHECK(values[boundary[i].element_offset + 1] == samples[(i + 1) * 2 + 1]);
        }
        CHECK(last[0].buffer->get_elements<uint32_t>() == std::vector<uint32_t>{50, 51});
    }
    CHECK_THROWS_WITH(storage.reset(ctx.device, samples, 2, 7), doctest::Contains("per-buffer limit"));
    storage.reset(ctx.device, {}, 0);
    storage.reset(ctx.device, {60, 61}, 2);
    auto encoder = ctx.device->create_command_encoder();
    const auto source = storage.prepare(encoder, 0, 0);
    ctx.device->submit_command_buffer(encoder->finish());
    ctx.device->wait();
    CHECK(source[0].element_offset == 0);
    CHECK(source[0].buffer->get_elements<uint32_t>() == std::vector<uint32_t>{60, 61});
}

TEST_SUITE_END();
