// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/config.h"

#if FALCOR_ENABLE_STREAMING

#include "falcor2/streaming/webrtc_internal.h"
#include "falcor2/streaming/video_format.h"
#include <array>

using namespace falcor::streaming;
using namespace falcor::streaming::detail;

TEST_SUITE_BEGIN("webrtc");

TEST_CASE("Streaming H.264 levels cover padded frame sizes and macroblock rates")
{
    CHECK(h264_level(1280, 720, 30) == 31);
    CHECK(h264_level(1280, 720, 31) == 32);
    CHECK(h264_level(1280, 720, 60) == 32);
    CHECK(h264_level(1920, 1080, 30) == 40);
    CHECK(h264_level(1920, 1080, 60) == 42);
    CHECK(h264_level(2560, 1440, 30) == 50);
    CHECK(h264_level(2560, 1440, 60) == 51);
    CHECK(h264_level(3840, 2160, 30) == 51);
    CHECK(h264_level(3840, 2160, 60) == 52);
    CHECK(h264_level(1280, 722, 30) == 32);
    CHECK(h264_level(3840, 16, 1) == 40);
    CHECK_THROWS(h264_level(3842, 2160, 30));
    CHECK_THROWS(h264_level(1920, 1081, 30));
    CHECK_THROWS(h264_level(0, 720, 30));
    CHECK_THROWS(h264_level(1280, 720, 61));
    CHECK(h264_level(1280, 720, 30, 14000000) == 31);
    CHECK(h264_level(1280, 720, 30, 14000001) == 32);
    CHECK(h264_level(1280, 720, 60, 20000000) == 32);
    CHECK(h264_level(1280, 720, 60, 20000001) == 41);
    CHECK(h264_level(1280, 720, 60, 50000000) == 41);
    CHECK(h264_level(1920, 1080, 30, 20000000) == 40);
    CHECK(h264_level(1920, 1080, 30, 30000000) == 41);
    CHECK(h264_level(1920, 1080, 60, 30000000) == 42);
    CHECK(h264_level(3840, 2160, 60, 50000000) == 52);
    CHECK_THROWS(h264_level(1280, 720, 30, 99999));
    CHECK_THROWS(h264_level(1280, 720, 30, 50000001));
}

namespace {
EncodedFrame frame(uint64_t id, bool keyframe, size_t bytes = 10)
{
    return {std::vector<uint8_t>(bytes), id, id * 33333, 33333, keyframe};
}
} // namespace

TEST_CASE("Streaming queue discards dependent chains on byte overflow")
{
    EncodedQueue queue(25, 200);
    CHECK_FALSE(queue.push(frame(0, false), 0));
    REQUIRE(queue.push(frame(1, true), 0));
    REQUIRE(queue.push(frame(2, false), 0));
    CHECK_FALSE(queue.push(frame(3, false), 0));
    CHECK(queue.bytes() == 0);
    CHECK(queue.needs_keyframe());
    CHECK_FALSE(queue.pop(0).has_value());
    CHECK_FALSE(queue.push(frame(4, false), 0));
    REQUIRE(queue.push(frame(5, true), 0));
    REQUIRE(queue.push(frame(6, false), 0));
    CHECK(queue.pop(0)->frame_id == 5);
    CHECK(queue.pop(0)->frame_id == 6);
    CHECK(queue.dropped() == 5);
}

TEST_CASE("Streaming queue expires the entire stale chain and bounds frame count")
{
    EncodedQueue queue(100, 20, 2);
    REQUIRE(queue.push(frame(0, true), 100));
    REQUIRE(queue.push(frame(1, false), 110));
    CHECK_FALSE(queue.pop(121));
    CHECK(queue.needs_keyframe());
    CHECK(queue.dropped() == 2);
    CHECK_FALSE(queue.push(frame(2, true, 101), 122));
    REQUIRE(queue.push(frame(3, true), 130));
    REQUIRE(queue.push(frame(4, false), 130));
    REQUIRE(queue.push(frame(5, true), 130));
    CHECK(queue.size() == 1);
    CHECK(queue.pop(130)->frame_id == 5);
}

TEST_CASE("Streaming RTP timestamp preserves gaps and wraps at 32 bits")
{
    CHECK(rtp_timestamp(0) == 0);
    CHECK(rtp_timestamp(1000000) == 90000);
    CHECK(rtp_timestamp(10000000) == 900000);
    CHECK(rtp_timestamp(47721858845ULL) == 0);
    CHECK(rtp_timestamp(47721892178ULL) == 3000);
}

TEST_CASE("Streaming RTCP receiver report preserves high loss fractions")
{
    // Network-order SSRC followed by fraction and cumulative loss. In
    // particular, 32/256 (12.5%) must not be truncated to zero by a bad mask.
    std::array<uint8_t, 24> bytes{0x89, 0xab, 0xcd, 0xef, 32, 0x01, 0x02, 0x03};
    auto report = falcor::streaming::detail::decode_receiver_report(bytes);
    CHECK(report.ssrc == 0x89abcdefu);
    CHECK(report.fraction_lost == 32);
    CHECK(report.cumulative_lost == 0x010203);
    bytes[8] = bytes[9] = bytes[10] = 255;
    bytes[11] = 254;
    bytes[5] = bytes[6] = bytes[7] = 255;
    report = falcor::streaming::detail::decode_receiver_report(bytes);
    CHECK(report.sequence == 0xfffffffeu);
    CHECK(report.cumulative_lost == -1);
    bytes[4] = 255;
    CHECK(falcor::streaming::detail::decode_receiver_report(bytes).fraction_lost == 255);
    bytes[4] = 0;
    CHECK(falcor::streaming::detail::decode_receiver_report(bytes).fraction_lost == 0);
}

TEST_CASE("Streaming packet credit bounds bursts and follows rate changes")
{
    PacketBudget budget;
    budget.set_rate(800000, 0); // 100 bytes/ms.
    CHECK(budget.delay_us(1280, 0) == 0);
    budget.consume(1280);
    budget.consume(1280);
    CHECK(budget.delay_us(1000, 0) == 10000);
    CHECK(budget.delay_us(1000, 5000) == 5000);
    budget.set_rate(400000, 5000);
    CHECK(budget.delay_us(1000, 5000) == 10000);
    CHECK(budget.delay_us(1000, 15000) == 0);
    budget.consume(1000);
    // A long idle cannot fund a full-frame burst or catch-up transmission.
    CHECK(budget.delay_us(2560, 10000000) == 0);
    budget.consume(2560);
    CHECK(budget.delay_us(1000, 10000000) == 20000);
}

TEST_CASE("Streaming repairs expire and bound duplicates across sequence wrap")
{
    RepairWindow repair{1000, 0, 65535, 0, true, false};
    CHECK(repair.eligible(65535, 1000));
    CHECK_FALSE(repair.eligible(0, 1000));
    repair.queued = true;
    CHECK_FALSE(repair.eligible(65535, 2000));
    repair.queued = false;
    repair.attempts = 1;
    repair.repaired_us = 2000;
    CHECK_FALSE(repair.eligible(65535, 51999));
    CHECK(repair.eligible(65535, 52000));
    repair.attempts = 2;
    CHECK_FALSE(repair.eligible(65535, 100000));
    repair.attempts = 0;
    CHECK_FALSE(repair.eligible(65535, 251001));
    repair = RepairWindow{252000, 0, 0, 0, true, false};
    CHECK(repair.eligible(0, 252000));
    CHECK_FALSE(repair.eligible(65535, 252000));
}

TEST_SUITE_END();

#endif // FALCOR_ENABLE_STREAMING
