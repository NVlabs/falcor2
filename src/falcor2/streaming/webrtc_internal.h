// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Private WebRTC policies and helpers, shared with focused native tests.
// Keep these out of webrtc.h so callers do not depend on transport internals.

#include "video_encoder.h"

#include "falcor2/core/config.h"
#include "falcor2/core/error.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <span>
#include <vector>

#if FALCOR_ENABLE_STREAMING
#include <rtc/rtc.hpp>
#ifdef _WIN32
#include <windows.h>
#endif
#endif

namespace falcor::streaming::detail {

// Caller supplies monotonic milliseconds and synchronization. Losing an encoded
// frame invalidates its entire dependent chain, so resume only at an IDR.
class EncodedQueue {
public:
    EncodedQueue(size_t max_bytes, uint64_t max_age_ms, size_t max_frames = 8)
        : m_max_bytes(max_bytes)
        , m_max_age_ms(max_age_ms)
        , m_max_frames(max_frames)
    {
    }
    bool push(EncodedFrame frame, uint64_t now_ms)
    {
        expire(now_ms);
        if (frame.data.size() > m_max_bytes || m_bytes + frame.data.size() > m_max_bytes
            || m_frames.size() >= m_max_frames)
            recover();
        if (frame.data.empty() || frame.data.size() > m_max_bytes || (m_waiting && !frame.keyframe)) {
            ++m_dropped;
            return false;
        }
        m_waiting = false;
        m_bytes += frame.data.size();
        m_frames.push_back({std::move(frame), now_ms});
        return true;
    }
    std::optional<EncodedFrame> pop(uint64_t now_ms)
    {
        expire(now_ms);
        if (m_frames.empty())
            return {};
        auto frame = std::move(m_frames.front().frame);
        m_bytes -= frame.data.size();
        m_frames.pop_front();
        return frame;
    }
    void recover(size_t in_flight_drops = 0)
    {
        m_dropped += m_frames.size() + in_flight_drops;
        m_frames.clear();
        m_bytes = 0;
        m_waiting = true;
        ++m_recoveries;
    }
    void expire(uint64_t now_ms)
    {
        if (!m_frames.empty() && now_ms - m_frames.front().arrival_ms > m_max_age_ms)
            recover();
    }
    size_t bytes() const { return m_bytes; }
    size_t size() const { return m_frames.size(); }
    uint64_t age_ms(uint64_t now_ms) const { return m_frames.empty() ? 0 : now_ms - m_frames.front().arrival_ms; }
    bool needs_keyframe() const { return m_waiting; }
    uint64_t dropped() const { return m_dropped; }
    uint64_t recoveries() const { return m_recoveries; }

private:
    struct Entry {
        EncodedFrame frame;
        uint64_t arrival_ms;
    };
    size_t m_max_bytes;
    uint64_t m_max_age_ms;
    size_t m_max_frames;
    std::deque<Entry> m_frames;
    size_t m_bytes{0};
    bool m_waiting{true};
    uint64_t m_dropped{0};
    uint64_t m_recoveries{0};
};

// Split before multiplication to avoid overflow for long-running microsecond clocks.
inline uint32_t rtp_timestamp(uint64_t pts_us)
{
    return uint32_t((pts_us / 1000000) * 90000 + (pts_us % 1000000) * 9 / 100);
}

struct ReceiverReport {
    uint32_t ssrc;
    uint8_t fraction_lost;
    uint32_t sequence;
    int32_t cumulative_lost;
};

// The caller validates the compound packet and complete 24-byte report block.
// Read the full fraction byte directly: libdatachannel 0.24.5's accessor masks
// off its upper nibble, turning e.g. 32/256 loss into zero.
inline ReceiverReport decode_receiver_report(std::span<const uint8_t, 24> bytes)
{
    const auto word = [&](std::size_t offset)
    {
        return (uint32_t(bytes[offset]) << 24) | (uint32_t(bytes[offset + 1]) << 16)
            | (uint32_t(bytes[offset + 2]) << 8) | bytes[offset + 3];
    };
    const auto lost = int32_t(word(4) & 0xffffff);
    return {word(0), bytes[4], word(8), lost >= 0x800000 ? lost - 0x1000000 : lost};
}

// Caller supplies monotonic microseconds and synchronization. Credit is capped
// at two MTU-sized packets: idle time or a late wake never permits a frame burst.
class PacketBudget {
public:
    void set_rate(uint32_t bits_per_second, uint64_t now_us)
    {
        refill(now_us);
        m_bytes_per_us = double(bits_per_second) / 8000000.0;
    }
    uint64_t delay_us(uint32_t bytes, uint64_t now_us)
    {
        refill(now_us);
        return m_credit >= bytes ? 0 : uint64_t(std::ceil((bytes - m_credit) / m_bytes_per_us));
    }
    void consume(uint32_t bytes) { m_credit -= bytes; }

private:
    void refill(uint64_t now_us)
    {
        m_credit = std::min(2560.0, m_credit + double(now_us - m_updated_us) * m_bytes_per_us);
        m_updated_us = now_us;
    }
    double m_credit{2560};
    double m_bytes_per_us{1};
    uint64_t m_updated_us{0};
};

// Bound repair lifetime and repeated attempts independently of NACK frequency.
struct RepairWindow {
    uint64_t sent_us{0};
    uint64_t repaired_us{0};
    uint16_t sequence{0};
    uint8_t attempts{0};
    bool valid{false};
    bool queued{false};

    bool eligible(uint16_t requested, uint64_t now_us) const
    {
        return valid && sequence == requested && !queued && now_us - sent_us <= 250000 && attempts < 2
            && (attempts == 0 || now_us - repaired_us >= 50000);
    }
};

#if FALCOR_ENABLE_STREAMING

// At 125 Mbit/s, even minimum-size RTP packets (12 bytes plus 48 bytes
// transport overhead) cannot exhaust the sequence space within 250 ms,
// including the initial 2560-byte pacing credit. Use heap storage for this
// conservative bound for sequence-indexed repair bookkeeping. Packet data is
// retained only for the repair window, bounding it to about 4 MB at peak rate.
inline constexpr size_t REPAIR_HISTORY_SIZE = 65536;

// Windows condition-variable timeouts may round packet-scale waits to the
// system timer tick. A high-resolution waitable timer avoids changing global
// timer resolution or spinning. The event interrupts waits on close/rate changes.
class PacingWait {
public:
#ifdef _WIN32
    PacingWait()
    {
        m_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
        m_timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
        if (!m_event || !m_timer) {
            if (m_event)
                CloseHandle(m_event);
            if (m_timer)
                CloseHandle(m_timer);
            FALCOR_THROW("Cannot create high-resolution packet pacing timer");
        }
    }
    ~PacingWait()
    {
        CloseHandle(m_timer);
        CloseHandle(m_event);
    }
    void notify() { SetEvent(m_event); }
    void wait(std::unique_lock<std::mutex>& lock, uint64_t delay_us)
    {
        LARGE_INTEGER due;
        due.QuadPart = -static_cast<LONGLONG>(delay_us * 10);
        if (!SetWaitableTimer(m_timer, &due, 0, nullptr, nullptr, FALSE))
            FALCOR_THROW("Cannot arm packet pacing timer");
        HANDLE handles[] = {m_event, m_timer};
        lock.unlock();
        const auto result = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
        lock.lock();
        if (result == WAIT_FAILED)
            FALCOR_THROW("Packet pacing wait failed");
    }
#else
    PacingWait() = default;
    void notify() { m_condition.notify_one(); }
    void wait(std::unique_lock<std::mutex>& lock, uint64_t delay_us)
    {
        m_condition.wait_for(lock, std::chrono::microseconds(delay_us));
    }
#endif
    PacingWait(const PacingWait&) = delete;
    PacingWait& operator=(const PacingWait&) = delete;

private:
#ifdef _WIN32
    HANDLE m_timer{nullptr};
    HANDLE m_event{nullptr};
#else
    std::condition_variable m_condition;
#endif
};

// Outgoing runs only on Falcor's sender. Incoming callbacks enqueue bounded
// repairs without waiting or sending. No additional transport worker is needed.
class PacketPacer : public rtc::MediaHandler {
public:
    struct Stats {
        uint64_t packets{0}, bytes{0}, repair_packets{0}, repair_bytes{0}, repair_dropped{0};
        uint32_t bitrate{0};
        size_t repair_queue{0};
        uint64_t pending_bytes{0};
    };
    PacketPacer() { set_rate(8000000); }

    void set_rate(uint32_t encoder_bitrate)
    {
        std::lock_guard lock(m_mutex);
        m_encoder_bitrate = encoder_bitrate;
        update_rates();
    }
    void set_pacing_rate(uint32_t bitrate)
    {
        std::lock_guard lock(m_mutex);
        m_pacing_bitrate = bitrate;
        update_rates();
    }
    void stop()
    {
        std::lock_guard lock(m_mutex);
        m_stopped = true;
        m_repairs.clear();
        m_stats.pending_bytes = 0;
        m_wake.notify();
    }
    Stats stats()
    {
        std::lock_guard lock(m_mutex);
        auto result = m_stats;
        result.repair_queue = m_repairs.size();
        return result;
    }
    void enqueue_repair(rtc::message_ptr packet)
    {
        if (packet->size() < 12)
            return;
        enqueue_repair(sequence(packet));
    }
    void outgoing(rtc::message_vector& messages, const rtc::message_callback& send) override
    {
        m_last_frame_sent = false;
        m_send = send; // Library callback owns only a weak Track reference.
        {
            std::lock_guard lock(m_mutex);
            m_stats.pending_bytes = 0;
            for (const auto& packet : messages)
                if (packet->type != rtc::Message::Control)
                    m_stats.pending_bytes += packet->size() + 48;
            update_frame_deadline();
        }
        for (const auto& packet : messages) {
            drain_repairs();
            if (packet->type == rtc::Message::Control) {
                send(packet);
                continue;
            }
            if (!send_original(packet)) {
                messages.clear();
                std::lock_guard lock(m_mutex);
                m_stats.pending_bytes = 0;
                return;
            }
        }
        messages.clear(); // All packets went through the paced callback.
        m_last_frame_sent = true;
    }
    bool last_frame_sent() const { return m_last_frame_sent; }

    void incoming(rtc::message_vector& messages, const rtc::message_callback&) override
    {
        // Decode generic NACK blocks directly so the cache can expire by send
        // time. The library's packet-count cache retains expired data and its
        // linked ownership chain can overflow the stack at larger capacities.
        for (const auto& message : messages) {
            if (message->type != rtc::Message::Control)
                continue;
            const auto* bytes = reinterpret_cast<const uint8_t*>(message->data());
            for (size_t offset = 0; offset + 4 <= message->size();) {
                const auto* p = bytes + offset;
                const size_t length = ((size_t(p[2]) << 8) + p[3] + 1) * 4;
                if ((p[0] >> 6) != 2 || length > message->size() - offset)
                    break;
                size_t payload_length = length;
                if (p[0] & 0x20) {
                    const size_t padding = p[length - 1];
                    if (!padding || padding > length - 4)
                        break;
                    payload_length -= padding;
                }
                if (p[1] == 205 && (p[0] & 31) == 1 && payload_length >= 12 && payload_length % 4 == 0) {
                    for (size_t field = 12; field + 4 <= payload_length; field += 4) {
                        const auto seq = uint16_t((uint16_t(p[field]) << 8) | p[field + 1]);
                        const auto mask = uint16_t((uint16_t(p[field + 2]) << 8) | p[field + 3]);
                        enqueue_repair(seq);
                        for (uint16_t bit = 0; bit < 16; ++bit)
                            if (mask & (1u << bit))
                                enqueue_repair(uint16_t(seq + bit + 1));
                    }
                }
                offset += length;
            }
        }
    }

    // Also called while no new encoded frame is available. Never waits for
    // repair credit, so repair backlog cannot hold up an original frame.
    void drain_repairs()
    {
        std::unique_lock lock(m_mutex);
        while (!m_stopped && m_send && !m_repairs.empty()) {
            const auto packet = m_repairs.front();
            const auto seq = sequence(packet);
            auto& entry = m_history[seq].window;
            const auto now = now_us();
            if (!entry.valid || entry.sequence != seq || now - entry.sent_us > 250000) {
                if (entry.sequence == seq)
                    entry.queued = false;
                m_repairs.pop_front();
                ++m_stats.repair_dropped;
                continue;
            }
            const auto cost = uint32_t(packet->size() + 48);
            if (m_repair.delay_us(cost, now) || m_total.delay_us(cost, now))
                return;
            m_repairs.pop_front();
            entry.queued = false;
            entry.repaired_us = now;
            ++entry.attempts;
            m_repair.consume(cost);
            m_total.consume(cost);
            ++m_stats.repair_packets;
            m_stats.repair_bytes += packet->size();
            lock.unlock();
            m_send(packet);
            lock.lock();
        }
    }

private:
    struct SentPacket {
        RepairWindow window;
        rtc::message_ptr packet;
    };
    void enqueue_repair(uint16_t seq)
    {
        std::lock_guard lock(m_mutex);
        auto& entry = m_history[seq];
        if (m_stopped || m_repairs.size() >= 32 || !entry.window.eligible(seq, now_us())) {
            ++m_stats.repair_dropped;
            return;
        }
        entry.window.queued = true;
        m_repairs.push_back(entry.packet);
        m_wake.notify();
    }
    void expire_history(uint64_t now)
    {
        while (!m_cached_sequences.empty()) {
            auto& entry = m_history[m_cached_sequences.front()];
            if (now - entry.window.sent_us <= 250000)
                break;
            entry = {};
            m_cached_sequences.pop_front();
        }
    }
    // Called under m_mutex. An explicit transport allowance survives encoder
    // target changes; repairs still compete within it and retain their own cap.
    void update_rates()
    {
        m_stats.bitrate = m_pacing_bitrate ? m_pacing_bitrate : m_encoder_bitrate * 5 / 4;
        m_total.set_rate(m_stats.bitrate, now_us());
        m_repair.set_rate(m_encoder_bitrate * 15 / 100, now_us());
        if (m_stats.pending_bytes)
            update_frame_deadline();
        m_wake.notify();
    }
    // Called under m_mutex at frame start and on rate changes. Using the
    // encoder rate leaves room for the 15% repair budget within the 125%
    // transport rate. Keep one second of scheduling slack and a finite bound.
    void update_frame_deadline()
    {
        m_frame_deadline
            = now_us() + 1000000 + (m_stats.pending_bytes * 8000000 + m_encoder_bitrate - 1) / m_encoder_bitrate;
    }
    static uint64_t now_us()
    {
        return uint64_t(
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch())
                .count()
        );
    }
    static uint16_t sequence(const rtc::message_ptr& packet)
    {
        const auto* p = reinterpret_cast<const uint8_t*>(packet->data());
        return uint16_t((uint16_t(p[2]) << 8) | p[3]);
    }
    bool send_original(const rtc::message_ptr& packet)
    {
        std::unique_lock lock(m_mutex);
        const auto cost = uint32_t(packet->size() + 48);
        while (!m_stopped) {
            // A repair must compete when shared credit becomes available, not
            // just before we wait. Otherwise every wake funds the next original
            // packet and repairs starve until the entire frame has been sent.
            lock.unlock();
            drain_repairs();
            lock.lock();
            if (m_stopped)
                return false;
            const auto now = now_us();
            if (now >= m_frame_deadline)
                return false;
            const auto delay = m_total.delay_us(cost, now);
            if (!delay) {
                m_total.consume(cost);
                ++m_stats.packets;
                m_stats.bytes += packet->size();
                m_stats.pending_bytes -= cost;
                const auto seq = sequence(packet);
                expire_history(now);
                m_history[seq] = {RepairWindow{now, 0, seq, 0, true, false}, packet};
                m_cached_sequences.push_back(seq);
                lock.unlock();
                m_send(packet);
                return true;
            }
            m_wake.wait(lock, std::min(delay, m_frame_deadline - now));
        }
        return false;
    }
    std::mutex m_mutex;
    PacingWait m_wake;
    bool m_stopped{false};
    bool m_last_frame_sent{false}; // Sender-thread only.
    PacketBudget m_total, m_repair;
    uint32_t m_encoder_bitrate{8000000};
    uint32_t m_pacing_bitrate{0};
    uint64_t m_frame_deadline{0};
    std::vector<SentPacket> m_history = std::vector<SentPacket>(REPAIR_HISTORY_SIZE);
    std::deque<uint16_t> m_cached_sequences;
    std::deque<rtc::message_ptr> m_repairs;
    rtc::message_callback m_send; // Sender-thread only.
    Stats m_stats;
};

#endif // FALCOR_ENABLE_STREAMING

} // namespace falcor::streaming::detail
