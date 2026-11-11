// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "webrtc.h"
#include "webrtc_internal.h"
#include "video_format.h"

#include "falcor2/core/config.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <condition_variable>
#include <cstring>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>

#if FALCOR_ENABLE_STREAMING
#include <rtc/rtc.hpp>
#endif

namespace falcor::streaming {

using namespace detail;

bool WebRTCPeer::available()
{
    return FALCOR_ENABLE_STREAMING != 0;
}

#if FALCOR_ENABLE_STREAMING

namespace {
using Clock = std::chrono::steady_clock;
uint64_t now_ms()
{
    return uint64_t(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count());
}
constexpr uint32_t VIDEO_SSRC = 42;
constexpr uint8_t VIDEO_PAYLOAD = 102;

struct Feedback {
    std::mutex mutex;
    std::optional<std::string> offer;
    std::deque<std::string> input;
    bool input_reset{false};
    std::atomic<bool> keyframe{true};
    std::atomic<bool> failed{false};
    std::atomic<uint64_t> pli{0}, nack{0}, reports{0};
    std::atomic<uint32_t> loss{0}, bitrate{0};
    std::atomic<uint64_t> bitrate_reports{0};
    uint32_t receiver_sequence{0}; // Coherent report snapshot under mutex.
    int32_t receiver_lost{0};
};

// Validate compound RTCP lengths before passing messages to library handlers.
// Read RR fields only from complete report blocks for our video SSRC.
class ReceiverFeedback : public rtc::MediaHandler {
public:
    explicit ReceiverFeedback(std::shared_ptr<Feedback> feedback)
        : m_feedback(std::move(feedback))
    {
    }
    void incoming(rtc::message_vector& messages, const rtc::message_callback&) override
    {
        std::erase_if(
            messages,
            [&](const rtc::message_ptr& message)
            {
                if (message->type != rtc::Message::Control)
                    return true;
                size_t offset = 0;
                while (offset < message->size()) {
                    if (message->size() - offset < 4)
                        return true;
                    const auto* p = reinterpret_cast<const uint8_t*>(message->data() + offset);
                    const size_t length = ((size_t(p[2]) << 8) + p[3] + 1) * 4;
                    if ((p[0] >> 6) != 2 || length < 4 || length > message->size() - offset)
                        return true;
                    const size_t count = p[0] & 31;
                    if (p[1] == 201) {
                        if (length < 8 + count * 24)
                            return true;
                        for (size_t i = 0; i < count; ++i) {
                            const auto block = decode_receiver_report(std::span<const uint8_t, 24>(p + 8 + i * 24, 24));
                            if (block.ssrc == VIDEO_SSRC) {
                                std::lock_guard lock(m_feedback->mutex);
                                m_feedback->loss = block.fraction_lost;
                                m_feedback->receiver_sequence = block.sequence;
                                m_feedback->receiver_lost = block.cumulative_lost;
                                ++m_feedback->reports;
                            }
                        }
                    }
                    if (p[1] == 205 || p[1] == 206) {
                        if (length < 12)
                            return true;
                        if (p[1] == 205 && count == 1)
                            ++m_feedback->nack;
                    }
                    offset += length;
                }
                return false;
            }
        );
    }

private:
    std::shared_ptr<Feedback> m_feedback;
};

std::string candidate_type(const rtc::Candidate& candidate)
{
    std::ostringstream stream;
    stream << candidate.type();
    return stream.str();
}
} // namespace

struct WebRTCPeer::Impl {
    std::shared_ptr<Feedback> feedback = std::make_shared<Feedback>();
    std::shared_ptr<rtc::PeerConnection> peer;
    std::shared_ptr<rtc::Track> track;
    std::shared_ptr<rtc::DataChannel> input_channel;
    std::shared_ptr<rtc::RtpPacketizationConfig> packetization;
    std::shared_ptr<PacketPacer> pacer = std::make_shared<PacketPacer>();
    std::mutex mutex;
    std::condition_variable wake;
    EncodedQueue queue;
    std::thread sender;
    bool stopped{false};
    bool answered{false};
    uint64_t sent{0};
    bool in_flight{false};
    uint64_t last_keyframe_request{0};
    uint64_t last_pts{0};
    bool have_pts{false};
    std::string error;
    std::string level;

    explicit Impl(const WebRTCConfig& config)
        : queue(config.max_queue_bytes, config.max_queue_age_ms)
    {
        level = fmt::format("{:02x}", h264_level(config.width, config.height, config.fps, config.max_bitrate));
        FALCOR_CHECK(config.port_begin > 0 && config.port_end >= config.port_begin, "Invalid ICE port range");
        FALCOR_CHECK(
            config.max_queue_bytes >= 65536 && config.max_queue_bytes <= 16 * 1024 * 1024,
            "Media queue must contain 64 KiB to 16 MiB"
        );
        FALCOR_CHECK(
            config.max_queue_age_ms >= 10 && config.max_queue_age_ms <= 1000,
            "Media queue age must be 10 to 1000 ms"
        );
        rtc::Configuration settings;
        settings.portRangeBegin = config.port_begin;
        settings.portRangeEnd = config.port_end;
        settings.mtu = 1280;
        settings.maxMessageSize = 4096;
        settings.disableAutoNegotiation = true;
        settings.iceTransportPolicy = config.force_relay ? rtc::TransportPolicy::Relay : rtc::TransportPolicy::All;
        if (!config.bind_address.empty())
            settings.bindAddress = config.bind_address;
        bool have_turn = false;
        for (const auto& item : config.ice_servers) {
            rtc::IceServer server(item.url);
            if (server.type == rtc::IceServer::Type::Turn) {
                FALCOR_CHECK(
                    server.relayType == rtc::IceServer::RelayType::TurnUdp,
                    "The pinned libjuice backend supports TURN/UDP only"
                );
                have_turn = true;
                FALCOR_CHECK(
                    !item.username.empty() && !item.credential.empty(),
                    "TURN requires username and credential"
                );
                server.username = item.username;
                server.password = item.credential;
            }
            settings.iceServers.push_back(std::move(server));
        }
        FALCOR_CHECK(!config.force_relay || have_turn, "Forced relay requires a configured TURN server");
        // There is one controlling viewer. The library default uses
        // hardware concurrency, which is excessive on large cluster hosts and
        // spreads lazy per-worker initialization over successive sessions.
        static std::once_flag initialize_transport;
        std::call_once(
            initialize_transport,
            []
            {
                rtc::SetThreadPoolSize(4);
            }
        );
        peer = std::make_shared<rtc::PeerConnection>(settings);
        auto state = feedback;
        std::weak_ptr<rtc::PeerConnection> weak_peer = peer;
        peer->onGatheringStateChange(
            [state, weak_peer](rtc::PeerConnection::GatheringState value)
            {
                if (value == rtc::PeerConnection::GatheringState::Complete) {
                    if (auto peer = weak_peer.lock()) {
                        std::lock_guard lock(state->mutex);
                        if (auto description = peer->localDescription())
                            state->offer = std::string(*description);
                    }
                }
            }
        );
        peer->onStateChange(
            [state](rtc::PeerConnection::State value)
            {
                if (value == rtc::PeerConnection::State::Failed || value == rtc::PeerConnection::State::Disconnected)
                    state->failed = true;
            }
        );
        rtc::Description::Video video("video", rtc::Description::Direction::SendOnly);
        video.addH264Codec(
            VIDEO_PAYLOAD,
            "profile-level-id=42c0" + level + ";packetization-mode=1;level-asymmetry-allowed=1"
        );
        video.addSSRC(VIDEO_SSRC, "falcor", "falcor-stream", "falcor-video");
        track = peer->addTrack(video);
        packetization = std::make_shared<rtc::RtpPacketizationConfig>(VIDEO_SSRC, "falcor", VIDEO_PAYLOAD, 90000);
        auto packetizer
            = std::make_shared<rtc::H264RtpPacketizer>(rtc::NalUnit::Separator::StartSequence, packetization, 1200);
        packetizer->addToChain(std::make_shared<rtc::RtcpSrReporter>(packetization));
        packetizer->addToChain(
            std::make_shared<rtc::PliHandler>(
                [state]
                {
                    ++state->pli;
                    state->keyframe = true;
                }
            )
        );
        packetizer->addToChain(
            std::make_shared<rtc::RembHandler>(
                [state](unsigned int bitrate)
                {
                    state->bitrate = bitrate;
                    ++state->bitrate_reports;
                }
            )
        );
        packetizer->addToChain(pacer);
        // incomingChain runs in reverse: validate before any library parser.
        packetizer->addToChain(std::make_shared<ReceiverFeedback>(state));
        track->setMediaHandler(packetizer);
        if (config.enable_input) {
            input_channel = peer->createDataChannel("falcor-input-v1");
            input_channel->onMessage(
                [state](rtc::message_variant message)
                {
                    std::lock_guard lock(state->mutex);
                    auto text = std::get_if<std::string>(&message);
                    if (!text || text->size() > 4096 || state->input.size() >= 128) {
                        state->input.clear();
                        state->input_reset = true;
                        return;
                    }
                    state->input.push_back(std::move(*text));
                }
            );
            input_channel->onClosed(
                [state]
                {
                    std::lock_guard lock(state->mutex);
                    state->input.clear();
                    state->input_reset = true;
                }
            );
        }
    }
    ~Impl() { stop(); }
    void start()
    {
        peer->setLocalDescription();
        sender = std::thread(
            [this]
            {
                run();
            }
        );
    }
    void stop()
    {
        {
            std::lock_guard lock(mutex);
            stopped = true;
        }
        wake.notify_one();
        pacer->stop();
        if (sender.joinable())
            sender.join();
        if (input_channel) {
            input_channel->resetCallbacks();
            input_channel->close();
            input_channel.reset();
        }
        if (track) {
            track->resetCallbacks();
            track->close();
            track.reset();
        }
        if (peer) {
            peer->resetCallbacks();
            peer->close();
            peer.reset();
        }
    }
    void run() noexcept
    {
        try {
            while (true) {
                std::optional<EncodedFrame> frame;
                {
                    std::unique_lock lock(mutex);
                    wake.wait_for(
                        lock,
                        std::chrono::milliseconds(5),
                        [&]
                        {
                            return stopped || queue.size() > 0;
                        }
                    );
                    if (stopped)
                        break;
                    frame = queue.pop(now_ms());
                    in_flight = frame.has_value();
                }
                pacer->drain_repairs();
                if (!frame)
                    continue;
                if (!track->isOpen()) {
                    std::lock_guard lock(mutex);
                    in_flight = false;
                    queue.recover(1);
                    continue;
                }
                packetization->timestamp = packetization->startTimestamp + rtp_timestamp(frame->pts_us);
                track->send(reinterpret_cast<const std::byte*>(frame->data.data()), frame->data.size());
                // The handler consumes messages after sending through the
                // callback; libdatachannel then returns false for an empty list.
                const bool sent_ok = pacer->last_frame_sent();
                std::lock_guard lock(mutex);
                in_flight = false;
                if (sent_ok)
                    ++sent;
                else
                    queue.recover(1);
            }
        } catch (const std::exception& exception) {
            std::lock_guard lock(mutex);
            error = exception.what();
            feedback->failed = true;
        }
    }
};

WebRTCPeer::WebRTCPeer(WebRTCConfig config)
    : m_impl(std::make_unique<Impl>(config))
{
    m_impl->start();
}

std::optional<std::string> WebRTCPeer::poll_offer()
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    std::lock_guard lock(m_impl->feedback->mutex);
    auto offer = std::move(m_impl->feedback->offer);
    m_impl->feedback->offer.reset();
    return offer;
}

void WebRTCPeer::set_answer(const std::string& sdp)
{
    FALCOR_CHECK(m_impl && !m_impl->answered, "Peer is closed or already answered");
    FALCOR_CHECK(sdp.size() <= 65536, "SDP exceeds 64 KiB");
    rtc::Description answer(sdp, "answer");
    FALCOR_CHECK(answer.mediaCount() == (m_impl->input_channel ? 2 : 1), "Unexpected media section count");
    auto media_variant = answer.media(0);
    auto media_ptr = std::get_if<rtc::Description::Media*>(&media_variant);
    FALCOR_CHECK(media_ptr && !(*media_ptr)->isRemoved(), "Browser rejected video");
    const auto* codec = (*media_ptr)->rtpMap(VIDEO_PAYLOAD);
    FALCOR_CHECK(
        codec && codec->format == "H264" && codec->clockRate == 90000,
        "Browser did not accept H.264 payload 102"
    );
    std::unordered_map<std::string, std::string> parameters;
    for (const auto& fmtp : codec->fmtps) {
        std::istringstream fields(fmtp);
        std::string field;
        while (std::getline(fields, field, ';')) {
            auto trim = [](std::string value)
            {
                const auto begin = value.find_first_not_of(" \t\r\n");
                return begin == std::string::npos ? std::string{}
                                                  : value.substr(begin, value.find_last_not_of(" \t\r\n") - begin + 1);
            };
            field = trim(field);
            if (field.empty())
                continue;
            const auto separator = field.find('=');
            FALCOR_CHECK(separator != std::string::npos, "Invalid H.264 format parameter");
            auto name = trim(field.substr(0, separator));
            auto value = trim(field.substr(separator + 1));
            FALCOR_CHECK(
                !name.empty() && !value.empty() && parameters.emplace(name, value).second,
                "Empty or duplicate H.264 format parameter"
            );
        }
    }
    auto profile = parameters["profile-level-id"];
    std::transform(
        profile.begin(),
        profile.end(),
        profile.begin(),
        [](unsigned char value)
        {
            return char(std::tolower(value));
        }
    );
    const bool supported_profile = parameters["packetization-mode"] == "1"
        && (profile == "42c0" + m_impl->level || profile == "42e0" + m_impl->level);
    FALCOR_CHECK(
        supported_profile,
        "Browser must accept the offered constrained-baseline H.264 level, packetization mode 1"
    );
    if (m_impl->input_channel) {
        auto application_variant = answer.media(1);
        auto application = std::get_if<rtc::Description::Application*>(&application_variant);
        FALCOR_CHECK(application && !(*application)->isRemoved(), "Browser rejected the input data channel");
    }
    m_impl->peer->setRemoteDescription(answer);
    m_impl->answered = true;
}

bool WebRTCPeer::send(const EncodedFrame& frame)
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    std::lock_guard lock(m_impl->mutex);
    FALCOR_CHECK(!m_impl->feedback->failed && !m_impl->stopped, "Peer has failed or stopped");
    FALCOR_CHECK(!m_impl->have_pts || frame.pts_us > m_impl->last_pts, "Video timestamps must increase strictly");
    m_impl->last_pts = frame.pts_us;
    m_impl->have_pts = true;
    bool accepted = m_impl->queue.push(frame, now_ms());
    m_impl->wake.notify_one();
    return accepted;
}

bool WebRTCPeer::poll_keyframe_request()
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    std::lock_guard lock(m_impl->mutex);
    const auto now = now_ms();
    if (now - m_impl->last_keyframe_request < 250)
        return false;
    if (!m_impl->queue.needs_keyframe() && !m_impl->feedback->keyframe)
        return false;
    m_impl->last_keyframe_request = now;
    m_impl->feedback->keyframe = false;
    return true;
}

void WebRTCPeer::set_target_bitrate(uint32_t bits_per_second)
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    FALCOR_CHECK(bits_per_second >= 100000 && bits_per_second <= 100000000, "Invalid pacing bitrate");
    m_impl->pacer->set_rate(bits_per_second);
}

void WebRTCPeer::set_pacing_bitrate(uint32_t bits_per_second)
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    FALCOR_CHECK(
        bits_per_second == 0 || (bits_per_second >= 100000 && bits_per_second <= 100000000),
        "Invalid pacing bitrate"
    );
    m_impl->pacer->set_pacing_rate(bits_per_second);
}

PeerStats WebRTCPeer::stats()
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    PeerStats report;
    auto& impl = *m_impl;
    {
        std::lock_guard lock(impl.mutex);
        report.sent_frames = impl.sent;
        report.dropped_frames = impl.queue.dropped();
        report.recoveries = impl.queue.recoveries();
        report.queue_bytes = impl.queue.bytes();
        report.queue_frames = impl.queue.size();
        report.queue_age_ms = impl.queue.age_ms(now_ms());
        report.in_flight_frames = impl.in_flight ? 1 : 0;
        report.diagnostic = impl.error;
    }
    report.connected = impl.track->isOpen();
    report.failed = impl.feedback->failed;
    report.pli_count = impl.feedback->pli;
    report.nack_count = impl.feedback->nack;
    {
        std::lock_guard lock(impl.feedback->mutex);
        report.receiver_reports = impl.feedback->reports;
        report.loss_fraction = impl.feedback->loss;
        report.receiver_sequence = impl.feedback->receiver_sequence;
        report.receiver_lost = impl.feedback->receiver_lost;
    }
    report.receiver_bitrate = impl.feedback->bitrate;
    report.receiver_bitrate_reports = impl.feedback->bitrate_reports;
    const auto pacing = impl.pacer->stats();
    report.pacing_bitrate = pacing.bitrate;
    report.pacing_delay_ms = (pacing.pending_bytes * 8000 + pacing.bitrate - 1) / pacing.bitrate;
    report.rtp_packets = pacing.packets;
    report.rtp_bytes = pacing.bytes;
    report.repair_packets = pacing.repair_packets;
    report.repair_bytes = pacing.repair_bytes;
    report.repair_dropped = pacing.repair_dropped;
    report.repair_queue_packets = pacing.repair_queue;
    rtc::Candidate local, remote;
    if (impl.peer->getSelectedCandidatePair(&local, &remote)) {
        report.local_candidate_type = candidate_type(local);
        report.remote_candidate_type = candidate_type(remote);
    }
    return report;
}
void WebRTCPeer::close()
{
    m_impl.reset();
}

std::optional<std::string> WebRTCPeer::poll_input()
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    std::lock_guard lock(m_impl->feedback->mutex);
    auto& input = m_impl->feedback->input;
    if (input.empty())
        return {};
    auto message = std::move(input.front());
    input.pop_front();
    return message;
}

bool WebRTCPeer::poll_input_reset()
{
    FALCOR_CHECK(m_impl, "Peer is closed");
    std::lock_guard lock(m_impl->feedback->mutex);
    return std::exchange(m_impl->feedback->input_reset, false);
}

#else // FALCOR_ENABLE_STREAMING

struct WebRTCPeer::Impl { };

WebRTCPeer::WebRTCPeer(WebRTCConfig)
{
    FALCOR_THROW("WebRTC not built; configure FALCOR_ENABLE_STREAMING=ON");
}

std::optional<std::string> WebRTCPeer::poll_offer()
{
    FALCOR_THROW("WebRTC not built");
}

void WebRTCPeer::set_answer(const std::string&)
{
    FALCOR_THROW("WebRTC not built");
}

bool WebRTCPeer::send(const EncodedFrame&)
{
    FALCOR_THROW("WebRTC not built");
}

bool WebRTCPeer::poll_keyframe_request()
{
    FALCOR_THROW("WebRTC not built");
}

void WebRTCPeer::set_target_bitrate(uint32_t)
{
    FALCOR_THROW("WebRTC not built");
}

void WebRTCPeer::set_pacing_bitrate(uint32_t)
{
    FALCOR_THROW("WebRTC not built");
}

PeerStats WebRTCPeer::stats()
{
    FALCOR_THROW("WebRTC not built");
}

void WebRTCPeer::close() { }

std::optional<std::string> WebRTCPeer::poll_input()
{
    FALCOR_THROW("WebRTC not built");
}

bool WebRTCPeer::poll_input_reset()
{
    FALCOR_THROW("WebRTC not built");
}

#endif // FALCOR_ENABLE_STREAMING

WebRTCPeer::~WebRTCPeer() = default;

} // namespace falcor::streaming
