// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "video_encoder.h"

namespace falcor::streaming {

/// STUN or TURN server used to discover or relay a peer connection.
struct ICEServerConfig {
    /// ICE server URL. The transport supports TURN over UDP only.
    std::string url;
    /// TURN authentication username; required for TURN, ignored for STUN.
    std::string username;
    /// TURN authentication password; required for TURN, ignored for STUN.
    std::string credential;
};

/// Connection and buffering settings copied when a WebRTCPeer is constructed.
/// The encoded queue holds at most eight frames in addition to the in-flight frame.
struct WebRTCConfig {
    /// Encoded frame width in pixels (even, 16..3840); must match VideoEncoderConfig::width.
    uint32_t width{1280};
    /// Encoded frame height in pixels (even, 16..2160); must match VideoEncoderConfig::height.
    uint32_t height{720};
    /// Encoded frame rate (1..60); must match VideoEncoderConfig::fps. Frame size
    /// and rate, together with max_bitrate, select the advertised H.264 level (3.1..5.2).
    uint32_t fps{30};
    /// Session encoder bitrate ceiling in bits/s (100000..50000000); must match
    /// VideoEncoderConfig::max_bitrate. Fixes the advertised level, not pacing.
    uint32_t max_bitrate{14000000};
    /// STUN/TURN servers; an empty list uses local host candidates only.
    std::vector<ICEServerConfig> ice_servers;
    /// Require a relay connection. At least one TURN server must be configured.
    bool force_relay{false};
    /// Create the reliable, ordered "falcor-input-v1" data channel for incoming text.
    bool enable_input{false};
    /// Local interface address to bind; empty lets the transport choose interfaces.
    std::string bind_address;
    /// First local ICE port, inclusive; must be nonzero.
    uint16_t port_begin{50000};
    /// Last local ICE port, inclusive; must be at least port_begin.
    uint16_t port_end{50100};
    /// Encoded queue byte limit, from 64 KiB to 16 MiB, excluding the in-flight frame.
    uint32_t max_queue_bytes{2 * 1024 * 1024};
    /// Maximum queued frame age in milliseconds, from 10 to 1000, before recovery.
    uint32_t max_queue_age_ms{200};
};

/// Transport counters and state sampled by stats(). Counters accumulate for the
/// peer's lifetime; fields from different subsystems may reflect different instants.
struct PeerStats {
    /// Whether the video track is open; this does not confirm receiver playback.
    bool connected{false};
    /// Latched after a transport failure, disconnect, or sender exception.
    bool failed{false};
    /// Last sender exception message, or empty if none; transport failures may leave this empty.
    std::string diagnostic;
    /// Frames fully handed to the paced transport callback, without delivery acknowledgement.
    uint64_t sent_frames{0};
    /// Frames rejected or discarded by queue recovery, including aborted in-flight frames.
    uint64_t dropped_frames{0};
    /// Queue resets that discard the dependent frame chain and wait for an IDR.
    uint64_t recoveries{0};
    /// Encoded bytes waiting in the queue, excluding the in-flight frame.
    uint64_t queue_bytes{0};
    /// Frames waiting in the queue, excluding the in-flight frame.
    uint64_t queue_frames{0};
    /// Age of the oldest queued frame in milliseconds, or zero for an empty queue.
    uint64_t queue_age_ms{0};
    /// Whether the sender currently holds a frame outside the queue (zero or one).
    uint64_t in_flight_frames{0};
    /// Received picture-loss indications or legacy full-intra requests handled as keyframe requests.
    uint64_t pli_count{0};
    /// Received generic NACK feedback blocks, not the number of missing packets.
    uint64_t nack_count{0};
    /// Received RTCP receiver report blocks for the video stream.
    uint64_t receiver_reports{0};
    /// Latest reported loss fraction, 0..255; divide by 256 to obtain a fraction.
    uint32_t loss_fraction{0};
    /// Latest reported extended highest RTP sequence number; zero before a report arrives.
    uint32_t receiver_sequence{0};
    /// Latest reported signed cumulative packet loss; negative values indicate excess receptions.
    int32_t receiver_lost{0};
    /// Latest receiver-estimated maximum bitrate in bits per second, or zero before feedback.
    uint32_t receiver_bitrate{0};
    /// Number of receiver-estimated maximum bitrate (REMB) messages received.
    uint64_t receiver_bitrate_reports{0};
    /// Total pacing budget in bits per second, including overhead and retransmissions.
    uint32_t pacing_bitrate{0};
    /// Estimated time to pace the remaining original packets of the current frame, in milliseconds.
    uint64_t pacing_delay_ms{0};
    /// Original RTP packet send attempts, excluding retransmissions and RTCP.
    uint64_t rtp_packets{0};
    /// Original RTP bytes attempted, including RTP headers but excluding SRTP/network overhead.
    uint64_t rtp_bytes{0};
    /// Retransmission packet send attempts.
    uint64_t repair_packets{0};
    /// Retransmitted RTP bytes attempted, excluding SRTP/network overhead.
    uint64_t repair_bytes{0};
    /// Repairs rejected or expired by the pacer, including duplicate requests.
    uint64_t repair_dropped{0};
    /// Retransmission packets currently waiting for pacing credit.
    uint64_t repair_queue_packets{0};
    /// Selected local ICE candidate type, or empty before a candidate pair is selected.
    std::string local_candidate_type;
    /// Selected remote ICE candidate type, or empty before a candidate pair is selected.
    std::string remote_candidate_type;
};

/// Sends one Annex-B H.264 constrained-baseline level-3.1..5.2 stream using RTP payload
/// 102 and a 90 kHz clock. The application supplies signaling and encoded frames.
/// Serialize all calls and destruction. Unless stated otherwise, instance methods
/// throw after close(). Networking callbacks use only native state.
class FALCOR_API WebRTCPeer {
public:
    /// Validate settings, start ICE gathering, and start the sender worker.
    /// Throws if streaming support is disabled or connection setup fails.
    explicit WebRTCPeer(WebRTCConfig config);

    /// Close the connection and join the sender worker.
    ~WebRTCPeer();

    /// Whether this build includes WebRTC support; does not test network connectivity.
    static bool available();

    /// Consume the local SDP offer once ICE gathering completes; empty if not ready or already consumed.
    std::optional<std::string> poll_offer();

    /// Apply one remote SDP answer, at most 64 KiB. It must accept the offered H.264
    /// profile, packetization mode, and input channel if enabled. Throws on rejection
    /// or a second successfully applied answer; renegotiation is not supported.
    void set_answer(const std::string& sdp);

    /// Copy a frame into the bounded queue without waiting for network transmission.
    /// Returns false if rejected; true does not guarantee delivery. Timestamps must
    /// strictly increase across calls, including rejected frames. Queue overflow or
    /// expiry discards dependent frames until an IDR arrives; poll_keyframe_request()
    /// reports this recovery need. Start sending when stats().connected is true.
    /// Throws if the peer has failed or timestamps do not increase.
    bool send(const EncodedFrame& frame);

    /// Request an encoder IDR when true. Requests coalesce and are throttled to one
    /// per 250 ms; recovery repeats requests until an IDR is accepted.
    bool poll_keyframe_request();

    /// Set the encoder bitrate used for pacing, from 100000 to 100000000 bits/s.
    /// Initially 8000000 bits/s; adds a 25% transport allowance. Does not change the encoder or automatically
    /// apply receiver feedback; the caller must coordinate encoder rate changes.
    /// An explicit set_pacing_bitrate() overrides the total transport allowance.
    void set_target_bitrate(uint32_t bits_per_second);

    /// Override total transport pacing, including overhead and repairs, in bits/s.
    /// Accepts 100000..100000000, or zero (default) to use 125% of the encoder target.
    /// Persists across set_target_bitrate() calls without changing the encoder or
    /// its repair budget. A higher allowance can reduce frame delivery delay on
    /// fast links; the caller must choose a rate the connection can sustain.
    void set_pacing_bitrate(uint32_t bits_per_second);

    /// Consume the oldest input text message, or return empty if none is queued.
    /// Input is limited to 4096 bytes per message and 128 queued messages. Invalid
    /// messages or overflow clear the queue and signal poll_input_reset().
    std::optional<std::string> poll_input();

    /// Consume the input-reset flag, set on invalid input, overflow, or channel closure.
    /// When true, clear application-held input state such as pressed keys.
    bool poll_input_reset();

    /// Sample transport state, counters, receiver feedback, and selected ICE candidate types.
    PeerStats stats();

    /// Stop sending, discard pending frames and input, join the worker, and release
    /// transport resources. Safe to call repeatedly; the peer cannot be reopened.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace falcor::streaming
