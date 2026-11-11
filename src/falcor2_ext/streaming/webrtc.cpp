// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "falcor2/streaming/webrtc.h"

FALCOR_PY_EXPORT(streaming_webrtc)
{
    namespace streaming = falcor::streaming;
    using namespace falcor::streaming;
    using namespace nb::literals;

    nb::module_ module = nb::module_::import_("falcor2.streaming");

    nb::class_<ICEServerConfig>(module, "ICEServerConfig", D(streaming, ICEServerConfig))
        .def(nb::init<>())
        .DEF_RW_2(streaming, ICEServerConfig, url)
        .DEF_RW_2(streaming, ICEServerConfig, username)
        .DEF_RW_2(streaming, ICEServerConfig, credential);

    nb::class_<WebRTCConfig>(module, "WebRTCConfig", D(streaming, WebRTCConfig))
        .def(nb::init<>())
        .DEF_RW_2(streaming, WebRTCConfig, fps)
        .DEF_RW_2(streaming, WebRTCConfig, max_bitrate)
        .DEF_RW_2(streaming, WebRTCConfig, width)
        .DEF_RW_2(streaming, WebRTCConfig, height)
        .DEF_RW_2(streaming, WebRTCConfig, ice_servers)
        .DEF_RW_2(streaming, WebRTCConfig, force_relay)
        .DEF_RW_2(streaming, WebRTCConfig, enable_input)
        .DEF_RW_2(streaming, WebRTCConfig, bind_address)
        .DEF_RW_2(streaming, WebRTCConfig, port_begin)
        .DEF_RW_2(streaming, WebRTCConfig, port_end)
        .DEF_RW_2(streaming, WebRTCConfig, max_queue_bytes)
        .DEF_RW_2(streaming, WebRTCConfig, max_queue_age_ms);

    nb::class_<PeerStats>(module, "PeerStats", D(streaming, PeerStats))
        .DEF_RO_2(streaming, PeerStats, connected)
        .DEF_RO_2(streaming, PeerStats, failed)
        .DEF_RO_2(streaming, PeerStats, diagnostic)
        .DEF_RO_2(streaming, PeerStats, sent_frames)
        .DEF_RO_2(streaming, PeerStats, dropped_frames)
        .DEF_RO_2(streaming, PeerStats, recoveries)
        .DEF_RO_2(streaming, PeerStats, queue_bytes)
        .DEF_RO_2(streaming, PeerStats, queue_frames)
        .DEF_RO_2(streaming, PeerStats, queue_age_ms)
        .DEF_RO_2(streaming, PeerStats, in_flight_frames)
        .DEF_RO_2(streaming, PeerStats, pli_count)
        .DEF_RO_2(streaming, PeerStats, nack_count)
        .DEF_RO_2(streaming, PeerStats, receiver_reports)
        .DEF_RO_2(streaming, PeerStats, loss_fraction)
        .DEF_RO_2(streaming, PeerStats, receiver_sequence)
        .DEF_RO_2(streaming, PeerStats, receiver_lost)
        .DEF_RO_2(streaming, PeerStats, receiver_bitrate)
        .DEF_RO_2(streaming, PeerStats, receiver_bitrate_reports)
        .DEF_RO_2(streaming, PeerStats, pacing_bitrate)
        .DEF_RO_2(streaming, PeerStats, pacing_delay_ms)
        .DEF_RO_2(streaming, PeerStats, rtp_packets)
        .DEF_RO_2(streaming, PeerStats, rtp_bytes)
        .DEF_RO_2(streaming, PeerStats, repair_packets)
        .DEF_RO_2(streaming, PeerStats, repair_bytes)
        .DEF_RO_2(streaming, PeerStats, repair_dropped)
        .DEF_RO_2(streaming, PeerStats, repair_queue_packets)
        .DEF_RO_2(streaming, PeerStats, local_candidate_type)
        .DEF_RO_2(streaming, PeerStats, remote_candidate_type);

    // Instance calls retain the GIL to serialize access with close(). Native
    // callbacks never acquire it, so joining the sender while holding it is safe.
    nb::class_<WebRTCPeer>(module, "WebRTCPeer", D(streaming, WebRTCPeer))
        .def_static("available", &WebRTCPeer::available, D(streaming, WebRTCPeer, available))
        .def(nb::init<WebRTCConfig>(), "config"_a, D(streaming, WebRTCPeer, WebRTCPeer))
        .def("poll_offer", &WebRTCPeer::poll_offer, D(streaming, WebRTCPeer, poll_offer))
        .def("poll_input", &WebRTCPeer::poll_input, D(streaming, WebRTCPeer, poll_input))
        .def("poll_input_reset", &WebRTCPeer::poll_input_reset, D(streaming, WebRTCPeer, poll_input_reset))
        .def("set_answer", &WebRTCPeer::set_answer, "sdp"_a, D(streaming, WebRTCPeer, set_answer))
        .def("send", &WebRTCPeer::send, "frame"_a, D(streaming, WebRTCPeer, send))
        .def(
            "set_target_bitrate",
            &WebRTCPeer::set_target_bitrate,
            "bits_per_second"_a,
            D(streaming, WebRTCPeer, set_target_bitrate)
        )
        .def(
            "set_pacing_bitrate",
            &WebRTCPeer::set_pacing_bitrate,
            "bits_per_second"_a,
            D(streaming, WebRTCPeer, set_pacing_bitrate)
        )
        .def(
            "poll_keyframe_request",
            &WebRTCPeer::poll_keyframe_request,
            D(streaming, WebRTCPeer, poll_keyframe_request)
        )
        .def("stats", &WebRTCPeer::stats, D(streaming, WebRTCPeer, stats))
        .def("close", &WebRTCPeer::close, D(streaming, WebRTCPeer, close));
}
