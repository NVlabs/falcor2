# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from falcor2 import streaming
import time

import pytest


@pytest.mark.parametrize(
    "width, height, fps, ceiling, level",
    [
        (1280, 720, 30, 14000000, "1f"),
        (1280, 720, 60, 14000000, "20"),
        (1920, 1080, 30, 14000000, "28"),
        (1920, 1080, 60, 14000000, "2a"),
        (3840, 2160, 30, 14000000, "33"),
        (3840, 2160, 60, 14000000, "34"),
        (1280, 720, 30, 20000000, "20"),
        (1280, 720, 60, 30000000, "29"),
        (1280, 720, 60, 50000000, "29"),
    ],
)
def test_native_peer_offer_level(
    width: int, height: int, fps: int, ceiling: int, level: str
) -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    config = streaming.WebRTCConfig()
    config.fps = fps
    config.max_bitrate = ceiling
    config.width, config.height = width, height
    peer = streaming.WebRTCPeer(config)
    try:
        deadline = time.monotonic() + 10
        offer = None
        while offer is None and time.monotonic() < deadline:
            offer = peer.poll_offer()
            time.sleep(0.01)
        assert offer is not None and f"profile-level-id=42c0{level}" in offer
    finally:
        peer.close()


@pytest.mark.parametrize("enable_input", [False, True])
def test_native_peer_answer_validation_and_close(enable_input: bool) -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    config = streaming.WebRTCConfig()
    config.enable_input = enable_input
    config.fps = 60
    peer = streaming.WebRTCPeer(config)
    try:
        assert peer.poll_input() is None
        assert not peer.poll_input_reset()
        with pytest.raises(RuntimeError):
            peer.set_answer("invalid SDP")
        # Codec validation precedes transport setup, so it needs no gathered offer.
        answer = (
            "v=0\r\n"
            "o=- 1 1 IN IP4 127.0.0.1\r\n"
            "s=-\r\n"
            "t=0 0\r\n"
            "m=video 9 UDP/TLS/RTP/SAVPF 102\r\n"
            "c=IN IP4 0.0.0.0\r\n"
            "a=mid:video\r\n"
            "a=rtpmap:102 H264/90000\r\n"
            "a=fmtp:102 packetization-mode=1;profile-level-id=42c020\r\n"
        )
        if enable_input:
            answer += (
                "m=application 9 UDP/DTLS/SCTP webrtc-datachannel\r\n"
                "a=mid:input\r\n"
                "a=sctp-port:5000\r\n"
            )
        for name, invalid in (
            ("packetization-mode=1", "packetization-mode=10"),
            ("profile-level-id=42c020", "profile-level-id=42c020ff"),
            ("profile-level-id=42c020", "profile-level-id=42c01e"),
            ("profile-level-id=42c020", "profile-level-id=42c01f"),
        ):
            with pytest.raises(RuntimeError, match="constrained-baseline"):
                peer.set_answer(answer.replace(name, invalid))
        assert not peer.stats().connected
        assert peer.stats().repair_queue_packets == 0
    finally:
        peer.close()
    peer.close()
    with pytest.raises(RuntimeError, match="closed"):
        peer.stats()


@pytest.mark.parametrize("fps", [0, 61])
def test_peer_rejects_invalid_fps(fps: int) -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    config = streaming.WebRTCConfig()
    config.fps = fps
    with pytest.raises(RuntimeError, match="FPS"):
        streaming.WebRTCPeer(config)


def test_independent_pacing_survives_encoder_target_changes() -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    peer = streaming.WebRTCPeer(streaming.WebRTCConfig())
    try:
        assert peer.stats().pacing_bitrate == 10000000
        peer.set_target_bitrate(1000000)
        assert peer.stats().pacing_bitrate == 1250000
        for invalid in (0, 99999, 100000001):
            with pytest.raises(RuntimeError, match="pacing bitrate"):
                peer.set_target_bitrate(invalid)
        for rate in (100000, 50000000, 100000000):
            peer.set_pacing_bitrate(bits_per_second=rate)
            assert peer.stats().pacing_bitrate == rate
            peer.set_target_bitrate(4000000)
            assert peer.stats().pacing_bitrate == rate
        for invalid in (1, 99999, 100000001):
            with pytest.raises(RuntimeError, match="pacing bitrate"):
                peer.set_pacing_bitrate(invalid)
            assert peer.stats().pacing_bitrate == 100000000
        peer.set_pacing_bitrate(0)
        assert peer.stats().pacing_bitrate == 5000000
        peer.set_target_bitrate(8000000)
        assert peer.stats().pacing_bitrate == 10000000
    finally:
        peer.close()
    with pytest.raises(RuntimeError, match="closed"):
        peer.set_pacing_bitrate(50000000)


def test_relay_requires_configuration() -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    config = streaming.WebRTCConfig()
    config.force_relay = True
    with pytest.raises(RuntimeError, match="TURN"):
        streaming.WebRTCPeer(config)


@pytest.mark.parametrize("ceiling", [0, 99999, 50000001])
def test_peer_rejects_invalid_bitrate_ceiling(ceiling: int) -> None:
    if not streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=ON")
    config = streaming.WebRTCConfig()
    config.max_bitrate = ceiling
    with pytest.raises(RuntimeError, match="Maximum bitrate"):
        streaming.WebRTCPeer(config)


def test_disabled_peer_diagnostic() -> None:
    if streaming.WebRTCPeer.available():
        pytest.skip("Requires FALCOR_ENABLE_STREAMING=OFF")
    assert not streaming.VideoEncoder.check_available()
    with pytest.raises(RuntimeError, match="FALCOR_ENABLE_STREAMING=ON"):
        streaming.WebRTCPeer(streaming.WebRTCConfig())
