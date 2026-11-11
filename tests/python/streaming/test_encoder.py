# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
import shutil
import time
from pathlib import Path

import numpy as np
from falcor2 import streaming

import pytest
import slangpy as spy
from falcor2.testing.helpers import DEFAULT_DEVICE_TYPES, get_device
from falcor2.streaming import openh264


def _make_pattern(device: spy.Device, width: int, height: int) -> spy.Texture:
    image = device.create_texture(
        type=spy.TextureType.texture_2d,
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
    )
    kernel = device.create_compute_kernel(
        device.load_program("streaming/pattern.slang", ["make_pattern"])
    )
    kernel.dispatch(
        thread_count=[width, height, 1], vars={"pattern": image, "width": width, "height": height}
    )
    return image


@pytest.fixture
def streaming_runtime() -> None:
    if not streaming.VideoEncoder.check_available():
        pytest.skip("Requires native streaming support and an installed OpenH264 codec")


@pytest.fixture(params=DEFAULT_DEVICE_TYPES)
def streaming_device(request: pytest.FixtureRequest) -> spy.Device:
    if not spy.Device.enumerate_adapters(request.param):
        pytest.skip(f"No GPU available for {request.param}")
    return get_device(request.param)


@pytest.fixture
def encoder_device(streaming_runtime: None, streaming_device: spy.Device) -> spy.Device:
    return streaming_device


def _config(slots: int = 3) -> streaming.VideoEncoderConfig:
    config = streaming.VideoEncoderConfig()
    config.width, config.height = 256, 144
    config.max_frame_slots = slots
    return config


def _receive(encoder: streaming.VideoEncoder) -> streaming.EncodedFrame:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        packet = encoder.poll_output()
        if packet is not None:
            return packet
        time.sleep(0.001)
    pytest.fail("Native encoder did not produce a packet within ten seconds")


def _nal_types(data: bytes) -> set[int]:
    return {part[0] & 31 for part in re.split(b"\x00\x00\x00?\x01", data) if part}


def test_bounded_pool_timestamps_and_controls(encoder_device: spy.Device) -> None:
    encoder = streaming.VideoEncoder(encoder_device, _config())
    try:
        image = _make_pattern(encoder_device, 256, 144)
        for frame_id in range(3):
            assert encoder.try_submit(
                image, frame_id, frame_id * 33333, streaming.InputTransfer.linear
            )
        # Even completed output occupies a slot until consumed. This bounds both
        # queued GPU work and compressed packets when the consumer stalls.
        for frame_id in range(3, 6):
            assert not encoder.try_submit(
                image, frame_id, frame_id * 33333, streaming.InputTransfer.linear
            )
        del image  # The native pool must own in-flight sources.
        packets = [_receive(encoder) for _ in range(3)]
        assert [packet.frame_id for packet in packets] == [0, 1, 2]
        assert [packet.pts_us for packet in packets] == [0, 33333, 66666]
        assert all(packet.duration_us == 33333 for packet in packets)
        for packet in packets:
            assert packet.byte_size == len(packet.data) > 0
            assert packet.gpu_prepare_ms >= 0
            assert 0 < packet.encode_wall_ms <= packet.submit_to_output_ms
        assert packets[0].keyframe and {5, 7, 8} <= _nal_types(packets[0].data)
        sps = next(
            part
            for part in re.split(b"\x00\x00\x00?\x01", packets[0].data)
            if part and part[0] & 31 == 7
        )
        assert sps[1] == 66 and sps[2] & 0xC0 == 0xC0 and sps[3] <= 31
        assert not packets[1].keyframe and 1 in _nal_types(packets[1].data)
        # Busy frames never collapse the capture time line. Subsequent output
        # keeps the caller's frame ID and timestamp, including this large gap.
        encoder.request_keyframe()
        image = _make_pattern(encoder_device, 256, 144)
        assert encoder.try_submit(image, 1003, 1003 * 33333, streaming.InputTransfer.linear)
        packet = _receive(encoder)
        assert packet.frame_id == 1003 and packet.pts_us == 1003 * 33333
        assert packet.keyframe and {5, 7, 8} <= _nal_types(packet.data)
        encoder.set_target_bitrate(4000000)
        assert encoder.try_submit(image, 1004, 1004 * 33333, streaming.InputTransfer.srgb)
        assert _receive(encoder).keyframe
        with pytest.raises(RuntimeError, match="increase strictly"):
            encoder.try_submit(image, 1004, 1004 * 33333, streaming.InputTransfer.linear)
        with pytest.raises(RuntimeError, match="Bitrate"):
            encoder.set_target_bitrate(1)
    finally:
        encoder.close()
    encoder.close()
    with pytest.raises(RuntimeError, match="closed"):
        encoder.poll_output()


def test_close_with_work_and_reinitialize(encoder_device: spy.Device) -> None:
    for _ in range(3):
        encoder = streaming.VideoEncoder(encoder_device, _config())
        for frame_id in range(3):
            image = _make_pattern(encoder_device, 256, 144)
            assert encoder.try_submit(
                image, frame_id, frame_id * 33333, streaming.InputTransfer.linear
            )
        del image
        encoder.close()  # No output consumer; must drain in-flight work without deadlocking.
    encoder = streaming.VideoEncoder(encoder_device, _config(1))
    image = _make_pattern(encoder_device, 256, 144)
    assert encoder.try_submit(image, 0, 0, streaming.InputTransfer.linear)
    del encoder  # Destructor also owns shutdown.


def test_config_validation(streaming_device: spy.Device) -> None:
    for field, value, diagnostic in [
        ("width", 255, "Dimensions"),
        ("width", 3842, "Dimensions"),
        ("height", 2162, "Dimensions"),
        ("fps", 0, "FPS"),
        ("fps", 61, "FPS"),
        ("max_frame_slots", 0, "Frame pool"),
        ("target_bitrate", 0, "Bitrate"),
        ("target_bitrate", 14000001, "configured maximum"),
        ("max_bitrate", 99999, "Maximum bitrate"),
        ("max_bitrate", 50000001, "Maximum bitrate"),
        ("max_bitrate", 1000000, "configured maximum"),
    ]:
        config = _config()
        setattr(config, field, value)
        with pytest.raises(RuntimeError, match=diagnostic):
            streaming.VideoEncoder(streaming_device, config)


@pytest.mark.parametrize(
    "width, height, fps, level",
    [
        (1280, 720, 30, 31),
        (1280, 720, 31, 32),
        (1280, 720, 60, 32),
        (1920, 1080, 30, 40),
        (1920, 1080, 60, 42),
        (3840, 2160, 30, 51),
        (3840, 2160, 60, 52),
    ],
)
def test_frame_rate_and_bitstream_level(
    encoder_device: spy.Device, width: int, height: int, fps: int, level: int
) -> None:
    config = _config()
    config.fps = fps
    config.width, config.height = width, height
    encoder = streaming.VideoEncoder(encoder_device, config)
    try:
        image = _make_pattern(encoder_device, config.width, config.height)
        assert encoder.try_submit(image, 0, 0, streaming.InputTransfer.linear)
        packet = _receive(encoder)
        assert packet.duration_us == 1000000 // fps
        parameter_sets = [
            part
            for part in re.split(b"\x00\x00\x00?\x01", packet.data)
            if part and part[0] & 31 == 7
        ]
        assert parameter_sets and parameter_sets[0][3] == level
    finally:
        encoder.close()


@pytest.mark.parametrize(
    "ceiling, level", [(14000000, 31), (20000000, 32), (30000000, 41), (50000000, 41)]
)
def test_bitrate_ceiling_fixes_bitstream_level(
    encoder_device: spy.Device, ceiling: int, level: int
) -> None:
    config = _config()
    config.max_bitrate = ceiling
    config.target_bitrate = ceiling
    encoder = streaming.VideoEncoder(encoder_device, config)
    try:
        image = _make_pattern(encoder_device, config.width, config.height)
        for frame_id, target in enumerate((ceiling, 1000000, ceiling)):
            encoder.set_target_bitrate(target)
            assert encoder.try_submit(
                image, frame_id, frame_id * 33333, streaming.InputTransfer.linear
            )
            packet = _receive(encoder)
            assert packet.keyframe
            sps = [
                part
                for part in re.split(b"\x00\x00\x00?\x01", packet.data)
                if part and part[0] & 31 == 7
            ]
            assert sps and sps[0][3] == level
        with pytest.raises(RuntimeError, match="configured maximum"):
            encoder.set_target_bitrate(ceiling + 1)
    finally:
        encoder.close()


def test_source_validation(encoder_device: spy.Device) -> None:
    encoder = streaming.VideoEncoder(encoder_device, _config())
    try:
        image = _make_pattern(encoder_device, 258, 144)
        with pytest.raises(RuntimeError, match="dimensions"):
            encoder.try_submit(image, 0, 0, streaming.InputTransfer.linear)
        image = encoder_device.create_texture(
            format=spy.Format.rgba8_unorm_srgb,
            width=256,
            height=144,
            usage=spy.TextureUsage.shader_resource,
        )
        with pytest.raises(RuntimeError, match="Unsupported source format"):
            encoder.try_submit(image, 0, 0, streaming.InputTransfer.srgb)
    finally:
        encoder.close()


@pytest.mark.parametrize("complexity", ["low", "medium", "high"])
@pytest.mark.parametrize("threads", [1, 4, 8, 16])
def test_software_encoding_controls(
    encoder_device: spy.Device, complexity: str, threads: int
) -> None:
    config = _config()
    config.complexity = getattr(streaming.VideoEncoderComplexity, complexity)
    config.thread_count = threads
    encoder = streaming.VideoEncoder(encoder_device, config)
    try:
        image = _make_pattern(encoder_device, config.width, config.height)
        for frame_id in range(2):
            assert encoder.try_submit(
                image, frame_id, frame_id * 33333, streaming.InputTransfer.linear
            )
            packet = _receive(encoder)
            assert packet.frame_id == frame_id and packet.byte_size > 0
            assert (7 if frame_id == 0 else 1) in _nal_types(packet.data)
    finally:
        encoder.close()


@pytest.mark.parametrize("threads", [0, 17])
def test_invalid_thread_count(encoder_device: spy.Device, threads: int) -> None:
    config = _config()
    config.thread_count = threads
    with pytest.raises(RuntimeError, match="Thread count"):
        streaming.VideoEncoder(encoder_device, config)


def test_software_encoding_defaults() -> None:
    config = streaming.VideoEncoderConfig()
    assert config.complexity == streaming.VideoEncoderComplexity.high
    assert config.thread_count == 4


def test_unavailable_runtime(streaming_device: spy.Device) -> None:
    if streaming.VideoEncoder.check_available():
        pytest.skip("Requires a build or installation without OpenH264 support")
    with pytest.raises(RuntimeError, match="OpenH264"):
        streaming.VideoEncoder(streaming_device, _config())


@pytest.mark.parametrize("srgb", [False, True])
def test_transfer_ramp_orientation_and_chroma(streaming_device: spy.Device, srgb: bool) -> None:
    width, height = 258, 18
    data = np.ones((height, width, 4), dtype=np.float32)
    data[..., :3] = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    data[0:2, 0:2, :3] = [1, 0, 0]
    data[-2:, 0:2, :3] = [0, 0, 1]
    image = streaming_device.create_texture(
        format=spy.Format.rgba32_float,
        width=width,
        height=height,
        usage=spy.TextureUsage.shader_resource,
        data=data,
    )
    pitch = (width + 255) & ~255
    size = pitch * height * 3 // 2
    buffer = streaming_device.create_buffer(size=size, usage=spy.BufferUsage.unordered_access)
    kernel = streaming_device.create_compute_kernel(
        streaming_device.load_program("falcor2/streaming/convert.slang", ["convert_i420"])
    )
    kernel.dispatch(
        thread_count=[pitch // 8, height // 2, 1],
        vars={
            "source": image,
            "i420": buffer,
            "width": width,
            "height": height,
            "pitch": pitch,
            "input_is_srgb": int(srgb),
        },
    )
    # Compare conversion directly before lossy compression.
    actual = buffer.to_numpy().view(np.uint8).reshape(height * 3 // 2, pitch)
    rgb = np.clip(data[..., :3], 0, 1)
    if not srgb:
        rgb = np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * rgb ** (1 / 2.4) - 0.055)
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    np.testing.assert_allclose(actual[:height, :width], np.round(16 + 219 * luminance), atol=1)
    u = 128 + 224 * (rgb[..., 2] - luminance) / 1.8556
    v = 128 + 224 * (rgb[..., 0] - luminance) / 1.5748
    for offset, expected in enumerate([u, v]):
        expected = expected.reshape(height // 2, 2, width // 2, 2).mean(axis=(1, 3))
        start = height * pitch + offset * height * pitch // 4
        chroma = actual.reshape(-1)[start : start + height * pitch // 4].reshape(
            height // 2, pitch // 2
        )
        converted = chroma[:, : width // 2]
        np.testing.assert_allclose(converted, np.round(expected), atol=1)
    assert np.all(actual[:height, width:] == 235)


def test_openh264_missing_runtime(tmp_path: Path) -> None:
    library = tmp_path / "missing-codec"
    assert streaming.VideoEncoder.check_available(library) is False


def test_openh264_rejects_relative_path() -> None:
    assert streaming.VideoEncoder.check_available("codec.dll") is False


def test_check_available_releases_library(streaming_runtime: None, tmp_path: Path) -> None:
    name, _, _ = openh264._artifact()
    library = tmp_path / name
    shutil.copy2(openh264.runtime_directory() / name, library)
    assert streaming.VideoEncoder.check_available(library_path=library) is True
    # Windows refuses deletion if the query retains the loaded library.
    library.unlink()
    assert streaming.VideoEncoder.check_available(library_path=library) is False


@pytest.mark.parametrize("relative", [False, True])
def test_encoder_library_path_diagnostics(
    streaming_runtime: None, encoder_device: spy.Device, tmp_path: Path, relative: bool
) -> None:
    config = _config()
    config.library_path = "codec.dll" if relative else tmp_path / "missing-codec"
    diagnostic = "must be absolute" if relative else "python -m falcor2.streaming.openh264 install"
    with pytest.raises(RuntimeError, match=diagnostic):
        streaming.VideoEncoder(encoder_device, config)
