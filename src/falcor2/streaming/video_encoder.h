// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/core/enum.h"

#include <sgl/device/fwd.h>

#include <memory>
#include <optional>
#include <filesystem>
#include <vector>

namespace falcor::streaming {

/// Transfer function of the stored RGB values, independent of the texture format.
enum class InputTransfer {
    linear,
    srgb,
};
SGL_ENUM_INFO(
    InputTransfer,
    {
        {InputTransfer::linear, "linear"},
        {InputTransfer::srgb, "srgb"},
    }
);
SGL_ENUM_REGISTER(InputTransfer);

/// Software encoding effort; lower effort favors speed over compression efficiency.
enum class VideoEncoderComplexity {
    low,
    medium,
    high,
};
SGL_ENUM_INFO(
    VideoEncoderComplexity,
    {
        {VideoEncoderComplexity::low, "low"},
        {VideoEncoderComplexity::medium, "medium"},
        {VideoEncoderComplexity::high, "high"},
    }
);
SGL_ENUM_REGISTER(VideoEncoderComplexity);

/// Fixed, even dimensions up to 3840x2160, at 1..60 FPS. H.264 baseline level
/// 3.1..5.2 is selected from the frame size, macroblock rate and bitrate ceiling.
struct VideoEncoderConfig {
    /// Frame width in pixels; even and between 16 and 3840.
    uint32_t width{1280};
    /// Frame height in pixels; even and between 16 and 2160.
    uint32_t height{720};
    /// Frames per second (1..60), used for rate control and output duration.
    uint32_t fps{30};
    /// Initial target bitrate in bits per second (100000..max_bitrate).
    uint32_t target_bitrate{8000000};
    /// Session bitrate ceiling in bits/s (100000..50000000), used to select a fixed
    /// H.264 level. Must match WebRTCConfig::max_bitrate when using that transport.
    uint32_t max_bitrate{14000000};
    /// Maximum outstanding frames (1..8), including encoded output not yet polled.
    uint32_t max_frame_slots{3};
    VideoEncoderComplexity complexity{VideoEncoderComplexity::high};
    /// OpenH264 worker threads (1..16). Slices are size-limited; frames remain ordered.
    uint32_t thread_count{4};
    /// Optional absolute OpenH264 2.6.0 library path; otherwise use the managed
    /// installation from python -m falcor2.streaming.openh264 install.
    std::filesystem::path library_path;
};

/// Owned encoded packet and capture metadata returned by poll_output().
struct EncodedFrame {
    /// Annex-B H.264 data, including parameter sets on keyframes.
    std::vector<uint8_t> data;
    /// Caller-provided capture identifier, preserved from try_submit().
    uint64_t frame_id{0};
    /// Caller-provided presentation timestamp in microseconds.
    uint64_t pts_us{0};
    /// Nominal duration in microseconds, computed as 1000000 / fps.
    uint64_t duration_us{0};
    /// True for an independently decodable IDR frame.
    bool keyframe{false};
    /// Size of the compressed packet in bytes.
    size_t byte_size() const { return data.size(); }
    /// GPU timestamp interval covering conversion and readback, in milliseconds.
    float gpu_prepare_ms{0};
    /// CPU encoding wall time, excluding GPU preparation, in milliseconds.
    double encode_wall_ms{0};
    /// Submission through encoded output completion, including readiness polling, in milliseconds.
    double submit_to_output_ms{0};
};

/// OpenH264 encoder with bounded GPU conversion/readback and a native CPU worker.
/// Rendering must already be submitted on the device's default graphics queue.
/// Serialize all calls and destruction on the rendering owner thread. Poll output
/// regularly to advance GPU-ready frames to the worker. Close the encoder before
/// closing its device. CUDA, D3D12 and Vulkan use the same readback path.
class FALCOR_API VideoEncoder {
public:
    /// Validates settings, loads the codec, allocates the frame pool and starts the worker.
    /// Throws if the runtime, settings or device resources are unavailable or invalid.
    VideoEncoder(ref<sgl::Device> device, VideoEncoderConfig config);
    /// Waits for pending work and releases resources without propagating errors.
    ~VideoEncoder();

    /// True if streaming support is built and the codec library loads with the required
    /// entry points and version. Uses the same path selection as VideoEncoderConfig.
    /// Creates no device or encoder session. Encoder construction separately validates
    /// device resources and session settings, and reports detailed errors on failure.
    static bool check_available(const std::filesystem::path& library_path = {});

    /// False means the bounded pool is full. Never waits for encoded output.
    /// Accepts display-referred SDR rgba32_float/rgba16_float/rgba8_unorm only.
    /// The explicit tag describes the stored values, not the texture format.
    /// The texture must match the configured dimensions and is retained until output
    /// is polled or the encoder closes. Accepted IDs and timestamps must strictly
    /// increase; rejected submissions do not advance them.
    bool try_submit(sgl::Texture* texture, uint64_t frame_id, uint64_t pts_us, InputTransfer transfer);

    /// Advances completed readbacks without waiting, then returns the next encoded frame if ready.
    std::optional<EncodedFrame> poll_output();

    /// Requests an IDR for the next frame picked up by the worker; requests coalesce.
    void request_keyframe();
    /// Requests a bitrate in bits per second (100000..configured max_bitrate).
    /// The session level remains fixed; exceeding the ceiling throws.
    /// The worker applies
    /// the latest request before its next frame and emits an IDR if the rate changes.
    void set_target_bitrate(uint32_t bits_per_second);

    /// Stops acceptance, finishes submitted GPU work, joins the worker and frees
    /// all native resources. Discards unpolled packets. Idempotent; errors surface
    /// here or on the next API call, never from the destructor.
    void close();

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    FALCOR_NON_COPYABLE(VideoEncoder);
};

} // namespace falcor::streaming
