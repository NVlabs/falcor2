// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "video_encoder.h"
#include "video_format.h"
#include "falcor2/core/config.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>
#include <chrono>

#if FALCOR_ENABLE_STREAMING
#include "frame_converter.h"
#include <sgl/core/platform.h>
#include <sgl/device/command.h>
#include <sgl/device/query.h>
#include "falcor2/core/platform.h"
#include <wels/codec_api.h>
#include <wels/codec_ver.h>
#include <algorithm>
static_assert(OPENH264_MAJOR == 2 && OPENH264_MINOR == 6 && OPENH264_REVISION == 0);
#endif

namespace falcor::streaming {
namespace {
void validate_bitrate(uint32_t value, uint32_t maximum)
{
    FALCOR_CHECK(
        value >= 100000 && value <= maximum,
        "Bitrate must be between 100000 and the configured maximum of {} bits/s",
        maximum
    );
}
void validate_config(sgl::Device* device, const VideoEncoderConfig& config)
{
    FALCOR_CHECK_NOT_NULL(device);
    detail::h264_level(config.width, config.height, config.fps, config.max_bitrate);
    FALCOR_CHECK(config.max_frame_slots >= 1 && config.max_frame_slots <= 8, "Frame pool must contain 1 to 8 slots");
    validate_bitrate(config.target_bitrate, config.max_bitrate);
    FALCOR_CHECK(config.thread_count >= 1 && config.thread_count <= 16, "Thread count must be between 1 and 16");
    FALCOR_CHECK(
        config.complexity == VideoEncoderComplexity::low || config.complexity == VideoEncoderComplexity::medium
            || config.complexity == VideoEncoderComplexity::high,
        "Invalid encoder complexity"
    );
}
} // namespace

#if FALCOR_ENABLE_STREAMING
namespace {
class OpenH264Library {
public:
    OpenH264Library() = default;
    ~OpenH264Library() { close(); }

    void load(const std::filesystem::path& library_path);
    void close()
    {
        if (m_handle) {
            sgl::platform::release_shared_library(m_handle);
            m_handle = nullptr;
        }
    }

    decltype(&WelsCreateSVCEncoder) create{nullptr};
    decltype(&WelsDestroySVCEncoder) destroy{nullptr};

private:
    sgl::SharedLibraryHandle m_handle{nullptr};
    FALCOR_NON_COPYABLE(OpenH264Library);
};
} // namespace

struct VideoEncoder::Impl {
    enum class SlotState { free, conversion_pending, encode_pending, output_pending };
    struct Slot {
        SlotState state{SlotState::free};
        ref<sgl::Buffer> buffer;
        ref<sgl::Texture> source;
        ref<sgl::Buffer> readback;
        ref<sgl::QueryPool> timestamps;
        uint64_t submit_id{0};
        std::chrono::steady_clock::time_point submitted;
        void* host{nullptr};
        EncodedFrame packet;
    };

    ref<sgl::Device> device;
    VideoEncoderConfig config;
    uint32_t pitch;
    std::unique_ptr<FrameConverter> converter;
    OpenH264Library library;
    ISVCEncoder* encoder{nullptr};
    uint32_t active_bitrate{0};
    std::vector<Slot> slots;
    std::mutex mutex;
    std::condition_variable wake;
    std::deque<size_t> gpu_pending; // Rendering owner only.
    std::deque<size_t> pending;
    std::deque<size_t> output;
    std::thread worker;
    bool stopping{false};
    bool closed{false};
    bool keyframe_requested{true};
    uint32_t requested_bitrate;
    bool have_frame{false};
    uint64_t last_id{0};
    uint64_t last_pts{0};
    std::exception_ptr failure;

    Impl(ref<sgl::Device> value, VideoEncoderConfig settings)
        : device(std::move(value))
        , config(settings)
        , pitch((settings.width + 255) & ~255U)
        , requested_bitrate(settings.target_bitrate)
    {
    }

    ~Impl() { shutdown(); }

    // Called only by the rendering owner; the worker never calls SGL or releases
    // references that might belong to Python.
    void advance_ready()
    {
        while (!gpu_pending.empty()) {
            const auto index = gpu_pending.front();
            auto& slot = slots[index];
            // CUDA's RHI currently signals submission fences on the host, before GPU
            // completion. Keep the query fallback there until its fences are fixed.
            const bool finished = device->type() == sgl::DeviceType::cuda
                ? slot.timestamps->get_result_state(0, 2) == sgl::QueryResultState::resolved
                : device->is_submit_finished(slot.submit_id);
            if (!finished)
                break;
            const auto times = slot.timestamps->get_timestamp_results(0, 2);
            slot.packet.gpu_prepare_ms = float((times[1] - times[0]) * 1000.0);
            slot.host = slot.readback->map();
            gpu_pending.pop_front();
            {
                std::lock_guard lock(mutex);
                slot.state = SlotState::encode_pending;
                pending.push_back(index);
            }
            wake.notify_one();
        }
    }

    void shutdown() noexcept
    {
        if (closed)
            return;
        // Even a failed submission may have launched work. Drain before mapped
        // memory or source references are released, including during construction.
        std::exception_ptr cleanup_failure;
        try {
            device->wait_for_idle();
            advance_ready();
        } catch (...) {
            cleanup_failure = std::current_exception();
        }
        stop();
        for (auto& slot : slots) {
            try {
                if (slot.readback && slot.readback->is_mapped())
                    slot.readback->unmap();
            } catch (...) {
                if (!cleanup_failure)
                    cleanup_failure = std::current_exception();
            }
        }
        if (encoder) {
            encoder->Uninitialize();
            library.destroy(encoder);
            encoder = nullptr;
        }
        library.close();
        if (!failure)
            failure = cleanup_failure;
        closed = true;
    }

    void check(int status, const char* operation)
    {
        FALCOR_CHECK(status == 0, "{} failed (OpenH264 status {})", operation, status);
    }
    void check_failure() const
    {
        if (failure)
            std::rethrow_exception(failure);
    }

    void initialize();

    void stop()
    {
        {
            std::lock_guard lock(mutex);
            stopping = true;
        }
        wake.notify_one();
        if (worker.joinable())
            worker.join();
    }

    void run() noexcept
    {
        try {
            while (true) {
                size_t index;
                bool force_idr;
                uint32_t bitrate;
                {
                    std::unique_lock lock(mutex);
                    wake.wait(
                        lock,
                        [&]
                        {
                            return stopping || !pending.empty();
                        }
                    );
                    if (pending.empty())
                        break;
                    index = pending.front();
                    pending.pop_front();
                    force_idr = keyframe_requested;
                    keyframe_requested = false;
                    bitrate = requested_bitrate;
                }
                auto& slot = slots[index];
                const auto encode_start = std::chrono::steady_clock::now();
                if (bitrate != active_bitrate) {
                    SBitrateInfo change{SPATIAL_LAYER_ALL, int(bitrate)};
                    check(encoder->SetOption(ENCODER_OPTION_BITRATE, &change), "Set bitrate");
                    active_bitrate = bitrate;
                    force_idr = true;
                }
                if (force_idr)
                    check(encoder->ForceIntraFrame(true), "Force IDR");
                SSourcePicture picture{};
                picture.iColorFormat = videoFormatI420;
                picture.iPicWidth = config.width;
                picture.iPicHeight = config.height;
                picture.iStride[0] = pitch;
                picture.iStride[1] = picture.iStride[2] = pitch / 2;
                picture.pData[0] = static_cast<unsigned char*>(slot.host);
                picture.pData[1] = picture.pData[0] + pitch * config.height;
                picture.pData[2] = picture.pData[1] + pitch * config.height / 4;
                picture.uiTimeStamp = slot.packet.pts_us / 1000;
                SFrameBSInfo bitstream{};
                check(encoder->EncodeFrame(&picture, &bitstream), "EncodeFrame");
                FALCOR_CHECK(bitstream.eFrameType != videoFrameTypeSkip, "OpenH264 unexpectedly skipped a frame");
                for (int layer = 0; layer < bitstream.iLayerNum; ++layer) {
                    const auto& data = bitstream.sLayerInfo[layer];
                    size_t size = 0;
                    for (int nal = 0; nal < data.iNalCount; ++nal)
                        size += data.pNalLengthInByte[nal];
                    slot.packet.data.insert(slot.packet.data.end(), data.pBsBuf, data.pBsBuf + size);
                }
                FALCOR_CHECK(!slot.packet.data.empty(), "OpenH264 returned empty output");
                slot.packet.keyframe = bitstream.eFrameType == videoFrameTypeIDR;
                {
                    std::lock_guard lock(mutex);
                    const auto completed = std::chrono::steady_clock::now();
                    slot.packet.encode_wall_ms
                        = std::chrono::duration<double, std::milli>(completed - encode_start).count();
                    slot.packet.submit_to_output_ms
                        = std::chrono::duration<double, std::milli>(completed - slot.submitted).count();
                    slot.state = SlotState::output_pending;
                    output.push_back(index);
                }
            }
        } catch (...) {
            std::lock_guard lock(mutex);
            failure = std::current_exception();
            stopping = true;
        }
    }
};

namespace {
// Loading is separated from construction so failures also release the library.
void OpenH264Library::load(const std::filesystem::path& requested_path)
{
#if SGL_WINDOWS
    const char* managed_library = "openh264-2.6.0-win64.dll";
#elif defined(__aarch64__)
    const char* managed_library = "libopenh264-2.6.0-linux-arm64.8.so";
#else
    const char* managed_library = "libopenh264-2.6.0-linux64.8.so";
#endif
    const auto directory = falcor::platform::app_data_directory() / "codecs" / "openh264" / "2.6.0";
    const auto library_path = requested_path.empty() ? directory / managed_library : requested_path;
    FALCOR_CHECK(library_path.is_absolute(), "OpenH264 library_path must be absolute");
    m_handle = sgl::platform::load_shared_library(library_path);
    FALCOR_CHECK(
        m_handle,
        "Cannot load OpenH264 2.6.0 library {}; run python -m falcor2.streaming.openh264 install "
        "or set VideoEncoderConfig.library_path to an existing library",
        library_path
    );
    create = reinterpret_cast<decltype(&WelsCreateSVCEncoder)>(
        sgl::platform::get_proc_address(m_handle, "WelsCreateSVCEncoder")
    );
    destroy = reinterpret_cast<decltype(&WelsDestroySVCEncoder)>(
        sgl::platform::get_proc_address(m_handle, "WelsDestroySVCEncoder")
    );
    auto version = reinterpret_cast<decltype(&WelsGetCodecVersionEx)>(
        sgl::platform::get_proc_address(m_handle, "WelsGetCodecVersionEx")
    );
    FALCOR_CHECK(create && destroy && version, "OpenH264 entry points missing");
    OpenH264Version actual{};
    version(&actual);
    FALCOR_CHECK(
        actual.uMajor == 2 && actual.uMinor == 6 && actual.uRevision == 0,
        "OpenH264 runtime must be version 2.6.0"
    );
}
} // namespace

// Session initialization is separated so Impl's destructor also cleans up any
// resources created before an initialization failure.
void VideoEncoder::Impl::initialize()
{
    library.load(config.library_path);
    check(library.create(&encoder), "Create encoder");
    SEncParamExt params{};
    check(encoder->GetDefaultParams(&params), "Get defaults");
    params.iUsageType = CAMERA_VIDEO_REAL_TIME;
    params.iComplexityMode = ECOMPLEXITY_MODE(config.complexity);
    params.iPicWidth = config.width;
    params.iPicHeight = config.height;
    params.fMaxFrameRate = float(config.fps);
    params.iTargetBitrate = config.target_bitrate;
    params.iRCMode = RC_BITRATE_MODE;
    params.bEnableFrameSkip = false;
    params.iTemporalLayerNum = 1;
    params.iSpatialLayerNum = 1;
    params.uiIntraPeriod = config.fps * 2;
    params.iMultipleThreadIdc = uint16_t(config.thread_count);
    params.iEntropyCodingModeFlag = 0;
    auto& layer = params.sSpatialLayers[0];
    layer.iVideoWidth = config.width;
    layer.iVideoHeight = config.height;
    layer.fFrameRate = float(config.fps);
    layer.iSpatialBitrate = config.target_bitrate;
    layer.uiProfileIdc = PRO_BASELINE;
    layer.uiLevelIdc = ELevelIdc(detail::h264_level(config.width, config.height, config.fps, config.max_bitrate));
    // Fixed-count slices can exhaust OpenH264's per-slice output buffer on
    // detailed frames. Split by encoded size instead, independently of threads.
    params.uiMaxNalSize = 64 * 1024;
    layer.sSliceArgument.uiSliceMode = SM_SIZELIMITED_SLICE;
    layer.sSliceArgument.uiSliceSizeConstraint = params.uiMaxNalSize;
    layer.bVideoSignalTypePresent = true;
    layer.uiVideoFormat = 5;
    layer.bFullRange = false;
    layer.bColorDescriptionPresent = true;
    layer.uiColorPrimaries = 1;
    layer.uiTransferCharacteristics = 13;
    layer.uiColorMatrix = 1;
    check(encoder->InitializeExt(&params), "Initialize encoder");
    active_bitrate = config.target_bitrate;
    converter = std::make_unique<FrameConverter>(device);
    slots.resize(config.max_frame_slots);
    for (auto& slot : slots) {
        const size_t size = size_t(pitch) * config.height * 3 / 2;
        slot.buffer = device->create_buffer(
            {.size = size,
             .usage
             = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access | sgl::BufferUsage::copy_source,
             .label = "Streaming I420"}
        );
        slot.readback = device->create_buffer({
            .size = size,
            .memory_type = sgl::MemoryType::read_back,
            .usage = sgl::BufferUsage::copy_destination,
            .label = "Streaming I420 readback",
        });
        slot.timestamps = device->create_query_pool({.type = sgl::QueryType::timestamp, .count = 2});
    }
    worker = std::thread(
        [this]
        {
            run();
        }
    );
}

VideoEncoder::VideoEncoder(ref<sgl::Device> device, VideoEncoderConfig config)
{
    validate_config(device, config);
    m_impl = std::make_unique<Impl>(std::move(device), config);
    m_impl->initialize();
}

bool VideoEncoder::try_submit(sgl::Texture* texture, uint64_t frame_id, uint64_t pts_us, InputTransfer transfer)
{
    FALCOR_CHECK(m_impl, "Encoder is closed");
    auto& state = *m_impl;
    state.advance_ready();
    FALCOR_CHECK_NOT_NULL(texture);
    FALCOR_CHECK(texture->device() == state.device.get(), "Source must belong to the encoder's device");
    FALCOR_CHECK(
        texture->type() == sgl::TextureType::texture_2d && texture->array_length() == 1 && texture->mip_count() == 1
            && texture->width() == state.config.width && texture->height() == state.config.height,
        "Source must be a single-mip 2D texture matching the stream dimensions"
    );
    FALCOR_CHECK(
        texture->format() == sgl::Format::rgba32_float || texture->format() == sgl::Format::rgba16_float
            || texture->format() == sgl::Format::rgba8_unorm,
        "Unsupported source format; require rgba32_float, rgba16_float or rgba8_unorm"
    );
    FALCOR_CHECK(transfer == InputTransfer::linear || transfer == InputTransfer::srgb, "Invalid input transfer");
    size_t index = state.slots.size();
    {
        std::lock_guard lock(state.mutex);
        state.check_failure();
        FALCOR_CHECK(!state.stopping, "Encoder is closing");
        FALCOR_CHECK(
            !state.have_frame || (frame_id > state.last_id && pts_us > state.last_pts),
            "Accepted frame IDs and timestamps must increase strictly"
        );
        for (size_t i = 0; i < state.slots.size(); ++i) {
            if (state.slots[i].state == Impl::SlotState::free) {
                index = i;
                break;
            }
        }
        if (index == state.slots.size())
            return false;
        state.slots[index].state = Impl::SlotState::conversion_pending;
    }
    auto& slot = state.slots[index];
    slot.source = ref<sgl::Texture>(texture);
    slot.packet = {{}, frame_id, pts_us, 1000000 / state.config.fps, false};
    slot.submitted = std::chrono::steady_clock::now();
    try {
        slot.timestamps->reset();
        auto commands = state.device->create_command_encoder();
        commands->write_timestamp(slot.timestamps, 0);
        state.converter->convert(commands, texture, slot.buffer, state.pitch, transfer);
        commands->copy_buffer(slot.readback, 0, slot.buffer, 0, slot.buffer->size());
        commands->write_timestamp(slot.timestamps, 1);
        slot.submit_id = state.device->submit_command_buffer(commands->finish());
    } catch (...) {
        std::lock_guard lock(state.mutex);
        state.failure = std::current_exception();
        state.stopping = true;
        state.wake.notify_one();
        throw;
    }
    {
        std::lock_guard lock(state.mutex);
        state.gpu_pending.push_back(index);
        state.have_frame = true;
        state.last_id = frame_id;
        state.last_pts = pts_us;
    }
    state.wake.notify_one();
    return true;
}

std::optional<EncodedFrame> VideoEncoder::poll_output()
{
    FALCOR_CHECK(m_impl, "Encoder is closed");
    m_impl->advance_ready();
    std::lock_guard lock(m_impl->mutex);
    m_impl->check_failure();
    if (m_impl->output.empty())
        return {};
    size_t index = m_impl->output.front();
    m_impl->output.pop_front();
    auto& slot = m_impl->slots[index];
    auto packet = std::move(slot.packet);
    slot.readback->unmap();
    slot.host = nullptr;
    slot.source.reset();
    slot.state = Impl::SlotState::free;
    return packet;
}

void VideoEncoder::request_keyframe()
{
    FALCOR_CHECK(m_impl, "Encoder is closed");
    std::lock_guard lock(m_impl->mutex);
    m_impl->check_failure();
    m_impl->keyframe_requested = true;
}

void VideoEncoder::set_target_bitrate(uint32_t bits_per_second)
{
    FALCOR_CHECK(m_impl, "Encoder is closed");
    validate_bitrate(bits_per_second, m_impl->config.max_bitrate);
    std::lock_guard lock(m_impl->mutex);
    m_impl->check_failure();
    m_impl->requested_bitrate = bits_per_second;
}

void VideoEncoder::close()
{
    if (!m_impl)
        return;
    m_impl->shutdown();
    auto failure = m_impl->failure;
    m_impl.reset();
    if (failure)
        std::rethrow_exception(failure);
}
#else
struct VideoEncoder::Impl { };
VideoEncoder::VideoEncoder(ref<sgl::Device> device, VideoEncoderConfig config)
{
    validate_config(device, config);
    FALCOR_THROW("OpenH264 not built; configure FALCOR_ENABLE_STREAMING=ON");
}
bool VideoEncoder::try_submit(sgl::Texture*, uint64_t, uint64_t, InputTransfer)
{
    FALCOR_THROW("OpenH264 not built");
}
std::optional<EncodedFrame> VideoEncoder::poll_output()
{
    FALCOR_THROW("OpenH264 not built");
}
void VideoEncoder::request_keyframe()
{
    FALCOR_THROW("OpenH264 not built");
}
void VideoEncoder::set_target_bitrate(uint32_t)
{
    FALCOR_THROW("OpenH264 not built");
}
void VideoEncoder::close() { }
#endif

VideoEncoder::~VideoEncoder() = default;

bool VideoEncoder::check_available(const std::filesystem::path& library_path)
{
#if FALCOR_ENABLE_STREAMING
    try {
        OpenH264Library library;
        library.load(library_path);
        return true;
    } catch (const std::exception&) {
        return false;
    }
#else
    (void)library_path;
    return false;
#endif
}

} // namespace falcor::streaming
