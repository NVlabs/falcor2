// SPDX-License-Identifier: Apache-2.0

#include "optix_denoiser.h"

#include "falcor2/core/error.h"
#include "falcor2/core/logger.h"

#include "sgl/device/device.h"
#include "sgl/device/resource.h"
#include "sgl/device/cuda_utils.h"
#include "sgl/device/command.h"

// Instead of using the OptiX SDK, we use a wrapper exposed by slang-rhi.
// This is due to slang-rhi supporting multiple versions of OptiX under the hood.
// The wrapper exposes the standard OptiX denoiser interface, however:
// - OptiX types in the rhi::optix_denoiser instead of the global namespace
// - OptiX functions are exposed through the rhi::optix_denoiser::IOptixDenoiserAPI interface
#include <slang-rhi/optix-denoiser.h>

namespace falcor {


// TODO: SGL has a similar-but-not-quite-right implementation of an interop buffer
// that does the opposite job (marshals incoming CUDA buffers), but ideally we'd
// work out how to share the implementation between the 2.
struct InteropBuffer : public Object {
    ref<sgl::Buffer> buffer;
    ref<sgl::Buffer> shared_buffer;
    ref<sgl::cuda::ExternalMemory> external_memory;

    InteropBuffer(sgl::Device* device, size_t size)
    {
        // Create a shared buffer that can be used for interop.
        sgl::BufferDesc desc;
        desc.size = size;
        desc.usage = sgl::BufferUsage::shared | sgl::BufferUsage::copy_source | sgl::BufferUsage::copy_destination;
        desc.memory_type = sgl::MemoryType::device_local;
        desc.data = nullptr;
        desc.data_size = 0;
        shared_buffer = device->create_buffer(desc);

        // Create external memory from the shared buffer's native handle
        external_memory = make_ref<sgl::cuda::ExternalMemory>(shared_buffer);
    }

    ~InteropBuffer()
    {
        // Make sure external memory is released before the shared buffer
        external_memory = nullptr;
        shared_buffer = nullptr;
        buffer = nullptr;
    }
};

class OptixDenoiser::Impl {
public:
    Impl(ref<sgl::Device> device, const OptixDenoiserDesc& desc);
    ~Impl();

    void denoise(
        const optix::DenoiserParams& params,
        const optix::DenoiserGuideLayer& guide_layer,
        std::span<optix::DenoiserLayer> layers,
        sgl::NativeHandle stream
    );

    ref<InteropBuffer> alloc_interop_buffer();

private:
    void create_denoiser(const OptixDenoiserDesc& desc);
    void cleanup();

    // Device and CUDA context
    ref<sgl::Device> m_device;
    CUdevice m_cuda_device{-1};
    CUcontext m_cuda_context{nullptr};

    // Intermediate interop buffers for CUDA interop
    std::vector<ref<InteropBuffer>> m_interop_free_buffers;
    size_t m_interop_buffer_size{0};

    // Handle to rhi OptiX denoiser API
    Slang::ComPtr<rhi::optix_denoiser::IOptixDenoiserAPI> m_api;

    // Denoiser state
    rhi::optix_denoiser::OptixDeviceContext m_optix_context{nullptr};
    rhi::optix_denoiser::OptixDenoiser m_denoiser{nullptr};
    rhi::optix_denoiser::OptixDenoiserSizes m_denoiser_sizes{};
    bool m_owns_optix_context{false};

    // Temporary buffers for denoising
    CUdeviceptr m_state_buffer{0};
    CUdeviceptr m_scratch_buffer{0};
    size_t m_state_buffer_size{0};
    size_t m_scratch_buffer_size{0};
};


typedef sgl::short_vector<ref<InteropBuffer>, 32> InteropBuffers;

inline rhi::optix_denoiser::OptixDenoiserAlphaMode to_optix(optix::AlphaMode mode)
{
    switch (mode) {
    case optix::AlphaMode::copy:
        return rhi::optix_denoiser::OPTIX_DENOISER_ALPHA_MODE_COPY;
    case optix::AlphaMode::denoise:
        return rhi::optix_denoiser::OPTIX_DENOISER_ALPHA_MODE_DENOISE;
    default:
        FALCOR_THROW("Unsupported OptiX denoiser alpha mode");
    }
}

inline rhi::optix_denoiser::OptixDenoiserModelKind to_optix(optix::ModelKind kind)
{
    switch (kind) {
    case optix::ModelKind::aov:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_AOV;
    case optix::ModelKind::temporal_aov:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV;
    case optix::ModelKind::upscale2x:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_UPSCALE2X;
    case optix::ModelKind::temporal_upscale2x:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_TEMPORAL_UPSCALE2X;
    case optix::ModelKind::ldr:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_LDR;
    case optix::ModelKind::hdr:
        return rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_HDR;
    default:
        FALCOR_THROW("Unsupported OptiX denoiser model kind");
    }
}

inline rhi::optix_denoiser::OptixPixelFormat to_optix(optix::PixelFormat format)
{
    switch (format) {
    case optix::PixelFormat::unknown:
        return rhi::optix_denoiser::OptixPixelFormat(0);
    case optix::PixelFormat::half1:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_HALF1;
    case optix::PixelFormat::half2:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_HALF2;
    case optix::PixelFormat::half3:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_HALF3;
    case optix::PixelFormat::half4:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_HALF4;
    case optix::PixelFormat::float1:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_FLOAT1;
    case optix::PixelFormat::float2:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_FLOAT2;
    case optix::PixelFormat::float3:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_FLOAT3;
    case optix::PixelFormat::float4:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_FLOAT4;
    case optix::PixelFormat::uchar3:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_UCHAR3;
    case optix::PixelFormat::uchar4:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_UCHAR4;
    case optix::PixelFormat::guide_layer:
        return rhi::optix_denoiser::OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER;
    default:
        FALCOR_THROW("Unsupported OptiX pixel format");
    }
}

inline rhi::optix_denoiser::OptixDenoiserAOVType to_optix(optix::DenoiserAOVType type)
{
    switch (type) {
    case optix::DenoiserAOVType::none:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_NONE;
    case optix::DenoiserAOVType::beauty:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_BEAUTY;
    case optix::DenoiserAOVType::specular:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_SPECULAR;
    case optix::DenoiserAOVType::reflection:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_REFLECTION;
    case optix::DenoiserAOVType::refraction:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_REFRACTION;
    case optix::DenoiserAOVType::diffuse:
        return rhi::optix_denoiser::OPTIX_DENOISER_AOV_TYPE_DIFFUSE;
    default:
        FALCOR_THROW("Unsupported OptiX denoiser AOV type");
    }
}

inline uint32_t get_pixel_size_in_bytes(optix::PixelFormat format)
{
    switch (format) {
    case optix::PixelFormat::half1:
        return 2;
    case optix::PixelFormat::half2:
        return 4;
    case optix::PixelFormat::half3:
        return 6;
    case optix::PixelFormat::half4:
        return 8;
    case optix::PixelFormat::float1:
        return 4;
    case optix::PixelFormat::float2:
        return 8;
    case optix::PixelFormat::float3:
        return 12;
    case optix::PixelFormat::float4:
        return 16;
    case optix::PixelFormat::uchar3:
        return 3;
    case optix::PixelFormat::uchar4:
        return 4;
    default:
        FALCOR_THROW("Unsupported OptiX pixel format");
    }
}

inline rhi::optix_denoiser::OptixImage2D
to_optix(const optix::Image2D& image, OptixDenoiser::Impl* denoiser, InteropBuffers* interop_buffers)
{
    rhi::optix_denoiser::OptixImage2D optix_image = {};

    if (image.address) {
        // User provided an explicit device address to use (only works for CUDA devices)
        optix_image.data = image.address;
    } else if (image.buffer) {
        // User provided an SGL buffer.
        if (interop_buffers) {
            // If interop buffer list provided, this is a none-CUDA device so create/add an interop buffer
            // and use its mapped pointer.
            auto interop_buffer = denoiser->alloc_interop_buffer();
            interop_buffer->buffer = image.buffer;
            interop_buffers->push_back(interop_buffer);
            optix_image.data = reinterpret_cast<CUdeviceptr>(interop_buffer->external_memory->mapped_data());
        } else {
            // If no interop buffer list provided, assume CUDA device and use the buffer's device address directly.
            optix_image.data = image.buffer->device_address();
        }
    } else {
        optix_image.data = 0;
    }

    optix_image.width = image.width;
    optix_image.height = image.height;
    optix_image.rowStrideInBytes = image.row_stride_in_bytes;
    optix_image.pixelStrideInBytes = image.pixel_stride_in_bytes;
    optix_image.format = to_optix(image.format);

    if (image.format != optix::PixelFormat::unknown) {
        if (optix_image.pixelStrideInBytes == 0) {
            optix_image.pixelStrideInBytes = get_pixel_size_in_bytes(image.format);
        }
        if (optix_image.rowStrideInBytes == 0) {
            optix_image.rowStrideInBytes = optix_image.pixelStrideInBytes * image.width;
        }
    }

    return optix_image;
}

inline rhi::optix_denoiser::OptixDenoiserOptions to_optix(const OptixDenoiserDesc& desc)
{
    rhi::optix_denoiser::OptixDenoiserOptions options = {};
    options.guideAlbedo = desc.albedo_guide_layer ? 1 : 0;
    options.guideNormal = desc.normal_guide_layer ? 1 : 0;
    options.denoiseAlpha = to_optix(desc.alpha_mode);
    return options;
}

inline rhi::optix_denoiser::OptixDenoiserParams to_optix(const optix::DenoiserParams& params)
{
    rhi::optix_denoiser::OptixDenoiserParams optix_params = {};
    optix_params.blendFactor = params.blend_factor;
    optix_params.temporalModeUsePreviousLayers = params.temporal_mode_use_previous_layers ? 1 : 0;
    // TODO: Handle CUDA pointer info
    optix_params.hdrIntensity = 0;    // params.hdr_intensity;
    optix_params.hdrAverageColor = 0; // params.hdr_average_color;
    return optix_params;
}

inline rhi::optix_denoiser::OptixDenoiserLayer to_optix(
    const optix::DenoiserLayer& layer,
    OptixDenoiser::Impl* denoiser,
    InteropBuffers* interop_input_buffers,
    InteropBuffers* interop_output_buffers
)
{
    rhi::optix_denoiser::OptixDenoiserLayer optix_layer = {};
    optix_layer.input = to_optix(layer.input, denoiser, interop_input_buffers);
    optix_layer.previousOutput = to_optix(layer.previous_output, denoiser, interop_input_buffers);
    optix_layer.output = to_optix(layer.output, denoiser, interop_output_buffers);
    optix_layer.type = to_optix(layer.type);
    return optix_layer;
}

inline rhi::optix_denoiser::OptixDenoiserGuideLayer
to_optix(const optix::DenoiserGuideLayer& layer, OptixDenoiser::Impl* denoiser, InteropBuffers* interop_input_buffers)
{
    rhi::optix_denoiser::OptixDenoiserGuideLayer optix_layer = {};
    optix_layer.albedo = to_optix(layer.albedo, denoiser, interop_input_buffers);
    optix_layer.normal = to_optix(layer.normal, denoiser, interop_input_buffers);
    optix_layer.flow = to_optix(layer.flow, denoiser, interop_input_buffers);
    optix_layer.previousOutputInternalGuideLayer
        = to_optix(layer.previous_output_internal_guide_layer, denoiser, interop_input_buffers);
    optix_layer.outputInternalGuideLayer = to_optix(layer.output_internal_guide_layer, denoiser, interop_input_buffers);
    optix_layer.flowTrustworthiness = to_optix(layer.flow_trustworthiness, denoiser, interop_input_buffers);
    return optix_layer;
}


OptixDenoiser::Impl::Impl(ref<sgl::Device> device, const OptixDenoiserDesc& desc)
    : m_device(std::move(device))
{
    if (m_device->type() != sgl::DeviceType::cuda && !m_device->supports_cuda_interop()) {
        SGL_THROW("OptixDenoiser requires either a CUDA device or a device with CUDA interop enabled");
    }

#if !SGL_MACOS
    // Create the OptiX denoiser API.
    if (SLANG_FAILED(rhiCreateOptixDenoiserAPI(m_device->info().optix_version, m_api.writeRef()))) {
        SGL_THROW("Failed to create the OptiX denoiser API");
    }
#else
    SGL_THROW("OptiX denoiser API is not supported on macOS");
#endif

    // Get CUDA handles
    if (m_device->type() == sgl::DeviceType::cuda) {
        auto native_handles = m_device->native_handles();
        for (const auto& nh : native_handles) {
            switch (nh.type()) {
            case sgl::NativeHandleType::CUdevice: {
                m_cuda_device = static_cast<CUdevice>(nh.value());
                break;
            }
            case sgl::NativeHandleType::CUcontext: {
                m_cuda_context = reinterpret_cast<CUcontext>(nh.value());
                break;
            }
            case sgl::NativeHandleType::OptixDeviceContext: {
                m_optix_context = reinterpret_cast<rhi::optix_denoiser::OptixDeviceContext>(nh.value());
                break;
            }
            default:
                break;
            }
        }
    } else if (m_device->supports_cuda_interop()) {
        // If the device supports CUDA interop, we can get the CUcontext from it
        m_cuda_context = m_device->cuda_device()->context();
        m_cuda_device = m_device->cuda_device()->device();

        // Create OptiX context, because SGL CUDA interop doesn't create one
        SGL_CU_SCOPE(m_device);
        rhi::optix_denoiser::OptixDeviceContextOptions options = {};
        options.logCallbackFunction = nullptr; // Could add logging callback here
        options.logCallbackLevel = 0;
        auto result = m_api->optixDeviceContextCreate(m_cuda_context, &options, &m_optix_context);
        if (result != rhi::optix_denoiser::OPTIX_SUCCESS) {
            SGL_THROW("Failed to create OptiX device context: error code {}", static_cast<int>(result));
        }
        m_owns_optix_context = true;

        // Assume max 4-channel float buffers
        m_interop_buffer_size = desc.max_width * desc.max_height * 4 * sizeof(float);
    }

    if (!m_cuda_context) {
        SGL_THROW("OptixDenoiser requires a CUDA context");
    }
    if (m_cuda_device < 0) {
        SGL_THROW("OptixDenoiser requires a CUDA device");
    }
    if (!m_optix_context) {
        SGL_THROW("OptixDenoiser requires an Optix context");
    }

    create_denoiser(desc);
}

OptixDenoiser::Impl::~Impl()
{
    cleanup();
}

ref<InteropBuffer> OptixDenoiser::Impl::alloc_interop_buffer()
{
    // Reuse existing interop buffers if possible
    if (!m_interop_free_buffers.empty()) {
        ref<InteropBuffer> buffer = m_interop_free_buffers.back();
        m_interop_free_buffers.pop_back();
        return buffer;
    }

    // Create a new interop buffer
    return make_ref<InteropBuffer>(m_device, m_interop_buffer_size);
}

void OptixDenoiser::Impl::denoise(
    const optix::DenoiserParams& params,
    const optix::DenoiserGuideLayer& guide_layer,
    std::span<optix::DenoiserLayer> layers,
    sgl::NativeHandle cuda_stream
)
{
    SGL_CU_SCOPE(m_device);

    CUstream cuda_stream_ptr = nullptr;
    if (cuda_stream.is_valid()) {
        cuda_stream_ptr = reinterpret_cast<CUstream>(cuda_stream.value());
    }

    InteropBuffers interop_inputs;
    InteropBuffers interop_outputs;
    InteropBuffers* p_interop_input_buffers = nullptr;
    InteropBuffers* p_interop_output_buffers = nullptr;
    if (m_device->type() != sgl::DeviceType::cuda) {
        p_interop_input_buffers = &interop_inputs;
        p_interop_output_buffers = &interop_outputs;
    }

    rhi::optix_denoiser::OptixDenoiserParams optix_params = to_optix(params);
    rhi::optix_denoiser::OptixDenoiserGuideLayer optix_guide_layer
        = to_optix(guide_layer, this, p_interop_input_buffers);
    sgl::short_vector<rhi::optix_denoiser::OptixDenoiserLayer, 32> optix_layers;
    for (const auto& layer : layers) {
        optix_layers.push_back(to_optix(layer, this, p_interop_input_buffers, p_interop_output_buffers));
    }

    // Perform any copies for interop inputs
    if (interop_inputs.size() > 0) {
        ref<sgl::CommandEncoder> encoder = m_device->create_command_encoder();
        for (const auto& ib : interop_inputs) {
            if (ib->shared_buffer != ib->buffer) {
                encoder->copy_buffer(ib->shared_buffer, 0, ib->buffer, 0, ib->buffer->size());
            }
        }
        m_device->submit_command_buffer(encoder->finish(), sgl::CommandQueueType::graphics, cuda_stream);
    }

    // For none-CUDA devices, ensure CUDA waits for device work to finish before denoising.
    // This includes any interop copies, but also any other device work that may have been
    // populating already shared buffers.
    m_device->sync_to_device(cuda_stream_ptr);

    // Invoke denoiser
    auto result = m_api->optixDenoiserInvoke(
        m_denoiser,
        cuda_stream_ptr,
        &optix_params,
        m_state_buffer,
        m_state_buffer_size,
        &optix_guide_layer,
        &optix_layers[0],
        (uint32_t)optix_layers.size(), // num layers
        0,                             // input offset X
        0,                             // input offset Y
        m_scratch_buffer,
        m_scratch_buffer_size
    );
    if (result != rhi::optix_denoiser::OPTIX_SUCCESS) {
        SGL_THROW("OptixDenoiser: Denoising failed with error code {}", static_cast<int>(result));
    }

    // For none-CUDA devices, ensure device waits for denoising to finish before copying back
    // interop outputs or returning control to the user.
    m_device->sync_to_cuda(cuda_stream_ptr);

    // Perform any copies for interop outputs
    if (interop_outputs.size() > 0) {
        ref<sgl::CommandEncoder> encoder = m_device->create_command_encoder();
        for (const auto& ib : interop_outputs) {
            if (ib->shared_buffer != ib->buffer) {
                encoder->copy_buffer(ib->buffer, 0, ib->shared_buffer, 0, ib->buffer->size());
            }
        }
        m_device->submit_command_buffer(encoder->finish(), sgl::CommandQueueType::graphics, cuda_stream);
    }

    // Release any interop buffers back to the pool
    for (const auto& ib : interop_inputs) {
        ib->buffer = nullptr;
        m_interop_free_buffers.push_back(ib);
    }
    for (const auto& ib : interop_outputs) {
        ib->buffer = nullptr;
        m_interop_free_buffers.push_back(ib);
    }
}

void OptixDenoiser::Impl::create_denoiser(const OptixDenoiserDesc& desc)
{
    SGL_CU_SCOPE(m_device);

    // Create denoiser
    rhi::optix_denoiser::OptixDenoiserOptions options = to_optix(desc);
    rhi::optix_denoiser::OptixResult result = m_api->optixDenoiserCreate(
        m_optix_context,
        rhi::optix_denoiser::OPTIX_DENOISER_MODEL_KIND_LDR, // Could be configurable
        &options,
        &m_denoiser
    );
    if (result != rhi::optix_denoiser::OPTIX_SUCCESS) {
        SGL_THROW(
            "OptixDenoiser: Failed to create denoiser: {} ({})",
            m_api->optixGetErrorString(result),
            m_api->optixGetErrorName(result)
        );
    }

    // Compute memory requirements
    result = m_api->optixDenoiserComputeMemoryResources(m_denoiser, desc.max_width, desc.max_height, &m_denoiser_sizes);
    if (result != rhi::optix_denoiser::OPTIX_SUCCESS) {
        m_api->optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
        SGL_THROW(
            "OptixDenoiser: Failed to compute memory resources: {} ({})",
            m_api->optixGetErrorString(result),
            m_api->optixGetErrorName(result)
        );
    }

    // Allocate state buffer
    m_state_buffer_size = m_denoiser_sizes.stateSizeInBytes;
    CUresult cu_result = cuMemAlloc(&m_state_buffer, m_state_buffer_size);
    if (cu_result != CUDA_SUCCESS) {
        m_api->optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
        SGL_THROW("OptixDenoiser: Failed to allocate state buffer");
    }

    // Allocate scratch buffer
    m_scratch_buffer_size = m_denoiser_sizes.withoutOverlapScratchSizeInBytes;
    cu_result = cuMemAlloc(&m_scratch_buffer, m_scratch_buffer_size);
    if (cu_result != CUDA_SUCCESS) {
        cuMemFree(m_state_buffer);
        m_state_buffer = 0;
        m_api->optixDenoiserDestroy(m_denoiser);
        m_denoiser = nullptr;
        SGL_THROW("OptixDenoiser: Failed to allocate scratch buffer");
    }

    // To avoid issues with, e.g., pytorch using multiple threads/streams,
    // just sync the context before attempting to run denoiser setup
    cuCtxSynchronize();

    // Setup denoiser
    result = m_api->optixDenoiserSetup(
        m_denoiser,
        nullptr,
        desc.max_width,
        desc.max_height,
        m_state_buffer,
        m_state_buffer_size,
        m_scratch_buffer,
        m_scratch_buffer_size
    );
    if (result != rhi::optix_denoiser::OPTIX_SUCCESS) {
        cleanup();
        SGL_THROW(
            "OptixDenoiser: Failed to setup denoiser: {} ({})",
            m_api->optixGetErrorString(result),
            m_api->optixGetErrorName(result)
        );
    }
}

void OptixDenoiser::Impl::cleanup()
{
    if (m_device && !m_device->is_closed()) {
        SGL_CU_SCOPE(m_device);

        cuCtxSynchronize();

        if (m_denoiser) {
            m_api->optixDenoiserDestroy(m_denoiser);
            m_denoiser = nullptr;
        }

        if (m_scratch_buffer) {
            cuMemFree(m_scratch_buffer);
            m_scratch_buffer = 0;
        }

        if (m_state_buffer) {
            cuMemFree(m_state_buffer);
            m_state_buffer = 0;
        }

        if (m_owns_optix_context && m_optix_context) {
            m_api->optixDeviceContextDestroy(m_optix_context);
            m_optix_context = nullptr;
        }

        m_interop_free_buffers.clear();
    }
}

OptixDenoiser::OptixDenoiser(ref<sgl::Device> device, const OptixDenoiserDesc& desc)
    : m_device(std::move(device))
{
    FALCOR_ASSERT(m_device);
    m_impl = std::make_unique<OptixDenoiser::Impl>(m_device, desc);
}

OptixDenoiser::~OptixDenoiser() = default;

void OptixDenoiser::denoise(
    const optix::DenoiserParams& params,
    const optix::DenoiserGuideLayer& guide_layer,
    std::span<optix::DenoiserLayer> layers,
    sgl::NativeHandle cuda_stream
)
{
    FALCOR_ASSERT(m_impl);
    m_impl->denoise(params, guide_layer, layers, cuda_stream);
}

} // namespace falcor
