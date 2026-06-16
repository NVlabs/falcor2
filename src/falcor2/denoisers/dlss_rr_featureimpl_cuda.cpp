// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_rr_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <nvsdk_ngx_helpers_dlssd_cuda.h>

#include <string>
#include <utility>

namespace falcor::ngx {
namespace {

uint64_t get_cuda_texture_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_ro();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::texture && handle.value,
        "CUDA DLSS-RR evaluation {} must expose a read CUtexObject.",
        name
    );
    return handle.value;
}

uint64_t get_cuda_surface_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_rw();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::rw_texture && handle.value,
        "CUDA DLSS-RR evaluation {} must expose a write CUsurfObject.",
        name
    );
    return handle.value;
}

class CudaDLSSRRFeatureImpl final : public DLSSRRFeature::Impl {
public:
    /// Create the CUDA NGX DLSS-RR feature on the active CUDA command stream.
    CudaDLSSRRFeatureImpl(ref<NGX> ngx, DLSSRRFeatureDesc desc)
        : DLSSRRFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the CUDA NGX feature and parameter map.
    ~CudaDLSSRRFeatureImpl() override { release_resources(); }

    /// Record a CUDA NGX DLSS-RR evaluation callback.
    void record_evaluate(
        ref<DLSSRRFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSRREvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSRREvaluateResources resources = retain_evaluate_resources(desc);
        prepare_evaluate_resources(command_encoder, resources);

        reset_last_evaluate_result();
        command_encoder->execute_callback(
            [this, resources = std::move(resources), owner = std::move(owner)](sgl::NativeHandle native_handle)
            {
                FALCOR_UNUSED(owner);
                if (native_handle.type() != sgl::NativeHandleType::CUstream) {
                    set_last_evaluate_result(NVSDK_NGX_Result_FAIL_InvalidParameter);
                    return;
                }

                m_cuda_device->cudaStream = reinterpret_cast<void*>(native_handle.value());

                uint64_t diffuse_albedo = get_cuda_texture_object(resources.diffuse_albedo, "diffuse_albedo");
                uint64_t specular_albedo = get_cuda_texture_object(resources.specular_albedo, "specular_albedo");
                uint64_t normals = get_cuda_texture_object(resources.normals, "normals");
                uint64_t roughness = get_cuda_texture_object(resources.roughness, "roughness");
                uint64_t color = get_cuda_texture_object(resources.color, "color");
                uint64_t depth = get_cuda_texture_object(resources.depth, "depth");
                uint64_t motion_vectors = get_cuda_texture_object(resources.motion_vectors, "motion_vectors");
                uint64_t output = get_cuda_surface_object(resources.output, "output");
                uint64_t motion_vectors_reflections = resources.motion_vectors_reflections
                    ? get_cuda_texture_object(resources.motion_vectors_reflections, "motion_vectors_reflections")
                    : 0;
                uint64_t specular_hit_distance = resources.specular_hit_distance
                    ? get_cuda_texture_object(resources.specular_hit_distance, "specular_hit_distance")
                    : 0;

                NVSDK_NGX_CUDA_DLSSD_Eval_Params eval_params{};
                eval_params.pInDiffuseAlbedo = &diffuse_albedo;
                eval_params.pInSpecularAlbedo = &specular_albedo;
                eval_params.pInNormals = &normals;
                eval_params.pInRoughness = &roughness;
                eval_params.pInColor = &color;
                eval_params.pInOutput = &output;
                eval_params.pInDepth = &depth;
                eval_params.pInMotionVectors = &motion_vectors;
                eval_params.InJitterOffsetX = resources.jitter_offset_x;
                eval_params.InJitterOffsetY = resources.jitter_offset_y;
                eval_params.InRenderSubrectDimensions.Width = m_desc.render_width;
                eval_params.InRenderSubrectDimensions.Height = m_desc.render_height;
                eval_params.InReset = resources.reset ? 1 : 0;
                eval_params.InMVScaleX = resources.motion_vector_scale_x;
                eval_params.InMVScaleY = resources.motion_vector_scale_y;
                eval_params.InPreExposure = resources.pre_exposure;
                eval_params.InExposureScale = resources.exposure_scale;
                eval_params.InFrameTimeDeltaInMsec = resources.frame_time_delta_ms;
                if (resources.motion_vectors_reflections) {
                    eval_params.pInMotionVectorsReflections = &motion_vectors_reflections;
                }
                if (resources.specular_hit_distance) {
                    eval_params.pInSpecularHitDistance = &specular_hit_distance;
                    eval_params.pInWorldToViewMatrix = const_cast<float*>(resources.view_from_world->data());
                    eval_params.pInViewToClipMatrix = const_cast<float*>(resources.clip_from_view->data());
                }

                set_last_evaluate_result(NGX_CUDA_EVALUATE_DLSSD_EXT(m_feature_handle, m_params, &eval_params));
            }
        );
    }

private:
    /// Allocate the NGX parameter map and create the feature through a native command callback.
    void initialize()
    {
        validate_feature_desc();

        NGX::Impl* ngx_impl = get_ngx_impl(m_ngx.get());
        m_cuda_device = ngx_impl ? static_cast<NVSDK_NGX_CUDADevice*>(ngx_impl->ngx_device()) : nullptr;
        FALCOR_CHECK(m_cuda_device && m_cuda_device->cudaContext, "CUDA DLSS-RR feature creation requires CUDA NGX.");

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_CUDA_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "CUDA DLSS-RR parameter allocation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(allocate_result))
        );

        sgl::Device* device = m_ngx->device();
        ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();

        bool callback_called = false;
        std::string callback_error;
        NVSDK_NGX_Result create_result = NVSDK_NGX_Result_Fail;

        command_encoder->execute_callback(
            [&](sgl::NativeHandle native_handle)
            {
                callback_called = true;
                if (native_handle.type() != sgl::NativeHandleType::CUstream) {
                    callback_error = "CUDA DLSS-RR feature creation requires an active CUstream.";
                    return;
                }

                m_cuda_device->cudaStream = reinterpret_cast<void*>(native_handle.value());

                // DLSSD CUDA feature creation succeeds without a scratch buffer.
                // The generic CUDA scratch-size query reports UnableToInitializeFeature
                // for RayReconstruction, so do not use it as a precondition here.
                NVSDK_NGX_CUDA_DLSSD_Create_Params create_params{};
                create_params.Feature = make_create_params();
                create_params.InCUContext = m_cuda_device->cudaContext;
                create_params.InCUStream = m_cuda_device->cudaStream;

                create_result = NGX_CUDA_CREATE_DLSSD_EXT1(m_cuda_device, &m_feature_handle, m_params, &create_params);
            }
        );

        ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
        device->submit_command_buffer(command_buffer);
        device->wait();

        FALCOR_CHECK(callback_called, "CUDA DLSS-RR feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "CUDA DLSS-RR feature creation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(create_result))
        );
    }

    /// Release backend resources in the order expected by NGX.
    void release_resources() noexcept
    {
        if (m_feature_handle) {
            FALCOR_UNUSED(NVSDK_NGX_CUDA_ReleaseFeature(m_feature_handle));
            m_feature_handle = nullptr;
        }
        if (m_params) {
            FALCOR_UNUSED(NVSDK_NGX_CUDA_DestroyParameters(m_params));
            m_params = nullptr;
        }
    }

    NVSDK_NGX_CUDADevice* m_cuda_device{nullptr};
    NVSDK_NGX_Parameter* m_params{nullptr};
    NVSDK_NGX_Handle* m_feature_handle{nullptr};
};

} // namespace

std::unique_ptr<DLSSRRFeature::Impl> make_cuda_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc)
{
    return std::make_unique<CudaDLSSRRFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
