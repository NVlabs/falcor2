// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/denoisers/dlss_sr_feature.h"

#if FALCOR_ENABLE_NGX
#include "falcor2/denoisers/dlss_sr_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#endif
#include "falcor2/denoisers/ngx.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <utility>

namespace falcor::ngx {
#if FALCOR_ENABLE_NGX
#if SGL_WINDOWS
std::unique_ptr<DLSSSRFeature::Impl> make_d3d12_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc);
#endif
std::unique_ptr<DLSSSRFeature::Impl> make_vulkan_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc);
std::unique_ptr<DLSSSRFeature::Impl> make_cuda_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc);

DLSSSRFeature::Impl::Impl(ref<NGX> ngx, DLSSSRFeatureDesc desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
}

DLSSSRFeature::Impl::~Impl() = default;

void DLSSSRFeature::Impl::reset_last_evaluate_result()
{
    m_last_evaluate_result.store(static_cast<NVSDK_NGX_Result>(0), std::memory_order_release);
}

void DLSSSRFeature::Impl::set_last_evaluate_result(NVSDK_NGX_Result result)
{
    m_last_evaluate_result.store(result, std::memory_order_release);
}

NVSDK_NGX_Result DLSSSRFeature::Impl::last_evaluate_result() const
{
    return m_last_evaluate_result.load(std::memory_order_acquire);
}

NVSDK_NGX_Result DLSSSRFeature::Impl::get_last_evaluate_result()
{
    return m_last_evaluate_result.exchange(static_cast<NVSDK_NGX_Result>(0), std::memory_order_acq_rel);
}

void DLSSSRFeature::Impl::validate_feature_desc() const
{
    FALCOR_CHECK(m_ngx, "DLSSSRFeature creation requires a parent NGX object.");
    FALCOR_CHECK(m_ngx->info().dlss_supported, "DLSSSRFeature creation requires supported NGX SuperSampling.");
    FALCOR_CHECK(
        m_desc.render_width != 0 && m_desc.render_height != 0,
        "DLSSSRFeature requires non-zero render dimensions."
    );
    FALCOR_CHECK(
        m_desc.target_width != 0 && m_desc.target_height != 0,
        "DLSSSRFeature requires non-zero target dimensions."
    );
    FALCOR_CHECK(
        m_desc.render_width <= m_desc.target_width,
        "DLSSSRFeature render width must not exceed target width."
    );
    FALCOR_CHECK(
        m_desc.render_height <= m_desc.target_height,
        "DLSSSRFeature render height must not exceed target height."
    );
}

DLSSSREvaluateResources DLSSSRFeature::Impl::retain_evaluate_resources(const DLSSSREvaluateDesc& desc)
{
    DLSSSREvaluateResources resources;
    resources.color = desc.color;
    resources.depth = desc.depth;
    resources.motion_vectors = desc.motion_vectors;
    resources.output = desc.output;
    resources.exposure = desc.exposure;
    resources.render_subrect_width = desc.render_subrect_width;
    resources.render_subrect_height = desc.render_subrect_height;
    resources.jitter_offset_x = desc.jitter_offset_x;
    resources.jitter_offset_y = desc.jitter_offset_y;
    resources.reset = desc.reset;
    resources.motion_vector_scale_x = desc.motion_vector_scale_x;
    resources.motion_vector_scale_y = desc.motion_vector_scale_y;
    resources.pre_exposure = desc.pre_exposure;
    resources.exposure_scale = desc.exposure_scale;
    return resources;
}

uint32_t DLSSSRFeature::Impl::evaluate_render_subrect_width(const DLSSSREvaluateResources& resources) const
{
    return resources.render_subrect_width != 0 ? resources.render_subrect_width : m_desc.render_width;
}

uint32_t DLSSSRFeature::Impl::evaluate_render_subrect_height(const DLSSSREvaluateResources& resources) const
{
    return resources.render_subrect_height != 0 ? resources.render_subrect_height : m_desc.render_height;
}

void DLSSSRFeature::Impl::validate_evaluate_desc(const DLSSSREvaluateDesc& desc) const
{
    FALCOR_CHECK(m_ngx, "DLSSSRFeature evaluation requires a parent NGX object.");

    auto check_texture = [&](sgl::Texture* texture, const char* name, uint32_t width, uint32_t height)
    {
        FALCOR_CHECK(texture, "DLSSSRFeature evaluation requires {}.", name);
        FALCOR_CHECK(texture->device() == m_ngx->device(), "DLSSSRFeature evaluation {} is on the wrong device.", name);
        FALCOR_CHECK(texture->width() == width, "DLSSSRFeature evaluation {} width does not match.", name);
        FALCOR_CHECK(texture->height() == height, "DLSSSRFeature evaluation {} height does not match.", name);
    };

    check_texture(desc.color, "color", m_desc.render_width, m_desc.render_height);
    check_texture(desc.depth, "depth", m_desc.render_width, m_desc.render_height);
    check_texture(desc.motion_vectors, "motion_vectors", m_desc.render_width, m_desc.render_height);
    check_texture(desc.output, "output", m_desc.target_width, m_desc.target_height);
    if (desc.exposure) {
        FALCOR_CHECK(
            desc.exposure->device() == m_ngx->device(),
            "DLSSSRFeature evaluation exposure is on the wrong device."
        );
    }

    const uint32_t render_subrect_width
        = desc.render_subrect_width != 0 ? desc.render_subrect_width : m_desc.render_width;
    const uint32_t render_subrect_height
        = desc.render_subrect_height != 0 ? desc.render_subrect_height : m_desc.render_height;
    FALCOR_CHECK(
        render_subrect_width <= m_desc.render_width,
        "DLSSSRFeature evaluation render_subrect_width must not exceed render_width."
    );
    FALCOR_CHECK(
        render_subrect_height <= m_desc.render_height,
        "DLSSSRFeature evaluation render_subrect_height must not exceed render_height."
    );
}

void DLSSSRFeature::Impl::prepare_evaluate_resources(
    sgl::CommandEncoder* command_encoder,
    const DLSSSREvaluateResources& resources
)
{
    FALCOR_CHECK(command_encoder, "DLSSSRFeature evaluation requires a command encoder.");

    command_encoder->set_texture_state(resources.color, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.depth, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.motion_vectors, sgl::ResourceState::shader_resource);
    if (resources.exposure)
        command_encoder->set_texture_state(resources.exposure, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.output, sgl::ResourceState::unordered_access);
}

NVSDK_NGX_DLSS_Create_Params DLSSSRFeature::Impl::make_create_params() const
{
    NVSDK_NGX_DLSS_Create_Params create_params{};
    create_params.Feature.InWidth = m_desc.render_width;
    create_params.Feature.InHeight = m_desc.render_height;
    create_params.Feature.InTargetWidth = m_desc.target_width;
    create_params.Feature.InTargetHeight = m_desc.target_height;
    create_params.Feature.InPerfQualityValue = to_ngx_perf_quality(m_desc.quality);
    create_params.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_None;
    if (m_desc.is_hdr)
        create_params.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_IsHDR;
    if (m_desc.depth_inverted)
        create_params.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_DepthInverted;
    create_params.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    if (m_desc.motion_vectors_jittered)
        create_params.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_MVJittered;
    if (m_desc.auto_exposure)
        create_params.InFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    create_params.InEnableOutputSubrects = m_desc.enable_output_subrects;
    return create_params;
}

std::unique_ptr<DLSSSRFeature::Impl> make_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc)
{
    switch (ngx->device()->type()) {
    case sgl::DeviceType::d3d12:
#if SGL_WINDOWS
        return make_d3d12_dlss_sr_feature_impl(std::move(ngx), desc);
#else
        FALCOR_THROW("DLSS-SR feature creation is not implemented for D3D12 on this platform.");
#endif
    case sgl::DeviceType::vulkan:
        return make_vulkan_dlss_sr_feature_impl(std::move(ngx), desc);
    case sgl::DeviceType::cuda:
        return make_cuda_dlss_sr_feature_impl(std::move(ngx), desc);
    default:
        FALCOR_THROW("DLSS-SR feature creation is not implemented for this device type.");
    }
}
#endif

DLSSSRFeature::DLSSSRFeature(ref<NGX> ngx, const DLSSSRFeatureDesc& desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
    FALCOR_CHECK(m_ngx, "DLSSSRFeature requires a parent NGX object.");
#if FALCOR_ENABLE_NGX
    m_impl = make_dlss_sr_feature_impl(m_ngx, m_desc).release();
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

DLSSSRFeature::~DLSSSRFeature()
{
#if FALCOR_ENABLE_NGX
    delete m_impl;
#endif
}

sgl::Device* DLSSSRFeature::device() const
{
    return m_ngx ? m_ngx->device() : nullptr;
}

void DLSSSRFeature::evaluate(const DLSSSREvaluateDesc& desc)
{
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    ref<sgl::CommandEncoder> command_encoder = device()->create_command_encoder();
    m_impl->record_evaluate(ref<DLSSSRFeature>(this), command_encoder.get(), desc);
    ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
    device()->submit_command_buffer(command_buffer);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

void DLSSSRFeature::evaluate(sgl::CommandEncoder* command_encoder, const DLSSSREvaluateDesc& desc)
{
    FALCOR_CHECK(command_encoder, "DLSSSRFeature evaluation requires a command encoder.");
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    m_impl->record_evaluate(ref<DLSSSRFeature>(this), command_encoder, desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

Result DLSSSRFeature::last_evaluate_result()
{
#if FALCOR_ENABLE_NGX
    return to_result(m_impl->get_last_evaluate_result());
#else
    return Result::none;
#endif
}

void DLSSSRFeature::throw_if_last_evaluate_failed() const
{
#if FALCOR_ENABLE_NGX
    NVSDK_NGX_Result result = m_impl->last_evaluate_result();
    if (result != static_cast<NVSDK_NGX_Result>(0) && result != NVSDK_NGX_Result_Success) {
        FALCOR_THROW(
            "Previous DLSS-SR evaluation failed asynchronously: {}. Catch exceptions from evaluate() immediately; "
            "wait for the device, then call last_evaluate_result() to retrieve and clear the NGX result before "
            "recording another DLSS-SR evaluation.",
            ngx_result_to_string(static_cast<int32_t>(result))
        );
    }
#endif
}

} // namespace falcor::ngx
