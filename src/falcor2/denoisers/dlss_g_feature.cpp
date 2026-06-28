// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/denoisers/dlss_g_feature.h"

#if FALCOR_ENABLE_NGX
#include "falcor2/denoisers/dlss_g_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#endif
#include "falcor2/denoisers/ngx.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <algorithm>
#include <utility>

namespace falcor::ngx {
#if FALCOR_ENABLE_NGX
#if SGL_WINDOWS
std::unique_ptr<DLSSGFeature::Impl> make_d3d12_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc);
#endif
std::unique_ptr<DLSSGFeature::Impl> make_vulkan_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc);
std::unique_ptr<DLSSGFeature::Impl> make_cuda_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc);

namespace {

void copy_matrix(float dst[4][4], const float4x4& src)
{
    const float* data = src.data();
    for (uint32_t row = 0; row < 4; ++row) {
        for (uint32_t column = 0; column < 4; ++column)
            dst[row][column] = data[row * 4 + column];
    }
}

void copy_float2(float dst[2], const float2& src)
{
    dst[0] = src.x;
    dst[1] = src.y;
}

void copy_float3(float dst[3], const float3& src)
{
    dst[0] = src.x;
    dst[1] = src.y;
    dst[2] = src.z;
}

} // namespace

DLSSGFeature::Impl::Impl(ref<NGX> ngx, DLSSGFeatureDesc desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
}

DLSSGFeature::Impl::~Impl() = default;

void DLSSGFeature::Impl::reset_last_evaluate_result()
{
    m_last_evaluate_result.store(static_cast<NVSDK_NGX_Result>(0), std::memory_order_release);
}

void DLSSGFeature::Impl::set_last_evaluate_result(NVSDK_NGX_Result result)
{
    m_last_evaluate_result.store(result, std::memory_order_release);
}

NVSDK_NGX_Result DLSSGFeature::Impl::last_evaluate_result() const
{
    return m_last_evaluate_result.load(std::memory_order_acquire);
}

NVSDK_NGX_Result DLSSGFeature::Impl::get_last_evaluate_result()
{
    return m_last_evaluate_result.exchange(static_cast<NVSDK_NGX_Result>(0), std::memory_order_acq_rel);
}

uint32_t DLSSGFeature::Impl::effective_render_width() const
{
    return m_desc.render_width != 0 ? m_desc.render_width : m_desc.width;
}

uint32_t DLSSGFeature::Impl::effective_render_height() const
{
    return m_desc.render_height != 0 ? m_desc.render_height : m_desc.height;
}

void DLSSGFeature::Impl::validate_feature_desc() const
{
    FALCOR_CHECK(m_ngx, "DLSSGFeature creation requires a parent NGX object.");
    FALCOR_CHECK(
        m_ngx->info().frame_generation_supported,
        "DLSSGFeature creation requires supported NGX FrameGeneration."
    );
    FALCOR_CHECK(m_desc.width != 0 && m_desc.height != 0, "DLSSGFeature requires non-zero output dimensions.");
    FALCOR_CHECK(
        effective_render_width() != 0 && effective_render_height() != 0,
        "DLSSGFeature requires non-zero render dimensions."
    );
    FALCOR_CHECK(effective_render_width() <= m_desc.width, "DLSSGFeature render width must not exceed output width.");
    FALCOR_CHECK(
        effective_render_height() <= m_desc.height,
        "DLSSGFeature render height must not exceed output height."
    );
}

DLSSGEvaluateResources DLSSGFeature::Impl::retain_evaluate_resources(const DLSSGEvaluateDesc& desc)
{
    DLSSGEvaluateResources resources;
    resources.backbuffer = desc.backbuffer;
    resources.depth = desc.depth;
    resources.motion_vectors = desc.motion_vectors;
    resources.output_interpolated_frame = desc.output_interpolated_frame;
    resources.camera = desc.camera;
    return resources;
}

void DLSSGFeature::Impl::validate_evaluate_desc(const DLSSGEvaluateDesc& desc) const
{
    FALCOR_CHECK(m_ngx, "DLSSGFeature evaluation requires a parent NGX object.");

    auto check_texture = [&](sgl::Texture* texture, const char* name, uint32_t width, uint32_t height)
    {
        FALCOR_CHECK(texture, "DLSSGFeature evaluation requires {}.", name);
        FALCOR_CHECK(texture->device() == m_ngx->device(), "DLSSGFeature evaluation {} is on the wrong device.", name);
        FALCOR_CHECK(texture->width() == width, "DLSSGFeature evaluation {} width does not match.", name);
        FALCOR_CHECK(texture->height() == height, "DLSSGFeature evaluation {} height does not match.", name);
    };

    check_texture(desc.backbuffer, "backbuffer", m_desc.width, m_desc.height);
    check_texture(desc.depth, "depth", effective_render_width(), effective_render_height());
    check_texture(desc.motion_vectors, "motion_vectors", effective_render_width(), effective_render_height());
    check_texture(desc.output_interpolated_frame, "output_interpolated_frame", m_desc.width, m_desc.height);

    const uint32_t max_multi_frame_count = std::max(1u, m_ngx->info().frame_generation_multi_frame_count_max);
    FALCOR_CHECK(
        desc.camera.multi_frame_count >= 1 && desc.camera.multi_frame_count <= max_multi_frame_count,
        "DLSSGFeature evaluation multi_frame_count is outside NGX capability."
    );
    FALCOR_CHECK(
        desc.camera.multi_frame_index >= 1 && desc.camera.multi_frame_index <= desc.camera.multi_frame_count,
        "DLSSGFeature evaluation multi_frame_index must be in [1, multi_frame_count]."
    );
}

void DLSSGFeature::Impl::prepare_evaluate_resources(
    sgl::CommandEncoder* command_encoder,
    const DLSSGEvaluateResources& resources
)
{
    FALCOR_CHECK(command_encoder, "DLSSGFeature evaluation requires a command encoder.");

    command_encoder->set_texture_state(resources.backbuffer, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.depth, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.motion_vectors, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.output_interpolated_frame, sgl::ResourceState::unordered_access);
}

NVSDK_NGX_DLSSG_Create_Params DLSSGFeature::Impl::make_create_params(uint32_t native_backbuffer_format) const
{
    NVSDK_NGX_DLSSG_Create_Params create_params{};
    create_params.Width = m_desc.width;
    create_params.Height = m_desc.height;
    create_params.NativeBackbufferFormat = native_backbuffer_format;
    create_params.RenderWidth = effective_render_width();
    create_params.RenderHeight = effective_render_height();
    return create_params;
}

NVSDK_NGX_DLSSG_Opt_Eval_Params DLSSGFeature::Impl::make_opt_eval_params(const DLSSGEvaluateResources& resources) const
{
    const DLSSGCameraDesc& camera = resources.camera;
    NVSDK_NGX_DLSSG_Opt_Eval_Params params{};
    params.multiFrameCount = camera.multi_frame_count;
    params.multiFrameIndex = camera.multi_frame_index;

    copy_matrix(params.cameraViewToClip, camera.camera_view_to_clip);
    copy_matrix(params.clipToCameraView, camera.clip_to_camera_view);
    copy_matrix(params.clipToLensClip, camera.clip_to_lens_clip);
    copy_matrix(params.clipToPrevClip, camera.clip_to_prev_clip);
    copy_matrix(params.prevClipToClip, camera.prev_clip_to_clip);
    copy_float2(params.jitterOffset, camera.jitter_offset);
    copy_float2(params.mvecScale, camera.motion_vector_scale);
    copy_float2(params.cameraPinholeOffset, camera.camera_pinhole_offset);
    copy_float3(params.cameraPos, camera.camera_pos);
    copy_float3(params.cameraUp, camera.camera_up);
    copy_float3(params.cameraRight, camera.camera_right);
    copy_float3(params.cameraFwd, camera.camera_fwd);

    params.cameraNear = camera.camera_near;
    params.cameraFar = camera.camera_far;
    params.cameraFOV = camera.camera_fov;
    params.cameraAspectRatio = camera.camera_aspect_ratio;
    params.colorBuffersHDR = camera.color_buffers_hdr;
    params.depthInverted = camera.depth_inverted;
    params.cameraMotionIncluded = camera.camera_motion_included;
    params.reset = camera.reset;
    params.automodeOverrideReset = camera.automode_override_reset;
    params.notRenderingGameFrames = camera.not_rendering_game_frames;
    params.orthoProjection = camera.ortho_projection;
    params.motionVectorsInvalidValue = camera.motion_vectors_invalid_value;
    params.motionVectorsDilated = camera.motion_vectors_dilated;
    params.menuDetectionEnabled = camera.menu_detection_enabled;

    params.mvecsSubrectSize = {effective_render_width(), effective_render_height()};
    params.depthSubrectSize = {effective_render_width(), effective_render_height()};
    params.backbufferSubrectSize = {m_desc.width, m_desc.height};
    params.outputInterpSubrectSize = {m_desc.width, m_desc.height};
    return params;
}

std::unique_ptr<DLSSGFeature::Impl> make_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc)
{
    switch (ngx->device()->type()) {
    case sgl::DeviceType::d3d12:
#if SGL_WINDOWS
        return make_d3d12_dlss_g_feature_impl(std::move(ngx), desc);
#else
        FALCOR_THROW("DLSS-G feature creation is not implemented for D3D12 on this platform.");
#endif
    case sgl::DeviceType::vulkan:
        return make_vulkan_dlss_g_feature_impl(std::move(ngx), desc);
    case sgl::DeviceType::cuda:
        return make_cuda_dlss_g_feature_impl(std::move(ngx), desc);
    default:
        FALCOR_THROW("DLSS-G feature creation is not implemented for this device type.");
    }
}
#endif

DLSSGFeature::DLSSGFeature(ref<NGX> ngx, const DLSSGFeatureDesc& desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
    FALCOR_CHECK(m_ngx, "DLSSGFeature requires a parent NGX object.");
#if FALCOR_ENABLE_NGX
    m_impl = make_dlss_g_feature_impl(m_ngx, m_desc).release();
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

DLSSGFeature::~DLSSGFeature()
{
#if FALCOR_ENABLE_NGX
    delete m_impl;
#endif
}

sgl::Device* DLSSGFeature::device() const
{
    return m_ngx ? m_ngx->device() : nullptr;
}

void DLSSGFeature::evaluate(const DLSSGEvaluateDesc& desc)
{
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    ref<sgl::CommandEncoder> command_encoder = device()->create_command_encoder();
    m_impl->record_evaluate(ref<DLSSGFeature>(this), command_encoder.get(), desc);
    ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
    device()->submit_command_buffer(command_buffer);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

void DLSSGFeature::evaluate(sgl::CommandEncoder* command_encoder, const DLSSGEvaluateDesc& desc)
{
    FALCOR_CHECK(command_encoder, "DLSSGFeature evaluation requires a command encoder.");
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    m_impl->record_evaluate(ref<DLSSGFeature>(this), command_encoder, desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

Result DLSSGFeature::last_evaluate_result()
{
#if FALCOR_ENABLE_NGX
    return to_result(m_impl->get_last_evaluate_result());
#else
    return Result::none;
#endif
}

void DLSSGFeature::throw_if_last_evaluate_failed() const
{
#if FALCOR_ENABLE_NGX
    NVSDK_NGX_Result result = m_impl->last_evaluate_result();
    if (result != static_cast<NVSDK_NGX_Result>(0) && result != NVSDK_NGX_Result_Success) {
        FALCOR_THROW(
            "Previous DLSS-G evaluation failed asynchronously: {}. Catch exceptions from evaluate() immediately; "
            "wait for the device, then call last_evaluate_result() to retrieve and clear the NGX result before "
            "recording another DLSS-G evaluation.",
            ngx_result_to_string(static_cast<int32_t>(result))
        );
    }
#endif
}

} // namespace falcor::ngx
