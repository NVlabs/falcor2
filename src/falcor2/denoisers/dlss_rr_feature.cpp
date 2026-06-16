// SPDX-License-Identifier: Apache-2.0

#include "falcor2/denoisers/dlss_rr_feature.h"

#if FALCOR_ENABLE_NGX
#include "falcor2/denoisers/dlss_rr_feature_impl.h"
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
std::unique_ptr<DLSSRRFeature::Impl> make_d3d12_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc);
#endif
std::unique_ptr<DLSSRRFeature::Impl> make_vulkan_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc);
std::unique_ptr<DLSSRRFeature::Impl> make_cuda_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc);

DLSSRRFeature::Impl::Impl(ref<NGX> ngx, DLSSRRFeatureDesc desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
}

DLSSRRFeature::Impl::~Impl() = default;

void DLSSRRFeature::Impl::reset_last_evaluate_result()
{
    m_last_evaluate_result.store(static_cast<NVSDK_NGX_Result>(0), std::memory_order_release);
}

void DLSSRRFeature::Impl::set_last_evaluate_result(NVSDK_NGX_Result result)
{
    m_last_evaluate_result.store(result, std::memory_order_release);
}

NVSDK_NGX_Result DLSSRRFeature::Impl::last_evaluate_result() const
{
    return m_last_evaluate_result.load(std::memory_order_acquire);
}

NVSDK_NGX_Result DLSSRRFeature::Impl::get_last_evaluate_result()
{
    return m_last_evaluate_result.exchange(static_cast<NVSDK_NGX_Result>(0), std::memory_order_acq_rel);
}

void DLSSRRFeature::Impl::validate_feature_desc() const
{
    FALCOR_CHECK(m_ngx, "DLSSRRFeature creation requires a parent NGX object.");
    FALCOR_CHECK(
        m_ngx->info().dlssd_supported,
        "DLSSRRFeature creation requires supported NGX SuperSamplingDenoising."
    );
    FALCOR_CHECK(
        m_desc.render_width != 0 && m_desc.render_height != 0,
        "DLSSRRFeature requires non-zero render dimensions."
    );
    FALCOR_CHECK(
        m_desc.target_width != 0 && m_desc.target_height != 0,
        "DLSSRRFeature requires non-zero target dimensions."
    );
    FALCOR_CHECK(
        m_desc.render_width <= m_desc.target_width,
        "DLSSRRFeature render width must not exceed target width."
    );
    FALCOR_CHECK(
        m_desc.render_height <= m_desc.target_height,
        "DLSSRRFeature render height must not exceed target height."
    );
}

DLSSRREvaluateResources DLSSRRFeature::Impl::retain_evaluate_resources(const DLSSRREvaluateDesc& desc)
{
    DLSSRREvaluateResources resources;
    resources.diffuse_albedo = desc.diffuse_albedo;
    resources.specular_albedo = desc.specular_albedo;
    resources.normals = desc.normals;
    resources.roughness = desc.roughness;
    resources.color = desc.color;
    resources.depth = desc.depth;
    resources.motion_vectors = desc.motion_vectors;
    resources.output = desc.output;
    resources.motion_vectors_reflections = desc.motion_vectors_reflections;
    resources.specular_hit_distance = desc.specular_hit_distance;
    resources.view_from_world = desc.view_from_world;
    resources.clip_from_view = desc.clip_from_view;
    resources.jitter_offset_x = desc.jitter_offset_x;
    resources.jitter_offset_y = desc.jitter_offset_y;
    resources.reset = desc.reset;
    resources.motion_vector_scale_x = desc.motion_vector_scale_x;
    resources.motion_vector_scale_y = desc.motion_vector_scale_y;
    resources.pre_exposure = desc.pre_exposure;
    resources.exposure_scale = desc.exposure_scale;
    resources.frame_time_delta_ms = desc.frame_time_delta_ms;
    return resources;
}

void DLSSRRFeature::Impl::validate_evaluate_desc(const DLSSRREvaluateDesc& desc) const
{
    FALCOR_CHECK(m_ngx, "DLSSRRFeature evaluation requires a parent NGX object.");

    auto check_texture = [&](sgl::Texture* texture, const char* name, uint32_t width, uint32_t height)
    {
        FALCOR_CHECK(texture, "DLSSRRFeature evaluation requires {}.", name);
        FALCOR_CHECK(texture->device() == m_ngx->device(), "DLSSRRFeature evaluation {} is on the wrong device.", name);
        FALCOR_CHECK(texture->width() == width, "DLSSRRFeature evaluation {} width does not match.", name);
        FALCOR_CHECK(texture->height() == height, "DLSSRRFeature evaluation {} height does not match.", name);
    };

    check_texture(desc.diffuse_albedo, "diffuse_albedo", m_desc.render_width, m_desc.render_height);
    check_texture(desc.specular_albedo, "specular_albedo", m_desc.render_width, m_desc.render_height);
    check_texture(desc.normals, "normals", m_desc.render_width, m_desc.render_height);
    check_texture(desc.roughness, "roughness", m_desc.render_width, m_desc.render_height);
    check_texture(desc.color, "color", m_desc.render_width, m_desc.render_height);
    check_texture(desc.depth, "depth", m_desc.render_width, m_desc.render_height);
    check_texture(desc.motion_vectors, "motion_vectors", m_desc.render_width, m_desc.render_height);
    check_texture(desc.output, "output", m_desc.target_width, m_desc.target_height);
    const bool has_motion_vectors_reflections = desc.motion_vectors_reflections != nullptr;
    const bool has_specular_hit_distance = desc.specular_hit_distance != nullptr;
    FALCOR_CHECK(
        has_motion_vectors_reflections != has_specular_hit_distance,
        "DLSSRRFeature evaluation requires exactly one of motion_vectors_reflections or specular_hit_distance."
    );
    if (desc.motion_vectors_reflections) {
        check_texture(
            desc.motion_vectors_reflections,
            "motion_vectors_reflections",
            m_desc.render_width,
            m_desc.render_height
        );
    }
    if (desc.specular_hit_distance) {
        check_texture(desc.specular_hit_distance, "specular_hit_distance", m_desc.render_width, m_desc.render_height);
        FALCOR_CHECK(
            desc.view_from_world.has_value(),
            "DLSSRRFeature evaluation with specular_hit_distance requires view_from_world."
        );
        FALCOR_CHECK(
            desc.clip_from_view.has_value(),
            "DLSSRRFeature evaluation with specular_hit_distance requires clip_from_view."
        );
    } else {
        FALCOR_CHECK(
            !desc.view_from_world.has_value() && !desc.clip_from_view.has_value(),
            "DLSSRRFeature evaluation matrices are only valid with specular_hit_distance."
        );
    }
}

void DLSSRRFeature::Impl::prepare_evaluate_resources(
    sgl::CommandEncoder* command_encoder,
    const DLSSRREvaluateResources& resources
)
{
    FALCOR_CHECK(command_encoder, "DLSSRRFeature evaluation requires a command encoder.");

    command_encoder->set_texture_state(resources.diffuse_albedo, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.specular_albedo, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.normals, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.roughness, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.color, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.depth, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.motion_vectors, sgl::ResourceState::shader_resource);
    if (resources.motion_vectors_reflections)
        command_encoder->set_texture_state(resources.motion_vectors_reflections, sgl::ResourceState::shader_resource);
    if (resources.specular_hit_distance)
        command_encoder->set_texture_state(resources.specular_hit_distance, sgl::ResourceState::shader_resource);
    command_encoder->set_texture_state(resources.output, sgl::ResourceState::unordered_access);
}

NVSDK_NGX_DLSSD_Create_Params DLSSRRFeature::Impl::make_create_params() const
{
    NVSDK_NGX_DLSSD_Create_Params create_params{};
    create_params.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    create_params.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
    create_params.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;
    create_params.InWidth = m_desc.render_width;
    create_params.InHeight = m_desc.render_height;
    create_params.InTargetWidth = m_desc.target_width;
    create_params.InTargetHeight = m_desc.target_height;
    create_params.InPerfQualityValue = to_ngx_perf_quality(m_desc.quality);
    create_params.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    create_params.InEnableOutputSubrects = false;
    return create_params;
}

std::unique_ptr<DLSSRRFeature::Impl> make_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc)
{
    // Feature implementations are backend-specific because feature creation and
    // evaluation will eventually bind native resources from that device API.
    switch (ngx->device()->type()) {
    case sgl::DeviceType::d3d12:
#if SGL_WINDOWS
        return make_d3d12_dlss_rr_feature_impl(std::move(ngx), desc);
#else
        FALCOR_THROW("DLSS-RR feature creation is not implemented for D3D12 on this platform.");
#endif
    case sgl::DeviceType::vulkan:
        return make_vulkan_dlss_rr_feature_impl(std::move(ngx), desc);
    case sgl::DeviceType::cuda:
        return make_cuda_dlss_rr_feature_impl(std::move(ngx), desc);
    default:
        FALCOR_THROW("DLSS-RR feature creation is not implemented for this device type.");
    }
}
#endif

DLSSRRFeature::DLSSRRFeature(ref<NGX> ngx, const DLSSRRFeatureDesc& desc)
    : m_ngx(std::move(ngx))
    , m_desc(desc)
{
    FALCOR_CHECK(m_ngx, "DLSSRRFeature requires a parent NGX object.");
#if FALCOR_ENABLE_NGX
    m_impl = make_dlss_rr_feature_impl(m_ngx, m_desc).release();
#else
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

DLSSRRFeature::~DLSSRRFeature()
{
#if FALCOR_ENABLE_NGX
    delete m_impl;
#endif
}

sgl::Device* DLSSRRFeature::device() const
{
    return m_ngx ? m_ngx->device() : nullptr;
}

void DLSSRRFeature::evaluate(const DLSSRREvaluateDesc& desc)
{
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    ref<sgl::CommandEncoder> command_encoder = device()->create_command_encoder();
    m_impl->record_evaluate(ref<DLSSRRFeature>(this), command_encoder.get(), desc);
    ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
    device()->submit_command_buffer(command_buffer);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

void DLSSRRFeature::evaluate(sgl::CommandEncoder* command_encoder, const DLSSRREvaluateDesc& desc)
{
    FALCOR_CHECK(command_encoder, "DLSSRRFeature evaluation requires a command encoder.");
#if FALCOR_ENABLE_NGX
    throw_if_last_evaluate_failed();
    m_impl->record_evaluate(ref<DLSSRRFeature>(this), command_encoder, desc);
#else
    FALCOR_UNUSED(desc);
    FALCOR_THROW("Falcor2 was built without NGX. Reconfigure with FALCOR_ENABLE_NGX=ON.");
#endif
}

Result DLSSRRFeature::last_evaluate_result()
{
#if FALCOR_ENABLE_NGX
    return to_result(m_impl->get_last_evaluate_result());
#else
    return Result::none;
#endif
}

void DLSSRRFeature::throw_if_last_evaluate_failed() const
{
#if FALCOR_ENABLE_NGX
    NVSDK_NGX_Result result = m_impl->last_evaluate_result();
    if (result != static_cast<NVSDK_NGX_Result>(0) && result != NVSDK_NGX_Result_Success) {
        FALCOR_THROW(
            "Previous DLSS-RR evaluation failed asynchronously: {}. Catch exceptions from evaluate() immediately; "
            "wait for the device, then call last_evaluate_result() to retrieve and clear the NGX result before "
            "recording another DLSS-RR evaluation.",
            ngx_result_to_string(static_cast<int32_t>(result))
        );
    }
#endif
}

} // namespace falcor::ngx
