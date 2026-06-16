// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include <sgl/device/formats.h>
#include <sgl/device/fwd.h>

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace falcor::ngx {

enum class QualityMode {
    performance,
    balanced,
    quality,
    ultra_performance,
    dlaa,
};
SGL_ENUM_INFO(
    QualityMode,
    {
        {QualityMode::performance, "performance"},
        {QualityMode::balanced, "balanced"},
        {QualityMode::quality, "quality"},
        {QualityMode::ultra_performance, "ultra_performance"},
        {QualityMode::dlaa, "dlaa"},
    }
);
SGL_ENUM_REGISTER(QualityMode);

enum class Result : uint32_t {
    none = 0,
    success = 0x1,
    fail = 0xBAD00000,
    fail_feature_not_supported = 0xBAD00001,
    fail_platform_error = 0xBAD00002,
    fail_feature_already_exists = 0xBAD00003,
    fail_feature_not_found = 0xBAD00004,
    fail_invalid_parameter = 0xBAD00005,
    fail_scratch_buffer_too_small = 0xBAD00006,
    fail_not_initialized = 0xBAD00007,
    fail_unsupported_input_format = 0xBAD00008,
    fail_rw_flag_missing = 0xBAD00009,
    fail_missing_input = 0xBAD0000A,
    fail_unable_to_initialize_feature = 0xBAD0000B,
    fail_out_of_date = 0xBAD0000C,
    fail_out_of_gpu_memory = 0xBAD0000D,
    fail_unsupported_format = 0xBAD0000E,
    fail_unable_to_write_to_app_data_path = 0xBAD0000F,
    fail_unsupported_parameter = 0xBAD00010,
    fail_denied = 0xBAD00011,
    fail_not_implemented = 0xBAD00012,
};
SGL_ENUM_INFO(
    Result,
    {
        {Result::none, "none"},
        {Result::success, "success"},
        {Result::fail, "fail"},
        {Result::fail_feature_not_supported, "fail_feature_not_supported"},
        {Result::fail_platform_error, "fail_platform_error"},
        {Result::fail_feature_already_exists, "fail_feature_already_exists"},
        {Result::fail_feature_not_found, "fail_feature_not_found"},
        {Result::fail_invalid_parameter, "fail_invalid_parameter"},
        {Result::fail_scratch_buffer_too_small, "fail_scratch_buffer_too_small"},
        {Result::fail_not_initialized, "fail_not_initialized"},
        {Result::fail_unsupported_input_format, "fail_unsupported_input_format"},
        {Result::fail_rw_flag_missing, "fail_rw_flag_missing"},
        {Result::fail_missing_input, "fail_missing_input"},
        {Result::fail_unable_to_initialize_feature, "fail_unable_to_initialize_feature"},
        {Result::fail_out_of_date, "fail_out_of_date"},
        {Result::fail_out_of_gpu_memory, "fail_out_of_gpu_memory"},
        {Result::fail_unsupported_format, "fail_unsupported_format"},
        {Result::fail_unable_to_write_to_app_data_path, "fail_unable_to_write_to_app_data_path"},
        {Result::fail_unsupported_parameter, "fail_unsupported_parameter"},
        {Result::fail_denied, "fail_denied"},
        {Result::fail_not_implemented, "fail_not_implemented"},
    }
);
SGL_ENUM_REGISTER(Result);

struct NGXInfo {

    // Deep Learning SuperSampling Denoising (DLSSD) capability and requirements
    bool dlssd_available{false};
    bool dlssd_supported{false};
    bool dlssd_needs_updated_driver{false};
    uint32_t dlssd_minimum_driver_version_major{0};
    uint32_t dlssd_minimum_driver_version_minor{0};
    int32_t dlssd_feature_init_result{0};

    // Deep Learning Super Sampling / Super Resolution capability and requirements.
    bool dlss_available{false};
    bool dlss_supported{false};
    bool dlss_needs_updated_driver{false};
    uint32_t dlss_minimum_driver_version_major{0};
    uint32_t dlss_minimum_driver_version_minor{0};
    int32_t dlss_feature_init_result{0};

    // DLSS Frame Generation capability and requirements.
    bool frame_generation_available{false};
    bool frame_generation_supported{false};
    bool frame_generation_needs_updated_driver{false};
    uint32_t frame_generation_minimum_driver_version_major{0};
    uint32_t frame_generation_minimum_driver_version_minor{0};
    int32_t frame_generation_feature_init_result{0};
    uint32_t frame_generation_multi_frame_count_max{1};

    // Vulkan-specific device requirements for NGX support.
    std::vector<std::string> required_vulkan_instance_extensions;
    std::vector<std::string> required_vulkan_device_extensions;
};

struct DLSSDOptimalSettings {
    uint32_t render_width{0};
    uint32_t render_height{0};
    uint32_t max_render_width{0};
    uint32_t max_render_height{0};
    uint32_t min_render_width{0};
    uint32_t min_render_height{0};
    float sharpness{0.f};
};

struct DLSSOptimalSettings {
    uint32_t render_width{0};
    uint32_t render_height{0};
    uint32_t max_render_width{0};
    uint32_t max_render_height{0};
    uint32_t min_render_width{0};
    uint32_t min_render_height{0};
    float sharpness{0.f};
};

struct DLSSRRFeatureDesc {
    QualityMode quality{QualityMode::quality};
    uint32_t render_width{0};
    uint32_t render_height{0};
    uint32_t target_width{0};
    uint32_t target_height{0};
};

struct DLSSRREvaluateDesc {
    ref<sgl::Texture> diffuse_albedo;
    ref<sgl::Texture> specular_albedo;
    ref<sgl::Texture> normals;
    ref<sgl::Texture> roughness;
    ref<sgl::Texture> color;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output;
    ref<sgl::Texture> motion_vectors_reflections;
    ref<sgl::Texture> specular_hit_distance;
    std::optional<float4x4> view_from_world;
    std::optional<float4x4> clip_from_view;

    float jitter_offset_x{0.f};
    float jitter_offset_y{0.f};
    bool reset{false};
    float motion_vector_scale_x{1.f};
    float motion_vector_scale_y{1.f};
    float pre_exposure{1.f};
    float exposure_scale{1.f};
    float frame_time_delta_ms{0.f};
};

struct DLSSSRFeatureDesc {
    QualityMode quality{QualityMode::quality};
    uint32_t render_width{0};
    uint32_t render_height{0};
    uint32_t target_width{0};
    uint32_t target_height{0};
    bool is_hdr{true};
    bool depth_inverted{false};
    bool motion_vectors_jittered{false};
    bool auto_exposure{true};
    bool enable_output_subrects{false};
    float sharpness{0.f};
};

struct DLSSSREvaluateDesc {
    ref<sgl::Texture> color;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output;
    ref<sgl::Texture> exposure;

    uint32_t render_subrect_width{0};
    uint32_t render_subrect_height{0};

    float jitter_offset_x{0.f};
    float jitter_offset_y{0.f};
    bool reset{false};
    float motion_vector_scale_x{1.f};
    float motion_vector_scale_y{1.f};
    float pre_exposure{1.f};
    float exposure_scale{1.f};
};

struct DLSSGFeatureDesc {
    uint32_t width{0};
    uint32_t height{0};
    uint32_t render_width{0};
    uint32_t render_height{0};
    sgl::Format backbuffer_format{sgl::Format::rgba16_float};
};

struct DLSSGCameraDesc {
    float4x4 camera_view_to_clip;
    float4x4 clip_to_camera_view;
    float4x4 clip_to_lens_clip;
    float4x4 clip_to_prev_clip;
    float4x4 prev_clip_to_clip;
    float2 jitter_offset{0.f};
    float2 motion_vector_scale{1.f};
    float2 camera_pinhole_offset{0.f};
    float3 camera_pos{0.f};
    float3 camera_up{0.f, 1.f, 0.f};
    float3 camera_right{1.f, 0.f, 0.f};
    float3 camera_fwd{0.f, 0.f, -1.f};
    float camera_near{0.1f};
    float camera_far{1000.f};
    float camera_fov{0.f};
    float camera_aspect_ratio{1.f};
    bool color_buffers_hdr{false};
    bool depth_inverted{false};
    bool camera_motion_included{true};
    bool reset{false};
    bool automode_override_reset{false};
    bool not_rendering_game_frames{false};
    bool ortho_projection{false};
    float motion_vectors_invalid_value{0.f};
    bool motion_vectors_dilated{false};
    bool menu_detection_enabled{false};
    uint32_t multi_frame_count{1};
    uint32_t multi_frame_index{1};
};

struct DLSSGEvaluateDesc {
    ref<sgl::Texture> backbuffer;
    ref<sgl::Texture> depth;
    ref<sgl::Texture> motion_vectors;
    ref<sgl::Texture> output_interpolated_frame;
    DLSSGCameraDesc camera;
};

} // namespace falcor::ngx
