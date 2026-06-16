// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/denoisers/ngx_types.h"

#include <sgl/device/resource.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(denoisers_ngx_types)
{
    using namespace falcor;
    using namespace falcor::ngx;

    nb::module_ ngx_module = nb::module_::import_("falcor2.ngx");

    nb::sgl_enum<QualityMode>(ngx_module, "QualityMode", D(ngx, QualityMode));
    nb::sgl_enum<Result>(ngx_module, "Result", D(ngx, Result));

    nb::class_<NGXInfo>(ngx_module, "NGXInfo", D(ngx, NGXInfo))
        .def(nb::init<>())
        .def_rw("dlssd_available", &NGXInfo::dlssd_available, D(ngx, NGXInfo, dlssd_available))
        .def_rw("dlssd_supported", &NGXInfo::dlssd_supported, D(ngx, NGXInfo, dlssd_supported))
        .def_rw(
            "dlssd_needs_updated_driver",
            &NGXInfo::dlssd_needs_updated_driver,
            D(ngx, NGXInfo, dlssd_needs_updated_driver)
        )
        .def_rw(
            "dlssd_minimum_driver_version_major",
            &NGXInfo::dlssd_minimum_driver_version_major,
            D(ngx, NGXInfo, dlssd_minimum_driver_version_major)
        )
        .def_rw(
            "dlssd_minimum_driver_version_minor",
            &NGXInfo::dlssd_minimum_driver_version_minor,
            D(ngx, NGXInfo, dlssd_minimum_driver_version_minor)
        )
        .def_rw(
            "dlssd_feature_init_result",
            &NGXInfo::dlssd_feature_init_result,
            D(ngx, NGXInfo, dlssd_feature_init_result)
        )
        .def_rw("dlss_available", &NGXInfo::dlss_available, D_NA(ngx, NGXInfo, dlss_available))
        .def_rw("dlss_supported", &NGXInfo::dlss_supported, D_NA(ngx, NGXInfo, dlss_supported))
        .def_rw(
            "dlss_needs_updated_driver",
            &NGXInfo::dlss_needs_updated_driver,
            D_NA(ngx, NGXInfo, dlss_needs_updated_driver)
        )
        .def_rw(
            "dlss_minimum_driver_version_major",
            &NGXInfo::dlss_minimum_driver_version_major,
            D_NA(ngx, NGXInfo, dlss_minimum_driver_version_major)
        )
        .def_rw(
            "dlss_minimum_driver_version_minor",
            &NGXInfo::dlss_minimum_driver_version_minor,
            D_NA(ngx, NGXInfo, dlss_minimum_driver_version_minor)
        )
        .def_rw(
            "dlss_feature_init_result",
            &NGXInfo::dlss_feature_init_result,
            D_NA(ngx, NGXInfo, dlss_feature_init_result)
        )
        .def_rw(
            "frame_generation_available",
            &NGXInfo::frame_generation_available,
            D_NA(ngx, NGXInfo, frame_generation_available)
        )
        .def_rw(
            "frame_generation_supported",
            &NGXInfo::frame_generation_supported,
            D_NA(ngx, NGXInfo, frame_generation_supported)
        )
        .def_rw(
            "frame_generation_needs_updated_driver",
            &NGXInfo::frame_generation_needs_updated_driver,
            D_NA(ngx, NGXInfo, frame_generation_needs_updated_driver)
        )
        .def_rw(
            "frame_generation_minimum_driver_version_major",
            &NGXInfo::frame_generation_minimum_driver_version_major,
            D_NA(ngx, NGXInfo, frame_generation_minimum_driver_version_major)
        )
        .def_rw(
            "frame_generation_minimum_driver_version_minor",
            &NGXInfo::frame_generation_minimum_driver_version_minor,
            D_NA(ngx, NGXInfo, frame_generation_minimum_driver_version_minor)
        )
        .def_rw(
            "frame_generation_feature_init_result",
            &NGXInfo::frame_generation_feature_init_result,
            D_NA(ngx, NGXInfo, frame_generation_feature_init_result)
        )
        .def_rw(
            "frame_generation_multi_frame_count_max",
            &NGXInfo::frame_generation_multi_frame_count_max,
            D_NA(ngx, NGXInfo, frame_generation_multi_frame_count_max)
        )
        .def_rw(
            "required_vulkan_instance_extensions",
            &NGXInfo::required_vulkan_instance_extensions,
            D(ngx, NGXInfo, required_vulkan_instance_extensions)
        )
        .def_rw(
            "required_vulkan_device_extensions",
            &NGXInfo::required_vulkan_device_extensions,
            D(ngx, NGXInfo, required_vulkan_device_extensions)
        );

    nb::class_<DLSSDOptimalSettings>(ngx_module, "DLSSDOptimalSettings", D(ngx, DLSSDOptimalSettings))
        .def(nb::init<>())
        .def_rw("render_width", &DLSSDOptimalSettings::render_width, D(ngx, DLSSDOptimalSettings, render_width))
        .def_rw("render_height", &DLSSDOptimalSettings::render_height, D(ngx, DLSSDOptimalSettings, render_height))
        .def_rw(
            "max_render_width",
            &DLSSDOptimalSettings::max_render_width,
            D(ngx, DLSSDOptimalSettings, max_render_width)
        )
        .def_rw(
            "max_render_height",
            &DLSSDOptimalSettings::max_render_height,
            D(ngx, DLSSDOptimalSettings, max_render_height)
        )
        .def_rw(
            "min_render_width",
            &DLSSDOptimalSettings::min_render_width,
            D(ngx, DLSSDOptimalSettings, min_render_width)
        )
        .def_rw(
            "min_render_height",
            &DLSSDOptimalSettings::min_render_height,
            D(ngx, DLSSDOptimalSettings, min_render_height)
        )
        .def_rw("sharpness", &DLSSDOptimalSettings::sharpness, D(ngx, DLSSDOptimalSettings, sharpness));

    nb::class_<DLSSOptimalSettings>(ngx_module, "DLSSOptimalSettings", D_NA(ngx, DLSSOptimalSettings))
        .def(nb::init<>())
        .def_rw("render_width", &DLSSOptimalSettings::render_width, D_NA(ngx, DLSSOptimalSettings, render_width))
        .def_rw("render_height", &DLSSOptimalSettings::render_height, D_NA(ngx, DLSSOptimalSettings, render_height))
        .def_rw(
            "max_render_width",
            &DLSSOptimalSettings::max_render_width,
            D_NA(ngx, DLSSOptimalSettings, max_render_width)
        )
        .def_rw(
            "max_render_height",
            &DLSSOptimalSettings::max_render_height,
            D_NA(ngx, DLSSOptimalSettings, max_render_height)
        )
        .def_rw(
            "min_render_width",
            &DLSSOptimalSettings::min_render_width,
            D_NA(ngx, DLSSOptimalSettings, min_render_width)
        )
        .def_rw(
            "min_render_height",
            &DLSSOptimalSettings::min_render_height,
            D_NA(ngx, DLSSOptimalSettings, min_render_height)
        )
        .def_rw("sharpness", &DLSSOptimalSettings::sharpness, D_NA(ngx, DLSSOptimalSettings, sharpness));

    nb::class_<DLSSRRFeatureDesc>(ngx_module, "DLSSRRFeatureDesc", D(ngx, DLSSRRFeatureDesc))
        .def(nb::init<>())
        .def_rw("quality", &DLSSRRFeatureDesc::quality, D(ngx, DLSSRRFeatureDesc, quality))
        .def_rw("render_width", &DLSSRRFeatureDesc::render_width, D(ngx, DLSSRRFeatureDesc, render_width))
        .def_rw("render_height", &DLSSRRFeatureDesc::render_height, D(ngx, DLSSRRFeatureDesc, render_height))
        .def_rw("target_width", &DLSSRRFeatureDesc::target_width, D(ngx, DLSSRRFeatureDesc, target_width))
        .def_rw("target_height", &DLSSRRFeatureDesc::target_height, D(ngx, DLSSRRFeatureDesc, target_height));

    nb::class_<DLSSRREvaluateDesc>(ngx_module, "DLSSRREvaluateDesc", D(ngx, DLSSRREvaluateDesc))
        .def(nb::init<>())
        .def_rw("diffuse_albedo", &DLSSRREvaluateDesc::diffuse_albedo, D(ngx, DLSSRREvaluateDesc, diffuse_albedo))
        .def_rw("specular_albedo", &DLSSRREvaluateDesc::specular_albedo, D(ngx, DLSSRREvaluateDesc, specular_albedo))
        .def_rw("normals", &DLSSRREvaluateDesc::normals, D(ngx, DLSSRREvaluateDesc, normals))
        .def_rw("roughness", &DLSSRREvaluateDesc::roughness, D(ngx, DLSSRREvaluateDesc, roughness))
        .def_rw("color", &DLSSRREvaluateDesc::color, D(ngx, DLSSRREvaluateDesc, color))
        .def_rw("depth", &DLSSRREvaluateDesc::depth, D(ngx, DLSSRREvaluateDesc, depth))
        .def_rw("motion_vectors", &DLSSRREvaluateDesc::motion_vectors, D(ngx, DLSSRREvaluateDesc, motion_vectors))
        .def_rw("output", &DLSSRREvaluateDesc::output, D(ngx, DLSSRREvaluateDesc, output))
        .def_rw(
            "motion_vectors_reflections",
            &DLSSRREvaluateDesc::motion_vectors_reflections,
            D(ngx, DLSSRREvaluateDesc, motion_vectors_reflections)
        )
        .def_rw(
            "specular_hit_distance",
            &DLSSRREvaluateDesc::specular_hit_distance,
            D(ngx, DLSSRREvaluateDesc, specular_hit_distance)
        )
        .def_rw("view_from_world", &DLSSRREvaluateDesc::view_from_world, D(ngx, DLSSRREvaluateDesc, view_from_world))
        .def_rw("clip_from_view", &DLSSRREvaluateDesc::clip_from_view, D(ngx, DLSSRREvaluateDesc, clip_from_view))
        .def_rw("jitter_offset_x", &DLSSRREvaluateDesc::jitter_offset_x, D(ngx, DLSSRREvaluateDesc, jitter_offset_x))
        .def_rw("jitter_offset_y", &DLSSRREvaluateDesc::jitter_offset_y, D(ngx, DLSSRREvaluateDesc, jitter_offset_y))
        .def_rw("reset", &DLSSRREvaluateDesc::reset, D(ngx, DLSSRREvaluateDesc, reset))
        .def_rw(
            "motion_vector_scale_x",
            &DLSSRREvaluateDesc::motion_vector_scale_x,
            D(ngx, DLSSRREvaluateDesc, motion_vector_scale_x)
        )
        .def_rw(
            "motion_vector_scale_y",
            &DLSSRREvaluateDesc::motion_vector_scale_y,
            D(ngx, DLSSRREvaluateDesc, motion_vector_scale_y)
        )
        .def_rw("pre_exposure", &DLSSRREvaluateDesc::pre_exposure, D(ngx, DLSSRREvaluateDesc, pre_exposure))
        .def_rw("exposure_scale", &DLSSRREvaluateDesc::exposure_scale, D(ngx, DLSSRREvaluateDesc, exposure_scale))
        .def_rw(
            "frame_time_delta_ms",
            &DLSSRREvaluateDesc::frame_time_delta_ms,
            D(ngx, DLSSRREvaluateDesc, frame_time_delta_ms)
        );

    nb::class_<DLSSSRFeatureDesc>(ngx_module, "DLSSSRFeatureDesc", D_NA(ngx, DLSSSRFeatureDesc))
        .def(nb::init<>())
        .def_rw("quality", &DLSSSRFeatureDesc::quality, D_NA(ngx, DLSSSRFeatureDesc, quality))
        .def_rw("render_width", &DLSSSRFeatureDesc::render_width, D_NA(ngx, DLSSSRFeatureDesc, render_width))
        .def_rw("render_height", &DLSSSRFeatureDesc::render_height, D_NA(ngx, DLSSSRFeatureDesc, render_height))
        .def_rw("target_width", &DLSSSRFeatureDesc::target_width, D_NA(ngx, DLSSSRFeatureDesc, target_width))
        .def_rw("target_height", &DLSSSRFeatureDesc::target_height, D_NA(ngx, DLSSSRFeatureDesc, target_height))
        .def_rw("is_hdr", &DLSSSRFeatureDesc::is_hdr, D_NA(ngx, DLSSSRFeatureDesc, is_hdr))
        .def_rw("depth_inverted", &DLSSSRFeatureDesc::depth_inverted, D_NA(ngx, DLSSSRFeatureDesc, depth_inverted))
        .def_rw(
            "motion_vectors_jittered",
            &DLSSSRFeatureDesc::motion_vectors_jittered,
            D_NA(ngx, DLSSSRFeatureDesc, motion_vectors_jittered)
        )
        .def_rw("auto_exposure", &DLSSSRFeatureDesc::auto_exposure, D_NA(ngx, DLSSSRFeatureDesc, auto_exposure))
        .def_rw(
            "enable_output_subrects",
            &DLSSSRFeatureDesc::enable_output_subrects,
            D_NA(ngx, DLSSSRFeatureDesc, enable_output_subrects)
        )
        .def_rw("sharpness", &DLSSSRFeatureDesc::sharpness, D_NA(ngx, DLSSSRFeatureDesc, sharpness));

    nb::class_<DLSSSREvaluateDesc>(ngx_module, "DLSSSREvaluateDesc", D_NA(ngx, DLSSSREvaluateDesc))
        .def(nb::init<>())
        .def_rw("color", &DLSSSREvaluateDesc::color, D_NA(ngx, DLSSSREvaluateDesc, color))
        .def_rw("depth", &DLSSSREvaluateDesc::depth, D_NA(ngx, DLSSSREvaluateDesc, depth))
        .def_rw("motion_vectors", &DLSSSREvaluateDesc::motion_vectors, D_NA(ngx, DLSSSREvaluateDesc, motion_vectors))
        .def_rw("output", &DLSSSREvaluateDesc::output, D_NA(ngx, DLSSSREvaluateDesc, output))
        .def_rw("exposure", &DLSSSREvaluateDesc::exposure, D_NA(ngx, DLSSSREvaluateDesc, exposure))
        .def_rw(
            "render_subrect_width",
            &DLSSSREvaluateDesc::render_subrect_width,
            D_NA(ngx, DLSSSREvaluateDesc, render_subrect_width)
        )
        .def_rw(
            "render_subrect_height",
            &DLSSSREvaluateDesc::render_subrect_height,
            D_NA(ngx, DLSSSREvaluateDesc, render_subrect_height)
        )
        .def_rw("jitter_offset_x", &DLSSSREvaluateDesc::jitter_offset_x, D_NA(ngx, DLSSSREvaluateDesc, jitter_offset_x))
        .def_rw("jitter_offset_y", &DLSSSREvaluateDesc::jitter_offset_y, D_NA(ngx, DLSSSREvaluateDesc, jitter_offset_y))
        .def_rw("reset", &DLSSSREvaluateDesc::reset, D_NA(ngx, DLSSSREvaluateDesc, reset))
        .def_rw(
            "motion_vector_scale_x",
            &DLSSSREvaluateDesc::motion_vector_scale_x,
            D_NA(ngx, DLSSSREvaluateDesc, motion_vector_scale_x)
        )
        .def_rw(
            "motion_vector_scale_y",
            &DLSSSREvaluateDesc::motion_vector_scale_y,
            D_NA(ngx, DLSSSREvaluateDesc, motion_vector_scale_y)
        )
        .def_rw("pre_exposure", &DLSSSREvaluateDesc::pre_exposure, D_NA(ngx, DLSSSREvaluateDesc, pre_exposure))
        .def_rw("exposure_scale", &DLSSSREvaluateDesc::exposure_scale, D_NA(ngx, DLSSSREvaluateDesc, exposure_scale));

    nb::class_<DLSSGFeatureDesc>(ngx_module, "DLSSGFeatureDesc", D_NA(ngx, DLSSGFeatureDesc))
        .def(nb::init<>())
        .def_rw("width", &DLSSGFeatureDesc::width, D_NA(ngx, DLSSGFeatureDesc, width))
        .def_rw("height", &DLSSGFeatureDesc::height, D_NA(ngx, DLSSGFeatureDesc, height))
        .def_rw("render_width", &DLSSGFeatureDesc::render_width, D_NA(ngx, DLSSGFeatureDesc, render_width))
        .def_rw("render_height", &DLSSGFeatureDesc::render_height, D_NA(ngx, DLSSGFeatureDesc, render_height))
        .def_rw(
            "backbuffer_format",
            &DLSSGFeatureDesc::backbuffer_format,
            D_NA(ngx, DLSSGFeatureDesc, backbuffer_format)
        );

    nb::class_<DLSSGCameraDesc>(ngx_module, "DLSSGCameraDesc", D_NA(ngx, DLSSGCameraDesc))
        .def(nb::init<>())
        .def_rw(
            "camera_view_to_clip",
            &DLSSGCameraDesc::camera_view_to_clip,
            D_NA(ngx, DLSSGCameraDesc, camera_view_to_clip)
        )
        .def_rw(
            "clip_to_camera_view",
            &DLSSGCameraDesc::clip_to_camera_view,
            D_NA(ngx, DLSSGCameraDesc, clip_to_camera_view)
        )
        .def_rw("clip_to_lens_clip", &DLSSGCameraDesc::clip_to_lens_clip, D_NA(ngx, DLSSGCameraDesc, clip_to_lens_clip))
        .def_rw("clip_to_prev_clip", &DLSSGCameraDesc::clip_to_prev_clip, D_NA(ngx, DLSSGCameraDesc, clip_to_prev_clip))
        .def_rw("prev_clip_to_clip", &DLSSGCameraDesc::prev_clip_to_clip, D_NA(ngx, DLSSGCameraDesc, prev_clip_to_clip))
        .def_rw("jitter_offset", &DLSSGCameraDesc::jitter_offset, D_NA(ngx, DLSSGCameraDesc, jitter_offset))
        .def_rw(
            "motion_vector_scale",
            &DLSSGCameraDesc::motion_vector_scale,
            D_NA(ngx, DLSSGCameraDesc, motion_vector_scale)
        )
        .def_rw(
            "camera_pinhole_offset",
            &DLSSGCameraDesc::camera_pinhole_offset,
            D_NA(ngx, DLSSGCameraDesc, camera_pinhole_offset)
        )
        .def_rw("camera_pos", &DLSSGCameraDesc::camera_pos, D_NA(ngx, DLSSGCameraDesc, camera_pos))
        .def_rw("camera_up", &DLSSGCameraDesc::camera_up, D_NA(ngx, DLSSGCameraDesc, camera_up))
        .def_rw("camera_right", &DLSSGCameraDesc::camera_right, D_NA(ngx, DLSSGCameraDesc, camera_right))
        .def_rw("camera_fwd", &DLSSGCameraDesc::camera_fwd, D_NA(ngx, DLSSGCameraDesc, camera_fwd))
        .def_rw("camera_near", &DLSSGCameraDesc::camera_near, D_NA(ngx, DLSSGCameraDesc, camera_near))
        .def_rw("camera_far", &DLSSGCameraDesc::camera_far, D_NA(ngx, DLSSGCameraDesc, camera_far))
        .def_rw("camera_fov", &DLSSGCameraDesc::camera_fov, D_NA(ngx, DLSSGCameraDesc, camera_fov))
        .def_rw(
            "camera_aspect_ratio",
            &DLSSGCameraDesc::camera_aspect_ratio,
            D_NA(ngx, DLSSGCameraDesc, camera_aspect_ratio)
        )
        .def_rw("color_buffers_hdr", &DLSSGCameraDesc::color_buffers_hdr, D_NA(ngx, DLSSGCameraDesc, color_buffers_hdr))
        .def_rw("depth_inverted", &DLSSGCameraDesc::depth_inverted, D_NA(ngx, DLSSGCameraDesc, depth_inverted))
        .def_rw(
            "camera_motion_included",
            &DLSSGCameraDesc::camera_motion_included,
            D_NA(ngx, DLSSGCameraDesc, camera_motion_included)
        )
        .def_rw("reset", &DLSSGCameraDesc::reset, D_NA(ngx, DLSSGCameraDesc, reset))
        .def_rw(
            "automode_override_reset",
            &DLSSGCameraDesc::automode_override_reset,
            D_NA(ngx, DLSSGCameraDesc, automode_override_reset)
        )
        .def_rw(
            "not_rendering_game_frames",
            &DLSSGCameraDesc::not_rendering_game_frames,
            D_NA(ngx, DLSSGCameraDesc, not_rendering_game_frames)
        )
        .def_rw("ortho_projection", &DLSSGCameraDesc::ortho_projection, D_NA(ngx, DLSSGCameraDesc, ortho_projection))
        .def_rw(
            "motion_vectors_invalid_value",
            &DLSSGCameraDesc::motion_vectors_invalid_value,
            D_NA(ngx, DLSSGCameraDesc, motion_vectors_invalid_value)
        )
        .def_rw(
            "motion_vectors_dilated",
            &DLSSGCameraDesc::motion_vectors_dilated,
            D_NA(ngx, DLSSGCameraDesc, motion_vectors_dilated)
        )
        .def_rw(
            "menu_detection_enabled",
            &DLSSGCameraDesc::menu_detection_enabled,
            D_NA(ngx, DLSSGCameraDesc, menu_detection_enabled)
        )
        .def_rw("multi_frame_count", &DLSSGCameraDesc::multi_frame_count, D_NA(ngx, DLSSGCameraDesc, multi_frame_count))
        .def_rw(
            "multi_frame_index",
            &DLSSGCameraDesc::multi_frame_index,
            D_NA(ngx, DLSSGCameraDesc, multi_frame_index)
        );

    nb::class_<DLSSGEvaluateDesc>(ngx_module, "DLSSGEvaluateDesc", D_NA(ngx, DLSSGEvaluateDesc))
        .def(nb::init<>())
        .def_rw("backbuffer", &DLSSGEvaluateDesc::backbuffer, D_NA(ngx, DLSSGEvaluateDesc, backbuffer))
        .def_rw("depth", &DLSSGEvaluateDesc::depth, D_NA(ngx, DLSSGEvaluateDesc, depth))
        .def_rw("motion_vectors", &DLSSGEvaluateDesc::motion_vectors, D_NA(ngx, DLSSGEvaluateDesc, motion_vectors))
        .def_rw(
            "output_interpolated_frame",
            &DLSSGEvaluateDesc::output_interpolated_frame,
            D_NA(ngx, DLSSGEvaluateDesc, output_interpolated_frame)
        )
        .def_rw("camera", &DLSSGEvaluateDesc::camera, D_NA(ngx, DLSSGEvaluateDesc, camera));
}
