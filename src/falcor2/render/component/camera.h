// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/render/component.h"
#include "falcor2/core/reflection.h"

#include <sgl/device/shader_cursor.h>

namespace falcor {

struct FALCOR_API CameraUniforms {
    FALCOR_STATIC_WRITE_TO_CURSOR(CameraUniforms);

    /// Viewport dimensions in pixels.
    uint2 dims{0, 0};
    /// Computed camera position (world space).
    float3 position{0.f};
    /// Computed image_u vector (right direction scaled by FOV and aspect ratio).
    float3 image_u{0.f};
    /// Computed image_v vector (up direction scaled by FOV).
    float3 image_v{0.f};
    /// Computed image_w vector (forward direction).
    float3 image_w{0.f};
    /// Thin-lens aperture radius in world units. Zero selects the pinhole model.
    float aperture_radius{0.f};
    /// Focus-plane distance along image_w in world units. Zero selects the pinhole model.
    float focus_distance{0.f};

    /// Viewport width in pixels.
    int width() const { return static_cast<int>(dims.x); }

    /// Viewport height in pixels.
    int height() const { return static_cast<int>(dims.y); }

    bool operator==(const CameraUniforms&) const = default;

    /// Write camera uniforms to a cursor.
    template<typename TCursor>
    void write_to_cursor(TCursor cursor) const
    {
        cursor["dims"] = dims;
        cursor["position"] = position;
        cursor["image_u"] = image_u;
        cursor["image_v"] = image_v;
        cursor["image_w"] = image_w;
        cursor["aperture_radius"] = aperture_radius;
        cursor["focus_distance"] = focus_distance;
    }
};

/// Perspective camera using a vertical film-fit model.
///
/// Focal length and sensor height determine the vertical field of view. The
/// viewport aspect ratio determines the corresponding horizontal field of view.
class FALCOR_API Camera : public Component {
    FALCOR_SCENE_OBJECT(Camera, Component)
public:
    FALCOR_STATIC_WRITE_TO_CURSOR(Camera);

    ~Camera() override;

    /// Focal length in mm.
    float focal_length() const { return m_focal_length; }
    void set_focal_length(float length);

    /// f-stop value.
    float fstop() const { return m_fstop; }
    void set_fstop(float fstop);

    /// Physical vertical sensor aperture in mm.
    float sensor_height() const { return m_sensor_height; }
    void set_sensor_height(float height);

    /// Vertical field of view in degrees, derived from focal length and sensor height.
    /// Setting this preserves sensor height and adjusts focal length.
    float fov_y() const;
    void set_fov_y(float degrees);

    /// Whether thin-lens depth of field is enabled.
    bool enable_depth_of_field() const { return m_enable_depth_of_field; }
    void set_enable_depth_of_field(bool enabled);

    /// Focus distance in world units.
    float focus_distance() const { return m_focus_distance; }
    void set_focus_distance(float distance);

    /// Depth range (near, far) in world units.
    float2 depth_range() const { return m_depth_range; }
    void set_depth_range(float2 depth_range);

    /// Viewport width in pixels.
    int width() const { return m_width; }
    void set_width(int width);

    /// Viewport height in pixels.
    int height() const { return m_height; }
    void set_height(int height);

    /// Recompute camera basis vectors from the entity's world transform.
    void recompute() const;

    /// Calculate camera uniforms for the camera's viewport dimensions.
    CameraUniforms calc_uniforms() const;

    /// Calculate camera uniforms for the given viewport dimensions.
    /// The vertical field of view is preserved while the horizontal field of view follows the viewport aspect ratio.
    CameraUniforms calc_uniforms(int width, int height) const;

    /// Calculate the view-from-world matrix.
    float4x4 calc_view_from_world() const;

    /// Calculate the clip-from-view projection matrix.
    float4x4 calc_clip_from_view() const;

    /// Calculate the clip-from-view projection matrix for the given viewport dimensions.
    float4x4 calc_clip_from_view(int width, int height) const;

    /// Write camera uniforms to a cursor.
    template<typename TCursor>
    void write_to_cursor(TCursor cursor) const
    {
        recompute();
        m_uniforms.write_to_cursor(cursor);
    }

    // Component interface

    void on_entity_transform_changed() override;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_prop_rw(
                "focal_length",
                &Camera::focal_length,
                &Camera::set_focal_length,
                "Focal length in mm.",
                reflection::ui_label("Focal Length")
            )
            .def_prop_rw(
                "fstop",
                &Camera::fstop,
                &Camera::set_fstop,
                "F-Stop value.",
                reflection::value_range(MIN_FSTOP, MAX_FSTOP),
                reflection::ui_label("F-Stop")
            )
            .def_prop_rw(
                "sensor_height",
                &Camera::sensor_height,
                &Camera::set_sensor_height,
                "Sensor height in mm.",
                reflection::ui_label("Sensor Height")
            )
            .def_prop_rw(
                "fov_y",
                &Camera::fov_y,
                &Camera::set_fov_y,
                "Vertical field of view in degrees.",
                reflection::value_range(MIN_FOV_Y, MAX_FOV_Y),
                reflection::ui_label("Vertical FOV")
            )
            .def_prop_rw(
                "enable_depth_of_field",
                &Camera::enable_depth_of_field,
                &Camera::set_enable_depth_of_field,
                "Whether thin-lens depth of field is enabled.",
                reflection::ui_label("Depth of Field")
            )
            .def_prop_rw(
                "focus_distance",
                &Camera::focus_distance,
                &Camera::set_focus_distance,
                "Focus distance in world units.",
                reflection::value_range_positive(),
                reflection::ui_label("Focus Distance"),
                reflection::ui_enable_if<Camera>(
                    [](const Camera& camera)
                    {
                        return camera.enable_depth_of_field();
                    }
                )
            )
            .def_prop_rw(
                "depth_range",
                &Camera::depth_range,
                &Camera::set_depth_range,
                "Depth range (near, far) in world units.",
                reflection::ui_label("Depth Range")
            )
            .def_prop_rw(
                "width",
                &Camera::width,
                &Camera::set_width,
                "Viewport width in pixels.",
                reflection::value_range(MIN_VIEWPORT_DIMENSION, MAX_VIEWPORT_DIMENSION),
                reflection::ui_label("Width")
            )
            .def_prop_rw(
                "height",
                &Camera::height,
                &Camera::set_height,
                "Viewport height in pixels.",
                reflection::value_range(MIN_VIEWPORT_DIMENSION, MAX_VIEWPORT_DIMENSION),
                reflection::ui_label("Height")
            );
    }

private:
    static constexpr int MIN_VIEWPORT_DIMENSION = 1;
    static constexpr int MAX_VIEWPORT_DIMENSION = 16384;
    static constexpr float MIN_FOV_Y = 0.f;
    static constexpr float MAX_FOV_Y = 180.f;
    static constexpr float MIN_FSTOP = 0.5f;
    static constexpr float MAX_FSTOP = 128.f;

    void needs_recompute() { m_needs_recompute = true; }

    /// Focal length in mm.
    float m_focal_length{50.f};
    /// f-stop value.
    float m_fstop{8.f};
    /// Sensor height in mm.
    float m_sensor_height{24.f};
    /// Whether thin-lens depth of field is enabled.
    bool m_enable_depth_of_field{false};
    /// Focus distance in world units.
    float m_focus_distance{1.f};
    /// Depth range (near, far) in world units.
    float2 m_depth_range{0.1f, 1000.f};
    /// Viewport width in pixels.
    int m_width{1280};
    /// Viewport height in pixels.
    int m_height{720};

    /// Flag indicating if the computed values need to be recomputed.
    mutable bool m_needs_recompute{true};

    /// Computed camera uniforms for the camera's viewport dimensions.
    mutable CameraUniforms m_uniforms;
};

} // namespace falcor
