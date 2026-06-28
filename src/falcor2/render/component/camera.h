// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/component.h"
#include "falcor2/core/reflection.h"
#include "falcor2/utils/idictionary.h"

#include <sgl/device/shader_cursor.h>

namespace falcor {

struct FALCOR_API CameraUniforms {
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

    /// Viewport width in pixels.
    int width() const { return static_cast<int>(dims.x); }

    /// Viewport height in pixels.
    int height() const { return static_cast<int>(dims.y); }

    /// Write camera uniforms to a dictionary.
    void to_dictionary(IDictionary& dict) const;

    /// Bind camera uniforms to a shader cursor.
    void bind(sgl::ShaderCursor cursor) const;
};

class FALCOR_API Camera : public Component {
    FALCOR_SCENE_OBJECT(Camera, Component)
public:
    ~Camera() override;

    /// Viewport width in pixels.
    int width() const { return m_width; }
    void set_width(int width);

    /// Viewport height in pixels.
    int height() const { return m_height; }
    void set_height(int height);

    /// Vertical field of view in degrees.
    float fov_y() const { return m_fov_y; }
    void set_fov_y(float degrees);

    /// Focal length in mm.
    float focal_length() const { return m_focal_length; }
    void set_focal_length(float length);

    /// Focus distance in world units.
    float focus_distance() const { return m_focus_distance; }
    void set_focus_distance(float distance);

    /// f-stop value.
    float fstop() const { return m_fstop; }
    void set_fstop(float fstop);

    /// Recompute camera basis vectors from the entity's world transform.
    void recompute() const;

    /// Calculate camera uniforms for the camera's viewport dimensions.
    CameraUniforms calc_uniforms() const;

    /// Calculate camera uniforms for the given viewport dimensions.
    CameraUniforms calc_uniforms(int width, int height) const;

    /// Calculate the view-from-world matrix.
    float4x4 calc_view_from_world() const;

    /// Calculate the clip-from-view projection matrix.
    float4x4 calc_clip_from_view() const;

    /// Calculate the clip-from-view projection matrix for the given viewport dimensions.
    float4x4 calc_clip_from_view(int width, int height) const;

    /// Write camera uniforms to a dictionary.
    void to_dictionary(IDictionary& dict);

    /// Bind camera uniforms to a shader cursor.
    void bind(sgl::ShaderCursor cursor) const;

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
                reflection::value_range(1.0, 500.0),
                reflection::ui_label("Focal Length")
            )
            .def_prop_rw(
                "focus_distance",
                &Camera::focus_distance,
                &Camera::set_focus_distance,
                "Focus distance in world units.",
                reflection::value_range(0.01, 10000.0),
                reflection::ui_label("Focus Distance")
            )
            .def_prop_rw(
                "fstop",
                &Camera::fstop,
                &Camera::set_fstop,
                "F-Stop value.",
                reflection::value_range(0.5, 128.0),
                reflection::ui_label("F-Stop")
            )
            .def_prop_rw(
                "fov_y",
                &Camera::fov_y,
                &Camera::set_fov_y,
                "Vertical field of view in degrees.",
                reflection::value_range(1.0, 179.0),
                reflection::ui_label("Vertical FOV")
            )
            .def_prop_rw(
                "width",
                &Camera::width,
                &Camera::set_width,
                "Viewport width in pixels.",
                reflection::value_range(1, 16384),
                reflection::ui_label("Width")
            )
            .def_prop_rw(
                "height",
                &Camera::height,
                &Camera::set_height,
                "Viewport height in pixels.",
                reflection::value_range(1, 16384),
                reflection::ui_label("Height")
            );
    }

private:
    void needs_recompute() { m_needs_recompute = true; }

    /// Viewport width in pixels.
    int m_width{1280};
    /// Viewport height in pixels.
    int m_height{720};
    /// Vertical field of view in degrees.
    float m_fov_y{70.0f};

    /// Focal length in mm.
    float m_focal_length{0.f};
    /// Focus distance in world units.
    float m_focus_distance{0.f};
    /// f-stop value.
    float m_fstop{0.f};

    /// Flag indicating if the computed values need to be recomputed.
    mutable bool m_needs_recompute{true};

    /// Computed camera uniforms for the camera's viewport dimensions.
    mutable CameraUniforms m_uniforms;
};

} // namespace falcor
