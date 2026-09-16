// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera.h"

#include "falcor2/render/entity.h"

#include <sgl/device/cursor_utils.h>
#include <sgl/math/matrix_math.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace falcor {

namespace {

float focal_length_from_fov_y(float fov_y, float sensor_height)
{
    return sensor_height / (2.f * std::tan(sgl::math::radians(fov_y) * 0.5f));
}

float fov_y_from_focal_length(float focal_length, float sensor_height)
{
    return sgl::math::degrees(2.f * std::atan(sensor_height / (2.f * focal_length)));
}

} // namespace

Camera::~Camera() { }

void Camera::set_focal_length(float length)
{
    length = std::clamp(length, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    if (length != m_focal_length) {
        m_focal_length = length;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_fstop(float fstop)
{
    fstop = std::clamp(fstop, MIN_FSTOP, MAX_FSTOP);
    if (fstop != m_fstop) {
        m_fstop = fstop;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_sensor_height(float height)
{
    height = std::clamp(height, std::numeric_limits<float>::min(), std::numeric_limits<float>::max());
    if (height != m_sensor_height) {
        m_sensor_height = height;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_fov_y(float degrees)
{
    // Keep the physical conversion away from the singular endpoints. The closest
    // float above zero underflows when converted from degrees to radians.
    degrees = std::clamp(degrees, std::numeric_limits<float>::epsilon(), std::nextafter(MAX_FOV_Y, MIN_FOV_Y));
    set_focal_length(focal_length_from_fov_y(degrees, m_sensor_height));
}

float Camera::fov_y() const
{
    return fov_y_from_focal_length(m_focal_length, m_sensor_height);
}

void Camera::set_enable_depth_of_field(bool enabled)
{
    if (enabled != m_enable_depth_of_field) {
        m_enable_depth_of_field = enabled;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_focus_distance(float distance)
{
    distance = std::max(distance, 0.f);
    if (distance != m_focus_distance) {
        m_focus_distance = distance;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_depth_range(float2 depth_range)
{
    depth_range.x = std::max(depth_range.x, std::numeric_limits<float>::min());
    depth_range.y = std::max(depth_range.y, std::nextafter(depth_range.x, std::numeric_limits<float>::infinity()));
    if (depth_range != m_depth_range) {
        m_depth_range = depth_range;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_width(int width)
{
    width = std::clamp(width, MIN_VIEWPORT_DIMENSION, MAX_VIEWPORT_DIMENSION);
    if (width != m_width) {
        m_width = width;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_height(int height)
{
    height = std::clamp(height, MIN_VIEWPORT_DIMENSION, MAX_VIEWPORT_DIMENSION);
    if (height != m_height) {
        m_height = height;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::recompute() const
{
    if (!m_needs_recompute)
        return;

    float aspect_ratio = static_cast<float>(m_width) / static_cast<float>(m_height);
    float half_tan = m_sensor_height / (2.f * m_focal_length);

    float3 pos{0.f};
    float3 right{1.f, 0.f, 0.f};
    float3 up{0.f, 1.f, 0.f};
    float3 fwd{0.f, 0.f, -1.f};

    if (m_entity) {
        // Extract position and orientation from entity world transform.
        // Columns: 0=right, 1=up, 2=forward(+Z), 3=translation.
        float4x4 world = m_entity->world_from_object_matrix();
        pos = world.get_col(3).xyz();
        right = sgl::math::normalize(world.get_col(0).xyz());
        up = sgl::math::normalize(world.get_col(1).xyz());
        fwd = -sgl::math::normalize(world.get_col(2).xyz());
    }

    m_uniforms.dims = uint2(m_width, m_height);
    m_uniforms.position = pos;
    m_uniforms.image_u = right * half_tan * aspect_ratio;
    m_uniforms.image_v = up * half_tan;
    m_uniforms.image_w = fwd;
    m_uniforms.aperture_radius = m_enable_depth_of_field ? 0.5f * m_focal_length * 0.001f / m_fstop : 0.f;
    m_uniforms.focus_distance = m_focus_distance;

    m_needs_recompute = false;
}

CameraUniforms Camera::calc_uniforms() const
{
    return calc_uniforms(m_width, m_height);
}

CameraUniforms Camera::calc_uniforms(int width, int height) const
{
    FALCOR_ASSERT_GT(width, 0);
    FALCOR_ASSERT_GT(height, 0);
    recompute();

    CameraUniforms uniforms = m_uniforms;
    uniforms.dims = uint2(width, height);
    if (width != m_width || height != m_height) {
        float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
        float image_v_len = sgl::math::length(m_uniforms.image_v);
        float3 right = sgl::math::normalize(m_uniforms.image_u);
        uniforms.image_u = right * image_v_len * aspect_ratio;
    }
    return uniforms;
}

float4x4 Camera::calc_view_from_world() const
{
    recompute();
    float3 up = sgl::math::normalize(m_uniforms.image_v);
    return sgl::math::matrix_from_look_at(m_uniforms.position, m_uniforms.position + m_uniforms.image_w, up);
}

float4x4 Camera::calc_clip_from_view() const
{
    return calc_clip_from_view(m_width, m_height);
}

float4x4 Camera::calc_clip_from_view(int width, int height) const
{
    FALCOR_ASSERT_GT(width, 0);
    FALCOR_ASSERT_GT(height, 0);
    recompute();
    float aspect_ratio = static_cast<float>(width) / static_cast<float>(height);
    float fov_rad = sgl::math::radians(fov_y());
    return sgl::math::perspective(fov_rad, aspect_ratio, m_depth_range.x, m_depth_range.y);
}

void Camera::on_entity_transform_changed()
{
    needs_recompute();
    mark_dirty(DirtyFlags::render_state);
}

FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<CameraUniforms>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<Camera>());
FALCOR_SCENE_REGISTER_COMPONENT(Camera);

} // namespace falcor
