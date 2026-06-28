// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "camera.h"

#include "falcor2/render/entity.h"

#include <sgl/math/matrix_math.h>

#include <cmath>

namespace falcor {

void CameraUniforms::to_dictionary(IDictionary& dict) const
{
    dict["dims"] = dims;
    dict["position"] = position;
    dict["image_u"] = image_u;
    dict["image_v"] = image_v;
    dict["image_w"] = image_w;
}

void CameraUniforms::bind(sgl::ShaderCursor cursor) const
{
    cursor["dims"] = dims;
    cursor["position"] = position;
    cursor["image_u"] = image_u;
    cursor["image_v"] = image_v;
    cursor["image_w"] = image_w;
}

Camera::~Camera() { }

void Camera::set_width(int width)
{
    if (width != m_width) {
        m_width = width;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_height(int height)
{
    if (height != m_height) {
        m_height = height;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_fov_y(float degrees)
{
    if (degrees != m_fov_y) {
        m_fov_y = degrees;
        needs_recompute();
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_focal_length(float length)
{
    if (length != m_focal_length) {
        m_focal_length = length;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_focus_distance(float distance)
{
    if (distance != m_focus_distance) {
        m_focus_distance = distance;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::set_fstop(float fstop)
{
    if (fstop != m_fstop) {
        m_fstop = fstop;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Camera::recompute() const
{
    if (!m_needs_recompute)
        return;

    float aspect_ratio = static_cast<float>(m_width) / static_cast<float>(m_height);
    float fov_rad = m_fov_y * (static_cast<float>(M_PI) / 180.0f);
    float half_tan = std::tan(fov_rad * 0.5f);

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
    float fov_rad = m_fov_y * (static_cast<float>(M_PI) / 180.0f);
    return sgl::math::perspective(fov_rad, aspect_ratio, 0.1f, 1000.0f);
}

void Camera::to_dictionary(IDictionary& dict)
{
    recompute();
    m_uniforms.to_dictionary(dict);
}

void Camera::bind(sgl::ShaderCursor cursor) const
{
    recompute();
    m_uniforms.bind(cursor);
}

void Camera::on_entity_transform_changed()
{
    needs_recompute();
    mark_dirty(DirtyFlags::render_state);
}

FALCOR_SCENE_REGISTER_COMPONENT(Camera);

} // namespace falcor
