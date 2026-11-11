// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/entity.h"
#include "falcor2/utils/math/transform.h"

#include <sgl/device/kernel.h>
#include <sgl/device/sampler.h>
#include <sgl/core/signature_buffer.h>
#include <sgl/device/cursor_utils.h>

#include <algorithm>
#include <cmath>

namespace falcor {

// ----------------------------------------------------------------------------
// Light
// ----------------------------------------------------------------------------

Light::Light() { }

Light::~Light() { }

void Light::on_entity_transform_changed()
{
    mark_dirty(DirtyFlags::render_state);
}

void Light::set_active(bool active)
{
    if (active != m_active) {
        m_active = active;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_color(float3 color)
{
    if (color != m_color) {
        m_color = color;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_intensity(float intensity)
{
    intensity = std::max(intensity, 0.f);
    if (intensity != m_intensity) {
        m_intensity = intensity;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_exposure(float exposure)
{
    if (exposure != m_exposure) {
        m_exposure = exposure;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_normalize(bool normalize)
{
    if (normalize != m_normalize) {
        m_normalize = normalize;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_enable_color_temperature(bool enable_color_temperature)
{
    if (enable_color_temperature != m_enable_color_temperature) {
        m_enable_color_temperature = enable_color_temperature;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_color_temperature(float color_temperature)
{
    color_temperature = std::clamp(color_temperature, MIN_COLOR_TEMPERATURE, MAX_COLOR_TEMPERATURE);
    if (color_temperature != m_color_temperature) {
        m_color_temperature = color_temperature;
        mark_dirty(DirtyFlags::render_state);
    }
}

void Light::set_geometry_visible(bool visible)
{
    if (visible != m_geometry_visible) {
        m_geometry_visible = visible;
        mark_dirty(DirtyFlags::render_state);
    }
}

float3 Light::evaluate_emission(float size_factor) const
{
    float3 value = m_color * m_intensity;
    if (m_enable_color_temperature)
        value *= color_temperature_to_rgb(m_color_temperature);
    value *= sgl::math::exp2(m_exposure);
    if (m_normalize && size_factor > 0.f && sgl::math::isfinite(size_factor))
        value /= size_factor;
    return value;
}

void Light::write_slangpy_signature(sgl::SignatureBuffer& signature, const Light* value)
{
    signature.add("falcor2.Light:");
    signature.add(value ? value->slang_type_name() : "<null>");
}

FALCOR_STATIC_ONCE(reflection::register_type<Light>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<Light>());

// ----------------------------------------------------------------------------
// ConstantLight
// ----------------------------------------------------------------------------

ConstantLight::ConstantLight()
{
    set_slang_type_name("ConstantLight");
    set_light_type(shared::LightType::constant_light);
    set_light_flags(shared::LightFlags::environment);
}

ConstantLight::~ConstantLight() { }

template<typename CursorT>
void ConstantLight::write_to_cursor_impl(CursorT cursor) const
{
    cursor["radiance"] = evaluate_emission();
}

template void ConstantLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void ConstantLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(ConstantLight);

// ----------------------------------------------------------------------------
// DistantLight
// ----------------------------------------------------------------------------

DistantLight::DistantLight()
{
    set_slang_type_name("DistantLight");
    set_light_type(shared::LightType::distant_light);
    set_light_flags(shared::LightFlags::environment);
}

DistantLight::~DistantLight() { }

void DistantLight::set_cutoff_angle(float cutoff_angle)
{
    cutoff_angle = sgl::math::clamp(cutoff_angle, MIN_LIGHT_ANGLE, MAX_LIGHT_ANGLE);
    if (cutoff_angle != m_cutoff_angle) {
        m_cutoff_angle = cutoff_angle;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void DistantLight::write_to_cursor_impl(CursorT cursor) const
{
    const float half_angle = sgl::math::radians(m_cutoff_angle);
    const float sin_half_angle = sgl::math::sin(half_angle);
    const float sin_squared = sin_half_angle * sin_half_angle;
    const float size_factor = half_angle == 0.f
        ? 1.f
        : static_cast<float>(M_PI) * (half_angle <= 0.5f * static_cast<float>(M_PI) ? sin_squared : 2.f - sin_squared);
    cursor["world_from_object"] = float3x3(m_entity->world_from_object_matrix());
    cursor["radiance"] = evaluate_emission(size_factor);
    cursor["cos_cutoff_angle"] = sgl::math::cos(sgl::math::radians(m_cutoff_angle));
}

template void DistantLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void DistantLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(DistantLight);

// ----------------------------------------------------------------------------
// EnvMapLight
// ----------------------------------------------------------------------------

EnvMapLight::EnvMapLight()
{
    set_slang_type_name("EnvMapLight");
    set_light_type(shared::LightType::env_map_light);
    set_light_flags(shared::LightFlags::environment);
}

EnvMapLight::~EnvMapLight() { }

void EnvMapLight::set_env_map_path(const std::filesystem::path& env_map_path)
{
    if (env_map_path != m_env_map_path || m_env_map_texture) {
        m_env_map_path = env_map_path;
        m_env_map_texture.reset();
        m_env_map_loaded = false;
        mark_dirty(DirtyFlags::render_state | DirtyFlags::resources);
    }
}

void EnvMapLight::set_env_map_texture(ref<sgl::Texture> env_map_texture)
{
    if (env_map_texture != m_env_map_texture) {
        m_env_map_path.clear();
        m_env_map_texture = std::move(env_map_texture);
        m_env_map_loaded = false;
        mark_dirty(DirtyFlags::render_state | DirtyFlags::resources);
    }
}

void EnvMapLight::on_load_resources()
{
    if (!m_env_map_loaded) {
        if (m_env_map_texture) {
            m_env_map_texture_handle = m_scene->texture_manager()->register_texture({.texture = m_env_map_texture});
        } else {
            m_env_map_texture_handle = m_scene->texture_manager()->load_texture({
                .path = m_env_map_path,
                .flags = TextureLoadFlags::environment,
                .load_deferred = true,
            });
        }
        m_env_map_loaded = true;
        m_importance_map_dirty = true;
    }
}

void EnvMapLight::set_pole_axis(EnvMapPoleAxis pole_axis)
{
    FALCOR_CHECK(pole_axis == EnvMapPoleAxis::y || pole_axis == EnvMapPoleAxis::z, "Invalid environment pole axis.");
    if (pole_axis != m_pole_axis) {
        m_pole_axis = pole_axis;
        mark_dirty(DirtyFlags::render_state);
    }
}

void EnvMapLight::update(SceneUpdateContext& ctx)
{
    if (m_importance_map_dirty) {
        if (!m_build_importance_map_kernel) {
            m_build_importance_map_kernel = m_scene->device()->create_compute_kernel({
                .program = m_scene->device()
                               ->load_program("falcor2/render/lights/env_map_kernels.slang", {"build_importance_map"}),
            });
        }

        if (!m_importance_map_texture) {
            m_importance_map_texture = m_scene->device()->create_texture({
                .format = sgl::Format::r32_float,
                .width = 512,
                .height = 512,
                .mip_count = sgl::ALL_MIPS,
                .usage = sgl::TextureUsage::shader_resource | sgl::TextureUsage::unordered_access,
            });
        }

        m_build_importance_map_kernel->dispatch(
            uint3(m_importance_map_texture->width(), m_importance_map_texture->height(), 1),
            [&](sgl::ShaderCursor cursor)
            {
                cursor = cursor.find_entry_point(0);
                cursor["env_map"] = m_env_map_texture_handle.texture();
                cursor["env_map_sampler"] = m_env_map_texture_handle.sampler();
                cursor["importance_map"] = m_importance_map_texture;
                cursor["sample_count"] = 16;
            },
            ctx.command_encoder()
        );

        ctx.command_encoder()->generate_mips(m_importance_map_texture);
        // Bindless reads do not trigger automatic state tracking. Make every mip readable,
        // including the last mip that generate_mips leaves as an unordered-access output.
        ctx.command_encoder()->set_texture_state(m_importance_map_texture, sgl::ResourceState::shader_resource);

        m_importance_map_dirty = false;
    }
}

template<typename CursorT>
void EnvMapLight::write_to_cursor_impl(CursorT cursor) const
{
    float3x3 world_from_object(m_entity->world_from_object_matrix());
    if (m_pole_axis == EnvMapPoleAxis::z) {
        const float3x3 local_from_object(sgl::math::matrix_from_rotation_x(sgl::math::radians(90.f)));
        world_from_object = mul(world_from_object, local_from_object);
    }
    cursor["world_from_object"] = world_from_object;
    cursor["scale"] = evaluate_emission();
    cursor["env_map_texture_handle"] = m_env_map_texture_handle;
    cursor["importance_map_texture"] = m_importance_map_texture->descriptor_handle_ro();
}

template void EnvMapLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void EnvMapLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(EnvMapLight);

// ----------------------------------------------------------------------------
// LightShaping
// ----------------------------------------------------------------------------

bool LightShaping::set_enabled(bool enabled)
{
    if (enabled == m_enabled)
        return false;
    m_enabled = enabled;
    return true;
}

bool LightShaping::set_cone_angle(float angle)
{
    angle = sgl::math::clamp(angle, MIN_CONE_ANGLE, MAX_CONE_ANGLE);
    if (angle == m_cone_angle)
        return false;
    m_cone_angle = angle;
    return true;
}

bool LightShaping::set_cone_softness(float softness)
{
    softness = sgl::math::clamp(softness, 0.f, 1.f);
    if (softness == m_cone_softness)
        return false;
    m_cone_softness = softness;
    return true;
}

bool LightShaping::set_focus(float focus)
{
    focus = std::max(focus, 0.f);
    if (focus == m_focus)
        return false;
    m_focus = focus;
    return true;
}

template<typename CursorT>
void LightShaping::write_to_cursor(CursorT cursor) const
{
    const float cutoff_angle = sgl::math::radians(m_cone_angle);
    const float falloff_angle = cutoff_angle * (1.f - m_cone_softness);
    const float angular_falloff_width = cutoff_angle - falloff_angle;
    const float angular_falloff_scale = angular_falloff_width > 0.f ? 1.f / angular_falloff_width : 0.f;
    cursor["enabled"] = m_enabled;
    cursor["cos_cutoff_angle"] = sgl::math::cos(cutoff_angle);
    cursor["cos_falloff_angle"] = sgl::math::cos(falloff_angle);
    cursor["focus"] = m_focus;
    cursor["cutoff_angle"] = cutoff_angle;
    cursor["angular_falloff_scale"] = angular_falloff_scale;
}

// ----------------------------------------------------------------------------
// PointLight
// ----------------------------------------------------------------------------

PointLight::PointLight()
{
    set_slang_type_name("PointLight");
    set_light_type(shared::LightType::point_light);
    set_light_flags(shared::LightFlags::delta_position);
}

PointLight::~PointLight() { }

template<typename CursorT>
void PointLight::write_to_cursor_impl(CursorT cursor) const
{
    float4x4 matrix = m_entity->world_from_object_matrix();
    cursor["pos_ws"] = matrix.get_col(3).xyz();
    cursor["dir_ws"] = -sgl::math::normalize(matrix.get_col(2).xyz());
    cursor["radiant_intensity"] = evaluate_emission();
    m_shaping.write_to_cursor(cursor["shaping"]);
}

template void PointLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void PointLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(PointLight);

// ----------------------------------------------------------------------------
// SphereLight
// ----------------------------------------------------------------------------

SphereLight::SphereLight()
{
    set_slang_type_name("SphereLight");
    set_light_type(shared::LightType::sphere_light);
    set_light_flags(shared::LightFlags::area);
}

SphereLight::~SphereLight() { }

void SphereLight::set_radius(float radius)
{
    radius = std::max(radius, 0.f);
    if (radius != m_radius) {
        m_radius = radius;
        mark_dirty(DirtyFlags::render_state);
    }
}

void SphereLight::set_enable_virtual_sphere_shrinking(bool enable)
{
    if (enable != m_enable_virtual_sphere_shrinking) {
        m_enable_virtual_sphere_shrinking = enable;
        mark_dirty(DirtyFlags::render_state);
    }
}

float SphereLight::world_radius() const
{
    return m_radius * volume_equivalent_uniform_scale(m_entity->world_from_object_matrix());
}

template<typename CursorT>
void SphereLight::write_to_cursor_impl(CursorT cursor) const
{
    const float4x4 matrix = m_entity->world_from_object_matrix();
    const float radius_ws = world_radius();
    cursor["center_ws"] = matrix.get_col(3).xyz();
    cursor["radius_ws"] = radius_ws;
    cursor["dir_ws"] = -sgl::math::normalize(matrix.get_col(2).xyz());
    const float surface_area = 4.f * static_cast<float>(M_PI) * radius_ws * radius_ws;
    cursor["radiance"] = evaluate_emission(surface_area);
    cursor["surface_area"] = surface_area;
    cursor["enable_virtual_sphere_shrinking"] = m_enable_virtual_sphere_shrinking;
    m_shaping.write_to_cursor(cursor["shaping"]);
}

template void SphereLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void SphereLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(SphereLight);

// ----------------------------------------------------------------------------
// CylinderLight
// ----------------------------------------------------------------------------

CylinderLight::CylinderLight()
{
    set_slang_type_name("CylinderLight");
    set_light_type(shared::LightType::cylinder_light);
    set_light_flags(shared::LightFlags::area);
}

CylinderLight::~CylinderLight() { }

void CylinderLight::set_radius(float radius)
{
    radius = std::max(radius, 0.f);
    if (radius != m_radius) {
        m_radius = radius;
        mark_dirty(DirtyFlags::render_state);
    }
}

void CylinderLight::set_length(float length)
{
    length = std::max(length, 0.f);
    if (length != m_length) {
        m_length = length;
        mark_dirty(DirtyFlags::render_state);
    }
}

CylinderLightWorldGeometry CylinderLight::world_geometry() const
{
    const float4x4 matrix = m_entity->world_from_object_matrix();
    const float3 axis_vector = matrix.get_col(0).xyz();
    const float3 radial_y_vector = matrix.get_col(1).xyz();
    const float3 radial_z_vector = matrix.get_col(2).xyz();
    const float axis_scale = sgl::math::length(axis_vector);
    const float radial_area_scale = sgl::math::length(sgl::math::cross(radial_y_vector, radial_z_vector));
    const float radius_ws = m_radius * std::sqrt(radial_area_scale);
    const float half_length_ws = 0.5f * m_length * axis_scale;

    const float3 axis_ws = axis_scale > 0.f ? axis_vector / axis_scale : float3(1.f, 0.f, 0.f);
    float3 radial_v_ws = radial_z_vector - sgl::math::dot(radial_z_vector, axis_ws) * axis_ws;
    if (sgl::math::length(radial_v_ws) == 0.f) {
        const float3 fallback = std::abs(axis_ws.z) < 0.9f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
        radial_v_ws = fallback - sgl::math::dot(fallback, axis_ws) * axis_ws;
    }
    radial_v_ws = sgl::math::normalize(radial_v_ws);
    const float3 radial_u_ws = sgl::math::normalize(sgl::math::cross(radial_v_ws, axis_ws));
    radial_v_ws = sgl::math::normalize(sgl::math::cross(axis_ws, radial_u_ws));

    return {
        .center_ws = matrix.get_col(3).xyz(),
        .axis_ws = axis_ws,
        .radius_u_ws = radius_ws * radial_u_ws,
        .radius_v_ws = radius_ws * radial_v_ws,
        .half_length_ws = half_length_ws,
        .radius_ws = radius_ws,
    };
}

template<typename CursorT>
void CylinderLight::write_to_cursor_impl(CursorT cursor) const
{
    const CylinderLightWorldGeometry geometry = world_geometry();

    cursor["center_ws"] = geometry.center_ws;
    cursor["axis_ws"] = geometry.axis_ws;
    cursor["radius_u_ws"] = geometry.radius_u_ws;
    cursor["radius_v_ws"] = geometry.radius_v_ws;
    cursor["half_length_ws"] = geometry.half_length_ws;
    cursor["radius_ws"] = geometry.radius_ws;
    cursor["dir_ws"] = -geometry.radius_v_ws / std::max(geometry.radius_ws, 1e-20f);
    const float surface_area = 4.f * static_cast<float>(M_PI) * geometry.radius_ws * geometry.half_length_ws;
    cursor["radiance"] = evaluate_emission(surface_area);
    cursor["surface_area"] = surface_area;
    m_shaping.write_to_cursor(cursor["shaping"]);
}

template void CylinderLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void CylinderLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(CylinderLight);

// ----------------------------------------------------------------------------
// DiskLight
// ----------------------------------------------------------------------------

DiskLight::DiskLight()
{
    set_slang_type_name("DiskLight");
    set_light_type(shared::LightType::disk_light);
    set_light_flags(shared::LightFlags::area);
}

DiskLight::~DiskLight() { }

void DiskLight::set_radius(float radius)
{
    radius = std::max(radius, 0.f);
    if (radius != m_radius) {
        m_radius = radius;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void DiskLight::write_to_cursor_impl(CursorT cursor) const
{
    const float4x4 matrix = m_entity->world_from_object_matrix();
    const float3 radius_u_ws = m_radius * matrix.get_col(0).xyz();
    const float3 radius_v_ws = m_radius * matrix.get_col(1).xyz();
    const float4x4 world_from_object_it = sgl::math::transpose(sgl::math::inverse(matrix));
    cursor["center_ws"] = matrix.get_col(3).xyz();
    cursor["radius_u_ws"] = radius_u_ws;
    cursor["radius_v_ws"] = radius_v_ws;
    cursor["normal_ws"] = -sgl::math::normalize(world_from_object_it.get_col(2).xyz());
    const float surface_area = static_cast<float>(M_PI) * length(cross(radius_u_ws, radius_v_ws));
    cursor["radiance"] = evaluate_emission(surface_area);
    cursor["surface_area"] = surface_area;
    m_shaping.write_to_cursor(cursor["shaping"]);
}

template void DiskLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void DiskLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(DiskLight);

// ----------------------------------------------------------------------------
// RectLight
// ----------------------------------------------------------------------------

RectLight::RectLight()
{
    set_slang_type_name("RectLight");
    set_light_type(shared::LightType::rect_light);
    set_light_flags(shared::LightFlags::area);
}

RectLight::~RectLight() { }

void RectLight::set_width(float width)
{
    width = std::max(width, 0.f);
    if (width != m_width) {
        m_width = width;
        mark_dirty(DirtyFlags::render_state);
    }
}

void RectLight::set_height(float height)
{
    height = std::max(height, 0.f);
    if (height != m_height) {
        m_height = height;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void RectLight::write_to_cursor_impl(CursorT cursor) const
{
    const float4x4 matrix = m_entity->world_from_object_matrix();
    const float3 half_extent_u_ws = 0.5f * m_width * matrix.get_col(0).xyz();
    const float3 half_extent_v_ws = 0.5f * m_height * matrix.get_col(1).xyz();
    const float4x4 world_from_object_it = sgl::math::transpose(sgl::math::inverse(matrix));
    cursor["center_ws"] = matrix.get_col(3).xyz();
    cursor["half_extent_u_ws"] = half_extent_u_ws;
    cursor["half_extent_v_ws"] = half_extent_v_ws;
    cursor["normal_ws"] = -sgl::math::normalize(world_from_object_it.get_col(2).xyz());
    const float surface_area = 4.f * length(cross(half_extent_u_ws, half_extent_v_ws));
    cursor["radiance"] = evaluate_emission(surface_area);
    cursor["surface_area"] = surface_area;
    m_shaping.write_to_cursor(cursor["shaping"]);
}

template void RectLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void RectLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(RectLight);

} // namespace falcor
