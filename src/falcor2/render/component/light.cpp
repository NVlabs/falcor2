// SPDX-License-Identifier: Apache-2.0

#include "light.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/entity.h"

#include <sgl/device/kernel.h>
#include <sgl/device/sampler.h>
#include <sgl/core/signature_buffer.h>
#include <sgl/device/cursor_utils.h>

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

void Light::set_exposure(float exposure)
{
    if (exposure != m_exposure) {
        m_exposure = exposure;
        mark_dirty(DirtyFlags::render_state);
    }
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

void ConstantLight::set_radiance(float3 radiance)
{
    if (radiance != m_radiance) {
        m_radiance = radiance;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void ConstantLight::write_to_cursor_impl(CursorT cursor) const
{
    cursor["radiance"] = m_radiance * sgl::math::exp2(m_exposure);
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

void DistantLight::set_radiance(float3 radiance)
{
    if (radiance != m_radiance) {
        m_radiance = radiance;
        mark_dirty(DirtyFlags::render_state);
    }
}

void DistantLight::set_cutoff_angle(float cutoff_angle)
{
    cutoff_angle = sgl::math::clamp(cutoff_angle, 0.f, 180.f);
    if (cutoff_angle != m_cutoff_angle) {
        m_cutoff_angle = cutoff_angle;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void DistantLight::write_to_cursor_impl(CursorT cursor) const
{
    cursor["world_from_object"] = float3x3(m_entity->world_from_object_matrix());
    cursor["radiance"] = m_radiance * sgl::math::exp2(m_exposure);
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
    if (env_map_path != m_env_map_path) {
        m_env_map_path = env_map_path;
        m_env_map_loaded = false;
        mark_dirty(DirtyFlags::render_state | DirtyFlags::resources);
    }
}

void EnvMapLight::on_load_resources()
{
    if (!m_env_map_loaded) {
        m_env_map_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_env_map_path,
            .generate_mips = false,
            .load_deferred = true,
        });
        m_env_map_loaded = true;
        m_importance_map_dirty = true;
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

        m_importance_map_dirty = false;
    }
}

template<typename CursorT>
void EnvMapLight::write_to_cursor_impl(CursorT cursor) const
{
    cursor["world_from_object"] = float3x3(m_entity->world_from_object_matrix());
    cursor["scale"] = float3(sgl::math::exp2(m_exposure));
    cursor["env_map_texture_handle"] = m_env_map_texture_handle;
    cursor["importance_map_texture"] = m_importance_map_texture->descriptor_handle_ro();
}

template void EnvMapLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void EnvMapLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(EnvMapLight);

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

void PointLight::set_intensity(float3 intensity)
{
    if (intensity != m_intensity) {
        m_intensity = intensity;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void PointLight::write_to_cursor_impl(CursorT cursor) const
{
    float4x4 matrix = m_entity->world_from_object_matrix();
    cursor["position_ws"] = matrix.get_col(3).xyz();
    cursor["intensity"] = m_intensity * sgl::math::exp2(m_exposure);
}

template void PointLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void PointLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(PointLight);

// ----------------------------------------------------------------------------
// SpotLight
// ----------------------------------------------------------------------------

SpotLight::SpotLight()
{
    set_slang_type_name("SpotLight");
    set_light_type(shared::LightType::spot_light);
    set_light_flags(shared::LightFlags::delta_position);
}

SpotLight::~SpotLight() { }

void SpotLight::set_intensity(float3 intensity)
{
    if (intensity != m_intensity) {
        m_intensity = intensity;
        mark_dirty(DirtyFlags::render_state);
    }
}

void SpotLight::set_cutoff_angle(float cutoff_angle)
{
    cutoff_angle = sgl::math::clamp(cutoff_angle, 0.f, 180.f);
    if (cutoff_angle != m_cutoff_angle) {
        m_cutoff_angle = cutoff_angle;
        mark_dirty(DirtyFlags::render_state);
    }
}

void SpotLight::set_falloff_angle(float falloff_angle)
{
    falloff_angle = sgl::math::clamp(falloff_angle, 0.f, 180.f);
    if (falloff_angle != m_falloff_angle) {
        m_falloff_angle = falloff_angle;
        mark_dirty(DirtyFlags::render_state);
    }
}

template<typename CursorT>
void SpotLight::write_to_cursor_impl(CursorT cursor) const
{
    float4x4 matrix = m_entity->world_from_object_matrix();
    cursor["position_ws"] = matrix.get_col(3).xyz();
    cursor["dir_ws"] = normalize(matrix.get_col(2).xyz());
    cursor["intensity"] = m_intensity * sgl::math::exp2(m_exposure);
    cursor["cos_cutoff_angle"] = sgl::math::cos(sgl::math::radians(m_cutoff_angle));
    cursor["cos_falloff_angle"] = sgl::math::cos(sgl::math::radians(m_falloff_angle));
}

template void SpotLight::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void SpotLight::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_COMPONENT(SpotLight);

} // namespace falcor
