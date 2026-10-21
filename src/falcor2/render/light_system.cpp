// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light_system.h"

#include "falcor2/core/error.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/component/light.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/kernel.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <array>
#include <bit>
#include <cmath>

namespace falcor {

namespace {

struct RenderAreaLightInstanceData {
    shared::LightID light_id;
    bool geometry_visible{false};
};

shared::GeometryInstanceDescPayload get_area_light_instance_payload(const RenderAreaLightInstanceData& data)
{
    shared::AreaLightGeometryInstanceDesc desc = {};
    desc.light_id = data.light_id;
    desc.flags = data.geometry_visible ? shared::AreaLightGeometryFlags::visible : shared::AreaLightGeometryFlags::none;
    return std::bit_cast<shared::GeometryInstanceDesc>(desc).payload;
}

uint8_t get_area_light_instance_mask(const RenderAreaLightInstanceData& data)
{
    uint8_t instance_mask = shared::INSTANCE_MASK_BIT_AREA_LIGHT;
    if (data.geometry_visible)
        instance_mask |= shared::INSTANCE_MASK_BIT_AREA_LIGHT_VISIBLE;
    return instance_mask;
}

float4x4 get_area_light_proxy_transform(const Light* light)
{
    enum class ProxyShape {
        planar,
        sphere,
        cylinder,
    };

    const float4x4 world_from_light = light->entity()->world_from_object_matrix();
    const float3 center_ws = world_from_light.get_col(3).xyz();
    float3 half_axis_u_ws = {};
    float3 half_axis_v_ws = {};
    float3 half_axis_w_ws = {};
    ProxyShape shape = ProxyShape::planar;

    if (const SphereLight* sphere = dynamic_cast<const SphereLight*>(light)) {
        shape = ProxyShape::sphere;
        const float radius = sphere->world_radius();
        half_axis_u_ws = float3(radius, 0.f, 0.f);
        half_axis_v_ws = float3(0.f, radius, 0.f);
        half_axis_w_ws = float3(0.f, 0.f, radius);
    } else if (const CylinderLight* cylinder = dynamic_cast<const CylinderLight*>(light)) {
        shape = ProxyShape::cylinder;
        const CylinderLightWorldGeometry geometry = cylinder->world_geometry();
        half_axis_u_ws = geometry.half_length_ws * geometry.axis_ws;
        half_axis_v_ws = geometry.radius_u_ws;
        half_axis_w_ws = geometry.radius_v_ws;
    } else if (const DiskLight* disk = dynamic_cast<const DiskLight*>(light)) {
        half_axis_u_ws = disk->radius() * world_from_light.get_col(0).xyz();
        half_axis_v_ws = disk->radius() * world_from_light.get_col(1).xyz();
    } else if (const RectLight* rect = dynamic_cast<const RectLight*>(light)) {
        half_axis_u_ws = 0.5f * rect->width() * world_from_light.get_col(0).xyz();
        half_axis_v_ws = 0.5f * rect->height() * world_from_light.get_col(1).xyz();
    } else {
        FALCOR_THROW("Area light type '{}' has no geometry transform implementation.", light->slang_type_name());
    }

    const float3 extent_ws = abs(half_axis_u_ws) + abs(half_axis_v_ws) + abs(half_axis_w_ws);
    const float magnitude = std::max({
        1.f,
        std::abs(center_ws.x),
        std::abs(center_ws.y),
        std::abs(center_ws.z),
        extent_ws.x,
        extent_ws.y,
        extent_ws.z,
    });
    const float padding = 1e-5f * magnitude;

    // The shared procedural geometry is the canonical [-1, 1]^3 AABB. Its instance transform must conservatively
    // contain the light and remain invertible, including for degenerate lights.
    const auto use_axis_aligned_proxy = [&]()
    {
        half_axis_u_ws = float3(std::max(extent_ws.x, padding), 0.f, 0.f);
        half_axis_v_ws = float3(0.f, std::max(extent_ws.y, padding), 0.f);
        half_axis_w_ws = float3(0.f, 0.f, std::max(extent_ws.z, padding));
    };

    switch (shape) {
    case ProxyShape::sphere: {
        const float radius = std::max(length(half_axis_w_ws), padding);
        half_axis_u_ws = float3(radius, 0.f, 0.f);
        half_axis_v_ws = float3(0.f, radius, 0.f);
        half_axis_w_ws = float3(0.f, 0.f, radius);
        break;
    }
    case ProxyShape::cylinder:
        if (length(half_axis_u_ws) <= padding || length(half_axis_v_ws) <= padding
            || length(half_axis_w_ws) <= padding) {
            use_axis_aligned_proxy();
        }
        break;
    case ProxyShape::planar: {
        const float3 area_vector = cross(half_axis_u_ws, half_axis_v_ws);
        if (length(area_vector) > padding * padding) {
            half_axis_w_ws = normalize(area_vector) * padding;
        } else {
            use_axis_aligned_proxy();
        }
        break;
    }
    }

    float4x4 transform = float4x4::identity();
    transform.set_col(0, float4(half_axis_u_ws, 0.f));
    transform.set_col(1, float4(half_axis_v_ws, 0.f));
    transform.set_col(2, float4(half_axis_w_ws, 0.f));
    transform.set_col(3, float4(center_ws, 1.f));
    return transform;
}

} // namespace

LightSystem::LightSystem(Scene* scene)
    : SceneSystem{scene}
    , m_components(scene->component_collection())
    , m_light_data({
          .label = "LightSystem::light_data",
      })
{
    m_type_conformances = {
        {"ILight", "ConstantLight"},
        {"ILight", "DistantLight"},
        {"ILight", "EnvMapLight"},
        {"ILight", "PointLight"},
        {"ILight", "SphereLight"},
        {"ILight", "CylinderLight"},
        {"ILight", "DiskLight"},
        {"ILight", "RectLight"},
    };
}

void LightSystem::create_kernels()
{
    if (m_compute_light_powers_kernel)
        return;

    auto module = m_scene->device()->load_module("falcor2/render/kernels/light_system_kernels.slang");
    m_compute_light_powers_kernel = m_scene->device()->create_compute_kernel({
        .program
        = m_scene->device()->link_program({module}, {module->entry_point("compute_light_powers", m_type_conformances)}),
    });
}

void LightSystem::sync_area_light_geometry()
{
    RenderScene* render_scene = m_scene->_render_scene();

    // Mark all existing area light geometry instances as unseen.
    for (auto& [_, instance] : m_area_light_geometry_instances)
        instance.seen = false;

    // Iterate over all lights and update or create area light geometry instances as needed.
    for (size_t i = 0; i < m_components.size(); ++i) {
        const Light* light = m_components[i]->as<Light>();
        if (!light || !light->is_valid() || !light->active() || !is_set(light->light_flags(), shared::LightFlags::area))
            continue;

        // Create the area light geometry if they don't already exist.
        if (!m_area_light_geometry) {
            m_area_light_geometry = render_scene->create_area_light_geometry({});
            std::array<RenderHandle, 1> geometries{m_area_light_geometry};
            m_area_light_geometry_group = render_scene->create_geometry_group({.geometries = geometries});
        }

        const shared::LightID light_id = light->light_id();
        const bool geometry_visible = light->geometry_visible();
        const RenderAreaLightInstanceData instance_data{
            .light_id = light_id,
            .geometry_visible = geometry_visible,
        };

        // Find or create the area light geometry instance for this light.
        auto instance_it = m_area_light_geometry_instances.find(light);
        if (instance_it == m_area_light_geometry_instances.end()) {
            const float4x4 transform = get_area_light_proxy_transform(light);
            const RenderTransformHandle render_transform
                = render_scene->create_transform({.world_from_object = transform});
            const RenderGeometryInstanceHandle geometry_instance = render_scene->create_geometry_instance({
                .geometry_group = m_area_light_geometry_group,
                .transform = render_transform,
                .payload_override = get_area_light_instance_payload(instance_data),
                .instance_mask = get_area_light_instance_mask(instance_data),
            });
            m_area_light_geometry_instances.emplace(
                light,
                AreaLightGeometryInstance{
                    geometry_instance,
                    render_transform,
                    transform,
                    true,
                }
            );
            continue;
        }

        // Update the existing area light geometry instance with the new transform and instance data.
        AreaLightGeometryInstance& instance = instance_it->second;
        instance.seen = true;
        // Another light may have triggered this sync, so reuse the cached proxy for unchanged lights.
        if (m_components.dirty_flags()[i] != Component::DirtyFlags::none) {
            const float4x4 transform = get_area_light_proxy_transform(light);
            if (instance.proxy_transform != transform) {
                render_scene->update_transform(instance.transform, transform);
                instance.proxy_transform = transform;
            }
        }
        render_scene->update_geometry_instance_payload(
            instance.geometry_instance,
            get_area_light_instance_payload(instance_data)
        );
        render_scene->update_geometry_instance_mask(
            instance.geometry_instance,
            get_area_light_instance_mask(instance_data)
        );
    }

    // Destroy area light geometry instances that are no longer seen.
    for (auto it = m_area_light_geometry_instances.begin(); it != m_area_light_geometry_instances.end();) {
        const AreaLightGeometryInstance& instance = it->second;
        if (instance.seen) {
            ++it;
            continue;
        }
        render_scene->destroy_geometry_instance(instance.geometry_instance);
        render_scene->destroy_transform(instance.transform);
        it = m_area_light_geometry_instances.erase(it);
    }

    // Destroy the area light geometry if no instances remain.
    if (m_area_light_geometry_instances.empty()) {
        if (m_area_light_geometry_group)
            render_scene->destroy_geometry_group(m_area_light_geometry_group);
        if (m_area_light_geometry)
            render_scene->destroy_area_light_geometry(m_area_light_geometry);
        m_area_light_geometry_group = {};
        m_area_light_geometry = {};
    }
}

SceneUpdateFlags LightSystem::update(SceneUpdateContext& ctx)
{
    // Check if lights have changed.
    // TODO: We need better ways to track changes in components.
    bool lights_changed = false;
    if (m_components.is_dirty()) {
        for (size_t i = 0; i < m_components.size(); ++i) {
            if (m_components.dirty_flags()[i] == Component::DirtyFlags::none)
                continue;
            if (Light* light = m_components[i]->as<Light>()) {
                lights_changed = true;
                break;
            }
        }
    }
    if (!lights_changed)
        return SceneUpdateFlags::none;

    uint32_t analytic_light_count = 0;
    uint32_t environment_light_count = 0;

    // Update lights.
    for (size_t i = 0; i < m_components.size(); ++i) {
        Light* light = m_components[i]->as<Light>();
        if (light && light->is_valid()) {
            light->update(ctx);
            if (light->active()) {
                if (is_set(light->light_flags(), shared::LightFlags::environment))
                    ++environment_light_count;
                else
                    ++analytic_light_count;
            }
        }
    }

    m_light_count = analytic_light_count + environment_light_count;
    m_analytic_light_count = analytic_light_count;
    m_environment_light_id = shared::LightID::invalid;
    m_environment_light_count = environment_light_count;
    m_light_data.resize(m_light_count);

    uint32_t analytic_light_index = 0;
    uint32_t environment_light_index = analytic_light_count;

    for (size_t i = 0; i < m_components.size(); ++i) {
        Light* light = m_components[i]->as<Light>();
        if (light && light->is_valid()) {
            if (light->active()) {
                uint32_t light_index = 0;
                if (is_set(light->light_flags(), shared::LightFlags::environment)) {
                    light_index = environment_light_index++;
                    if (m_environment_light_id == shared::LightID::invalid)
                        m_environment_light_id = shared::LightID{light_index};
                } else {
                    light_index = analytic_light_index++;
                }
                light->set_light_id(shared::LightID{light_index});
                m_light_data[light_index].header.type = light->light_type();
                auto layout = m_scene->render_module()->layout();
                std::string buffer_type_name = fmt::format("StructuredBuffer<{}>", light->slang_type_name());
                auto buffer_type = layout->find_type_by_name(buffer_type_name.c_str());
                auto buffer_layout = layout->get_type_layout(buffer_type);
                auto element_type_layout = buffer_layout->element_type_layout();
                auto buffer_cursor = make_ref<sgl::BufferCursor>(
                    m_scene->device()->type(),
                    element_type_layout,
                    &m_light_data[light_index],
                    sizeof(shared::LightData)
                );
                light->write_to_cursor((*buffer_cursor)[0]);
            } else {
                light->set_light_id(shared::LightID::invalid);
            }
        }
    }

    m_light_data.mark_all_dirty();
    m_light_data.update_buffer(ctx.command_encoder());

    sync_area_light_geometry();

    // Compute selection estimates for all active component lights. Environment-map estimates
    // depend on the 1x1 mip generated during light update, so evaluate every light
    // through the common shader interface. The CPU snapshot is read back lazily by power samplers.
    if (m_light_count > 0) {
        create_kernels();

        const size_t buffer_size = m_light_count * sizeof(float);
        if (!m_light_power_buffer || m_light_power_buffer->size() < buffer_size) {
            m_light_power_buffer = m_scene->device()->create_buffer({
                .size = buffer_size,
                .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
                .label = "LightSystem::light_powers",
            });
        }

        m_compute_light_powers_kernel->dispatch(
            uint3(m_light_count, 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                cursor = cursor.find_entry_point(0);
                auto light_system = cursor["light_system"];
                light_system["light_data"] = m_light_data.buffer();
                cursor["powers"] = m_light_power_buffer;
                cursor["light_count"] = m_light_count;
            },
            ctx.command_encoder()
        );
    }

    ++m_generations.topology;
    ++m_generations.geometry;
    ++m_generations.selection_weights;

    return SceneUpdateFlags::lights;
}

void LightSystem::bind_to_scene(const sgl::ShaderCursor& cursor) const
{
    auto light_system = cursor["light_system"];
    light_system["light_data"] = m_light_data.buffer();
    light_system["light_count"] = m_light_count;
    light_system["analytic_light_count"] = m_analytic_light_count;
    light_system["environment_light_id"] = static_cast<uint32_t>(m_environment_light_id);
    light_system["environment_light_count"] = m_environment_light_count;
}

std::span<const float> LightSystem::light_powers() const
{
    if (m_light_power_generations.topology != m_generations.topology
        || m_light_power_generations.selection_weights != m_generations.selection_weights) {
        m_light_powers.clear();
        if (m_light_count > 0) {
            FALCOR_CHECK(m_light_power_buffer, "Light power buffer is unavailable.");
            m_light_powers = m_light_power_buffer->get_elements<float>(0, m_light_count);
            for (float& power : m_light_powers) {
                if (!std::isfinite(power) || power <= 0.f)
                    power = 0.f;
            }
        }
        m_light_power_generations = m_generations;
    }
    return m_light_powers;
}

} // namespace falcor
