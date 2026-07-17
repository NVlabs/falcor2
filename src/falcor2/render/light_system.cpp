// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light_system.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/light.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/kernel.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

#include <algorithm>
#include <cmath>

namespace falcor {

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
        {"ILight", "SpotLight"},
    };
}

void LightSystem::create_kernels()
{
    if (m_compute_environment_light_powers_kernel)
        return;

    auto module = m_scene->device()->load_module("falcor2/render/kernels/light_system_kernels.slang");
    m_compute_environment_light_powers_kernel = m_scene->device()->create_compute_kernel({
        .program = m_scene->device()->link_program(
            {module},
            {module->entry_point("compute_environment_light_powers", m_type_conformances)}
        ),
    });
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
    std::vector<float> analytic_light_selection_pmf;

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
                    analytic_light_selection_pmf.push_back(1.f);
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

    // Build analytic light selection distribution.
    if (analytic_light_selection_pmf.empty()) {
        m_analytic_light_selection_distribution.reset();
    } else {
        m_analytic_light_selection_distribution.reset(new AliasTable1D(
            m_scene->device(),
            analytic_light_selection_pmf,
            "LightSystem::analytic_light_selection_distribution"
        ));
    }

    m_light_data.mark_all_dirty();
    m_light_data.update_buffer(ctx.command_encoder());

    // Build the environment light selection distribution when there are multiple
    // lights to choose from. Environment-map powers depend on the 1x1 mip generated
    // above, so compute all powers on the GPU and read them back before constructing
    // the alias table. Zero and single-light cases need no selection distribution.
    if (m_environment_light_count > 1) {
        create_kernels();

        const size_t buffer_size = m_environment_light_count * sizeof(float);
        if (!m_environment_light_power_buffer || m_environment_light_power_buffer->size() < buffer_size) {
            m_environment_light_power_buffer = m_scene->device()->create_buffer({
                .size = buffer_size,
                .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
                .label = "LightSystem::environment_light_powers",
            });
        }

        m_compute_environment_light_powers_kernel->dispatch(
            uint3(m_environment_light_count, 1, 1),
            [&](sgl::ShaderCursor cursor)
            {
                cursor = cursor.find_entry_point(0);
                auto light_system = cursor["light_system"];
                light_system["light_data"] = m_light_data.buffer();
                light_system["environment_light_id"] = static_cast<uint32_t>(m_environment_light_id);
                cursor["powers"] = m_environment_light_power_buffer;
                cursor["light_count"] = m_environment_light_count;
            },
            ctx.command_encoder()
        );

        // Read back emitted powers (stalls the host).
        ctx.submit();
        std::vector<float> environment_light_powers
            = m_environment_light_power_buffer->get_elements<float>(0, m_environment_light_count);

        double total_environment_light_power = 0.0;
        for (float& power : environment_light_powers) {
            if (!std::isfinite(power) || power <= 0.f)
                power = 0.f;
            total_environment_light_power += power;
        }
        if (total_environment_light_power == 0.0) {
            std::fill(environment_light_powers.begin(), environment_light_powers.end(), 1.f);
        } else {
            for (float& power : environment_light_powers)
                power = static_cast<float>(static_cast<double>(power) / total_environment_light_power);
        }

        m_environment_light_selection_distribution.reset(new AliasTable1D(
            m_scene->device(),
            environment_light_powers,
            "LightSystem::environment_light_selection_distribution"
        ));
    } else {
        m_environment_light_selection_distribution.reset();
    }

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
    if (m_analytic_light_selection_distribution)
        light_system["analytic_light_selection_distribution"] = *m_analytic_light_selection_distribution;
    if (m_environment_light_selection_distribution)
        light_system["environment_light_selection_distribution"] = *m_environment_light_selection_distribution;
}

} // namespace falcor
