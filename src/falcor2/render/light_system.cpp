// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "light_system.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/component/light.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/shader.h>
#include <sgl/device/shader_cursor.h>

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

    // Build light selection distribution.
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

    return SceneUpdateFlags::lights;
}

void LightSystem::bind_to_scene(const sgl::ShaderCursor& cursor) const
{
    auto light_system = cursor["light_system"];
    light_system["light_data"] = m_light_data.buffer();
    light_system["light_count"] = m_light_count;
    light_system["analytic_light_count"] = m_analytic_light_count;
    light_system["environment_light_id"] = static_cast<uint32_t>(m_environment_light_id);
    if (m_analytic_light_selection_distribution)
        light_system["analytic_light_selection_distribution"] = *m_analytic_light_selection_distribution;
}

} // namespace falcor
