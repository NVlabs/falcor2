// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "emission_only_material.h"

#include "falcor2/render/scene.h"

#include <algorithm>

namespace falcor {

EmissionOnlyMaterial::EmissionOnlyMaterial()
{
    set_slang_type_name("EmissionOnlyMaterial");
}

EmissionOnlyMaterial::~EmissionOnlyMaterial() { }

void EmissionOnlyMaterial::set_enable_color_temperature(bool enable_color_temperature)
{
    if (enable_color_temperature != m_enable_color_temperature) {
        m_enable_color_temperature = enable_color_temperature;
        mark_dirty_properties();
    }
}

void EmissionOnlyMaterial::set_color_temperature(float color_temperature)
{
    color_temperature = std::clamp(color_temperature, MIN_COLOR_TEMPERATURE, MAX_COLOR_TEMPERATURE);
    if (color_temperature != m_color_temperature) {
        m_color_temperature = color_temperature;
        mark_dirty_properties();
    }
}

void EmissionOnlyMaterial::on_load_resources()
{
    if (m_emission_color_texture) {
        m_emission_color_texture_handle = m_scene->texture_manager()->register_texture({
            .texture = m_emission_color_texture,
        });
    } else if (!m_emission_color_texture_path.empty()) {
        m_emission_color_texture_handle = m_scene->texture_manager()->load_texture({
            .path = m_emission_color_texture_path,
            .srgb = true,
            .load_deferred = true,
        });
    } else {
        m_emission_color_texture_handle = {};
    }
}

template<typename CursorT>
void EmissionOnlyMaterial::write_to_cursor_impl(CursorT cursor) const
{
    cursor["emission_color_texture"] = m_emission_color_texture_handle;
    cursor["emission_color"] = m_enable_color_temperature
        ? m_emission_color * color_temperature_to_rgb(m_color_temperature)
        : m_emission_color;
    cursor["emission_luminance"] = m_emission_luminance;
}

template void EmissionOnlyMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void EmissionOnlyMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

FALCOR_SCENE_REGISTER_MATERIAL(EmissionOnlyMaterial);

} // namespace falcor
