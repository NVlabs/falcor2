// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"
#include "falcor2/utils/color.h"

namespace falcor {

/// Material that emits radiance and has no scattering lobes.
class FALCOR_API EmissionOnlyMaterial : public Material {
    FALCOR_SCENE_OBJECT(EmissionOnlyMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    EmissionOnlyMaterial();
    virtual ~EmissionOnlyMaterial() override;

    virtual void on_load_resources() override;
    virtual shared::MaterialFlags flags() const override { return shared::MaterialFlags::none; }

    bool enable_color_temperature() const { return m_enable_color_temperature; }
    void set_enable_color_temperature(bool enable_color_temperature);

    float color_temperature() const { return m_color_temperature; }
    void set_color_temperature(float color_temperature);

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "emission_color",
                &EmissionOnlyMaterial::m_emission_color,
                "Unitless scene-linear emission color multiplier. HDR values are allowed.",
                reflection::default_value(float3(1.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&EmissionOnlyMaterial::mark_dirty_properties)
            )
            .def_rw(
                "emission_color_texture",
                &EmissionOnlyMaterial::m_emission_color_texture,
                "Unitless scene-linear emission color texture (takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&EmissionOnlyMaterial::mark_dirty_resources)
            )
            .def_rw(
                "emission_color_texture_path",
                &EmissionOnlyMaterial::m_emission_color_texture_path,
                "Emission color texture path. The texture is decoded from sRGB to scene-linear color.",
                reflection::UIFlags::advanced,
                reflection::on_change(&EmissionOnlyMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "enable_color_temperature",
                &EmissionOnlyMaterial::enable_color_temperature,
                &EmissionOnlyMaterial::set_enable_color_temperature,
                "Enable the luminance-normalized color temperature multiplier.",
                reflection::default_value(false)
            )
            .def_prop_rw(
                "color_temperature",
                &EmissionOnlyMaterial::color_temperature,
                &EmissionOnlyMaterial::set_color_temperature,
                "Blackbody color temperature in Kelvin. 6500 K is neutral white.",
                reflection::default_value(6500.f),
                reflection::value_range(MIN_COLOR_TEMPERATURE, MAX_COLOR_TEMPERATURE)
            )
            .def_rw(
                "emission_luminance",
                &EmissionOnlyMaterial::m_emission_luminance,
                "Emission luminance in cd/m^2 (nits) for white, untextured emission; scales emission color and "
                "texture.",
                reflection::default_value(0.f),
                reflection::value_range_positive(),
                reflection::on_change(&EmissionOnlyMaterial::mark_dirty_properties)
            );
    }

private:
    void mark_dirty_resources() { mark_dirty(DirtyFlags::resources); }
    void mark_dirty_properties() { mark_dirty(DirtyFlags::properties); }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    float3 m_emission_color{1.f};
    ref<sgl::Texture> m_emission_color_texture;
    std::filesystem::path m_emission_color_texture_path;
    bool m_enable_color_temperature{false};
    float m_color_temperature{6500.f};
    float m_emission_luminance{0.f};

    TextureHandle m_emission_color_texture_handle;
};

} // namespace falcor
