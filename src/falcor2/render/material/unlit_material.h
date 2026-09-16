// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/material_types.h"
#include "falcor2/render/texture_manager.h"

namespace falcor {

/// Direction-independent material whose accepted hits return baked color and terminate the path.
class FALCOR_API UnlitMaterial : public Material {
    FALCOR_SCENE_OBJECT(UnlitMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    UnlitMaterial();
    virtual ~UnlitMaterial() override;

    virtual void on_load_resources() override;
    virtual shared::MaterialFlags flags() const override;
    virtual OpacityDesc opacity_desc() const override;

    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "alpha_mode",
                &UnlitMaterial::m_alpha_mode,
                "Alpha handling mode.",
                reflection::default_value(AlphaMode::opaque),
                reflection::on_change(&UnlitMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_factor",
                &UnlitMaterial::m_alpha_factor,
                "Constant alpha factor.",
                reflection::default_value(1.f),
                reflection::value_range_unit(),
                reflection::on_change(&UnlitMaterial::mark_dirty_properties)
            )
            .def_rw(
                "alpha_cutoff",
                &UnlitMaterial::m_alpha_cutoff,
                "Alpha cutoff used for mask mode.",
                reflection::default_value(0.5f),
                reflection::value_range_unit(),
                reflection::on_change(&UnlitMaterial::mark_dirty_properties)
            )
            .def_rw(
                "base_color_texture",
                &UnlitMaterial::m_base_color_texture,
                "Baked color and opacity texture (takes precedence over path).",
                reflection::UIFlags::advanced,
                reflection::on_change(&UnlitMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_texture_path",
                &UnlitMaterial::m_base_color_texture_path,
                "Baked color and opacity texture path.",
                reflection::UIFlags::advanced,
                reflection::on_change(&UnlitMaterial::mark_dirty_resources)
            )
            .def_rw(
                "base_color_factor",
                &UnlitMaterial::m_base_color_factor,
                "Direction-independent color factor.",
                reflection::default_value(float3(1.f)),
                reflection::value_range_positive(),
                reflection::UIFlags::display_as_color,
                reflection::on_change(&UnlitMaterial::mark_dirty_properties)
            )
            .def_rw(
                "double_sided",
                &UnlitMaterial::m_double_sided,
                "Double-sided material.",
                reflection::default_value(false),
                reflection::on_change(&UnlitMaterial::mark_dirty_properties)
            );
    }

    float3 _base_color_factor() const { return m_base_color_factor; }
    bool _double_sided() const { return m_double_sided; }

private:
    void mark_dirty_resources() { mark_dirty(DirtyFlags::resources); }
    void mark_dirty_properties() { mark_dirty(DirtyFlags::properties); }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    AlphaMode m_alpha_mode{AlphaMode::opaque};
    float m_alpha_factor{1.f};
    float m_alpha_cutoff{0.5f};
    ref<sgl::Texture> m_base_color_texture;
    std::filesystem::path m_base_color_texture_path;
    float3 m_base_color_factor{1.f};
    bool m_double_sided{false};

    TextureHandle m_base_color_texture_handle;
};

} // namespace falcor
