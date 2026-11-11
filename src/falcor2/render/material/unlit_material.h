// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/scene_types.h"
#include "falcor2/render/texture_manager.h"

#include <sgl/device/sampler.h>

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
    virtual SceneCapabilities required_capabilities() const override;

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
                "base_color_texture_sampler",
                &UnlitMaterial::m_base_color_texture_sampler,
                "Texture sampler (null uses the scene default)",
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
                "base_color_texture_bytes",
                &UnlitMaterial::m_base_color_texture_bytes,
                "Encoded baked color and opacity texture bytes.",
                reflection::UIFlags::advanced,
                reflection::on_change(&UnlitMaterial::mark_dirty_resources)
            )
            .def_prop_rw(
                "base_color_texture_texcoord",
                &UnlitMaterial::base_color_texture_texcoord,
                &UnlitMaterial::set_base_color_texture_texcoord,
                "Texture-coordinate set used by the base color texture.",
                reflection::default_value(0u),
                reflection::value_range(0u, 7u),
                reflection::UIFlags::advanced
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
                "use_vertex_color",
                &UnlitMaterial::m_use_vertex_color,
                "Multiply the baked color and opacity by vertex color zero.",
                reflection::default_value(false),
                reflection::UIFlags::advanced,
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

    uint32_t base_color_texture_texcoord() const { return m_base_color_texture_texcoord; }
    void set_base_color_texture_texcoord(uint32_t value)
    {
        FALCOR_CHECK(value < 8, "base_color_texture_texcoord must be in the range [0, 7], got {}.", value);
        if (m_base_color_texture_texcoord != value) {
            m_base_color_texture_texcoord = value;
            mark_dirty_properties();
        }
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    AlphaMode m_alpha_mode{AlphaMode::opaque};
    float m_alpha_factor{1.f};
    float m_alpha_cutoff{0.5f};
    ref<sgl::Texture> m_base_color_texture;
    ref<sgl::Sampler> m_base_color_texture_sampler;
    std::filesystem::path m_base_color_texture_path;
    ref<Blob> m_base_color_texture_bytes;
    uint32_t m_base_color_texture_texcoord{0};
    float3 m_base_color_factor{1.f};
    bool m_use_vertex_color{false};
    bool m_double_sided{false};

    TextureHandle m_base_color_texture_handle;
};

} // namespace falcor
