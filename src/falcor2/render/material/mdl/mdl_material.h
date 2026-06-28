// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mdl_context.h"

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"
#include "falcor2/core/reflection.h"

#include "falcor2/utils/managed_buffer.h"

namespace falcor {

/// MDL material.
class FALCOR_API MDLMaterial : public Material {
    FALCOR_SCENE_OBJECT(MDLMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    MDLMaterial();

    /// Destructor.
    virtual ~MDLMaterial() override;

    // ReflectedObject interface

    virtual void set_properties(const Properties& props) override;

    virtual reflection::DynamicPropertySet* dynamic_properties() override { return &m_material_properties; }
    virtual const reflection::DynamicPropertySet* dynamic_properties() const override { return &m_material_properties; }

    // Material interface

    virtual void on_load_resources() override;
    virtual void update(SceneUpdateContext& ctx) override;
    virtual ref<sgl::SlangModule> required_module() const override;

    /// List the texture handles used by this material.
    /// Walks m_texture_resources and returns valid handles. Empty before
    /// on_load_resources() has run.
    std::vector<TextureHandle> build_texture_list() const;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "mdl_library_path",
                &MDLMaterial::m_mdl_library_path,
                "MDL library path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MDLMaterial::require_codegen)
            )
            .def_rw(
                "mdl_material_name",
                &MDLMaterial::m_mdl_material_name,
                "MDL material name.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MDLMaterial::require_codegen)
            )
            .def_rw(
                "mdl_class_compilation",
                &MDLMaterial::m_mdl_class_compilation,
                "Use MDL class compilation.",
                reflection::default_value(true),
                reflection::on_change(&MDLMaterial::require_codegen)
            )
            .def_rw(
                "learnable",
                &MDLMaterial::m_learnable,
                "Make this material learnable.",
                reflection::default_value(false),
                reflection::on_change(&MDLMaterial::require_codegen)
            )
            .def_rw(
                "debug_write_shader_path",
                &MDLMaterial::m_debug_write_shader_path,
                "Write the generated Slang shader to this path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MDLMaterial::require_codegen)
            )
            .def_rw(
                "debug_load_shader_path",
                &MDLMaterial::m_debug_load_shader_path,
                "Load a previously generate Slang shader from this path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MDLMaterial::require_codegen)
            );
    }

    // Accessors for Python Material.get_this() only.
    const ref<ManagedBuffer>& _mdl_data() const { return m_mdl_data; }
    uint32_t _arg_block_offset() const { return m_arg_block_offset; }
    uint32_t _texture_table_offset() const { return m_texture_table_offset; }
    uint32_t _surface_scatter_bsdf_count() const { return m_result.surface_scatter_handle_count; }

private:
    void require_codegen();
    void run_codegen();

    /// Attempts to resolve a path using the library_path_resolver,
    /// returning the path intact if that fails (deferring to the scene resolver).
    std::filesystem::path try_resolve(const std::filesystem::path& path) const
    {
        auto result = m_library_path_resolver->resolve_path(path);
        return result.empty() ? path : result;
    }

    template<typename T>
    void write_mdl_material_property(std::string_view name, size_t offset)
    {
        m_mdl_data->set_data(m_material_properties.get<T>(name), offset);
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    // Static properties.
    std::string m_mdl_library_path;
    std::string m_mdl_material_name;
    bool m_mdl_class_compilation{true};
    bool m_learnable{false};
    std::string m_debug_write_shader_path;
    std::string m_debug_load_shader_path;
    bool m_force_update{false};
    bool m_texture_table_needs_update{false};

    /// Dynamic properties (generated from MDL definition).
    reflection::DynamicPropertySet m_material_properties;

    /// True if we need to run codegen.
    bool m_require_codegen{true};
    size_t m_codegen_properties_hash{0};

    /// Attempts to resolve paths based on the library path.
    ref<AssetResolver> m_library_path_resolver;


    struct TextureResource {
        /// For Texture2D
        std::filesystem::path path;
        bool srgb;

        /// Name of the property that controls srgb
        std::string srgb_property_name;
        /// Handle of the texture, only valid for Type != Unused
        TextureHandle texture_handle;
    };

    std::vector<TextureResource> m_texture_resources;
    /// Buffer where MDL stores its data
    ref<ManagedBuffer> m_mdl_data;
    /// Byte offsets into the mdl_data buffer (readonly is at offset 0)
    uint32_t m_arg_block_offset;
    uint32_t m_texture_table_offset;

    /// Compilation request and result used for the actual shader.
    MDLContext::CompileRequest m_request;
    MDLContext::CompileResult m_result;

    /// Module with the MDL code
    ref<sgl::SlangModule> m_slang_module;

    /// Header stuff
    float m_ior{1.f};
};

} // namespace falcor
