// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mdl_context.h"

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"
#include "falcor2/core/reflection.h"

#include <cstring>
#include <optional>
#include <type_traits>

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
    virtual shared::MaterialFlags flags() const override
    {
        // MDL expressions may select transmissive lobes dynamically. Until host-side
        // expression analysis is available, conservatively allow two-sided interaction.
        auto result = shared::MaterialFlags::two_sided;
        // A constant MDL thin_walled value must also control the path tracer's
        // medium stack. Supporting a state-dependent value needs a runtime API;
        // leave that existing limitation explicit rather than guessing from lobes.
        if (m_result.thin_walled.value_or(false))
            result |= shared::MaterialFlags::thin_walled;
        return result;
    }
    virtual OpacityDesc opacity_desc() const override;
    virtual SceneCapabilities required_capabilities() const override;
    virtual ref<sgl::SlangModule> required_module() const override;

    /// List the unique texture handles used by this material.
    /// Walks m_texture_resources and returns each valid handle once. Empty before
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
                "Semicolon-separated MDL search roots, before configured defaults. Empty uses defaults, "
                "including the bundled OmniPBR fallback. Loaded modules retain their first implementation.",
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
                "mdl_source",
                &MDLMaterial::m_mdl_source,
                "Optional in-memory MDL module source; its name must identify this source uniquely.",
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
                "mdl_target_material_type",
                &MDLMaterial::m_mdl_target_material_type,
                "Optional declarative material-category struct used as the MDL compilation target.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MDLMaterial::require_codegen),
                reflection::UIFlags::advanced
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

    /// Copy bytes into a span obtained from mutable_data().
    static void set_material_data(std::span<uint8_t> buffer, size_t offset, const void* data, size_t size)
    {
        FALCOR_ASSERT_LE(offset, buffer.size());
        FALCOR_ASSERT_LE(size, buffer.size() - offset);
        if (size == 0)
            return;
        std::memcpy(buffer.data() + offset, data, size);
    }

    template<typename T>
    static void set_material_data(std::span<uint8_t> buffer, size_t offset, const T& data)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        set_material_data(buffer, offset, &data, sizeof(T));
    }

    template<typename T>
    void write_mdl_material_property(std::string_view name, size_t offset)
    {
        set_material_data(mutable_data(), offset, m_material_properties.get<T>(name));
    }

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    // Static properties.
    std::string m_mdl_library_path;
    std::string m_mdl_material_name;
    std::string m_mdl_source;
    bool m_mdl_class_compilation{true};
    std::string m_mdl_target_material_type;
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


    struct TextureParameter {
        std::string name;
        /// Offset within the MDL argument block, without m_arg_block_offset.
        uint32_t argument_offset;
    };

    struct TextureResource {
        MDLContext::TextureSource source;
        /// Present only for an editable texture argument.
        std::optional<TextureParameter> parameter;
        TextureHandle texture_handle;
    };

    /// Owns all load sources and handles; slot i corresponds to MDL resource ID i + 1.
    std::vector<TextureResource> m_texture_resources;
    /// Relative byte offsets into the material data (readonly is at offset 0).
    uint32_t m_arg_block_offset;
    uint32_t m_texture_table_offset;

    /// Compilation request and result used for the actual shader.
    MDLContext::CompileRequest m_request;
    MDLContext::CompileResult m_result;

    /// Module with the MDL code
    ref<sgl::SlangModule> m_slang_module;
};

} // namespace falcor
