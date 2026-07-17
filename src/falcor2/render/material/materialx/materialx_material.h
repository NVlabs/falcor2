// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"

#include "falcor2/utils/managed_buffer.h"

#include "codegen/codegen.h"

#include <sgl/core/crypto.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace falcor {
namespace materialx {
class MxParamList;
struct CodeGenResult;
} // namespace materialx

/// MaterialX material.
class FALCOR_API MaterialXMaterial : public Material {
    FALCOR_SCENE_OBJECT(MaterialXMaterial, Material)
    FALCOR_WRITE_TO_CURSOR_OVERRIDES();

public:
    /// Constructor.
    MaterialXMaterial();

    /// Destructor.
    virtual ~MaterialXMaterial() override;

    // ReflectedObject interface

    virtual void set_properties(const Properties& props) override;

    virtual reflection::DynamicPropertySet* dynamic_properties() override { return &m_material_properties; }
    virtual const reflection::DynamicPropertySet* dynamic_properties() const override { return &m_material_properties; }

    // Material interface

    virtual void on_load_resources() override;
    virtual void update(SceneUpdateContext& ctx) override;
    virtual ref<sgl::SlangModule> required_module() const override;

    /// List the texture handles used by this material's bound parameters.
    /// Extracts TextureInfo values from the codegen param list. Empty before
    /// on_load_resources() has run.
    std::vector<TextureHandle> build_texture_list() const;

    /// Reflect this class.
    template<reflection::ClassReflector R>
    static void reflect(R& r)
    {
        r //
            .def_rw(
                "mtlx_basepath",
                &MaterialXMaterial::m_mtlx_basepath,
                "Semicolon-separated base paths for MaterialX; mtlx_path and images in the mtlx are searched from "
                "these paths.",
                reflection::default_value(std::filesystem::path{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_buffer",
                &MaterialXMaterial::m_mtlx_buffer,
                "MaterialX xml passed as a buffer, used over path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_search_paths",
                &MaterialXMaterial::m_mtlx_search_paths,
                "Semicolon-separated MaterialX search roots used after mtlx_basepath.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_path",
                &MaterialXMaterial::m_mtlx_path,
                "Path to MaterialX file, relative to mtlx_basepath.",
                reflection::default_value(std::filesystem::path{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_editable_params",
                &MaterialXMaterial::m_mtlx_editable_params,
                // TODO(@tdavidovic): turn into bool.
                "Editable parameters, any value should expose all parameters.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_node_name",
                &MaterialXMaterial::m_mtlx_node_name,
                "Name of the mtlx material, in case the doc contains more than one.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "mtlx_layeringmethod",
                &MaterialXMaterial::m_mtlx_layeringmethod,
                // TODO(@tdavidovic): Turn into enum or bool
                "Layering method. (normal, positionfree).",
                reflection::default_value("normal"),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_transmissive_bsdfs",
                &MaterialXMaterial::m_mtlx_transmissive_bsdfs,
                // TODO(@tdavidovic, @aweidlich): Investigate possibility to bake this back into the nodes.
                "Comma separate list of BSDFs that are allowed to transmit through the layer stack.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_auto_transmission",
                &MaterialXMaterial::m_mtlx_auto_transmission,
                "Infer transmissive BSDFs from MaterialX transmission nodes.",
                reflection::default_value(true),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_autogamma",
                &MaterialXMaterial::m_mtlx_autogamma,
                "`true` loads sRGB when image node output is color. This is needed by some Houdini-authored assets, "
                "but double-decodes MaterialX files that already emit explicit colorspace transforms.",
                reflection::default_value(false),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_target_color_space_override",
                &MaterialXMaterial::m_mtlx_target_color_space_override,
                // TODO(mx139): Default MX139 to lin_rec709. MaterialX wrapper documents can omit a root
                // colorspace and then lose explicit srgb_texture -> lin_rec709 shader transforms.
                // Keep mtlx_autogamma false for MaterialX files that emit shader-side transforms.
                "Optional MaterialX shadergen target color space, for example lin_rec709. Empty uses the document "
                "color space.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_use_slang_derivatives",
                &MaterialXMaterial::m_mtlx_use_slang_derivatives,
                "Use upstream Slang ddx/ddy derivative implementations where available. Keep disabled for path "
                "tracing/ray-tracing materials.",
                reflection::default_value(false),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "mtlx_geomprop_names",
                &MaterialXMaterial::mtlx_geomprop_names_property,
                &MaterialXMaterial::set_mtlx_geomprop_names_property,
                "MaterialX geomprop stream names with explicit provider IDs.",
                reflection::default_value(detail::PropertyList::create(std::vector<std::string>{})),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_prop_rw(
                "mtlx_geomprop_ids",
                &MaterialXMaterial::mtlx_geomprop_ids_property,
                &MaterialXMaterial::set_mtlx_geomprop_ids_property,
                "MaterialX geomprop provider IDs matching mtlx_geomprop_names.",
                reflection::default_value(detail::PropertyList::create(std::vector<int64_t>{})),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_optimize_graph",
                &MaterialXMaterial::m_mtlx_optimize_graph,
                "Optimize MaterialX by pruning and simplifying statically known closures.",
                reflection::default_value(materialx::OptimizeGraphFlags::closure_pruning),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_layering_mode",
                &MaterialXMaterial::m_mtlx_layering_mode,
                "Mx139 layering mode: closure_tree or bsdf_mix. Performance guidance from the 2026-06 MX139 sweep: "
                "use bsdf_mix for the balanced public default.",
                reflection::default_value(materialx::default_layering_mode()),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "mtlx_compensation",
                &MaterialXMaterial::m_mtlx_compensation,
                "Microfacet compensation mode.",
                reflection::default_value(materialx::CompensationMode::turquin_analytic),
                reflection::on_change(&MaterialXMaterial::require_codegen),
                reflection::UIFlags::advanced
            )
            .def_rw(
                "debug_write_shader_path",
                &MaterialXMaterial::m_debug_write_shader_path,
                "Write the generated Slang shader to this path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            )
            .def_rw(
                "debug_load_shader_path",
                &MaterialXMaterial::m_debug_load_shader_path,
                "Load a previously generated Slang shader from this path.",
                reflection::default_value(std::string{}),
                reflection::on_change(&MaterialXMaterial::require_codegen)
            );
        /// TODO(@tdavidovic,@aweidlich): Investigate how to hook up homogeneous volumes properly.
        // .def_rw(
        //     "volume_sigmaA",
        //     &MaterialXMaterial::m_volume_sigmaA,
        //     "Volume sigmaA.",
        //     reflection::default_value(std::string{}),
        //     reflection::on_change(&MaterialXMaterial::require_codegen)
        // )
        // .def_rw(
        //     "volume_sigmaS",
        //     &MaterialXMaterial::m_volume_sigmaS,
        //     "Volume sigmaS.",
        //     reflection::default_value(std::string{}),
        //     reflection::on_change(&MaterialXMaterial::require_codegen)
        // )
        // .def_rw(
        //     "volume_hg",
        //     &MaterialXMaterial::m_volume_hg,
        //     "Volume hg.",
        //     reflection::default_value(std::string{}),
        //     reflection::on_change(&MaterialXMaterial::require_codegen)
        // )
        // .def_rw(
        //     "volume_2hg",
        //     &MaterialXMaterial::m_volume_2hg,
        //     "Volume 2hg.",
        //     reflection::default_value(std::string{}),
        //     reflection::on_change(&MaterialXMaterial::require_codegen)
        // )
    }

public:
    // Accessors for Python Material.get_this() only.
    bool _requires_data_buffer() const { return m_codegen_result && requires_data_buffer(); }
    const ref<ManagedBuffer>& _data_buffer() const { return m_data_buffer; }
    const std::vector<materialx::MxParamInfo>& _all_material_params() const
    {
        static const std::vector<materialx::MxParamInfo> k_empty;
        if (!m_codegen_result)
            return k_empty;
        return m_codegen_result->all_material_params.m_params;
    }

private:
    void require_codegen();
    void validate_device_support() const;
    void run_codegen();
    detail::PropertyList mtlx_geomprop_names_property() const;
    void set_mtlx_geomprop_names_property(const detail::PropertyList& value);
    detail::PropertyList mtlx_geomprop_ids_property() const;
    void set_mtlx_geomprop_ids_property(const detail::PropertyList& value);

    template<typename CursorT>
    void write_to_cursor_impl(CursorT cursor) const;

    bool requires_data_buffer() const { return m_codegen_result->data_struct_size > m_codegen_desc.payload_size; }
    size_t required_data_buffer_size() const { return requires_data_buffer() ? m_codegen_result->data_struct_size : 0; }

    // Static properties.
    std::filesystem::path m_mtlx_basepath;
    std::string m_mtlx_buffer;
    std::string m_mtlx_search_paths;
    std::filesystem::path m_mtlx_path;
    std::string m_mtlx_editable_params;
    std::string m_mtlx_node_name;
    std::string m_mtlx_layeringmethod;
    std::string m_mtlx_transmissive_bsdfs;
    bool m_mtlx_auto_transmission{true};
    materialx::OptimizeGraphFlags m_mtlx_optimize_graph{materialx::OptimizeGraphFlags::closure_pruning};
    materialx::LayeringMode m_mtlx_layering_mode{materialx::default_layering_mode()};
    materialx::CompensationMode m_mtlx_compensation{materialx::CompensationMode::turquin_analytic};
    bool m_mtlx_autogamma{false};
    std::string m_mtlx_target_color_space_override;
    bool m_mtlx_use_slang_derivatives{false};
    std::vector<std::string> m_mtlx_geomprop_names;
    std::vector<uint32_t> m_mtlx_geomprop_ids;
    std::string m_debug_write_shader_path;
    std::string m_debug_load_shader_path;
    // std::string m_volume_sigmaA;
    // std::string m_volume_sigmaS;
    // std::string m_volume_hg;
    // std::string m_volume_2hg;

    /// Dynamic properties (generated from MaterialX definition).
    reflection::DynamicPropertySet m_material_properties;

    bool m_require_codegen{true};
    size_t m_codegen_properties_hash{0};

    /// Attempts to resolve paths based on the library path
    ref<AssetResolver> m_mtlx_path_resolver;

    materialx::CodeGenDesc m_codegen_desc;
    std::unique_ptr<materialx::CodeGenResult> m_codegen_result;
    ref<ManagedBuffer> m_data_buffer;

    // When true, PathState must allow storing volume properties (sigmaA, sigmaS, and a flag they're set).
    // Left separated out so we don't forget about it, but it really should be only on-demand obtained,
    // from CodeGenResult::has_entry_point_volume_properties.
    bool m_has_entry_point_volume_properties = false;

    ref<SceneGlobals> m_lut_globals;
    ref<sgl::SlangModule> m_slang_module;
};

} // namespace falcor
