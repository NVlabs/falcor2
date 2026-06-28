// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"
#include "falcor2/core/object.h"

#include <sgl/device/formats.h>
#include <sgl/device/device.h>

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <span>

namespace falcor {

/// @brief MDL context wrapper.
///
/// This wrapper class implements the MDL material compilation pipeline.
class FALCOR_API MDLContext {
public:
    static MDLContext& get();

    ~MDLContext();

    /// MDL compilation request.
    struct CompileRequest {
        /// Qualified name of the material to compile (i.e. [::<package>]::<module>::<material>).
        std::string qualified_material_name;
        std::string instance_name;

        /// Use class compilation model instead of instance compilation model.
        bool use_class_compilation = false;

        /// Dump additional metadata into the generated source.
        bool dump_metadata = false;

        /// Compiler options.
        /// Note: These match the MDL compiler's command line options.
        struct {
            bool compile_constants = true;
            bool fast_math = true;
            uint32_t opt_level = 2; // 0 = none, 1 = medium, 2 = full optimization
            bool enable_auxiliary = true;
            bool use_renderer_adapt_normal = false;
            uint32_t df_handle_slot_mode = 0; // 0 = none, 1, 2, 4, 8 = fixed_X
            bool texture_runtime_with_derivs = false;
            uint32_t num_texture_results = 16;
            uint32_t num_texture_spaces = 4;
            bool jit_warn_spectrum_conversion = false;

            bool fold_ternary_on_df = false;
            bool fold_all_bool_parameters = false;
            bool fold_all_enum_parameters = false;
            bool ignore_noinline = true;

            // tag all leaf dfs and collect roughness/tint/tangent properties
            bool enable_tagged_df_properties = false;
            // replace specular dfs by glossy dfs
            bool speculars_as_glossy = false;
            // add a mollification parameter
            bool enable_param_mollification = false;
        } options;
    };

    /// MDL compilation result.
    struct CompileResult {
        /// True if compilation was successful.
        bool success = false;

        /// Error message if compilation failed.
        std::string error_msg;

        /// Generated source code.
        std::string code;

        /// Description of a data segment.
        struct DataSegment {
            std::string name;
            std::vector<uint8_t> data;
        };

        /// Description of a texture resource.
        struct Texture {
            enum class Type {
                Texture2D,
                Texture3D,
            };

            uint32_t index;

            std::string name;
            std::string path;
            bool srgb;

            Type type;
            uint32_t width;
            uint32_t height;
            uint32_t depth;
            sgl::Format format;
            std::vector<uint8_t> data;
        };

        struct ArgumentLayout {
            enum class Type {
                int_,
                bool_,
                float_,
                float2_,
                float3_,
                float4_,
                float4x4_,
                texture,
                /// Type is not supported by Properties (strings, arrays)
                unsupported,
            };

            Type type;
            uint32_t size;
            uint32_t offset;
        };

        /// Number of handles for the surface scatter function.
        uint32_t surface_scatter_handle_count;

        /// Read-only data segments used by the compiled material.
        std::vector<DataSegment> ro_data_segments;
        size_t ro_data_segment_size() const { return ro_data_segments.empty() ? 0 : ro_data_segments[0].data.size(); }
        /// Texture resources used by the compiled material.
        std::vector<Texture> textures;
        /// Argument block segments used by the compiled material.
        std::vector<DataSegment> arg_block_segments;
        size_t arg_block_segment_size() const
        {
            return arg_block_segments.empty() ? 0 : arg_block_segments[0].data.size();
        }
        /// Index of refraction in the interior of the material.
        float3 ior;

        // Description of argument block layout.
        std::map<std::string, ArgumentLayout, std::less<>> arg_block_layout;
    };

    class IImpl {
    public:
        virtual ~IImpl() = default;
        virtual void* get_neuray() = 0;
        virtual bool add_search_paths(const std::span<const std::filesystem::path>& paths) = 0;
        virtual bool remove_search_paths(const std::span<const std::filesystem::path>& paths) = 0;
    };

    /// @brief Compile an MDL material.
    CompileResult compile(const CompileRequest& request);

    void* get_neuray() { return m_impl ? m_impl->get_neuray() : nullptr; };
    bool add_search_paths(std::span<const std::filesystem::path> mdl_paths)
    {
        return m_impl ? m_impl->add_search_paths(mdl_paths) : false;
    };
    bool remove_search_paths(std::span<const std::filesystem::path> mdl_paths)
    {
        return m_impl ? m_impl->remove_search_paths(mdl_paths) : false;
    };

private:
    MDLContext();
    std::unique_ptr<MDLContext::IImpl> m_impl;
};

}; // namespace falcor
