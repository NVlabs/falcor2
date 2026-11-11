// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/types.h"
#include "falcor2/core/object.h"

#include <sgl/device/formats.h>
#include <sgl/device/device.h>

#include <memory>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <optional>
#include <span>
#include <mutex>

namespace falcor {

class MaterialAOVRegistry;

/// @brief MDL context wrapper.
///
/// This wrapper class implements the MDL material compilation pipeline.
class FALCOR_API MDLContext {
public:
    static MDLContext& get();

    ~MDLContext();

    /// Source data shared by compiled and runtime texture resources.
    struct TextureSource {
        /// Declared MDL shape; resource loading currently supports 2D images and BSDF lookup data.
        enum class Type {
            Unknown,
            Texture2D,
            Texture3D,
            TextureCube,
            TexturePtex,
            TextureBSDFData,
        };

        Type type{Type::Unknown};
        std::filesystem::path path;
        bool srgb{false};
        /// Dimensions, format, and bytes for embedded texture data.
        uint32_t width{0};
        uint32_t height{0};
        uint32_t depth{0};
        sgl::Format format{sgl::Format::undefined};
        std::vector<uint8_t> data;
    };

    /// MDL compilation request.
    struct CompileRequest {
        /// Qualified name of the material to compile (i.e. [::<package>]::<module>::<material>).
        std::string qualified_material_name;
        std::string instance_name;

        /// Optional source for the module named by qualified_material_name.
        std::string module_source;

        /// Optional declarative material-category struct used as the effective compilation target.
        std::string target_material_type;

        /// Optional registry for supported material AOVs; must outlive compilation.
        /// Null skips AOV discovery and generation; every registered output receives a generated callable.
        MaterialAOVRegistry* material_aov_registry{nullptr};

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

        struct MaterialAOVOutput {
            std::string name;
            uint32_t component_count{0};
            uint32_t id{0};
            std::string callable_name;
        };

        /// Registered top-level fields with generated callables from a user-defined material-category struct.
        std::vector<MaterialAOVOutput> material_aov_outputs;

        /// SDK scene-data name inventory; conservative when the SDK cannot resolve a name.
        std::vector<std::string> referenced_scene_data;

        /// SDK runtime string IDs index this table, including captured defaults (ID zero is invalid).
        std::vector<std::string> string_constants;

        /// SDK ITarget_code::State_usage flags; independent of Scene attribute IDs.
        uint32_t render_state_usage = 0;

        /// Description of a data segment.
        struct DataSegment {
            std::string name;
            std::vector<uint8_t> data;
        };

        /// Description of a texture resource.
        struct Texture {
            uint32_t index;
            std::string name;
            TextureSource source;
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
            /// Retain the declared texture type even when the default resource ID is zero.
            TextureSource::Type texture_type{TextureSource::Type::Unknown};
        };

        /// Number of handles for the surface scatter function.
        uint32_t surface_scatter_handle_count;

        /// Structural EDF metadata; authored intensity stays in the argument block.
        bool has_surface_emission{false};
        uint32_t surface_emission_handle_count{0};
        /// False when surface emission is absent or structurally proven zero, independent of editable arguments.
        bool potentially_emissive{false};
        struct EmissionZeroCheck {
            std::string value_function;
            std::string condition_function; // Empty for an unconditional check.
            bool condition_value;
        };
        /// State-independent scalar/color expressions whose zero value proves zero surface intensity.
        std::vector<EmissionZeroCheck> emission_zero_checks;

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
        /// Constant geometry cutout opacity, or empty if it requires runtime evaluation.
        std::optional<float> cutout_opacity;

        /// Constant MDL thin_walled expression, if known at compilation time.
        /// Dynamic expressions cannot be represented by the current per-material header.
        std::optional<bool> thin_walled;

        // Description of argument block layout.
        std::map<std::string, ArgumentLayout, std::less<>> arg_block_layout;
    };

    class IImpl {
    public:
        virtual ~IImpl() = default;
        virtual void* get_neuray() = 0;
    };

    /// @brief Compile an MDL material.
    CompileResult compile(const CompileRequest& request);

    void* get_neuray() { return m_impl ? m_impl->get_neuray() : nullptr; };

private:
    friend class MDLSearchPathScope;
    MDLContext();
    std::recursive_mutex m_mutex;
    std::unique_ptr<MDLContext::IImpl> m_impl;
};

/// Serializes SDK access and temporarily prepends module and resource roots.
/// Nested scopes restore their immediate predecessor. Loaded modules remain cached by name.
class FALCOR_API MDLSearchPathScope {
public:
    MDLSearchPathScope(MDLContext& context, std::span<const std::filesystem::path> paths);
    ~MDLSearchPathScope() noexcept;
    MDLSearchPathScope(const MDLSearchPathScope&) = delete;
    MDLSearchPathScope& operator=(const MDLSearchPathScope&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

}; // namespace falcor
