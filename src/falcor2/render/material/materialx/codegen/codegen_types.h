// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/properties.h"
#include "mx_parameters.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace falcor {

class Properties;
class AssetResolver;

namespace materialx {

struct RenderableElement {
    std::string name;
    std::string type;
};

// Declared outside CodeGen to allow forward declare.
struct CodeGenResult {
    struct GeometryStreamRequest {
        std::string name;
        std::string type;
        uint32_t id{};
    };

    // The name after `module` keyword.
    std::string module_name;
    // The generated source code.
    std::string module_source;
    // Name of the struct that implements `Material`.
    std::string material_name;
    // Name of the struct that implements `IMaterialInstance`.
    std::string material_instance_name;
    // Name of the material data struct from which instance is created.
    // This contains all the editable parameters.
    std::string material_data_name;
    // Size of the material data struct. If it is <= than `CodeGenDesc::payload_size`,
    // it is stored in the MaterialData, just after MaterialHeader.
    // Otherwise MaterialData contains BufferHandle that is used to obtain the struct
    // from BufferManager.
    uint32_t data_struct_size{};
    // All parameters the material could possibly expose.
    // Includes params that are exposed (and end up in material data struct),
    // as well as params that are compiled in, but could be exposed (for UI handling).
    MxParamList all_material_params;
    // The total byte size of the material instance.
    size_t material_instance_byte_size{};
    // True when the generated material needs path-state storage for entry-point volume properties.
    bool has_entry_point_volume_properties = false;
    // Metadata emitted by codegen for diagnostics and reports.
    Properties codegen_metadata;
    // Geometry streams needed by the generated graph. Entries are stable text descriptors such as
    // `geomprop:color_0:color4:12` or `geomprop:custom_stream:vector3:77` for diagnostics and reports.
    std::vector<std::string> required_geometry_streams;
    // Typed geomprop/primvar requests emitted through GeomPropProvider.
    std::vector<GeometryStreamRequest> required_geomprop_streams;
};

enum class OptimizeGraphFlags : uint32_t {
    none = 0,
    /// Remove BSDFs, mixes, and layers with provably zero contribution.
    closure_pruning = (1 << 0),
    /// Set IMxScatterModePolicy as Static rather than Runtime if possible.
    static_scatter_mode = (1 << 1),
    /// Set IMxFresnelPolicy to the simplest mode needed (Airy, Color82, standard).
    fresnel_policy = (1 << 2),
    all = closure_pruning | static_scatter_mode | fresnel_policy,
};
FALCOR_ENUM_CLASS_OPERATORS(OptimizeGraphFlags);
SGL_ENUM_FLAGS_INFO(
    OptimizeGraphFlags,
    {
        {OptimizeGraphFlags::none, "none"},
        {OptimizeGraphFlags::closure_pruning, "closure_pruning"},
        {OptimizeGraphFlags::static_scatter_mode, "static_scatter_mode"},
        {OptimizeGraphFlags::fresnel_policy, "fresnel_policy"},
        {OptimizeGraphFlags::all, "all"},
    }
);
SGL_ENUM_REGISTER(OptimizeGraphFlags);

enum class LayeringMode {
    closure_tree,
    bsdf_mix,
};
SGL_ENUM_INFO(
    LayeringMode,
    {
        {LayeringMode::closure_tree, "closure_tree"},
        {LayeringMode::bsdf_mix, "bsdf_mix"},
    }
);
SGL_ENUM_REGISTER(LayeringMode);

enum class CompensationMode {
    turquin_lut,
    turquin_analytic,
    no_compensation,
};
SGL_ENUM_INFO(
    CompensationMode,
    {
        {CompensationMode::turquin_lut, "turquin_lut"},
        {CompensationMode::turquin_analytic, "turquin_analytic"},
        {CompensationMode::no_compensation, "no_compensation"},
    }
);
SGL_ENUM_REGISTER(CompensationMode);

struct CodeGenDesc {
    using GeomPropIdCallback
        = std::function<std::optional<uint32_t>(std::string_view geomprop, std::string_view output_type)>;

    /// Either the mtlx directly, or a path to it.
    std::variant<std::string, std::filesystem::path> document;
    /// Name of the node to build from. Will try auto-detect if left empty.
    std::string node_name;
    /// Optional resolver used to resolve the document and MaterialX search paths.
    ref<AssetResolver> asset_resolver;

    /// When true, use position free layering instead of the standard one.
    bool positionfree_layering = false;

    /// Optional homogeneous volume properties.
    float3 sigma_a = float3(0);
    float3 sigma_s = float3(0);
    /// Phase function: Isotropic, 1 HenyeyGreenstain with a float parameter,
    /// or 2 HG with g0, g1, w0 (param, param, weight of the first one)
    std::variant<std::monostate, float, std::array<float3, 3>> phase_function_params;

    /// List of BSDFs that can transmit through the material.
    /// TODO(@tdavidovic/@aweidlich): Investigate the use the R/T/RT on bsdfs instead.
    std::vector<std::string> transmissive_bsdfs;
    /// When true, infer transmissive BSDFs from MaterialX node implementations and scatter modes.
    bool auto_transmission = true;

    /// Mx139 graph optimizations for the genslangpt backend.
    OptimizeGraphFlags optimize_graph_flags = OptimizeGraphFlags::all;

    /// Mx139 layering implementation used by the MaterialX 1.39 genslangpt backend.
    ///
    /// closure_tree emits the closure tree directly, and bsdf_mix emits a generated
    /// BSDF-mixture root.
    LayeringMode layering_mode = LayeringMode::bsdf_mix;

    /// Mx139 microfacet energy-compensation policy used by the MaterialX 1.39 genslangpt backend.
    CompensationMode compensation_mode = CompensationMode::turquin_analytic;


    /// When true, image nodes with colorN output ask the texture loader for srgb conversion.
    /// Some Houdini-authored assets rely on this, but MaterialX files with explicit colorspace
    /// transforms such as srgb_texture -> lin_rec709 will be double-decoded if this is also enabled.
    bool auto_gamma_image = false;

    /// Optional target colorspace for generated MaterialX color transforms.
    /// Empty leaves MaterialX to use the active document colorspace.
    /// TODO(mx139): Default MX139 to lin_rec709. Wrapper documents can omit a root
    /// colorspace and then lose explicit srgb_texture -> lin_rec709 shader transforms.
    /// Keep auto_gamma_image false for MaterialX files that emit shader-side transforms.
    std::string target_color_space_override;

    /// When true, inherit upstream Slang/GLSL derivative implementations such as heighttonormal.
    /// Leave false for path tracing because ray tracing shaders do not provide ddx/ddy derivatives.
    bool use_slang_derivatives = false;

    /// When set to true, all unbound parameters of the material are exposed.
    bool make_editable = false;
    /// Size of the payload we can fit into the MaterialPayload.
    /// If the data does not fit, we use a buffer instead.
    size_t payload_size = 0;

    /// Optional mapping from MaterialX geomprop string/type pairs to stable shader IDs.
    /// Missing mappings intentionally fall back to the legacy built-in codegen expressions.
    GeomPropIdCallback geomprop_id_callback;
};

} // namespace materialx
} // namespace falcor
