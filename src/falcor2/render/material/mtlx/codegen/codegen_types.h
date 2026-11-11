// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"
#include "falcor2/core/properties.h"
#include "falcor2/render/material_aov_registry.h"
#include "mx_parameters.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>
#include <vector>

namespace falcor {

class Properties;
class AssetResolver;

namespace mtlx {

struct RenderableElement {
    std::string name;
    std::string type;
};

// Declared outside CodeGen to allow forward declare.
struct CodeGenResult {
    struct OpacityResult {
        enum class Kind {
            opaque,
            constant,
            dynamic,
        };

        Kind kind{Kind::opaque};
        float constant{1.f};
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
    // Conservative target-independent upper bound for the material data struct size. If it is <=
    // `CodeGenDesc::payload_size`, the data is stored inline in MaterialData after MaterialHeader.
    // Otherwise MaterialData contains a BufferOffsetPair into shared material storage.
    // The loaded target module is reflected to verify this bound and obtain the exact buffer stride.
    size_t data_struct_size{};
    // All parameters the material could possibly expose.
    // Includes params that are exposed (and end up in material data struct),
    // as well as params that are compiled in, but could be exposed (for UI handling).
    MxParamList all_material_params;
    // The total byte size of the material instance.
    size_t material_instance_byte_size{};
    // True when the generated material needs path-state storage for entry-point volume properties.
    bool has_entry_point_volume_properties = false;
    // A supported terminal fiber closure. Recomputed
    // from the analyzed graph before every code-cache lookup, like opacity metadata.
    bool is_hair = false;
    // Classification of the generated root surface opacity.
    OpacityResult opacity;
    // Emitted operations need retained hit context, including in additional material outputs.
    bool uses_geometry_properties = false;
    // Metadata emitted by codegen for diagnostics and reports.
    Properties codegen_metadata;
    // Supported material AOV declarations with stable scene IDs.
    // Populated when a registry is supplied; every entry is emitted.
    std::vector<AOVDesc> material_aov_outputs;
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

inline constexpr LayeringMode default_layering_mode()
{
    LayeringMode mode = LayeringMode::bsdf_mix;
    return mode;
}

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

enum class ZeltnerSheenCoefficientMode {
    analytic_fit,
    lut,
};
SGL_ENUM_INFO(
    ZeltnerSheenCoefficientMode,
    {
        {ZeltnerSheenCoefficientMode::analytic_fit, "analytic_fit"},
        {ZeltnerSheenCoefficientMode::lut, "lut"},
    }
);
SGL_ENUM_REGISTER(ZeltnerSheenCoefficientMode);

struct CodeGenDesc {
    /// Either the mtlx directly, or a path to it.
    std::variant<std::string, std::filesystem::path> document;
    /// Name of the node to build from. Will try auto-detect if left empty.
    std::string node_name;
    /// Additional document outputs or selected root NodeDef outputs to retain in the same Graph. Supports '*' patterns;
    /// the default selects all eligible numeric outputs. Empty disables additional instance outputs.
    /// The primary root may be a named node/output or an automatically detected material.
    /// These are explicit code-generation roots and do not imply material ownership.
    std::vector<std::string> additional_output_names = {"*"};
    /// Optional resolver used to resolve the document and MaterialX search paths.
    ref<AssetResolver> asset_resolver;

    /// Legacy position-free layering request. The structured generator rejects
    /// this until a layering policy implements it.
    bool positionfree_layering = false;

    /// Legacy entry-point homogeneous volume properties, retained for source
    /// compatibility. Non-default values are rejected; author VDF and phase
    /// behavior in the MaterialX graph instead.
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

    /// Mtlx graph optimizations for the genslangpt backend.
    OptimizeGraphFlags optimize_graph_flags = OptimizeGraphFlags::all;

    /// Mtlx layering implementation used by the MaterialX 1.39 genslangpt backend.
    ///
    /// closure_tree emits the closure tree directly, and bsdf_mix emits a generated
    /// BSDF-mixture root.
    LayeringMode layering_mode = default_layering_mode();

    /// Mtlx microfacet energy-compensation policy used by the MaterialX 1.39 genslangpt backend.
    CompensationMode compensation_mode = CompensationMode::turquin_analytic;

    /// Zeltner sheen coefficient source. The merged MaterialX specification defines the analytic fit.
    ZeltnerSheenCoefficientMode zeltner_sheen_coefficients = ZeltnerSheenCoefficientMode::analytic_fit;


    /// When true, inherit upstream Slang/GLSL derivative implementations such as heighttonormal.
    /// Leave false for path tracing because ray tracing shaders do not provide ddx/ddy derivatives.
    bool use_slang_derivatives = false;

    /// When set to true, all supported unbound parameters of the material are exposed.
    bool make_editable = false;
    /// Canonicalize generated MaterialX identity without changing parameter liveness.
    bool canonicalize_names = false;
    /// Authored names of supported unbound parameters to expose unless make_editable is true.
    /// Omit the `inputs:` prefix. Selected-root inputs use bare names (base); other inputs use node_input
    /// (surface_base). Full authored paths (NG/surface/base) disambiguate names; ambiguous short names throw.
    std::unordered_set<std::string> editable_param_names;

    bool is_editable_param(const std::string& name) const
    {
        return make_editable || editable_param_names.contains(name);
    }

    /// Size of the payload we can fit into the MaterialPayload.
    /// If the data does not fit, we use a buffer instead.
    size_t payload_size = 0;

    /// Reserve the Scene-owned shader ID for a blind named geometry attribute.
    std::function<uint32_t(std::string_view)> reserve_geometry_attribute_id;

    /// Optional registry for float1-4 material-authored outputs; must outlive generation.
    /// Every supported candidate matched by additional_output_names is emitted.
    /// When supplied, the registry assigns stable IDs to all matched, runtime-supported outputs.
    MaterialAOVRegistry* material_aov_registry{nullptr};
};

} // namespace mtlx
} // namespace falcor
