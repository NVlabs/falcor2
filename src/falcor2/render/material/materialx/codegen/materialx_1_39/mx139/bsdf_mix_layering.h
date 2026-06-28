// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "layering_analysis.h"

#include <string>
#include <vector>

namespace falcor::materialx::mx139 {

enum class MxFlatCombinerMode {
    layer,
    mix,
    add,
};

// Selects which BSDL-compatible sample albedo bsdf_mix layering should feed
// into BSDF summaries and branch probabilities. Mx139 BSDFs report a
// physical reflection/transmission split, while legacy BSDL filter_o sometimes
// exposes only reflection, or full unity for transmissive microfacet scatter.
enum class MxFlatLayerPassThroughAlbedoMode {
    // Use reflection + transmission from eval_albedo().
    scattering = 0,
    // Use reflection only; pass-through remains in filter_o transmission.
    reflection = 1,
    // For BSDFs with active transmissive scatter, use unity; otherwise use
    // reflection. This mirrors BSDL microfacet albedo/filter_o behavior.
    transmissive_unity = 2,
};

// Selects which BSDL-compatible filter_o pass-through bsdf_mix layering should
// feed into layer() opacity and material-instance transmission reconstruction.
enum class MxFlatLayerPassThroughTransmissionMode {
    // Use physical eval_albedo().transmission.
    physical = 0,
    // Force an opaque filter_o pass-through.
    zero = 1,
    // Force a fully transparent filter_o pass-through.
    one = 2,
    // Treat BSDL absorption as layer pass-through in addition to transmission.
    absorption = 3,
    // Use absorption unless this is an active transmissive scatter BSDF.
    absorption_unless_transmissive = 4,
    // Use physical transmission unless this is an active transmissive scatter
    // BSDF, where BSDL layering makes the BSDF opaque to the base.
    physical_unless_transmissive = 5,
};

struct MxFlatBsdfDesc {
    std::string field_name;
    std::string type_name;
    MxFlatLayerPassThroughAlbedoMode layer_pass_through_albedo_mode = MxFlatLayerPassThroughAlbedoMode::scattering;
    MxFlatLayerPassThroughTransmissionMode layer_pass_through_transmission_mode
        = MxFlatLayerPassThroughTransmissionMode::physical;
    bool uses_transmission_tint_predicate = false;
};

struct MxFlatCombinerDesc {
    int index = 0;
    MxFlatCombinerMode mode = MxFlatCombinerMode::layer;

    // branch0/branch1 are bsdf_mix traversal branches, not MaterialX
    // input names. Mapping after adapter normalization:
    // layer(top, base): branch0 = top, branch1 = base
    // mix(fg, bg, amount): branch0 = bg, branch1 = fg
    // add(in1, in2): branch0 = in1, branch1 = in2
    ClosureRef branch0_ref;
    ClosureRef branch1_ref;
};

struct MxFlatRootDesc {
    std::string type_name;
    ClosureRef root_ref;
    std::vector<MxFlatBsdfDesc> bsdfs;
    std::vector<MxFlatCombinerDesc> combiners;
};

std::string emit_bsdf_mix_root_bsdf(const MxFlatRootDesc& desc);

} // namespace falcor::materialx::mx139
