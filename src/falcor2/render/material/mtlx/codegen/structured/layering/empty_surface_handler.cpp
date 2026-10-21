// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "empty_surface_handler.h"

#include "basic/material_wrapper.h"

#include "falcor2/core/error.h"

namespace falcor::mtlx::structured {

void EmptySurfaceHandler::build_representation(ClassifiedLayering&& layering)
{
    FALCOR_CHECK(!m_built, "Structured MaterialX empty-surface representation was built more than once.");
    FALCOR_CHECK(layering.root_kind == SurfaceRootKind::empty, "EmptySurfaceHandler received a surface BSDF root.");
    m_built = true;
}

GraphStoragePlan EmptySurfaceHandler::graph_storage_plan() const
{
    FALCOR_CHECK(m_built, "Structured MaterialX empty-surface storage was queried before representation build.");
    return {};
}

void EmptySurfaceHandler::emit_material_implementation(EmissionContext& context) const
{
    FALCOR_CHECK(m_built, "Structured MaterialX empty surface was emitted before representation build.");
    const std::string collect_properties
        = R"(    public MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        float3 wi_ls = shading_frame_ws.to_local(si.wi_ws);
        MaterialProperties result = {};
        result.emission = emissive_radiance;
        result.material_color = saturate(result.diffuse_reflection_albedo);
        return result;
    }
)";
    emit_basic_material_wrapper(
        context,
        BasicMaterialWrapperDesc{
            .aliases_name = "MxAliases_$HASH",
            .aliases_comment
            = "// Generated WeightedLayer<N> aliases below mean Weighted<Layer<N>> traversal wrappers.",
            .aliases
            = {{"Weighted<T : ILayerableBSDF>", "MxWeightedBSDF<T>"},
               {"FramedWeighted<T : ILayerableBSDF>", "MxFramedWeightedBSDF<T>"},
               {"RootBSDF", "MxNullBSDF"}},
            .lobe_flags = "BSDFFlags::none",
            .root_value = "mtlx_ns::RootBSDF()",
            .collect_properties_method = collect_properties,
        }
    );
}

} // namespace falcor::mtlx::structured
