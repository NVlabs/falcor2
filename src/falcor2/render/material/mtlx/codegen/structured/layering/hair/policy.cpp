// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../policy.h"
#include "../policy_factories.h"
#include "../basic/material_wrapper.h"

namespace falcor::mtlx::structured {
namespace {

/// Single-fiber scattering uses graph parameters without a surface layering representation.
class HairPolicy final : public ILayeringPolicy {
public:
    void analyze_original_layering(ClassifiedLayering&) override { }

    void build_representation(ClassifiedLayering&& layering, const CodeGenDesc&) override
    {
        FALCOR_CHECK(
            layering.bsdfs.size() == 1 && layering.bsdfs[0].kind == BsdfKind::chiang_hair
                && layering.main_layer == ClosureRef::bsdf(0) && layering.combiners.empty(),
            "MaterialX hair representation requires one terminal Chiang closure."
        );
    }

    GraphStoragePlan graph_storage_plan(const GeneratedNames&) const override { return {}; }

    void emit_representation_declarations(EmissionContext&) const override { }

    void emit_material_implementation(EmissionContext& context) const override
    {
        BasicMaterialWrapperDesc desc;
        desc.aliases_name = "MxHairAliases_$HASH";
        desc.aliases = {{"RootBSDF", "MxChiangHairBSDF"}};
        desc.root_value = "stack.bsdf0";
        // Graph frames are relative to graph_si. Preserve the fiber tangent when orthogonalizing.
        desc.root_value_lines
            = "result.shading_frame_ws = Frame(graph_si.shading_frame_ws.to_global(stack.bsdf_n[0]), "
              "graph_si.shading_frame_ws.to_global(stack.bsdf_t[0]), 1.0, Frame::Component::tangent);\n";
        emit_basic_material_wrapper(context, desc);
    }
};

} // namespace

std::unique_ptr<ILayeringPolicy> create_hair_policy()
{
    return std::make_unique<HairPolicy>();
}

} // namespace falcor::mtlx::structured
