// SPDX-License-Identifier: Apache-2.0

#include "material_emit.h"

#include "methods/basic/closure_tree_emit.h"

#include "../profile_policy.h"

#include "falcor2/core/macros.h"

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

void emit_material_structs(
    const MaterialX::ShaderGenerator& shadergen,
    const EmitRewriteSnippetFn& emit_rewrite_snippet,
    const MaterialStructsDesc& params,
    MaterialX::ShaderStage& stage
)
{
    FALCOR_UNUSED(emit_rewrite_snippet);


    const ProfileDesc& profile = profile_desc();
    emit_closure_tree_material_structs(
        shadergen,
        ClosureTreeMaterialStructsDesc{
            params.material_name,
            params.material_instance_name,
            params.material_data_name,
            params.graph_name,
            params.layering,
            params.codegen_desc,
            profile,
            params.graph_has_emission,
            params.materialx_payload_size,
        },
        stage
    );
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
