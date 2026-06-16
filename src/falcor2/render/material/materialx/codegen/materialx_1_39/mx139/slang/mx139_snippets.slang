// SPDX-License-Identifier: Apache-2.0

module mx139_snippets;

import falcor2.render;
import falcor2.utils;
import falcor2.render.materials.materialx.materialx_1_39.libraries.mx139.mx139;

namespace mx139 {

struct SnippetDirectStack
{
    float3 bsdf_n[1] = {};
    float3 bsdf_t[1] = {};
};

struct BSDF
{
    int get_index() { return 0; }
};

static SurfaceInteraction si = {};
static SnippetDirectStack mx_stack = {};

__include "mx_space_constants.slang";
__include "mx_space_helpers.slang";
__include "mx_normal_adjustment.slang";
__include "mx_direct_microfacet_helpers.slang";
__include "surface_unlit_transparent.slang";
__include "texture_sampling_helper.slang";

} // namespace mx139
