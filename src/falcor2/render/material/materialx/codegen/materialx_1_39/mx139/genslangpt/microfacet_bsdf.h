// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNodeImpl.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

namespace mx = MaterialX;

// Codegen factories for the GenslangPt microfacet BSDF nodes
// (dielectric_bsdf, conductor_bsdf, generalized_schlick_bsdf).
mx::ShaderNodeImplPtr create_microfacet_dielectric_node();
mx::ShaderNodeImplPtr create_microfacet_conductor_node();
mx::ShaderNodeImplPtr create_microfacet_generalized_schlick_node();

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
