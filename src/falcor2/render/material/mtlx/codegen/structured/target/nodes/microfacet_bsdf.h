// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNodeImpl.h>

namespace falcor::mtlx::structured {

/// Create structured target implementations for MaterialX microfacet BSDF
/// families. The implementations consume classified MxMicrofacetSelection facts
/// and emit the already-selected concrete Falcor BSDF constructor.
MaterialX::ShaderNodeImplPtr create_microfacet_dielectric_node();
MaterialX::ShaderNodeImplPtr create_microfacet_conductor_node();
MaterialX::ShaderNodeImplPtr create_microfacet_generalized_schlick_node();

} // namespace falcor::mtlx::structured
