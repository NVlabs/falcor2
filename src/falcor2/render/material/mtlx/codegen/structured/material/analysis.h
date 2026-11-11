// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Element.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderStage.h>

namespace falcor::mtlx::structured {

/// Report whether the graph's surface output can reach an EDF.
bool graph_has_emission(const MaterialX::ShaderGraph& graph);
/// Report whether generated graph source needs Falcor's space-conversion helpers.
bool graph_uses_space_helpers(const MaterialX::ShaderGraph& graph);
/// Report whether the first output is a surface shader or material value.
/// Such outputs expose the generated `.opacity` and `.edf.radiance` fields, so opacity analysis and material-wrapper
/// field access are applicable. This does not mean the material actually emits light or has non-opaque opacity.
bool output_block_has_material_output(const MaterialX::VariableBlock& outputs);
/// Conservatively determine whether a document element may describe emission.
bool element_may_emit(const MaterialX::ElementPtr& element);
} // namespace falcor::mtlx::structured
