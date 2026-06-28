// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "codegen_support/snippet_loader.h"
#include "graph_prepare/closure_pruning/closure_pruning_counters.h"
#include "layering_analysis.h"
#include "codegen.h"

#include <MaterialXCore/Document.h>
#include <MaterialXGenShader/GenUserData.h>

namespace falcor {
namespace materialx {
namespace mx139 {

// Read-only inputs supplied to the codegen pipeline before any shader is
// generated. Sub-struct of `CodegenUserData`.
struct CodegenInputs {
    const CodeGenDesc* desc = nullptr;
    MaterialX::DocumentPtr document;
    CodeGenResult* result = nullptr;
};

// Analysis results computed once per material before emission. Sub-struct of
// `CodegenUserData`. Owned by all emitters alike; method-specific scratch
// state should not be added here.
struct AnalysisState {
    LayeringDesc layering;
    bool graph_has_emission = false;
    graph_prepare::ClosurePruningCounters closure_pruning;
};

class CodegenUserData : public MaterialX::GenUserData {
public:
    CodegenInputs inputs;
    AnalysisState analysis;
    codegen_support::SnippetEmitState emit_state;
};

} // namespace mx139
} // namespace materialx
} // namespace falcor
