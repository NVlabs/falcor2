// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "../../codegen_types.h"

#include <MaterialXCore/Document.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/ShaderGraph.h>

#include <memory>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {

class CodegenUserData;

// Builds the same prepared MaterialX shader graph that MX139 codegen hands to
// graph-preparation passes, without running code emission. Native tests use it
// to call graph passes directly instead of installing production callbacks.
class PreparedCodegenGraph {
public:
    FALCOR_API PreparedCodegenGraph(MaterialX::DocumentPtr doc, CodeGenDesc desc);
    FALCOR_API ~PreparedCodegenGraph();

    FALCOR_API PreparedCodegenGraph(PreparedCodegenGraph&&) noexcept;
    FALCOR_API PreparedCodegenGraph& operator=(PreparedCodegenGraph&&) noexcept;

    PreparedCodegenGraph(const PreparedCodegenGraph&) = delete;
    PreparedCodegenGraph& operator=(const PreparedCodegenGraph&) = delete;

    FALCOR_API MaterialX::ShaderGraph& graph();
    FALCOR_API MaterialX::GenContext& context();
    FALCOR_API CodegenUserData& user_data();
    FALCOR_API CodeGenResult& result();
    FALCOR_API const std::string& shader_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace mx139
} // namespace materialx
} // namespace falcor
