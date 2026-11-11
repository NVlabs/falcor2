// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "user_data.h"

#include <memory>

namespace falcor::mtlx::structured {

EmissionUserData::EmissionUserData(
    const CodeGenDesc& request,
    const MaterialX::Document& document,
    const NodeEmissionFacts& node_emission_facts,
    EmissionScratch& emission_scratch,
    bool force_inline_graph_boundaries
)
    : m_request(&request)
    , m_document(&document)
    , m_node_emission_facts(&node_emission_facts)
    , m_emission_scratch(&emission_scratch)
    , m_force_inline_graph_boundaries(force_inline_graph_boundaries)
{
}

EmissionUserDataScope::EmissionUserDataScope(
    MaterialX::GenContext& context,
    const CodeGenDesc& request,
    const MaterialX::Document& document,
    const NodeEmissionFacts& node_emission_facts,
    EmissionScratch& emission_scratch,
    bool force_inline_graph_boundaries
)
    : m_context(&context)
{
    context.pushUserData(
        USER_DATA_NAME,
        std::shared_ptr<EmissionUserData>(new EmissionUserData(
            request,
            document,
            node_emission_facts,
            emission_scratch,
            force_inline_graph_boundaries
        ))
    );
}

EmissionUserDataScope::~EmissionUserDataScope()
{
    m_context->popUserData(USER_DATA_NAME);
}

} // namespace falcor::mtlx::structured
