// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "walker.h"

#include "../graph/static_input_query/input_static_values.h"
#include "../graph/static_input_query/static_input_query.h"

#include "falcor2/core/macros.h"

#include <MaterialXCore/Types.h>

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <map>
#include <set>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

constexpr const char* SURFACE_UNLIT_TRANSPARENT_IMPL = "IM_surface_unlit_transparent_genslangpt";
constexpr const char* WEIGHTED_LAYER_BSDF_IMPL = "IM_weighted_layer_bsdf_genslangpt";
constexpr const char* LAYER_VDF_IMPL = "IM_layer_vdf_genslangpt";

struct CombinerSpec {
    const char* implementation;
    CombinerMode mode;
    std::array<const char*, 2> inputs;
};

constexpr std::array<CombinerSpec, 6> COMBINER_SPECS = {{
    {"IM_layer_bsdf_genslangpt", CombinerMode::layer, {"top", "base"}},
    {WEIGHTED_LAYER_BSDF_IMPL, CombinerMode::weighted_layer, {"top", "base"}},
    {"IM_mix_bsdf_genslangpt", CombinerMode::mix, {"fg", "bg"}},
    {"IM_mix_bsdfC_genslangpt", CombinerMode::mix, {"fg", "bg"}},
    {"IM_mix_surfaceshader_genslangpt", CombinerMode::mix, {"fg", "bg"}},
    {"IM_add_bsdf_genslangpt", CombinerMode::add, {"in1", "in2"}},
}};

const CombinerSpec* find_combiner_spec(const std::string& implementation)
{
    const auto found = std::find_if(
        COMBINER_SPECS.begin(),
        COMBINER_SPECS.end(),
        [&](const CombinerSpec& spec)
        {
            return implementation == spec.implementation;
        }
    );
    return found == COMBINER_SPECS.end() ? nullptr : &*found;
}

std::string optional_input_value(const mx::ShaderNode& node, const std::string& name, const std::string& fallback)
{
    const mx::ShaderInput* input = node.getInput(name);
    return input && input->getValue() ? input->getValue()->getValueString() : fallback;
}

BsdfLobeFlags glossy_scatter_lobe_flags(std::string_view scatter_mode)
{
    BsdfLobeFlags flags = BsdfLobeFlags::none;
    if (scatter_mode_includes_reflection(scatter_mode))
        flags |= BsdfLobeFlags::glossy_reflection;
    if (scatter_mode_includes_transmission(scatter_mode))
        flags |= BsdfLobeFlags::glossy_transmission;
    return flags;
}

void reconcile_shared_microfacet_nodes(std::vector<ClassifiedLayering::BsdfDesc>& bsdfs)
{
    // MaterialX emits one function for an implementation graph even when the
    // graph is instantiated several times. Its concrete BSDF type must
    // therefore represent every call site: retain a specialization only when
    // all invocations agree, otherwise use the least-specialized correct type.
    std::unordered_map<const mx::ShaderNode*, MxMicrofacetSelection> shared_selections;
    for (const ClassifiedLayering::BsdfDesc& bsdf : bsdfs) {
        if (!bsdf.microfacet_selection)
            continue;
        auto [found, inserted] = shared_selections.emplace(bsdf.shader_node, *bsdf.microfacet_selection);
        if (!inserted)
            found->second = common_microfacet_selection(found->second, *bsdf.microfacet_selection);
    }

    for (ClassifiedLayering::BsdfDesc& bsdf : bsdfs) {
        if (!bsdf.microfacet_selection)
            continue;
        bsdf.microfacet_selection = shared_selections.at(bsdf.shader_node);
        bsdf.bsdf_type = microfacet_bsdf_type(*bsdf.microfacet_selection);
    }
}

class WalkerImpl {
    struct OutputKey {
        std::vector<mx::ShaderNode*> invocation_nodes;
        mx::ShaderOutput* output = nullptr;
        auto operator<=>(const OutputKey&) const = default;
    };

public:
    static ClassifiedLayering
    build(mx::ShaderGraph& graph, const StaticInputQuery* static_input_query, std::vector<std::string>& events)
    {
        WalkerImpl walker(static_input_query, events);
        walker.m_path_prefixes.push_back("");
        walker.record("begin:<root>");
        walker.visit_graph(graph);
        walker.finish(graph);
        return std::move(walker.m_result);
    }

private:
    static bool is_closure_type(mx::TypeDesc type)
    {
        return type == mx::Type::BSDF || type == mx::Type::SURFACESHADER || type == mx::Type::MATERIAL;
    }

    static bool is_emitted_closure_node(const mx::ShaderNode* node)
    {
        return node->hasClassification(mx::ShaderNode::Classification::CLOSURE)
            || node->hasClassification(mx::ShaderNode::Classification::SHADER)
            || node->hasClassification(mx::ShaderNode::Classification::MATERIAL);
    }

    template<typename T>
    static std::vector<T*> closure_ports(const std::vector<T*>& ports)
    {
        std::vector<T*> result;
        for (T* port : ports) {
            if (is_closure_type(port->getType()))
                result.push_back(port);
        }
        return result;
    }

    static void append_dependencies(std::vector<mx::ShaderNode*>& order, mx::ShaderNode* node)
    {
        if (node->hasClassification(mx::ShaderNode::Classification::MATERIAL)) {
            append_input(order, node->getInput(mx::ShaderNode::SURFACESHADER));
            order.push_back(node);
            return;
        }
        for (mx::ShaderInput* input : node->getInputs())
            append_input(order, input);
        order.push_back(node);
    }

    static void append_input(std::vector<mx::ShaderNode*>& order, mx::ShaderInput* input)
    {
        mx::ShaderNode* upstream = input ? input->getConnectedSibling() : nullptr;
        if (upstream && is_emitted_closure_node(upstream))
            append_dependencies(order, upstream);
    }

    static std::vector<mx::ShaderNode*> closure_emit_order(mx::ShaderGraph& graph)
    {
        std::vector<mx::ShaderNode*> result;
        for (mx::ShaderGraphOutputSocket* socket : graph.getOutputSockets()) {
            mx::ShaderNode* upstream = socket->getConnection() ? socket->getConnection()->getNode() : nullptr;
            if (upstream && upstream->getParent() == &graph && is_emitted_closure_node(upstream))
                append_dependencies(result, upstream);
        }
        return result;
    }

    void visit_graph(mx::ShaderGraph& graph)
    {
        std::set<mx::ShaderNode*> visited;
        for (mx::ShaderNode* node : closure_emit_order(graph)) {
            if (!visited.insert(node).second)
                continue;
            if (mx::ShaderGraph* implementation_graph = node->getImplementation().getGraph()) {
                enter(node);
                connect_inputs(node, *implementation_graph);
                visit_graph(*implementation_graph);
                connect_outputs(node, *implementation_graph);
                leave(node);
            } else {
                add_node(node);
            }
        }
    }

    void enter(mx::ShaderNode* node)
    {
        m_invocation_nodes.push_back(node);
        m_path_prefixes.push_back(m_path_prefixes.back() + node->getName() + ":");
        record("begin:" + node->getName());
    }

    void leave(mx::ShaderNode* node)
    {
        FALCOR_ASSERT(!m_invocation_nodes.empty() && m_invocation_nodes.back() == node);
        m_invocation_nodes.pop_back();
        m_path_prefixes.pop_back();
        record("end:" + node->getName());
    }

    OutputKey key(mx::ShaderOutput* output, int skip = 0) const
    {
        OutputKey result{m_invocation_nodes, output};
        while (skip-- > 0 && !result.invocation_nodes.empty())
            result.invocation_nodes.pop_back();
        return result;
    }

    OutputKey resolve_alias(OutputKey value) const
    {
        while (true) {
            auto found = m_aliases.find(value);
            if (found == m_aliases.end())
                return value;
            value = found->second;
        }
    }

    void connect_inputs(mx::ShaderNode* node, mx::ShaderGraph& subgraph)
    {
        for (mx::ShaderGraphInputSocket* socket : subgraph.getInputSockets()) {
            mx::ShaderInput* parent = is_closure_type(socket->getType()) ? node->getInput(socket->getName()) : nullptr;
            if (parent && parent->getConnection())
                m_aliases[key(socket)] = resolve_alias(key(parent->getConnection(), 1));
        }
    }

    void connect_outputs(mx::ShaderNode* node, mx::ShaderGraph& subgraph)
    {
        FALCOR_ASSERT_EQ(node->getOutputs().size(), subgraph.getOutputSockets().size());
        for (std::size_t i = 0; i < node->getOutputs().size(); ++i) {
            mx::ShaderOutput* outer = node->getOutputs()[i];
            mx::ShaderOutput* inner = subgraph.getOutputSockets()[i]->getConnection();
            if (outer && inner && is_closure_type(outer->getType()))
                m_aliases[key(outer, 1)] = key(inner);
            record("connect:" + port_name(outer) + "=" + port_name(inner));
        }
    }

    bool resolve(mx::ShaderInput* input, ClosureRef& ref, RootContextualWeight* weight = nullptr) const
    {
        if (!input || !input->getConnection())
            return false;
        const OutputKey resolved = resolve_alias(key(input->getConnection()));
        auto found = m_output_bindings.find(resolved);
        if (found == m_output_bindings.end())
            return false;
        ref = found->second.ref;
        if (weight)
            *weight = found->second.weight;
        return true;
    }

    static RootContextualWeight static_weight(const ScalarStaticValue& value)
    {
        RootContextualWeight result;
        if (value.kind == ScalarStaticValueKind::KnownOne)
            return result;
        if (value.kind == ScalarStaticValueKind::Unknown) {
            result.kind = RootContextualWeight::Kind::dynamic;
            return result;
        }

        result.kind = RootContextualWeight::Kind::constant;
        if (value.kind == ScalarStaticValueKind::KnownZero) {
            result.constant = {0.0f, 0.0f, 0.0f};
            return result;
        }

        FALCOR_ASSERT(value.kind == ScalarStaticValueKind::KnownConstant);
        if (value.value.size != 1 && value.value.size != 3) {
            result.kind = RootContextualWeight::Kind::dynamic;
            return result;
        }
        for (size_t i = 0; i < result.constant.size(); ++i)
            result.constant[i] = value.value.values[value.value.size == 1 ? 0 : i];
        return result;
    }

    static RootContextualWeight multiply_weights(RootContextualWeight lhs, const RootContextualWeight& rhs)
    {
        if (lhs.kind == RootContextualWeight::Kind::identity)
            return rhs;
        if (rhs.kind == RootContextualWeight::Kind::identity)
            return lhs;
        if (lhs.kind == RootContextualWeight::Kind::dynamic || rhs.kind == RootContextualWeight::Kind::dynamic)
            return RootContextualWeight{RootContextualWeight::Kind::dynamic};

        for (size_t i = 0; i < lhs.constant.size(); ++i)
            lhs.constant[i] *= rhs.constant[i];
        if (lhs.constant == std::array{1.0f, 1.0f, 1.0f})
            return {};
        return lhs;
    }

    RootContextualWeight factor_weight(const mx::ShaderNode& node) const
    {
        const mx::ShaderInput* input = node.getInput("in2");
        if (!input)
            return RootContextualWeight{RootContextualWeight::Kind::dynamic};
        const ScalarStaticValue value
            = m_static_input_query ? m_static_input_query->input_value(*input) : lift_static_value(input->getValue());
        return static_weight(value);
    }

    static mx::ShaderInput* find_input(const std::vector<mx::ShaderInput*>& inputs, const std::string& name)
    {
        auto found = std::find_if(
            inputs.begin(),
            inputs.end(),
            [&](const mx::ShaderInput* input)
            {
                return input->getName() == name;
            }
        );
        return found == inputs.end() ? nullptr : *found;
    }

    void add_node(mx::ShaderNode* node)
    {
        record("add:" + node->getName());
        auto inputs = closure_ports(node->getInputs());
        auto outputs = closure_ports(node->getOutputs());
        if (outputs.empty())
            return;
        FALCOR_CHECK(
            outputs.size() == 1,
            "Structured MaterialX node '{}' has multiple closure outputs.",
            node->getName()
        );

        const std::string implementation = node->getImplementation().getName();
        if (outputs[0]->getType() == mx::Type::MATERIAL) {
            add_passthrough(outputs[0], find_input(inputs, "surfaceshader"));
        } else if (implementation == LAYER_VDF_IMPL) {
            // A VDF boundary modifies the connected surface closure's
            // transmission in generated graph code; it does not introduce a
            // surface lobe of its own. An unconnected top therefore remains
            // an empty surface root while the VDF expression is still emitted.
            add_passthrough(outputs[0], find_input(inputs, "top"));
        } else if (const CombinerSpec* combiner = find_combiner_spec(implementation)) {
            add_combiner(node, inputs, outputs[0], *combiner);
        } else if (implementation == "IM_multiply_bsdfC_genslangpt"
                   || implementation == "IM_multiply_bsdfF_genslangpt") {
            add_multiply(node, outputs[0], find_input(inputs, "in1"));
        } else if (outputs[0]->getType() == mx::Type::SURFACESHADER) {
            add_surface(node, inputs, outputs[0]);
        } else {
            std::vector<std::pair<ClosureRef, RootContextualWeight>> upstream;
            for (mx::ShaderInput* input : inputs) {
                ClosureRef ref;
                RootContextualWeight weight;
                if (resolve(input, ref, &weight))
                    upstream.emplace_back(ref, weight);
            }
            if (upstream.empty())
                add_bsdf(node, outputs[0]);
            else if (upstream.size() == 1)
                set_output(outputs[0], upstream.front().first, upstream.front().second);
            else
                FALCOR_THROW(
                    "Structured MaterialX node '{}' has multiple unsupported closure inputs.",
                    node->getName()
                );
        }
    }

    void add_passthrough(mx::ShaderOutput* output, mx::ShaderInput* input)
    {
        ClosureRef ref;
        RootContextualWeight weight;
        if (resolve(input, ref, &weight))
            set_output(output, ref, weight);
        else
            set_output(output, ClosureRef::none(), {});
    }

    void add_multiply(mx::ShaderNode* node, mx::ShaderOutput* output, mx::ShaderInput* input)
    {
        ClosureRef ref;
        RootContextualWeight weight;
        if (resolve(input, ref, &weight))
            set_output(output, ref, multiply_weights(weight, factor_weight(*node)));
        else
            set_output(output, ClosureRef::none(), {});
    }

    void add_surface(mx::ShaderNode* node, const std::vector<mx::ShaderInput*>& inputs, mx::ShaderOutput* output)
    {
        ClosureRef ref;
        RootContextualWeight weight;
        if (resolve(find_input(inputs, "bsdf"), ref, &weight)) {
            set_output(output, ref, weight);
            return;
        }
        if (node->getImplementation().getName() == "IM_surface_unlit_genslangpt") {
            const mx::ShaderInput* transmission = node->getInput("transmission");
            float value = 0.0f;
            std::stringstream(optional_input_value(*node, "transmission", "0")) >> value;
            if ((transmission && transmission->getConnection()) || value != 0.0f) {
                add_bsdf(node, output, SURFACE_UNLIT_TRANSPARENT_IMPL);
                return;
            }
        }
        set_output(output, ClosureRef::none(), {});
    }

    void add_bsdf(mx::ShaderNode* node, mx::ShaderOutput* output, std::string implementation_override = {})
    {
        ClassifiedLayering::BsdfDesc bsdf;
        bsdf.shader_node = node;
        bsdf.invocation_chain = invocation_chain();
        for (const std::string& invocation : bsdf.invocation_chain)
            bsdf.nodegraph_instance_path += invocation + ":";
        bsdf.node_path = node_path(node);
        bsdf.node_impl
            = implementation_override.empty() ? node->getImplementation().getName() : implementation_override;
        bsdf.scatter_mode = optional_input_value(*node, "scatter_mode", "");
        set_output(output, ClosureRef::bsdf(m_result.bsdfs.size()), {});
        m_result.bsdfs.push_back(std::move(bsdf));
    }

    void add_combiner(
        mx::ShaderNode* node,
        const std::vector<mx::ShaderInput*>& inputs,
        mx::ShaderOutput* output,
        const CombinerSpec& spec
    )
    {
        ClassifiedLayering::CombinerDesc combiner;
        combiner.mode = spec.mode;
        combiner.node_impl = node->getImplementation().getName();
        combiner.node_path = node_path(node);
        for (std::size_t i = 0; i < spec.inputs.size(); ++i) {
            const char* input_name = spec.inputs[i];
            FALCOR_CHECK(
                resolve(find_input(inputs, input_name), combiner.children[i]),
                "Structured MaterialX input '{}.{}' is not connected to a known closure.",
                node->getName(),
                input_name
            );
            FALCOR_CHECK(
                !combiner.children[i].is_none(),
                "Structured MaterialX input '{}.{}' resolves to an empty closure; empty combiner children must "
                "be pruned before layering.",
                node->getName(),
                input_name
            );
        }
        set_output(output, ClosureRef::combiner(m_result.combiners.size()), {});
        m_result.combiners.push_back(std::move(combiner));
    }

    void set_output(mx::ShaderOutput* output, ClosureRef ref, const RootContextualWeight& weight)
    {
        m_output_bindings[key(output)] = OutputBinding{ref, weight};
    }

    std::vector<std::string> invocation_chain() const
    {
        std::vector<std::string> result;
        result.reserve(m_invocation_nodes.size());
        for (const mx::ShaderNode* node : m_invocation_nodes)
            result.push_back(node->getName());
        return result;
    }

    std::string node_path(const mx::ShaderNode* node) const { return m_path_prefixes.back() + node->getName(); }

    void finish(mx::ShaderGraph& graph)
    {
        if (!graph.getOutputSockets().empty() && graph.getOutputSockets()[0]->getConnection()) {
            const OutputKey resolved = resolve_alias(key(graph.getOutputSockets()[0]->getConnection()));
            auto found = m_output_bindings.find(resolved);
            if (found != m_output_bindings.end()) {
                m_result.main_layer = found->second.ref;
                m_result.root_contextual_weight = found->second.weight;
            }
        }
        m_result.root_kind = m_result.main_layer.is_none() ? SurfaceRootKind::empty : SurfaceRootKind::non_empty;
        record("end:<root>");
    }

    void record(std::string event) { m_events.push_back(std::move(event)); }

    explicit WalkerImpl(const StaticInputQuery* static_input_query, std::vector<std::string>& events)
        : m_static_input_query(static_input_query)
        , m_events(events)
    {
    }

    static std::string port_name(mx::ShaderOutput* output)
    {
        if (!output)
            return "<null>";
        return output->getNode()->getName() + "." + output->getName();
    }

    struct OutputBinding {
        ClosureRef ref;
        RootContextualWeight weight;
    };

    std::map<OutputKey, OutputBinding> m_output_bindings;
    std::map<OutputKey, OutputKey> m_aliases;
    std::vector<mx::ShaderNode*> m_invocation_nodes;
    std::vector<std::string> m_path_prefixes;
    ClassifiedLayering m_result;
    const StaticInputQuery* m_static_input_query = nullptr;
    std::vector<std::string>& m_events;
};

void count_reachable_uses(
    const ClassifiedLayering& layering,
    ClosureRef ref,
    std::map<ClosureRef, int>& uses,
    std::set<ClosureRef>& visited
)
{
    if (!ref.is_combiner() || !visited.insert(ref).second)
        return;
    const auto& combiner = layering.combiners.at(ref.combiner_index());
    for (ClosureRef child : combiner.children) {
        if (!child.is_none())
            ++uses[child];
        count_reachable_uses(layering, child, uses, visited);
    }
}

} // namespace

ClassifiedLayering LayeringWalker::build_and_classify(
    mx::ShaderGraph& graph,
    const CodeGenDesc& desc,
    const mx::VariableBlock* constants,
    const StaticInputQuery* static_input_query,
    std::vector<std::string>& traversal_events
)
{
    ClassifiedLayering result = WalkerImpl::build(graph, static_input_query, traversal_events);
    const std::set<std::string> explicit_transmissive(desc.transmissive_bsdfs.begin(), desc.transmissive_bsdfs.end());
    for (ClassifiedLayering::BsdfDesc& bsdf : result.bsdfs) {
        const BsdfInfo* info = find_bsdf_info(bsdf.node_impl);
        FALCOR_CHECK(info, "Structured MaterialX unsupported BSDF implementation '{}'.", bsdf.node_impl);
        bsdf.kind = info->kind;
        bsdf.bsdf_type = info->type;
        bsdf.lobe_flags = info->lobe_flags;
        bsdf.transmissive
            = desc.auto_transmission ? info->transmissive : explicit_transmissive.contains(bsdf.node_path);
        if (microfacet_kind(bsdf.kind) != MxMicrofacetKind::none) {
            bsdf.microfacet_selection
                = select_microfacet(desc, bsdf.kind, bsdf.shader_node, constants, static_input_query);
            bsdf.bsdf_type = microfacet_bsdf_type(*bsdf.microfacet_selection);
            if (bsdf.microfacet_selection->scatter != MxScatterSelection::runtime) {
                const std::string scatter_mode = scatter_selection_mode_string(bsdf.microfacet_selection->scatter);
                bsdf.lobe_flags = glossy_scatter_lobe_flags(scatter_mode);
                if (desc.auto_transmission)
                    bsdf.transmissive = scatter_mode_includes_transmission(scatter_mode);
            }
        }
        if (bsdf.kind == BsdfKind::sheen) {
            bsdf.sheen_selection = select_sheen(desc);
            bsdf.bsdf_type = sheen_bsdf_type(*bsdf.sheen_selection);
        }
        if (is_scatter_mode_bsdf(bsdf.kind)) {
            const std::string scatter_mode
                = bsdf.microfacet_selection && bsdf.microfacet_selection->scatter != MxScatterSelection::runtime
                ? scatter_selection_mode_string(bsdf.microfacet_selection->scatter)
                : bsdf.scatter_mode;
            bsdf.lobe_flags = glossy_scatter_lobe_flags(scatter_mode);
            if (desc.auto_transmission)
                bsdf.transmissive = scatter_mode_includes_transmission(scatter_mode);
        }
        bsdf.through_material_transmissive = bsdf.transmissive;
        bsdf.preserve_tangent_frame = info->preserve_tangent_frame;
    }
    reconcile_shared_microfacet_nodes(result.bsdfs);
    return result;
}

BsdfLobeFlags reachable_lobe_flags(const ClassifiedLayering& layering, ClosureRef root)
{
    if (root.is_none())
        return BsdfLobeFlags::none;
    if (root.is_bsdf())
        return layering.bsdfs.at(root.bsdf_index()).lobe_flags;

    BsdfLobeFlags flags = BsdfLobeFlags::none;
    for (ClosureRef child : layering.combiners.at(root.combiner_index()).children)
        flags |= reachable_lobe_flags(layering, child);
    return flags;
}

std::string format_bsdf_lobe_flags(BsdfLobeFlags flags)
{
    // Keep the lexical ordering and left-associated parentheses produced by
    // the former std::set<string> implementation so this typed host cleanup
    // remains byte-identical at the Slang emission boundary.
    static constexpr std::array<std::pair<BsdfLobeFlags, std::string_view>, 6> LOBES = {{
        {BsdfLobeFlags::delta_transmission, "BSDFFlags::delta_transmission"},
        {BsdfLobeFlags::diffuse_reflection, "BSDFFlags::diffuse_reflection"},
        {BsdfLobeFlags::diffuse_transmission, "BSDFFlags::diffuse_transmission"},
        {BsdfLobeFlags::glossy_curve, "BSDFFlags::glossy_curve"},
        {BsdfLobeFlags::glossy_reflection, "BSDFFlags::glossy_reflection"},
        {BsdfLobeFlags::glossy_transmission, "BSDFFlags::glossy_transmission"},
    }};

    std::string result;
    for (const auto& [lobe, spelling] : LOBES) {
        if (is_set(flags, lobe))
            result = result.empty() ? std::string(spelling) : "(" + result + " | " + std::string(spelling) + ")";
    }
    return result;
}

void validate_weighted_layer_top_refs_are_not_shared(const ClassifiedLayering& layering)
{
    std::map<ClosureRef, int> uses;
    std::set<ClosureRef> visited;
    count_reachable_uses(layering, layering.main_layer, uses, visited);
    for (ClosureRef ref : visited) {
        const auto& combiner = layering.combiners.at(ref.combiner_index());
        FALCOR_CHECK(
            combiner.mode != CombinerMode::weighted_layer || combiner.children[0].is_none()
                || uses[combiner.children[0]] <= 1,
            "Structured MaterialX weighted-layer top '{}' is reused.",
            combiner.node_path
        );
    }
}

void lower_weighted_layers_once(ClassifiedLayering& layering)
{
    FALCOR_CHECK(!layering.weighted_layers_lowered, "Structured MaterialX weighted layers were lowered twice.");
    for (ClassifiedLayering::CombinerDesc& combiner : layering.combiners) {
        if (combiner.mode == CombinerMode::weighted_layer)
            combiner.mode = CombinerMode::layer;
    }
    layering.weighted_layers_lowered = true;
}

} // namespace falcor::mtlx::structured
