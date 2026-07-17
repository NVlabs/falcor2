// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "layering_analysis.h"

#include "codegen_support/shader_input_value.h"
#include "policy.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Types.h>

#include <algorithm>
#include <array>
#include <compare>
#include <initializer_list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace mx = MaterialX;

namespace {

const char* k_surface_unlit_transparent_impl = "IM_surface_unlit_transparent_genslangpt";
const char* k_mx139_simple_btdf_impl = "__mx139_simple_btdf_synthetic";
const char* k_mx139_opacity_mix_impl = "__mx139_opacity_mix_synthetic";
const char* k_weighted_layer_bsdf_impl = "IM_weighted_layer_bsdf_genslangpt";

using codegen_support::optional_shader_input_value_string;

class LayeringDescBuilderImpl {
    struct OutputKey {
        std::vector<mx::ShaderNode*> path;
        mx::ShaderOutput* output = nullptr;
        auto operator<=>(const OutputKey&) const = default;
    };

public:
    static LayeringDesc build(mx::ShaderGraph& shader_graph, std::vector<std::string>* traversal_events)
    {
        LayeringDescBuilderImpl builder;
        builder.m_traversal_events = traversal_events;
        builder.reset_root_graph();
        builder.visit_graph(shader_graph);
        builder.finish_root_graph(shader_graph);
        return std::move(builder.m_layering_desc);
    }

private:
    void reset_root_graph()
    {
        m_layering_desc = {};
        m_output_to_index.clear();
        m_aliases.clear();
        m_path_prefix_stack = {""};
        m_prefix_nodes.clear();
        record_event("begin:<root>");
    }

    void enter_subgraph(mx::ShaderNode* shader_node)
    {
        FALCOR_ASSERT(shader_node);
        record_event("begin:" + shader_node->getName());
        m_path_prefix_stack.push_back(m_path_prefix_stack.back() + shader_node->getName() + ":");
        m_prefix_nodes.push_back(shader_node);
    }

    void leave_subgraph(mx::ShaderNode* shader_node)
    {
        FALCOR_ASSERT(shader_node == m_prefix_nodes.back());
        m_path_prefix_stack.pop_back();
        m_prefix_nodes.pop_back();
        record_event("end:" + shader_node->getName());
    }

    void finish_root_graph(mx::ShaderGraph& shader_graph)
    {
        std::vector<mx::ShaderInput*> outputs = filter(shader_graph.getOutputSockets());
        auto it = m_output_to_index.end();
        if (!outputs.empty() && shader_graph.getOutputSockets()[0]->getConnection())
            it = m_output_to_index.find(resolve_alias(shader_graph.getOutputSockets()[0]->getConnection()));
        m_layering_desc.main_layer = it != m_output_to_index.end() ? it->second : ClosureRef::none();

        m_path_prefix_stack.pop_back();
        record_event("end:<root>");
    }

    void visit_graph(mx::ShaderGraph& shader_graph)
    {
        std::vector<mx::ShaderNode*> emit_order = get_closure_emit_order(shader_graph);
        std::set<mx::ShaderNode*> already_visited;

        for (mx::ShaderNode* shader_node : emit_order) {
            if (!already_visited.insert(shader_node).second)
                continue;

            if (auto sub_graph = shader_node->getImplementation().getGraph()) {
                enter_subgraph(shader_node);
                connect_subgraph_inputs(shader_node, *sub_graph);
                visit_graph(*sub_graph);
                connect_subgraph_outputs(shader_node, *sub_graph);
                leave_subgraph(shader_node);
            } else {
                add_node(shader_node);
            }
        }
    }

    void connect_subgraph_inputs(mx::ShaderNode* shader_node, mx::ShaderGraph& sub_graph)
    {
        for (mx::ShaderGraphInputSocket* input_socket : sub_graph.getInputSockets()) {
            if (!accept_connection_type(input_socket->getType()))
                continue;

            mx::ShaderInput* parent_input = shader_node->getInput(input_socket->getName());
            if (!parent_input || !parent_input->getConnection())
                continue;

            m_aliases[make_key(input_socket)] = resolve_alias(make_key(parent_input->getConnection(), 1));
        }
    }

    void connect_subgraph_outputs(mx::ShaderNode* shader_node, mx::ShaderGraph& sub_graph)
    {
        FALCOR_ASSERT_EQ(shader_node->getOutputs().size(), sub_graph.getOutputSockets().size());
        for (size_t i = 0; i < shader_node->getOutputs().size(); ++i) {
            mx::ShaderOutput* graph_output = shader_node->getOutputs()[i];
            mx::ShaderOutput* inner_output = sub_graph.getOutputSockets()[i]->getConnection();
            connect_graph_socket(graph_output, inner_output);
            record_event("connect:" + port_name(graph_output) + "=" + port_name(inner_output));
        }
    }

    void add_node(mx::ShaderNode* node)
    {
        record_event("add:" + node->getName());
        std::vector<mx::ShaderInput*> inputs = filter(node->getInputs());
        std::vector<mx::ShaderOutput*> outputs = filter(node->getOutputs());
        if (outputs.empty())
            return;

        FALCOR_CHECK(
            outputs.size() == 1,
            "MX139 cannot handle multiple BSDF/surface outputs from node {}.",
            node->getName()
        );

        const std::string implementation = node->getImplementation().getName();
        if (outputs[0]->getType() == mx::Type::MATERIAL) {
            add_material(node, inputs, outputs);
        } else if (is_combiner_implementation(implementation)) {
            add_combiner(node, inputs, outputs);
        } else if (is_weight_implementation(implementation)) {
            add_named_passthrough(node, inputs, outputs, {"in1"});
        } else if (outputs[0]->getType() == mx::Type::SURFACESHADER) {
            add_surface(node, inputs, outputs);
        } else {
            add_bsdf_or_passthrough(node, inputs, outputs);
        }
    }

    void connect_graph_socket(mx::ShaderOutput* graph_output, mx::ShaderOutput* inner_output)
    {
        if (graph_output && inner_output && accept_connection_type(graph_output->getType()))
            m_aliases[make_key(graph_output, 1)] = make_key(inner_output);
    }

    static void append_dependent_closure_input(std::vector<mx::ShaderNode*>& result, mx::ShaderInput* input)
    {
        if (!input)
            return;

        mx::ShaderNode* upstream = input->getConnectedSibling();
        if (upstream && accept_emit_order_node(upstream))
            append_dependent_closure_nodes(result, upstream);
    }

    static void append_dependent_closure_nodes(std::vector<mx::ShaderNode*>& result, mx::ShaderNode* shader_node)
    {
        if (shader_node->hasClassification(mx::ShaderNode::Classification::MATERIAL)) {
            // MX139 currently emits the concrete MaterialX material through its
            // surfaceshader input only; keep the LayeringDesc in that same shape.
            append_dependent_closure_input(result, shader_node->getInput(mx::ShaderNode::SURFACESHADER));
            result.push_back(shader_node);
            return;
        }

        for (mx::ShaderInput* input : shader_node->getInputs()) {
            append_dependent_closure_input(result, input);
        }
        result.push_back(shader_node);
    }

    static bool accept_emit_order_node(mx::ShaderNode* shader_node)
    {
        return shader_node->hasClassification(mx::ShaderNode::Classification::CLOSURE)
            || shader_node->hasClassification(mx::ShaderNode::Classification::SHADER)
            || shader_node->hasClassification(mx::ShaderNode::Classification::MATERIAL);
    }

    static std::vector<mx::ShaderNode*> get_closure_emit_order(mx::ShaderGraph& shader_graph)
    {
        std::vector<mx::ShaderNode*> result;
        for (mx::ShaderGraphOutputSocket* socket : shader_graph.getOutputSockets()) {
            if (!socket->getConnection())
                continue;

            mx::ShaderNode* upstream = socket->getConnection()->getNode();
            if (upstream->getParent() == &shader_graph && accept_emit_order_node(upstream)) {
                append_dependent_closure_nodes(result, upstream);
            }
        }
        return result;
    }

    void record_event(std::string event)
    {
        if (m_traversal_events)
            m_traversal_events->push_back(std::move(event));
    }

    static std::string port_name(mx::ShaderOutput* output)
    {
        if (!output)
            return "<null>";
        return output->getNode()->getName() + "." + output->getName();
    }

    static bool is_combiner_implementation(const std::string& implementation)
    {
        return implementation == "IM_layer_bsdf_genslangpt" || implementation == k_weighted_layer_bsdf_impl
            || implementation == "IM_mix_bsdf_genslangpt" || implementation == "IM_mix_bsdfC_genslangpt"
            || implementation == "IM_mix_surfaceshader_genslangpt" || implementation == "IM_add_bsdf_genslangpt";
    }

    static bool is_weight_implementation(const std::string& implementation)
    {
        return implementation == "IM_multiply_bsdfC_genslangpt" || implementation == "IM_multiply_bsdfF_genslangpt";
    }

    void add_bsdf_or_passthrough(
        mx::ShaderNode* node,
        const std::vector<mx::ShaderInput*>& inputs,
        const std::vector<mx::ShaderOutput*>& outputs
    )
    {
        std::vector<std::pair<mx::ShaderInput*, ClosureRef>> resolved_inputs = collect_resolved_inputs(inputs);

        if (resolved_inputs.empty())
            add_bsdf(node, outputs);
        else if (resolved_inputs.size() == 1)
            map_output_to_index(outputs[0], resolved_inputs.front().second);
        else {
            FALCOR_THROW(
                "MX139 node {} has multiple connected BSDF-like inputs but is not a supported composition node.",
                node->getName()
            );
        }
    }

    template<typename T>
    std::vector<T*> filter(const std::vector<T*>& ports) const
    {
        std::vector<T*> result;
        for (T* port : ports) {
            if (accept_connection_type(port->getType()))
                result.push_back(port);
        }
        return result;
    }

    bool accept_connection_type(mx::TypeDesc type) const
    {
        return type == mx::Type::BSDF || type == mx::Type::SURFACESHADER || type == mx::Type::MATERIAL;
    }

    std::vector<std::pair<mx::ShaderInput*, ClosureRef>>
    collect_resolved_inputs(const std::vector<mx::ShaderInput*>& inputs) const
    {
        std::vector<std::pair<mx::ShaderInput*, ClosureRef>> result;
        for (mx::ShaderInput* input : inputs) {
            ClosureRef ref;
            if (try_resolve_input(input, ref))
                result.emplace_back(input, ref);
        }
        return result;
    }

    bool try_resolve_input(mx::ShaderInput* input, ClosureRef& ref) const
    {
        if (!input || !input->getConnection())
            return false;

        auto it = m_output_to_index.find(resolve_alias(input->getConnection()));
        if (it == m_output_to_index.end())
            return false;

        ref = it->second;
        return true;
    }

    mx::ShaderInput* find_input(const std::vector<mx::ShaderInput*>& inputs, const std::string& name) const
    {
        auto it = std::find_if(
            inputs.begin(),
            inputs.end(),
            [&](mx::ShaderInput* input)
            {
                return input->getName() == name;
            }
        );
        return it == inputs.end() ? nullptr : *it;
    }

    void map_output_to_index(mx::ShaderOutput* output, ClosureRef ref) { m_output_to_index[make_key(output)] = ref; }

    void add_bsdf(
        mx::ShaderNode* shader_node,
        const std::vector<mx::ShaderOutput*>& outputs,
        const std::string& node_impl_override = ""
    )
    {
        LayeringDesc::BsdfDesc desc;
        desc.index = int(m_layering_desc.bsdfs.size() + 1);
        desc.shader_node = shader_node;
        desc.node_impl = node_impl_override.empty() ? shader_node->getImplementation().getName() : node_impl_override;
        desc.node_path = node_path(shader_node);
        desc.scatter_mode = optional_shader_input_value_string(*shader_node, "scatter_mode", "");
        map_output_to_index(outputs[0], ClosureRef::bsdf(m_layering_desc.bsdfs.size()));
        m_layering_desc.bsdfs.push_back(std::move(desc));
    }

    void add_material(
        mx::ShaderNode*,
        const std::vector<mx::ShaderInput*>& inputs,
        const std::vector<mx::ShaderOutput*>& outputs
    )
    {
        mx::ShaderInput* input = find_input(inputs, "surfaceshader");
        ClosureRef ref;
        if (try_resolve_input(input, ref)) {
            map_output_to_index(outputs[0], ref);
            return;
        }

        map_output_to_index(outputs[0], ClosureRef::none());
    }

    void add_named_passthrough(
        mx::ShaderNode* shader_node,
        const std::vector<mx::ShaderInput*>& inputs,
        const std::vector<mx::ShaderOutput*>& outputs,
        std::initializer_list<const char*> names
    )
    {
        for (const char* name : names) {
            mx::ShaderInput* input = find_input(inputs, name);
            ClosureRef ref;
            if (try_resolve_input(input, ref)) {
                map_output_to_index(outputs[0], ref);
                return;
            }
        }

        std::vector<std::pair<mx::ShaderInput*, ClosureRef>> resolved_inputs = collect_resolved_inputs(inputs);
        if (resolved_inputs.size() == 1) {
            map_output_to_index(outputs[0], resolved_inputs.front().second);
            return;
        }

        FALCOR_THROW("MX139 node {} is not connected to a known BSDF output.", shader_node->getName());
    }

    bool surface_unlit_may_transmit(mx::ShaderNode* node) const
    {
        if (node->getImplementation().getName() != "IM_surface_unlit_genslangpt")
            return false;

        mx::ShaderInput* input = node->getInput("transmission");
        if (!input)
            return false;
        if (input->getConnection())
            return true;

        std::string value = optional_shader_input_value_string(*node, "transmission", "0");
        std::stringstream stream(value);
        float transmission = 0.0f;
        stream >> transmission;
        return stream.fail() || transmission != 0.0f;
    }

    void add_surface(
        mx::ShaderNode* node,
        const std::vector<mx::ShaderInput*>& inputs,
        const std::vector<mx::ShaderOutput*>& outputs
    )
    {
        mx::ShaderInput* input = find_input(inputs, "bsdf");
        ClosureRef ref;
        if (try_resolve_input(input, ref)) {
            map_output_to_index(outputs[0], ref);
            return;
        }

        if (surface_unlit_may_transmit(node)) {
            add_bsdf(node, outputs, k_surface_unlit_transparent_impl);
            return;
        }

        map_output_to_index(outputs[0], ClosureRef::none());
    }

    void add_combiner(
        mx::ShaderNode* shader_node,
        const std::vector<mx::ShaderInput*>& inputs,
        const std::vector<mx::ShaderOutput*>& outputs
    )
    {
        LayeringDesc::CombinerDesc desc;
        desc.index = -int(m_layering_desc.combiners.size() + 1);
        desc.node_impl = shader_node->getImplementation().getName();
        desc.node_path = node_path(shader_node);

        const std::array<const char*, 2> input_names = combiner_input_names(shader_node, desc.node_impl);
        for (std::size_t i = 0; i < input_names.size(); ++i) {
            const char* input_name = input_names[i];
            mx::ShaderInput* input = find_input(inputs, input_name);
            ClosureRef ref;
            FALCOR_CHECK(
                try_resolve_input(input, ref),
                "MX139 input {}.{} is not connected to a known BSDF output.",
                shader_node->getName(),
                input_name
            );
            desc.children[i] = ref;
        }

        map_output_to_index(outputs[0], ClosureRef::combiner(m_layering_desc.combiners.size()));
        m_layering_desc.combiners.push_back(std::move(desc));
    }

    OutputKey resolve_alias(mx::ShaderOutput* shader_output) const { return resolve_alias(make_key(shader_output)); }

    OutputKey resolve_alias(OutputKey output_key) const
    {
        while (true) {
            auto it = m_aliases.find(output_key);
            if (it == m_aliases.end())
                return output_key;
            output_key = it->second;
        }
    }

    OutputKey make_key(mx::ShaderOutput* shader_output, int skip = 0) const
    {
        OutputKey result{m_prefix_nodes, shader_output};
        for (int i = 0; i < skip && !result.path.empty(); ++i)
            result.path.pop_back();
        return result;
    }

    std::string node_path(mx::ShaderNode* shader_node) const
    {
        return m_path_prefix_stack.empty() ? shader_node->getName()
                                           : m_path_prefix_stack.back() + shader_node->getName();
    }

    static bool is_materialx_mix_lowered_add(mx::ShaderNode* shader_node)
    {
        const std::string& name = shader_node->getName();
        if (!name.ends_with("_add"))
            return false;

        return name.starts_with("mix_") || name.find("_mix_") != std::string::npos
            || name.find("mix_") != std::string::npos;
    }

    static std::array<const char*, 2>
    combiner_input_names(mx::ShaderNode* shader_node, const std::string& implementation)
    {
        if (implementation == "IM_layer_bsdf_genslangpt" || implementation == k_weighted_layer_bsdf_impl)
            return {"top", "base"};
        if (implementation == "IM_mix_bsdf_genslangpt" || implementation == "IM_mix_bsdfC_genslangpt"
            || implementation == "IM_mix_surfaceshader_genslangpt")
            return {"fg", "bg"};
        if (implementation == "IM_add_bsdf_genslangpt") {
            // MaterialX lowers some closure mix nodes into weighted fg/bg
            // BSDFs followed by an add combiner, for example base_mix_add.
            // Reverse only these generated mix-add nodes when requested so
            // recursive branch replay sees the authored mix(bg, fg, amount)
            // closure order instead of the lowering's fg-first add order.
            return is_materialx_mix_lowered_add(shader_node) ? std::array<const char*, 2>{"in2", "in1"}
                                                             : std::array<const char*, 2>{"in1", "in2"};
        }
        FALCOR_THROW("MX139 unsupported BSDF combiner implementation '{}'.", implementation);
    }

    std::map<OutputKey, ClosureRef> m_output_to_index;
    std::map<OutputKey, OutputKey> m_aliases;
    std::vector<std::string> m_path_prefix_stack;
    std::vector<mx::ShaderNode*> m_prefix_nodes;
    LayeringDesc m_layering_desc;
    std::vector<std::string>* m_traversal_events = nullptr;
};

struct BsdfNodeInfo {
    std::string type;
    std::string lobes;
    bool transmissive = false;
    bool curve_scattering = false;
    bool preserve_tangent_frame = false;
};

std::unordered_map<std::string, BsdfNodeInfo> make_bsdf_node_map(const ProfileDesc& profile, const CodeGenDesc& desc)
{
    return {
        {"IM_oren_nayar_diffuse_bsdf_genslangpt",
         {profile.bsdf_type("OrenNayarDiffuse"), "BSDFFlags::diffuse_reflection", false}},
        {"IM_burley_diffuse_bsdf_genslangpt",
         {profile.bsdf_type("BurleyDiffuse"), "BSDFFlags::diffuse_reflection", false}},
        {"IM_translucent_bsdf_genslangpt", {profile.bsdf_type("Translucent"), "BSDFFlags::diffuse_transmission", true}},
        {"IM_subsurface_bsdf_genslangpt",
         {profile.bsdf_type("SubsurfaceDiffuseFallback"), "BSDFFlags::diffuse_reflection", false}},
        {"IM_beer_bsdf_genslangpt", {profile.bsdf_type("Beer"), "BSDFFlags::delta_transmission", false}},
        {"IM_scratchconductor_brdf_genslangpt",
         {profile.bsdf_type("ScratchConductor"), "BSDFFlags::glossy_reflection", false}},
        {"IM_dielectric_bsdf_genslangpt",
         {fresnel_bsdf_type(profile, desc, "Dielectric"),
          "(BSDFFlags::glossy_reflection | BSDFFlags::glossy_transmission)",
          true}},
        {"IM_conductor_bsdf_genslangpt",
         {fresnel_bsdf_type(profile, desc, "Conductor"), "BSDFFlags::glossy_reflection", false}},
        {"IM_generalized_schlick_bsdf_genslangpt",
         {fresnel_bsdf_type(profile, desc, "GeneralizedSchlick"),
          "(BSDFFlags::glossy_reflection | BSDFFlags::glossy_transmission)",
          true}},
        {"IM_sheen_bsdf_genslangpt", {profile.bsdf_type("Sheen"), "BSDFFlags::glossy_reflection", false}},
        {"IM_chiang_hair_bsdf_genslangpt",
         {profile.bsdf_type("ChiangHair"), "BSDFFlags::glossy_curve", false, true, true}},
        {k_surface_unlit_transparent_impl, {"MxSurfaceUnlitTransparentBSDF", "BSDFFlags::delta_transmission", true}},
        {k_mx139_simple_btdf_impl, {"MxSimpleBTDF", "BSDFFlags::delta_transmission", true}},
    };
}

CombinerMode classify_combiner(const std::string& implementation)
{
    if (implementation == "IM_layer_bsdf_genslangpt")
        return CombinerMode::layer;
    if (implementation == k_weighted_layer_bsdf_impl)
        return CombinerMode::weighted_layer;
    if (implementation == "IM_mix_bsdf_genslangpt" || implementation == "IM_mix_bsdfC_genslangpt"
        || implementation == "IM_mix_surfaceshader_genslangpt")
        return CombinerMode::mix;
    if (implementation == k_mx139_opacity_mix_impl)
        return CombinerMode::mix;
    if (implementation == "IM_add_bsdf_genslangpt")
        return CombinerMode::add;
    FALCOR_THROW("MX139 unsupported BSDF combiner implementation '{}'.", implementation);
}

} // namespace

LayeringDesc LayeringDescBuilder::build(mx::ShaderGraph& shader_graph, std::vector<std::string>* traversal_events)
{
    return LayeringDescBuilderImpl::build(shader_graph, traversal_events);
}

void classify_layering(
    LayeringDesc& desc,
    const ProfileDesc& profile,
    const CodeGenDesc& codegen_desc,
    const mx::VariableBlock* constants,
    const StaticInputQuery* static_input_query
)
{
    const std::unordered_map<std::string, BsdfNodeInfo> node_map = make_bsdf_node_map(profile, codegen_desc);
    const std::set<std::string> explicit_transmissive_bsdfs(
        codegen_desc.transmissive_bsdfs.begin(),
        codegen_desc.transmissive_bsdfs.end()
    );
    for (LayeringDesc::BsdfDesc& bsdf : desc.bsdfs) {
        auto it = node_map.find(bsdf.node_impl);
        FALCOR_CHECK(
            it != node_map.end(),
            "{} unsupported BSDF implementation '{}'.",
            profile.display_name,
            bsdf.node_impl
        );
        bsdf.bsdf_type = it->second.type;
        bsdf.lobe_types = it->second.lobes;
        bsdf.transmissive = codegen_desc.auto_transmission
            ? it->second.transmissive
            : explicit_transmissive_bsdfs.find(bsdf.node_path) != explicit_transmissive_bsdfs.end();
        bsdf.curve_scattering = it->second.curve_scattering;
        bsdf.preserve_tangent_frame = it->second.preserve_tangent_frame;
        if (microfacet_kind(bsdf.node_impl) != MxMicrofacetKind::none) {
            MxMicrofacetSelection selection = select_mx139_microfacet(
                codegen_desc,
                bsdf.node_impl,
                bsdf.shader_node,
                constants,
                static_input_query
            );
            bsdf.has_mx139_microfacet_selection = true;
            bsdf.fresnel_selection = static_cast<int>(selection.fresnel);
            bsdf.scatter_selection = static_cast<int>(selection.scatter);
            bsdf.bsdf_type = microfacet_bsdf_type(codegen_desc, selection);
            if (selection.scatter != MxScatterSelection::runtime) {
                const std::string scatter_mode = scatter_selection_mode_string(selection.scatter);
                bsdf.lobe_types = glossy_scatter_lobe_types(scatter_mode);
                if (codegen_desc.auto_transmission)
                    bsdf.transmissive = scatter_mode_includes_transmission(scatter_mode);
            }
        }
        if (is_scatter_mode_bsdf(bsdf.node_impl)) {
            MxMicrofacetSelection selection = select_mx139_microfacet(
                codegen_desc,
                bsdf.node_impl,
                bsdf.shader_node,
                constants,
                static_input_query
            );
            if (bsdf.has_mx139_microfacet_selection) {
                selection.fresnel = static_cast<MxFresnelSelection>(bsdf.fresnel_selection);
                selection.scatter = static_cast<MxScatterSelection>(bsdf.scatter_selection);
            }
            const std::string scatter_mode = selection.scatter == MxScatterSelection::runtime
                ? bsdf.scatter_mode
                : scatter_selection_mode_string(selection.scatter);
            bsdf.lobe_types = glossy_scatter_lobe_types(scatter_mode);
            if (codegen_desc.auto_transmission)
                bsdf.transmissive = scatter_mode_includes_transmission(scatter_mode);
        }
        bsdf.through_material_transmissive = bsdf.transmissive;
    }

    for (LayeringDesc::CombinerDesc& combiner : desc.combiners)
        combiner.mode = classify_combiner(combiner.node_impl);
}

bool use_synthetic_opacity_mix(const LayeringDesc& desc, bool root_opacity_needs_mix)
{
    return root_opacity_needs_mix && !desc.main_layer.is_none();
}

void append_synthetic_opacity_mix(LayeringDesc& desc)
{
    const ClosureRef root = desc.main_layer;
    if (root.is_none())
        return;

    LayeringDesc::BsdfDesc transparent;
    transparent.index = int(desc.bsdfs.size() + 1);
    transparent.node_path = "synthetic_opacity_transparent";
    transparent.node_impl = k_mx139_simple_btdf_impl;
    transparent.bsdf_type = "MxSimpleBTDF";
    transparent.lobe_types = "BSDFFlags::delta_transmission";
    transparent.transmissive = true;
    transparent.through_material_transmissive = true;
    desc.bsdfs.push_back(std::move(transparent));

    LayeringDesc::CombinerDesc opacity_mix;
    opacity_mix.index = -int(desc.combiners.size() + 1);
    opacity_mix.mode = CombinerMode::mix;
    opacity_mix.children = {root, ClosureRef::bsdf(desc.bsdfs.size() - 1)};
    opacity_mix.node_path = "synthetic_opacity_mix";
    opacity_mix.node_impl = k_mx139_opacity_mix_impl;
    desc.combiners.push_back(std::move(opacity_mix));
    desc.main_layer = ClosureRef::combiner(desc.combiners.size() - 1);
}

static void count_reachable_ref_uses(
    const LayeringDesc& desc,
    ClosureRef ref,
    std::map<ClosureRef, int>& uses,
    std::set<ClosureRef>& visited_combiners
)
{
    if (!ref.is_combiner())
        return;

    const size_t combiner_index = ref.combiner_index();
    if (combiner_index >= desc.combiners.size() || !visited_combiners.insert(ref).second)
        return;

    const LayeringDesc::CombinerDesc& combiner = desc.combiners[combiner_index];
    for (ClosureRef child : combiner.children) {
        if (!child.is_none())
            ++uses[child];
        count_reachable_ref_uses(desc, child, uses, visited_combiners);
    }
}

void validate_weighted_layer_top_refs_are_not_shared(const LayeringDesc& desc)
{
    std::map<ClosureRef, int> reachable_ref_uses;
    std::set<ClosureRef> visited_combiners;
    count_reachable_ref_uses(desc, desc.main_layer, reachable_ref_uses, visited_combiners);

    for (ClosureRef combiner_ref : visited_combiners) {
        const size_t combiner_index = combiner_ref.combiner_index();
        const LayeringDesc::CombinerDesc& combiner = desc.combiners[combiner_index];
        if (combiner.mode != CombinerMode::weighted_layer)
            continue;
        const ClosureRef top_ref = combiner.children[0];
        if (top_ref.is_none())
            continue;
        FALCOR_CHECK(
            reachable_ref_uses[top_ref] <= 1,
            "Mx139 weighted_layer node {} cannot fold top_weight into reused top closure ref {}. "
            "Duplicate the weighted_layer top branch before lowering.",
            combiner.node_path,
            top_ref.raw_layering_index()
        );
    }
}

void lower_weighted_layers_to_layers_for_emission(LayeringDesc& desc)
{
    for (LayeringDesc::CombinerDesc& combiner : desc.combiners) {
        if (combiner.mode == CombinerMode::weighted_layer)
            combiner.mode = CombinerMode::layer;
    }
}

std::optional<std::size_t> synthetic_opacity_bsdf_index(const LayeringDesc& desc)
{
    for (std::size_t i = 0; i < desc.bsdfs.size(); ++i) {
        if (desc.bsdfs[i].node_impl == k_mx139_simple_btdf_impl)
            return i;
    }
    return std::nullopt;
}

std::optional<std::size_t> synthetic_opacity_combiner_index(const LayeringDesc& desc)
{
    for (std::size_t i = 0; i < desc.combiners.size(); ++i) {
        if (desc.combiners[i].node_impl == k_mx139_opacity_mix_impl)
            return i;
    }
    return std::nullopt;
}

bool is_unity_layer_pass_through_transmission_bsdf(const LayeringDesc::BsdfDesc& bsdf)
{
    return bsdf.node_impl == k_surface_unlit_transparent_impl || bsdf.node_impl == k_mx139_simple_btdf_impl;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
