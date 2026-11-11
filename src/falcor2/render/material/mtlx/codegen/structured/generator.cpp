// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "generator.h"
#include "../codegen_cache.h"

#include "document/loader.h"
#include "document/transforms.h"
#include "graph/closure_pruning/closure_pruning_report.h"
#include "graph/closure_pruning/run.h"
#include "graph/static_input_query/static_input_query.h"
#include "layering/empty_surface_handler.h"
#include "layering/policy.h"
#include "layering/policy_factories.h"
#include "material/analysis.h"
#include "material/emission_query.h"
#include "material/opacity_analysis.h"
#include "material/profile_policy.h"
#include "pipeline/generation_types.h"
#include "pipeline/request_validation.h"
#include "target/shader_generator.h"
#include "target/nodes/source_code_node.h"
#include "target/nodes/geometry_nodes.h"
#include "target/snippet_renderer.h"
#include "target/user_data.h"

#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"

#include <MaterialXCore/Unit.h>
#include <MaterialXCore/Interface.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenHw/HwConstants.h>

#include <fmt/format.h>
#include <sgl/core/crypto.h>

#include <algorithm>
#include <filesystem>
#include <functional>
#include <limits>
#include <regex>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace falcor::mtlx::structured {

namespace {

namespace mx = MaterialX;

constexpr const char* WORKING_COLOR_SPACE = "lin_rec709";

// Recover public controls from authored paths, independently of generated shader
// bindings. SDK color/unit transforms retain these paths when they replace inputs.
// Resolve names before pruning so editable values cannot be treated as constants.
// Keep root bare names and ordinary node_input names; qualify ambiguous names.
// Path-based lookup follows MaterialX's GlslMaterial::findUniform; the public
// naming and ambiguity rules here are Falcor-specific.
// https://github.com/AcademySoftwareFoundation/MaterialX/blob/7b64921ef1d42f2d57871e9d2c43dc11f041f26b/source/MaterialXRenderGlsl/GlslMaterial.cpp#L325
void collect_input_names(
    mx::ShaderGraph& graph,
    const mx::Document& document,
    const CodeGenDesc& desc,
    const std::unordered_map<std::string, std::string>& moved_inputs,
    PreCanonicalInputNames& names
)
{
    std::unordered_map<std::string, std::string> paths;
    std::unordered_set<std::string> ambiguous;
    for (mx::ShaderOutput* input : graph.getInputSockets()) {
        if (const auto found = moved_inputs.find(input->getPath()); found != moved_inputs.end())
            input->setPath(found->second);
        const std::string& path = input->getPath();
        std::string name = input->getName();
        // String enums are encoded as consumer-specific integer indices. Two
        // consumers can order the same enum differently, so retain separate controls.
        const mx::ElementPtr source = input->getType() == mx::Type::INTEGER ? document.getDescendant(path) : nullptr;
        const mx::TypedElementPtr typed_source = source ? source->asA<mx::TypedElement>() : nullptr;
        const bool remapped_enum = typed_source && typed_source->getType() == mx::STRING_TYPE_STRING;
        const size_t separator = path.rfind('/');
        if (!remapped_enum && separator != std::string::npos && name != path.substr(separator + 1)) {
            const size_t parent = separator ? path.rfind('/', separator - 1) : std::string::npos;
            name = path.substr(parent == std::string::npos ? 0 : parent + 1);
            std::replace(name.begin(), name.end(), '/', '_');
        }
        const auto [found, inserted] = paths.emplace(name, path);
        if (!inserted && found->second != path)
            ambiguous.insert(name);
        names.emplace(input, std::move(name));
    }
    for (const mx::ShaderOutput* input : graph.getInputSockets()) {
        std::string& name = names.at(input);
        if (!ambiguous.contains(name))
            continue;
        FALCOR_CHECK(
            !desc.editable_param_names.contains(name),
            "MaterialX editable input '{}' is ambiguous; use its full authored path.",
            name
        );
        name = input->getPath();
    }
}

/// MaterialX objects whose lifetimes are coupled by the library. Declaration
/// order keeps stages, shader, graph, and context dead before their owners.
struct MaterialXResources {
    mx::DocumentPtr document;
    mx::TypedElementPtr root;
    std::shared_ptr<GenslangPtShaderGenerator> shader_generator;
    std::unique_ptr<mx::GenContext> context;
    mx::ShaderGraphPtr graph;
    mx::ShaderPtr shader;
    mx::ShaderStagePtr vertex_stage;
    mx::ShaderStagePtr pixel_stage;
};

/// Single entry point for constructing the root shader graph. Ordinary requests
/// delegate directly to MaterialX. MaterialX's factory accepts one root and
/// finalizes immediately, so combined roots and multi-output material nodes use
/// this builder to put the primary surface in slot zero before finalization.
/// Repeated traversal reuses shared authored nodes through MaterialX's node map.
class AOVShaderGraph : public mx::ShaderGraph {
public:
    static mx::ShaderGraphPtr create(
        const std::string& name,
        const mx::TypedElementPtr& main_root,
        const std::vector<mx::OutputPtr>& additional_outputs,
        mx::GenContext& context
    )
    {
        FALCOR_CHECK(main_root, "Cannot build an AOV ShaderGraph without a main root.");
        const mx::NodePtr node = main_root->asA<mx::Node>();
        const mx::NodeDefPtr node_def = node ? node->getNodeDef() : nullptr;
        const auto outputs = node_def ? node_def->getActiveOutputs() : std::vector<mx::OutputPtr>{};
        const bool multioutput_material = outputs.size() > 1
            && std::any_of(outputs.begin(),
                           outputs.end(),
                           [&](const mx::OutputPtr& output)
                           {
                               return is_main_output_type(context.getTypeDesc(output->getType()));
                           });
        if (additional_outputs.empty() && !multioutput_material) {
            auto graph = mx::ShaderGraph::create(nullptr, name, main_root, context);
            // Omitted root inputs keep their NodeDef paths in the SDK. Public
            // controls belong to the selected instance, including its defaults.
            if (node_def) {
                const std::string prefix = node_def->getNamePath() + mx::NAME_PATH_SEPARATOR;
                for (mx::ShaderOutput* input : graph->getInputSockets()) {
                    if (input->getPath().starts_with(prefix))
                        input->setPath(
                            node->getNamePath() + mx::NAME_PATH_SEPARATOR + input->getPath().substr(prefix.size())
                        );
                }
            }
            return graph;
        }

        auto graph = std::make_shared<AOVShaderGraph>(nullptr, name, main_root->getDocument(), context);
        graph->setClassification(0);
        graph->initialize_main_root(main_root, context);
        graph->normalize_main_output(additional_outputs);
        for (const mx::OutputPtr& output : additional_outputs) {
            const mx::TypeDesc type = context.getTypeDesc(output->getType());
            FALCOR_CHECK(
                is_aov_value_type(type),
                "MTLX additional output '{}' has unsupported type '{}'; only the main output may be material or "
                "closure-shaped, and AOVs must be non-array numeric scalars, vectors, colors, or matrices.",
                output->getNamePath(),
                output->getType()
            );
            graph->append_output(output, context);
        }
        graph->finalize(context);
        return graph;
    }

    using mx::ShaderGraph::ShaderGraph;

private:
    static bool is_aov_value_type(const mx::TypeDesc& type)
    {
        const auto base_type = type.getBaseType();
        return !type.isArray()
            && (base_type == mx::TypeDesc::BASETYPE_BOOLEAN || base_type == mx::TypeDesc::BASETYPE_INTEGER
                || base_type == mx::TypeDesc::BASETYPE_FLOAT);
    }

    static bool is_main_output_type(const mx::TypeDesc& type)
    {
        return type == mx::Type::SURFACESHADER || type == mx::Type::MATERIAL;
    }

    void normalize_main_output(const std::vector<mx::OutputPtr>& additional_outputs)
    {
        // Only the generated root is reordered. Nested compound interfaces and
        // their existing connections retain the authored order.
        const auto is_main = [](const mx::ShaderInput* output)
        {
            return is_main_output_type(output->getType());
        };
        FALCOR_CHECK(
            std::count_if(_inputOrder.begin(), _inputOrder.end(), is_main) == 1,
            "MTLX primary root must expose exactly one surfaceshader or material output; select an explicit output "
            "otherwise."
        );
        const auto main = std::find_if(_inputOrder.begin(), _inputOrder.end(), is_main);
        std::rotate(_inputOrder.begin(), main, main + 1);

        // A standard material's synthetic 'out' socket must not hide an authored
        // numeric output of the same name. Only this internal primary name changes.
        mx::ShaderInput* primary = _inputOrder.front();
        const auto name_is_used = [&](const std::string& name)
        {
            return std::any_of(
                additional_outputs.begin(),
                additional_outputs.end(),
                [&](const mx::OutputPtr& output)
                {
                    return output->getName() == name;
                }
            );
        };
        if (name_is_used(primary->getName())) {
            std::string name = primary->getName();
            do {
                name += "_surface";
            } while (_inputMap.contains(name) || name_is_used(name));
            auto entry = _inputMap.extract(primary->getName());
            entry.key() = name;
            primary->setName(name);
            _inputMap.insert(std::move(entry));
        }
    }

    void initialize_main_root(const mx::TypedElementPtr& root, mx::GenContext& context)
    {
        if (mx::OutputPtr output = root->asA<mx::Output>()) {
            const mx::TypeDesc type = context.getTypeDesc(output->getType());
            FALCOR_CHECK(
                is_main_output_type(type),
                "MTLX main output '{}' has type '{}'; an AOV ShaderGraph requires a surfaceshader or material "
                "main output.",
                output->getNamePath(),
                output->getType()
            );

            mx::ElementPtr parent = output->getParent();
            mx::InterfaceElementPtr interface;
            if (mx::NodeGraphPtr node_graph = parent->asA<mx::NodeGraph>()) {
                interface = node_graph->getNodeDef();
                if (!interface)
                    interface = node_graph;
            } else if (parent->asA<mx::Document>()) {
                mx::NodePtr connected_node = output->getConnectedNode();
                interface = connected_node ? connected_node->asA<mx::InterfaceElement>() : nullptr;
            }
            FALCOR_CHECK(
                interface,
                "MTLX main output '{}' has no interface valid for shader generation.",
                output->getNamePath()
            );

            addInputSockets(*interface, context);
            append_output(output, context);
            return;
        }

        mx::NodePtr node = root->asA<mx::Node>();
        FALCOR_CHECK(node, "MTLX combined-output generation requires a node or output primary root.");
        mx::NodeDefPtr node_def = node->getNodeDef();
        FALCOR_CHECK(node_def, "MTLX primary node '{}' has no NodeDef.", node->getNamePath());

        // This is MaterialX ShaderGraph::create's Node path. Keeping the same
        // initialization preserves every NodeDef output; independent document
        // outputs are appended below before the graph is finalized once.
        addInputSockets(*node_def, context);
        addOutputSockets(*node_def, context);
        mx::ShaderNode* new_node = createNode(node->getName(), node->getNamePath(), node_def, context);
        setMetadata(new_node->getMetadata());

        for (size_t i = 0; i < new_node->numOutputs(); ++i) {
            mx::ShaderGraphOutputSocket* socket = getOutputSocket(i);
            socket->makeConnection(new_node->getOutput(i));
            socket->setPath(node->getNamePath());
        }

        for (const mx::InputPtr& node_def_input : node_def->getActiveInputs()) {
            mx::ShaderGraphInputSocket* input_socket = getInputSocket(node_def_input->getName());
            mx::ShaderInput* input = new_node->getInput(node_def_input->getName());
            FALCOR_CHECK(
                input_socket && input,
                "MTLX node input '{}' does not match an input on graph '{}'.",
                node_def_input->getName(),
                getName()
            );

            const std::string path = node->getNamePath() + mx::NAME_PATH_SEPARATOR + node_def_input->getName();
            input_socket->setPath(path);
            input->setPath(path);
            if (mx::InputPtr node_input = node->getInput(node_def_input->getName())) {
                mx::ValuePtr value = node_input->getResolvedValue();
                if (!value) {
                    if (mx::InputPtr interface_input = node_input->getInterfaceInput())
                        value = interface_input->getResolvedValue();
                }
                if (value) {
                    const std::string& value_string = value->getValueString();
                    std::pair<mx::TypeDesc, mx::ValuePtr> enum_result;
                    const mx::TypeDesc type = context.getTypeDesc(node_def_input->getType());
                    const std::string& enum_names = node_def_input->getAttribute(mx::ValueElement::ENUM_ATTRIBUTE);
                    if (context.getShaderGenerator()
                            .getSyntax()
                            .remapEnumeration(value_string, type, enum_names, enum_result))
                        input_socket->setValue(enum_result.second);
                    else
                        input_socket->setValue(value);
                }

                input->setBindInput();
                const std::string& unit = node_input->getUnit();
                if (!unit.empty()) {
                    input_socket->setUnit(unit);
                    input->setUnit(unit);
                }
                const std::string& color_space = node_input->getColorSpace();
                if (!color_space.empty()) {
                    input_socket->setColorSpace(color_space);
                    input->setColorSpace(color_space);
                }
            }

            input_socket->makeConnection(input);
            input_socket->setMetadata(input->getMetadata());
        }

        applyInputTransforms(node, new_node, context);
        if (context.getOptions().addUpstreamDependencies)
            addUpstreamDependencies(*node, context);
    }

    void append_output(const mx::OutputPtr& output, mx::GenContext& context)
    {
        FALCOR_CHECK(output, "Cannot append a null output to an AOV ShaderGraph.");
        mx::ShaderGraphOutputSocket* socket
            = addOutputSocket(output->getName(), context.getTypeDesc(output->getType()));
        socket->setPath(output->getNamePath());
        if (!output->getUnit().empty())
            socket->setUnit(output->getUnit());
        if (!output->getColorSpace().empty())
            socket->setColorSpace(output->getColorSpace());
        if (context.getOptions().addUpstreamDependencies)
            addUpstreamDependencies(*output, context);
    }
};

// Canonical naming removes user-authored shader identifiers while runtime
// parameter metadata retains authored names. Number each user graph in stable
// depth-first order. Trusted library graphs retain their own bindings, but we
// still traverse them because they can reach user-authored implementation graphs.
void canonicalize_graph_bindings(
    mx::ShaderGraph& graph,
    const GenslangPtShaderGenerator& shader_generator,
    bool canonicalize_this_graph,
    PreCanonicalInputNames& pre_canonical_names,
    std::unordered_set<const mx::ShaderGraph*>& visited_graphs
)
{
    if (!visited_graphs.insert(&graph).second)
        return;

    std::unordered_map<const mx::ShaderNode*, std::size_t> node_ids;
    std::vector<mx::ShaderNode*> ordered_nodes;
    std::function<void(mx::ShaderNode*)> visit = [&](mx::ShaderNode* node)
    {
        if (!node || node == &graph || node_ids.contains(node))
            return;
        node_ids.emplace(node, ordered_nodes.size());
        ordered_nodes.push_back(node);
        for (mx::ShaderInput* input : node->getInputs()) {
            if (mx::ShaderOutput* connection = input->getConnection())
                visit(connection->getNode());
        }
    };

    const auto& output_sockets = graph.getOutputSockets();
    for (mx::ShaderGraphOutputSocket* socket : output_sockets) {
        if (mx::ShaderOutput* connection = socket->getConnection())
            visit(connection->getNode());
    }
    for (mx::ShaderNode* node : graph.getNodes())
        visit(node);

    const auto& input_sockets = graph.getInputSockets();
    if (canonicalize_this_graph) {
        for (std::size_t i = 0; i < input_sockets.size(); ++i) {
            mx::ShaderGraphInputSocket* socket = input_sockets[i];
            pre_canonical_names.emplace(socket, socket->getName());
            socket->setVariable(fmt::format("mx_p{}", i));
        }
    }

    for (std::size_t node_index = 0; node_index < ordered_nodes.size(); ++node_index) {
        mx::ShaderNode* node = ordered_nodes[node_index];
        if (canonicalize_this_graph) {
            for (std::size_t input_index = 0; input_index < node->numInputs(); ++input_index) {
                mx::ShaderInput* input = node->getInput(input_index);
                pre_canonical_names.emplace(input, input->getVariable());
                input->setVariable(fmt::format("mx_n{}_i{}", node_index, input_index));
            }
            for (std::size_t output_index = 0; output_index < node->numOutputs(); ++output_index)
                node->getOutput(output_index)->setVariable(fmt::format("mx_n{}_o{}", node_index, output_index));
        }
        if (mx::ShaderGraph* implementation_graph = node->getImplementation().getGraph()) {
            canonicalize_graph_bindings(
                *implementation_graph,
                shader_generator,
                shader_generator.should_canonicalize_implementation_bindings(node->getImplementation()),
                pre_canonical_names,
                visited_graphs
            );
        }
    }

    if (canonicalize_this_graph) {
        for (std::size_t i = 0; i < output_sockets.size(); ++i)
            output_sockets[i]->setVariable(fmt::format("mx_o{}", i));
    }
}

void canonicalize_graph_bindings(
    mx::ShaderGraph& graph,
    const GenslangPtShaderGenerator& shader_generator,
    PreCanonicalInputNames& pre_canonical_names
)
{
    std::unordered_set<const mx::ShaderGraph*> visited_graphs;
    canonicalize_graph_bindings(graph, shader_generator, true, pre_canonical_names, visited_graphs);
}

void replace_all(std::string& text, std::string_view from, std::string_view to)
{
    std::size_t position = 0;
    while ((position = text.find(from, position)) != std::string::npos) {
        text.replace(position, from.size(), to);
        position += to.size();
    }
}

void remove_blank_line_indentation(std::string& source)
{
    std::size_t line_start = 0;
    while (line_start < source.size()) {
        const std::size_t line_end = source.find('\n', line_start);
        const std::size_t content_end = line_end == std::string::npos ? source.size() : line_end;
        const bool blank = std::all_of(
            source.begin() + static_cast<std::ptrdiff_t>(line_start),
            source.begin() + static_cast<std::ptrdiff_t>(content_end),
            [](char value)
            {
                return value == ' ' || value == '\t' || value == '\r';
            }
        );
        if (blank && content_end != line_start) {
            source.erase(line_start, content_end - line_start);
            line_start += line_end == std::string::npos ? 0 : 1;
        } else {
            line_start = line_end == std::string::npos ? source.size() : line_end + 1;
        }
    }
}

void preserve_generated_mutating_signature_shape(std::string& source)
{
    // The Slang formatter places attributes on their own line in maintained
    // source files. Generated MaterialX source historically keeps this small
    // attribute next to `void`; preserve that stable public source and module
    // identity after all MaterialX libraries and snippets have been emitted.
    source
        = std::regex_replace(source, std::regex(R"((^|\n)([ \t]*)\[mutating\]\n[ \t]*(void\b))"), "$1$2[mutating] $3");

    // The generated stack setter historically follows an injected member list
    // whose indentation is part of the generated-source contract. Reproduce
    // that single leading space here instead of keeping formatter-hostile
    // whitespace in the maintained snippet.
    source = std::regex_replace(
        source,
        std::regex(R"((^|\n)([ \t]*)(\[mutating\] void set_bsdf<TBSDF : ILayerableBSDF>))"),
        "$1$2 $3"
    );
}

std::string default_value(const GenslangPtShaderGenerator& target, mx::TypeDesc type)
{
    if (type == mx::Type::MATERIAL || type == mx::Type::SURFACESHADER)
        return "{}";
    std::string value = target.getSyntax().getDefaultValue(type, true);
    if (value.empty())
        return "{}";
    if (type.getSize() > 1 && value.find(',') != std::string::npos && value.find('(') == std::string::npos
        && !value.starts_with("{")) {
        value = target.getSyntax().getTypeName(type) + "(" + value + ")";
    }
    return value;
}

std::size_t calculate_data_size_upper_bound(const mx::VariableBlock& inputs)
{
    auto align_up = [](std::size_t value, std::size_t alignment)
    {
        FALCOR_ASSERT(alignment > 0);
        return (value + alignment - 1) / alignment * alignment;
    };

    // This is the same target-independent upper-bound contract as the current
    // backend. It selects inline versus buffered MaterialData before target
    // reflection is available; underestimating it would be an ABI error.
    std::size_t result = 0;
    std::size_t struct_alignment = 4;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const mx::TypeDesc type = inputs[i]->getType();
        std::size_t base_type_size = 0;
        switch (type.getBaseType()) {
        case mx::TypeDesc::BaseType::BASETYPE_INTEGER:
        case mx::TypeDesc::BaseType::BASETYPE_FLOAT:
        case mx::TypeDesc::BaseType::BASETYPE_BOOLEAN:
        case mx::TypeDesc::BaseType::BASETYPE_STRING:
            base_type_size = 4;
            break;
        default:
            FALCOR_THROW("Structured MaterialX unsupported material-data type '{}'.", type.getName());
        }
        const std::size_t element_count = type.getSize();
        FALCOR_CHECK(element_count > 0, "Structured MaterialX type '{}' has no elements.", type.getName());
        std::size_t alignment = base_type_size;
        if (type.getSemantic() == mx::TypeDesc::SEMANTIC_MATRIX || element_count >= 3)
            alignment = 16;
        else if (element_count == 2)
            alignment = 8;
        result += align_up(element_count * base_type_size, alignment) + alignment - 4;
        struct_alignment = std::max(struct_alignment, alignment);
    }
    return result + struct_alignment - 4;
}

float4 texture_border_color_from_default(std::string value, const mx::TypeDesc& output_type)
{
    std::replace(value.begin(), value.end(), ',', ' ');
    std::stringstream stream(value);
    std::vector<float> values;
    for (float component = 0.0f; stream >> component;)
        values.push_back(component);
    auto component = [&](std::size_t index)
    {
        return index < values.size() ? values[index] : 0.0f;
    };
    if (output_type == mx::Type::FLOAT)
        return float4(component(0), 0.0f, 0.0f, 1.0f);
    if (output_type == mx::Type::VECTOR2)
        return float4(component(0), component(1), 0.0f, 1.0f);
    if (output_type == mx::Type::COLOR4 || output_type == mx::Type::VECTOR4)
        return float4(component(0), component(1), component(2), component(3));
    return float4(component(0), component(1), component(2), 1.0f);
}

MxParamInfo make_param_info(
    const GenslangPtShaderGenerator& target,
    const mx::DocumentPtr& document,
    mx::ShaderPort* input,
    const CodeGenDesc& request,
    const PreCanonicalInputNames& pre_canonical_names
)
{
    auto authored_input_value
        = [](const mx::DocumentPtr& source_document, const mx::ShaderInput* source_input) -> std::optional<std::string>
    {
        if (!source_document || !source_input || source_input->getPath().empty())
            return std::nullopt;
        mx::ElementPtr element = source_document->getDescendant(source_input->getPath());
        mx::ValueElementPtr value_element = element ? element->asA<mx::ValueElement>() : nullptr;
        if (!value_element)
            return std::nullopt;
        const std::string value = value_element->getResolvedValueString();
        return value.empty() ? std::optional<std::string>{} : std::optional<std::string>{value};
    };
    auto source_sibling_value = [](const mx::DocumentPtr& source_document,
                                   const mx::ShaderPort* filename_socket,
                                   const std::string& input_name) -> std::optional<std::string>
    {
        if (!source_document || !filename_socket || filename_socket->getPath().empty())
            return std::nullopt;
        mx::ElementPtr file_input = source_document->getDescendant(filename_socket->getPath());
        mx::ElementPtr source_node = file_input ? file_input->getParent() : nullptr;
        mx::InterfaceElementPtr source_interface = source_node ? source_node->asA<mx::InterfaceElement>() : nullptr;
        mx::InputPtr source_input = source_interface ? source_interface->getInput(input_name) : nullptr;
        return source_input && source_input->hasValueString()
            ? std::optional<std::string>{source_input->getValueString()}
            : std::optional<std::string>{};
    };
    auto texture_input_value = [&](mx::ShaderNode* node, const std::string& input_name, const std::string& fallback)
    {
        const mx::ShaderInput* source_input = node ? node->getInput(input_name) : nullptr;
        if (std::optional<std::string> authored = authored_input_value(document, source_input))
            return *authored;
        if (source_input) {
            const std::string value = source_input->getValueString();
            if (!value.empty())
                return value;
        }
        if (std::optional<std::string> sibling = source_sibling_value(document, input, input_name))
            return *sibling;
        return fallback;
    };

    using ParamType = params::Type;
    MxParamInfo info;
    const mx::TypeDesc type = input->getType();
    info.param_type = ParamType::unsupported;
    info.mx_type_string = target.getSyntax().getTypeName(type);
    if (type.isFloat2())
        info.param_type = ParamType::float2_;
    else if (type.isFloat3())
        info.param_type = ParamType::float3_;
    else if (type.isFloat4())
        info.param_type = ParamType::float4_;
    else if (type.isScalar()) {
        if (type.getBaseType() == mx::TypeDesc::BASETYPE_BOOLEAN)
            info.param_type = ParamType::bool_;
        else if (type.getBaseType() == mx::TypeDesc::BASETYPE_INTEGER)
            info.param_type = ParamType::int_;
        else if (type.getBaseType() == mx::TypeDesc::BASETYPE_FLOAT)
            info.param_type = ParamType::float_;
        else if (type.getSemantic() == mx::TypeDesc::SEMANTIC_FILENAME)
            info.param_type = ParamType::texture;
    }

    const auto original = pre_canonical_names.find(input);
    info.param_name = original == pre_canonical_names.end() ? input->getName() : original->second;
    info.binding_name = input->getVariable();
    info.interface.folders = {"Params"};
    info.interface.name = info.param_name;

    if (input->getValue() && info.param_type != ParamType::unsupported) {
        if (input->getValue()->isA<int>())
            info.set_value<int>(input->getValue()->asA<int>());
        else if (input->getValue()->isA<bool>())
            info.set_value<bool>(input->getValue()->asA<bool>());
        else if (input->getValue()->isA<float>())
            info.set_value<float>(input->getValue()->asA<float>());
        else if (input->getValue()->isA<mx::Vector2>())
            info.set_value<float2>(input->getValue()->asA<mx::Vector2>());
        else if (input->getValue()->isA<mx::Vector3>())
            info.set_value<float3>(input->getValue()->asA<mx::Vector3>());
        else if (input->getValue()->isA<mx::Vector4>())
            info.set_value<float4>(input->getValue()->asA<mx::Vector4>());
        else if (input->getValue()->isA<mx::Color3>())
            info.set_value<float3>(input->getValue()->asA<mx::Color3>());
        else if (input->getValue()->isA<mx::Color4>())
            info.set_value<float4>(input->getValue()->asA<mx::Color4>());
    }

    if (type == mx::Type::FILENAME) {
        params::TextureInfo texture;
        texture.mx_input_name = input->getVariable();
        texture.file_name = target.getSyntax().getValue(input, true);
        mx::ShaderNode* node = input->getNode();
        texture.uaddressmode = texture_input_value(node, "uaddressmode", "periodic");
        texture.vaddressmode = texture_input_value(node, "vaddressmode", "periodic");
        texture.filtertype = texture_input_value(node, "filtertype", "linear");
        const mx::TypeDesc output_type
            = node && !node->getOutputs().empty() ? node->getOutputs()[0]->getType() : mx::Type::COLOR4;
        texture.border_color
            = texture_border_color_from_default(texture_input_value(node, "default", "0.0"), output_type);
        info.set_value(std::move(texture));
        if (original == pre_canonical_names.end())
            info.param_name = input->getVariable();
        info.interface.name = info.param_name;
        info.is_editable = true;
    } else {
        info.is_editable
            = request.is_editable_param(info.param_name) || request.editable_param_names.contains(input->getPath());
    }
    if (!info.holds_value())
        info.is_editable = false;
    return info;
}

bool finalize_filename_bindings(
    GenslangPtShaderGenerator& target,
    const mx::DocumentPtr& document,
    mx::ShaderGraph& root_graph,
    mx::ShaderStage& pixel_stage,
    const CodeGenDesc& request,
    const PreCanonicalInputNames& pre_canonical_names,
    CodeGenResult& result
)
{
    bool changed = false;
    mx::VariableBlock& public_uniforms = pixel_stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    std::vector<mx::ShaderGraph*> graphs = {&root_graph};
    std::unordered_set<const mx::ShaderGraph*> visited;
    while (!graphs.empty()) {
        mx::ShaderGraph* graph = graphs.back();
        graphs.pop_back();
        if (!visited.insert(graph).second)
            continue;
        for (mx::ShaderNode* node : graph->getNodes()) {
            if (node->hasClassification(mx::ShaderNode::Classification::FILETEXTURE)) {
                for (mx::ShaderInput* input : node->getInputs()) {
                    if (input->getConnection() || input->getType() != mx::Type::FILENAME)
                        continue;
                    result.all_material_params.add_param_info(
                        make_param_info(target, document, input, request, pre_canonical_names)
                    );
                    mx::ShaderPort* filename
                        = public_uniforms.add(mx::Type::FILENAME, input->getVariable(), input->getValue());
                    filename->setPath(input->getPath());
                    input->setValue(mx::Value::createValue(input->getVariable()));
                    changed = true;
                }
            }
            if (mx::ShaderGraph* implementation_graph = node->getImplementation().getGraph())
                graphs.push_back(implementation_graph);
        }
    }
    return changed;
}

void emit_ports(
    const GenslangPtShaderGenerator& target,
    const mx::VariableBlock& block,
    mx::ShaderStage& stage,
    bool public_members,
    bool as_constants,
    bool packed_order = false
)
{
    std::vector<const mx::ShaderPort*> ports;
    ports.reserve(block.size());
    for (std::size_t i = 0; i < block.size(); ++i)
        ports.push_back(block[i]);
    if (packed_order) {
        std::sort(
            ports.begin(),
            ports.end(),
            [](const mx::ShaderPort* lhs, const mx::ShaderPort* rhs)
            {
                const auto alignment_rank = [](const mx::TypeDesc& type)
                {
                    if (type.getSemantic() == mx::TypeDesc::SEMANTIC_MATRIX)
                        return 16u;
                    if (type.getSemantic() == mx::TypeDesc::SEMANTIC_VECTOR
                        || type.getSemantic() == mx::TypeDesc::SEMANTIC_COLOR) {
                        if (type.getSize() >= 3)
                            return 16u;
                        if (type.getSize() == 2)
                            return 8u;
                    }
                    return 4u;
                };
                const uint32_t lhs_alignment = alignment_rank(lhs->getType());
                const uint32_t rhs_alignment = alignment_rank(rhs->getType());
                return lhs_alignment != rhs_alignment ? lhs_alignment > rhs_alignment
                                                      : lhs->getVariable() < rhs->getVariable();
            }
        );
    }
    for (const mx::ShaderPort* port : ports) {
        std::string declaration;
        if (public_members)
            declaration += "public ";
        if (as_constants)
            declaration += "static const ";
        declaration += target.getSyntax().getTypeName(port->getType()) + " " + port->getVariable();
        if (as_constants && port->getType() != mx::Type::FILENAME) {
            std::string value = target.getSyntax().getValue(port, true);
            if (value.empty())
                value = default_value(target, port->getType());
            if (port->getType().getSize() > 1 && value.find(',') != std::string::npos
                && value.find('(') == std::string::npos && !value.starts_with("{")) {
                value = target.getSyntax().getTypeName(port->getType()) + "(" + value + ")";
            }
            declaration += " = " + value;
        }
        target.emitLine(declaration, stage);
    }
}

void emit_material_data_and_stack(
    const GenslangPtShaderGenerator& target,
    const GeneratedNames& names,
    const NodeEmissionFacts& emission_facts,
    const GraphStoragePlan& storage,
    CodeGenResult& result,
    mx::ShaderStage& stage
)
{
    const mx::VariableBlock& uniforms = stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    target.emitLine("public struct " + names.material_data_name, stage, false);
    target.emitScopeBegin(stage);
    target.emitLine("// Uniform graph inputs and texture handles.", stage, false);
    emit_ports(target, uniforms, stage, true, false, true);
    if (uniforms.empty())
        target.emitLine("public uint _empty_struct_padding", stage);
    target.emitScopeEnd(stage, true);
    target.emitLineBreak(stage);
    result.data_struct_size = calculate_data_size_upper_bound(uniforms);


    const std::size_t bsdf_count = std::max<std::size_t>(emission_facts.bsdf_types.size(), 1);
    const std::size_t layer_count = std::max<std::size_t>(emission_facts.combiner_count, 1);
    std::string bsdf_members;
    std::string setter_cases;
    for (std::size_t i = 0; i < emission_facts.bsdf_types.size(); ++i) {
        bsdf_members += fmt::format("        {} bsdf{} = {{}};\n", emission_facts.bsdf_types[i], i);
        setter_cases += fmt::format(
            "                {} bsdf{} = reinterpret<{}, TBSDF>(bsdf); break;\n",
            i + 1 < emission_facts.bsdf_types.size() ? fmt::format("case {}:", i) : "default:",
            i,
            emission_facts.bsdf_types[i]
        );
    }
    if (emission_facts.bsdf_types.empty())
        setter_cases = "                default: break;\n";
    std::string init_bsdf;
    std::string init_layer;
    for (std::size_t i = 0; i < bsdf_count; ++i)
        init_bsdf += fmt::format("            bsdf_transmission_scales[{}] = float3(1.0);\n", i);
    for (std::size_t i = 0; i < layer_count; ++i)
        init_layer += fmt::format("            layer_transmission_scales[{}] = float3(1.0);\n", i);
    if (storage.kind == GraphStorageKind::common_with_frames)
        target.emitLine("#define MTLX_STACK_HAS_BSDF_FRAMES 1", stage, false);
    std::string source = stage.getSourceCode();
    source += render_snippet(
        "stack_data_basic.slang",
        {
            {"$MxStackDataType", "MxStackData"},
            {"$MxBsdfCount", std::to_string(bsdf_count)},
            {"$MxLayerCount", std::to_string(layer_count)},
            {"$MxBsdfFramesLine",
             storage.kind == GraphStorageKind::common_with_frames
                 ? fmt::format("        Frame bsdf_frames[{}] = {{}};\n", bsdf_count)
                 : ""},
            {"$MxBsdfMemberLines", bsdf_members},
            {"$MxSetBsdfCases", setter_cases},
            {"$MxInitVdfBsdfLines", init_bsdf},
            {"$MxInitVdfLayerLines", init_layer},
        }
    );
    stage.setSourceCode(source);
    target.emitLineBreak(stage);
}

bool output_uses_empty_material(const mx::ShaderGraphOutputSocket* output)
{
    const mx::ShaderNode* material = output && output->getConnection() ? output->getConnection()->getNode() : nullptr;
    const mx::ShaderInput* surface = material ? material->getInput(mx::ShaderNode::SURFACESHADER) : nullptr;
    return surface && !surface->getConnection();
}

void emit_materialx_graph(
    const GenslangPtShaderGenerator& target,
    const GeneratedNames& names,
    const std::string& stack_type,
    bool force_inline_graph_boundaries,
    mx::ShaderGraph& graph,
    mx::GenContext& context,
    mx::ShaderStage& stage
)
{
    target.emitLine("struct " + names.graph_name + "<TLodSampler : ILodSampler>", stage, false);
    target.emitScopeBegin(stage);
    emit_ports(target, stage.getInputBlock("i"), stage, false, false);
    emit_ports(target, stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS), stage, false, false);
    target.emitLine("SurfaceInteraction si", stage);
    target.emitLine("uint hints", stage);
    target.emitLine("TLodSampler lod_sampler", stage);
    target.emitLine("ClosureData closure_data", stage);
    target.emitLine("int mx_stack_next_bsdf = 0", stage);
    target.emitLine("int mx_stack_next_layer = 0", stage);
    target.emitLine(stack_type + " mx_stack = {}", stage);
    for (std::size_t i = 0; i < stage.getOutputBlock("o").size(); ++i) {
        const mx::ShaderPort* output = stage.getOutputBlock("o")[i];
        target.emitLine(
            target.getSyntax().getTypeName(output->getType()) + " " + output->getVariable() + " = "
                + default_value(target, output->getType()),
            stage
        );
    }
    target.emitLineBreak(stage);

    target.emitLine(
        "__init(" + names.material_data_name
            + " data_, SurfaceInteraction si_, TLodSampler lod_sampler_, const uint hints_)",
        stage,
        false
    );
    target.emitFunctionBodyBegin(graph, context, stage);
    target.emitLine("si = si_", stage);
    target.emitLine("lod_sampler = lod_sampler_", stage);
    target.emitLine("hints = hints_", stage);
    std::vector<const mx::ShaderPort*> public_uniforms;
    const mx::VariableBlock& public_uniform_block = stage.getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    public_uniforms.reserve(public_uniform_block.size());
    for (std::size_t i = 0; i < public_uniform_block.size(); ++i)
        public_uniforms.push_back(public_uniform_block[i]);
    std::sort(
        public_uniforms.begin(),
        public_uniforms.end(),
        [](const mx::ShaderPort* lhs, const mx::ShaderPort* rhs)
        {
            return lhs->getVariable() < rhs->getVariable();
        }
    );
    for (const mx::ShaderPort* input : public_uniforms) {
        target.emitLine(input->getVariable() + " = data_." + input->getVariable(), stage);
    }
    target.emitLine("mx_stack.init_vdf_transmission_scales()", stage);
    target.emitLine("eval_shader_graph()", stage);
    target.emitFunctionBodyEnd(graph, context, stage);

    // Scalarized complex mix requires this boundary to disappear. Otherwise
    // CUDA may take the graph's address and materialize its state in local memory.
    if (force_inline_graph_boundaries)
        target.emitLine("[ForceInline]", stage, false);
    target.emitLine("[mutating] void eval_shader_graph()", stage, false);
    target.emitFunctionBodyBegin(graph, context, stage);
    if (!stage.getConstantBlock().empty()) {
        target.emitVariableDeclarations(
            stage.getConstantBlock(),
            target.getSyntax().getConstantQualifier(),
            mx::Syntax::SEMICOLON,
            context,
            stage
        );
        target.emitLineBreak(stage);
    }
    target.emitFunctionCalls(graph, context, stage, mx::ShaderNode::Classification::TEXTURE);
    for (mx::ShaderGraphOutputSocket* socket : graph.getOutputSockets()) {
        const mx::ShaderNode* upstream = socket->getConnection() ? socket->getConnection()->getNode() : nullptr;
        if (upstream && upstream->getParent() == &graph
            && (upstream->hasClassification(mx::ShaderNode::Classification::CLOSURE)
                || upstream->hasClassification(mx::ShaderNode::Classification::SHADER)
                || upstream->hasClassification(mx::ShaderNode::Classification::MATERIAL))) {
            target.emitFunctionCall(*upstream, context, stage);
        }
    }
    for (std::size_t i = 0; i < stage.getOutputBlock("o").size(); ++i) {
        mx::ShaderGraphOutputSocket* socket = graph.getOutputSocket(i);
        const std::string value = output_uses_empty_material(socket) ? default_value(target, socket->getType())
                                                                     : target.getUpstreamResult(socket, context);
        target.emitLine(socket->getVariable() + " = " + value, stage);
    }
    target.emitFunctionBodyEnd(graph, context, stage);

    std::string source = stage.getSourceCode();
    source += load_snippet("mx_normal_adjustment.slang");
    stage.setSourceCode(source);
    target.emitFunctionDefinitions(graph, context, stage);
    emit_ports(target, stage.getUniformBlock(mx::HW::PRIVATE_UNIFORMS), stage, false, true);
    target.emitScopeEnd(stage);
    target.emitLineBreak(stage);
}

void register_source_path_with_parent(mx::GenContext& context, const std::filesystem::path& path)
{
    if (path.empty())
        return;
    context.registerSourceCodeSearchPath(mx::FilePath(path.string()));
    if (!path.parent_path().empty())
        context.registerSourceCodeSearchPath(mx::FilePath(path.parent_path().string()));
}

void register_source_paths(mx::GenContext& context)
{
    register_source_path_with_parent(context, platform::runtime_directory() / "mtlx/standard");
    const std::filesystem::path library_path = platform::runtime_directory() / "mtlx/libraries";
    register_source_path_with_parent(context, library_path);
    register_source_path_with_parent(context, library_path / "pbrlib" / "genslangpt");
    register_source_path_with_parent(context, library_path / "stdlib" / "genslangpt");
    register_source_path_with_parent(context, platform::project_directory() / "external" / "MaterialX" / "libraries");
}

void install_unit_system(GenslangPtShaderGenerator& generator, const mx::DocumentPtr& document)
{
    mx::UnitSystemPtr unit_system = mx::UnitSystem::create(generator.getTarget());
    unit_system->loadLibrary(document);
    unit_system->setUnitConverterRegistry(mx::UnitConverterRegistry::create());
    if (mx::UnitTypeDefPtr distance = document->getUnitTypeDef("distance"))
        unit_system->getUnitConverterRegistry()->addUnitConverter(distance, mx::LinearUnitConverter::create(distance));
    if (mx::UnitTypeDefPtr angle = document->getUnitTypeDef("angle"))
        unit_system->getUnitConverterRegistry()->addUnitConverter(angle, mx::LinearUnitConverter::create(angle));
    generator.setUnitSystem(unit_system);
}

// Describe the prepared graph and resolved implementations, not the authored XML.
CodeGenCache::Key make_cache_key(
    const MaterialXResources& resources,
    const CodeGenDesc& request,
    const GraphAnalysis& analysis,
    const ClassifiedLayering& layering,
    const CodeGenResult& result,
    std::string_view graph_name
)
{
    sgl::SHA1 hash;
    // Frame strings and scalar bytes so adjacent fields cannot alias. This key
    // is process-local; native scalar representations need no portable encoding.
    auto add = [&](std::string_view value)
    {
        hash.update(uint8_t(0));
        hash.update(uint64_t(value.size()));
        hash.update(value);
    };
    auto number = [&](auto value)
    {
        hash.update(uint8_t(1));
        hash.update(uint8_t(sizeof(value)));
        hash.update(value);
    };
    add(graph_name);
    number(request.canonicalize_names);
    number(static_cast<int>(request.layering_mode));
    number(static_cast<int>(request.compensation_mode));
    number(static_cast<int>(request.zeltner_sheen_coefficients));
    number(static_cast<uint32_t>(request.optimize_graph_flags));
    number(request.use_slang_derivatives);
    number(request.payload_size);
    number(analysis.ancillary_facts.has_emission);
    number(static_cast<int>(result.opacity.kind));
    number(result.opacity.constant);
    // Editable defaults can still affect baked lobe classification (notably
    // scatter_mode). Key the actual classified facts, not an assumed split
    // between runtime values and every decision made by the generator.
    number(layering.bsdfs.size());
    for (const auto& bsdf : layering.bsdfs) {
        add(bsdf.bsdf_type);
        number(static_cast<uint32_t>(bsdf.lobe_flags));
        number(bsdf.transmissive);
        number(bsdf.preserve_tangent_frame);
    }
    number(static_cast<int>(layering.root_contextual_weight.kind));
    for (float value : layering.root_contextual_weight.constant)
        number(value);
    number(static_cast<int>(mx::Value::getFloatFormat()));
    number(mx::Value::getFloatPrecision());
    const auto value = [&](const mx::ShaderPort* port)
    {
        add(port->getValueString());
    };
    // MaterialX emits all registered type definitions, including authored structs.
    for (const auto& type : resources.shader_generator->syntax_ptr()->getTypeSyntaxes()) {
        add(type->getName());
        add(type->getTypeAlias());
        add(type->getTypeDefinition());
        add(type->getDefaultValue(false));
        add(type->getDefaultValue(true));
    }

    const auto& public_uniforms = resources.pixel_stage->getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    std::unordered_set<const mx::ShaderPort*> runtime_ports;
    number(public_uniforms.size());
    for (const mx::ShaderPort* port : public_uniforms.getVariableOrder()) {
        runtime_ports.insert(port);
        add(port->getVariable());
        add(port->getType().getName());
    }
    number(result.material_aov_outputs.size());
    for (const auto& output : result.material_aov_outputs) {
        number(output.id);
        add(output.name);
        number(output.component_count);
    }

    std::vector<const mx::ShaderGraph*> graphs{resources.graph.get()};
    std::unordered_map<const mx::ShaderGraph*, size_t> graph_ids{{resources.graph.get(), 0}};
    for (size_t graph_index = 0; graph_index < graphs.size(); ++graph_index) {
        const mx::ShaderGraph& graph = *graphs[graph_index];
        const auto& nodes = graph.getNodes();
        number(nodes.size() + 1);
        // The graph itself owns the interface sockets; visit those before its nodes.
        for (size_t node_index = 0; node_index <= nodes.size(); ++node_index) {
            const mx::ShaderNode* node = node_index == 0 ? &graph : nodes[node_index - 1];
            if (!request.canonicalize_names)
                add(node->getName());
            number(node->getClassification());
            if (node != &graph) {
                const auto& implementation = node->getImplementation();
                add(resources.shader_generator->implementation_name(implementation));
                // Registered C++ implementations can also consult the NodeDef category.
                const auto definition = resources.document->getImplementation(implementation.getName());
                const auto node_def = definition ? definition->getNodeDef() : nullptr;
                add(node_def ? node_def->getNodeString() : std::string());
                if (const auto* geometry = dynamic_cast<const GeomPropValueNode*>(&implementation))
                    add(geometry->resolve(*resources.document, request, *node, *resources.pixel_stage).expression);
                if (const mx::ShaderGraph* nested = implementation.getGraph()) {
                    auto [it, inserted] = graph_ids.emplace(nested, graphs.size());
                    if (inserted)
                        graphs.push_back(nested);
                    number(it->second);
                } else {
                    add("source");
                }
            }
            auto port_key = [&](const mx::ShaderPort* port, bool include_value)
            {
                add(port->getType().getName());
                add(port->getVariable());
                add(port->getSemantic());
                add(port->getColorSpace());
                add(port->getUnit());
                add(port->getGeomProp());
                number(port->getFlags());
                // Filenames are handles initialized from this request's fresh
                // parameter metadata, never source-code literals.
                include_value &= !runtime_ports.contains(port) && port->getType() != mx::Type::FILENAME;
                number(include_value && bool(port->getValue()));
                if (include_value && port->getValue())
                    value(port);
            };
            number(node->numInputs());
            for (const mx::ShaderInput* input : node->getInputs()) {
                const mx::ShaderOutput* connection = input->getConnection();
                port_key(input, !connection);
                number(connection != nullptr);
                if (connection)
                    add(connection->getVariable());
            }
            number(node->numOutputs());
            for (const mx::ShaderOutput* output : node->getOutputs())
                port_key(output, true);
        }
    }
    // Uniform blocks and constants can include variables created by node implementations.
    for (const mx::VariableBlock* block : {
             &resources.pixel_stage->getUniformBlock(mx::HW::PRIVATE_UNIFORMS),
             &resources.pixel_stage->getConstantBlock(),
         }) {
        number(block->size());
        for (const mx::ShaderPort* port : block->getVariableOrder()) {
            add(port->getType().getName());
            add(port->getVariable());
            number(bool(port->getValue()));
            if (port->getValue())
                value(port);
        }
    }
    return hash.digest();
}

/// Emit one Slang module from the completed analysis and representation.
void emit_module(
    MaterialXResources& resources,
    const CodeGenDesc& request,
    const GraphAnalysis& analysis,
    const GeneratedNames& names,
    const GraphStoragePlan& storage,
    ILayeringPolicy& policy,
    EmptySurfaceHandler& empty_surface_handler,
    CodeGenResult& result
)
{
    // Common module header and declarations.
    GenslangPtShaderGenerator& target = *resources.shader_generator;
    mx::ShaderStage& stage = *resources.pixel_stage;
    std::string source = stage.getSourceCode();
    const char* library_import = "falcor2.render.materials.mtlx.mtlx";
    source += render_snippet(
        "module_header_basic.slang",
        {
            {"$MxModuleName", names.module_name},
            {"$MxLibraryImport", library_import},
        }
    );
    stage.setSourceCode(source);
    target.emitLineBreak(stage);
    target.emitLine("namespace mtlx", stage, false);
    target.emitScopeBegin(stage);
    target.emitLineBreak(stage);

    emit_material_data_and_stack(target, names, analysis.node_emission_facts, storage, result, stage);
    target.emitTypeDefinitions(*resources.context, stage);
    target.emitLine("struct ClosureData { }", stage, false);
    source = stage.getSourceCode();
    if (graph_uses_space_helpers(*resources.graph)) {
        source += load_snippet("mx_space_helpers.slang");
        // Conservatively retain transforms even when a later optimizer removes an identity conversion.
        result.uses_geometry_properties = true;
    }
    stage.setSourceCode(source);
    target.emitLibraryInclude("stdlib/genslang/lib/mx_math.slang", *resources.context, stage);
    target.emitLineBreak(stage);

    EmissionContext emission{
        target,
        *resources.graph,
        *resources.context,
        stage,
        request,
        names,
        analysis.ancillary_facts,
        result,
    };

    // Representation declarations that must precede the generated graph.
    if (analysis.graph_facts.has_surface_root) {
        policy.emit_representation_declarations(emission);
    } else {
        // The empty-surface representation needs no pre-graph declarations.
    }

    // Emit the MaterialX graph exactly once for every representation.
    {
        EmissionScratch emission_scratch;
        bool force_inline_graph_boundaries = false;
        EmissionUserDataScope user_data_scope(
            *resources.context,
            request,
            *resources.document,
            analysis.node_emission_facts,
            emission_scratch,
            force_inline_graph_boundaries
        );
        emit_materialx_graph(
            target,
            names,
            storage.stack_type,
            force_inline_graph_boundaries,
            *resources.graph,
            *resources.context,
            stage
        );
        target.apply_compound_annotations(stage, emission_scratch);
        result.uses_geometry_properties |= emission_scratch.uses_geometry_properties;
    }

    if (!analysis.ancillary_facts.emission_query_source.empty()) {
        std::string query_source = analysis.ancillary_facts.emission_query_source;
        replace_all(query_source, "$MxMaterialDataName", names.material_data_name);
        target.emitString(query_source, stage);
    }

    // Material implementation, including its opacity, emission, volume, and
    // property methods. Empty surfaces use their dedicated implementation.
    if (analysis.graph_facts.has_surface_root) {
        policy.emit_material_implementation(emission);
    } else {
        empty_surface_handler.emit_material_implementation(emission);
    }

    // Close the module namespace. Finalization runs after this function so it
    // can apply the common ordered source rewrites and result metadata.
    target.emitScopeEnd(stage, false);
    target.emitLineBreak(stage);
}

void finalize_module(
    GenslangPtShaderGenerator& target,
    mx::ShaderStage& stage,
    const GeneratedNames& names,
    CodeGenResult& result
)
{
    auto rewrite_source = [&](auto&& rewrite)
    {
        std::string source = stage.getSourceCode();
        rewrite(source);
        stage.setSourceCode(source);
    };

    // Convert MaterialX's GLSL-shaped stage to Slang before applying Falcor
    // compatibility substitutions. Graph mutability is already stated at
    // each declaration owner.
    target.convert_glsl_to_slang(stage);
    rewrite_source(
        [](std::string& source)
        {
            remove_blank_line_indentation(source);
            preserve_generated_mutating_signature_shape(source);
        }
    );
    rewrite_source(
        [&](std::string& source)
        {
            replace_all(source, "M_PI_INV", "(1.0 / float.PI)");
            replace_all(source, "M_PI", "float.PI");
        }
    );
    // The public names are derived from the source after syntax and policy
    // rewrites, but before request-local hardware/token substitutions.
    sgl::SHA1 hash;
    hash.update(stage.getSourceCode());
    const std::string source_hash = hash.hex_digest().substr(0, 12);

    rewrite_source(
        [](std::string& source)
        {
            replace_all(source, "$vd.$positionWorld", "si.position_ws");
            replace_all(source, "$vd.$normalWorld", "si.shading_frame_ws.normal");
            replace_all(source, "$vd.$tangentWorld", "si.shading_frame_ws.tangent");
            replace_all(source, "$vd.$bitangentWorld", "si.shading_frame_ws.bitangent");
            replace_all(source, "$vd.$texcoord_0", "si.uv");
            replace_all(source, "$vd.$texcoord", "si.uv");
            source = std::regex_replace(source, std::regex(R"(\bclosureData\b)"), "closure_data");
        }
    );
    target.apply_token_substitutions(stage);
    rewrite_source(
        [&](std::string& source)
        {
            replace_all(source, "$HASH", source_hash);
        }
    );

    result.module_name = names.module_name;
    result.material_name = "mtlx::" + names.material_name;
    result.material_instance_name = "mtlx::" + names.material_instance_name;
    result.material_data_name = "mtlx::" + names.material_data_name;
    replace_all(result.module_name, "$HASH", source_hash);
    replace_all(result.material_name, "$HASH", source_hash);
    replace_all(result.material_instance_name, "$HASH", source_hash);
    replace_all(result.material_data_name, "$HASH", source_hash);
}

void write_result_metadata(const CodeGenDesc& request, const GraphAnalysis& analysis, CodeGenResult& result)
{
    result.codegen_metadata.set("bsdf_profile", std::string("mtlx"));
    result.codegen_metadata.set("target", std::string(GenslangPtShaderGenerator::TARGET));
    result.codegen_metadata.set("graph_emission", true);
    write_closure_pruning_report_fields(result.codegen_metadata, analysis.optimization.closure_pruning);
    result.codegen_metadata.set("requested_layering_mode", request.layering_mode);
    result.codegen_metadata.set("compensation_mode", request.compensation_mode);
    result.codegen_metadata.set("layering_root_valid", analysis.graph_facts.has_surface_root);
    result.codegen_metadata.set(
        "layering_combiner_count",
        static_cast<int>(analysis.node_emission_facts.combiner_count)
    );
    result.codegen_metadata.set(
        "layering_bsdf_count",
        static_cast<int>(analysis.node_emission_facts.bsdf_types.size())
    );
}

} // namespace

namespace {

/// Generate one structured MaterialX module through a visible causal waterfall.
///
/// The front end validates the request, selects the layering policy, prepares
/// the MaterialX document, builds the ShaderGraph, freezes its topology, and
/// derives the common shader interface and graph facts. Material analysis then
/// classifies the layering tree, validates the completed GraphAnalysis, and
/// transfers the still-live ClassifiedLayering to either the selected policy or
/// the explicit empty-surface handler. Code generation emits one module from
/// those completed products and applies the fixed finalization waterfall.
///
/// MaterialXResources exists only to express the coupled MaterialX ownership
/// and destruction order. Request routing, important products, mutations,
/// branch points, and joins stay as named locals in this function. Focused
/// helpers retain recursive or cohesive domain algorithms rather than hiding
/// phases of this waterfall.
std::unique_ptr<CodeGenResult> generate_impl(const CodeGenDesc& desc, CodeGenCache* cache)
{
    // Front end: prepare the document and freeze graph topology.
    validate_request(desc);
    EmptySurfaceHandler empty_surface_handler;
    auto result = std::make_unique<CodeGenResult>();

    LoadedDocument loaded = load_and_select_document(desc);
    result->material_aov_outputs = std::move(loaded.material_aov_outputs);
    // genslangpt normally overrides heighttonormal with this shader generator's
    // ray-tracing implementation, which resamples the height expression
    // without screen-space derivatives. In derivative mode, remove only that
    // target-specific declaration so MaterialX falls back through
    // genslangpt -> genslang -> genglsl to the inherited dFdx/dFdy version.
    // Shared libraries already selected their immutable derivative variant.
    if (desc.use_slang_derivatives && !loaded.document->hasDataLibrary())
        loaded.document->removeImplementation("IM_heighttonormal_vector3_genslangpt");

    // MaterialX normally supplies default geomprops implicitly. Materialize
    // them before ShaderGraph construction so graph optimization and static
    // analysis see the same explicit inputs as authored graph connections.
    DocumentTransformPass transforms = DocumentTransformPass::materialize_default_geomprops;
    const EditableInputPredicate is_editable = [&](const mx::Node& node, const std::string& name)
    {
        return desc.make_editable || desc.editable_param_names.contains(node.getNamePath() + "/" + name)
            || desc.editable_param_names.contains(&node == loaded.root.get() ? name : node.getName() + "_" + name);
    };
    MovedInputPaths moved_inputs;
    apply_document_transforms(
        transforms,
        *loaded.document,
        loaded.root,
        loaded.additional_outputs,
        is_editable,
        moved_inputs
    );

    MaterialXResources resources;
    resources.document = std::move(loaded.document);
    resources.root = std::move(loaded.root);
    resources.shader_generator = GenslangPtShaderGenerator::create(
        desc.use_slang_derivatives,
        desc.canonicalize_names,
        std::move(loaded.builtin_node_graphs)
    );
    resources.context = std::make_unique<mx::GenContext>(resources.shader_generator);
    register_source_paths(*resources.context);

    // MaterialX elides constant nodes during ShaderGraph construction, before we
    // collect public inputs and match the caller's editable parameter names. For
    // example, tint.value connected to surface.emission_color would be folded into
    // the downstream input, losing the requested tint_value control. Preserve node
    // identity whenever any inputs may be editable, including explicit selections.
    // This MaterialX option is graph-wide; it does not make every constant editable.
    // Unselected inputs still bake into generated code, and generation without
    // editable parameters retains normal constant-node elision.
    resources.context->getOptions().elideConstantNodes = !desc.make_editable && desc.editable_param_names.empty();

    // Search roots already point inside the installed libraries.
    resources.context->getOptions().libraryPrefix = mx::FilePath();
    // Keep MaterialX's hardware mix-to-weighted-leaves-plus-add rewrite. Besides
    // its original raster benefit, this folds mix factors into generated leaf
    // weights and substantially simplifies the structured flat-weight programs.
    // Add traversal below deliberately preserves the rewritten in1/in2 order:
    // sampling therefore differs from OSL's native mix CDF order, while eval/pdf
    // remain equivalent. Do not infer the old mix order from generated node names.
    resources.context->getOptions().premultipliedBsdfAdd = true;
    resources.shader_generator->setColorManagementSystem(
        mx::DefaultColorManagementSystem::create(GenslangPtShaderGenerator::TARGET)
    );
    resources.shader_generator->getColorManagementSystem()->loadLibrary(resources.document);
    install_unit_system(*resources.shader_generator, resources.document);
    resources.shader_generator->registerTypeDefs(resources.document);
    resources.context->getOptions().targetColorSpaceOverride = WORKING_COLOR_SPACE;
    resources.context->getOptions().targetDistanceUnit = "meter";
    resources.graph = AOVShaderGraph::create(
        loaded.shader_base_name,
        resources.root,
        loaded.additional_outputs,
        *resources.context
    );
    GraphAnalysis analysis{};

    // Simplify statically empty closure subgraphs before any analysis depends
    // on their topology. The retained pruning report describes output rewrites,
    // asymmetric-mix simplifications, removed nodes, and fixpoint iterations.
    collect_input_names(*resources.graph, *resources.document, desc, moved_inputs, analysis.pre_canonical_input_names);
    const RuntimeInputPredicate is_runtime_input = [&](const mx::ShaderOutput& input)
    {
        const auto found = analysis.pre_canonical_input_names.find(&input);
        return desc.is_editable_param(
                   found == analysis.pre_canonical_input_names.end() ? input.getName() : found->second
               )
            || desc.editable_param_names.contains(input.getPath());
    };
    run_closure_pruning(
        *resources.graph,
        *resources.context,
        desc,
        is_runtime_input,
        analysis.optimization.closure_pruning
    );

    if (desc.canonicalize_names)
        canonicalize_graph_bindings(*resources.graph, *resources.shader_generator, analysis.pre_canonical_input_names);

    // Build MaterialX's shader interface and facts derived from the frozen graph.
    resources.shader = std::make_shared<mx::Shader>(resources.graph->getName(), resources.graph);
    resources.vertex_stage = resources.shader_generator->create_stage(mx::Stage::VERTEX, *resources.shader);
    resources.vertex_stage->createInputBlock(mx::HW::VERTEX_INPUTS, "i_vs");
    resources.vertex_stage->createUniformBlock(mx::HW::PRIVATE_UNIFORMS, "u_prv");
    resources.vertex_stage->createUniformBlock(mx::HW::PUBLIC_UNIFORMS, "u_pub");
    resources.pixel_stage = resources.shader_generator->create_stage(mx::Stage::PIXEL, *resources.shader);
    resources.pixel_stage->createUniformBlock(mx::HW::PRIVATE_UNIFORMS);
    resources.pixel_stage->createUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    resources.pixel_stage->createInputBlock("i");
    resources.pixel_stage->createOutputBlock("o");
    mx::addStageConnectorBlock(
        mx::HW::VERTEX_DATA,
        mx::HW::T_VERTEX_DATA_INSTANCE,
        *resources.vertex_stage,
        *resources.pixel_stage
    );
    resources.shader_generator->create_variables(resources.graph, *resources.context, *resources.shader);

    mx::VariableBlock& public_uniforms = resources.pixel_stage->getUniformBlock(mx::HW::PUBLIC_UNIFORMS);
    mx::VariableBlock& private_uniforms = resources.pixel_stage->getUniformBlock(mx::HW::PRIVATE_UNIFORMS);
    for (mx::ShaderGraphInputSocket* input : resources.graph->getInputSockets()) {
        if (input->getConnections().empty() || !resources.graph->isEditable(*input))
            continue;
        MxParamInfo info = make_param_info(
            *resources.shader_generator,
            resources.document,
            input,
            desc,
            analysis.pre_canonical_input_names
        );
        mx::VariableBlock& destination = info.is_editable ? public_uniforms : private_uniforms;
        destination.add(input->getSelf());
        result->all_material_params.add_param_info(std::move(info));
    }
    mx::VariableBlock& outputs = resources.pixel_stage->getOutputBlock("o");
    for (mx::ShaderGraphOutputSocket* output : resources.graph->getOutputSockets())
        outputs.add(output->getSelf());

    const bool needs_static_microfacet_constants = is_set(
        desc.optimize_graph_flags,
        OptimizeGraphFlags::fresnel_policy | OptimizeGraphFlags::static_scatter_mode
    );
    // Surface shader and material outputs expose `.opacity` and `.edf.radiance` fields. This is only an output-shape
    // property, not evidence of transparency or light emission; their opacity still needs classification as opaque,
    // constant, or dynamic for the generated material wrapper.
    const bool has_material_output = output_block_has_material_output(outputs);
    const bool needs_static_input_query = needs_static_microfacet_constants || has_material_output;

    std::optional<StaticInputQuery> static_input_query;

    // Microfacet specialization and static opacity ask the same question:
    // which graph inputs remain compile-time values after editable parameters
    // are treated as runtime inputs. Solve that shared graph problem at most
    // once and let both consumers query the same result.
    if (needs_static_input_query) {
        const StaticValueAnalysisResult& static_values
            = analysis.static_values.emplace(analyze_graph_static_values(*resources.graph, is_runtime_input));
        static_input_query.emplace(static_values.output_static_values, &resources.pixel_stage->getConstantBlock());

        if (has_material_output) {
            if (auto static_opacity = materialx_graph_static_opacity(*resources.graph, *static_input_query)) {
                result->opacity.constant = *static_opacity;
                if (*static_opacity < 1.0f)
                    result->opacity.kind = CodeGenResult::OpacityResult::Kind::constant;
            } else {
                result->opacity.kind = CodeGenResult::OpacityResult::Kind::dynamic;
            }
        }
    }
    analysis.ancillary_facts.has_emission = graph_has_emission(*resources.graph) && element_may_emit(resources.root);
    if (analysis.ancillary_facts.has_emission && static_input_query) {
        const auto emission_query = build_emission_query(
            *resources.graph,
            public_uniforms,
            *static_input_query,
            *resources.shader_generator,
            *resources.context
        );
        analysis.ancillary_facts.emission_predicate = emission_query.predicate;
        analysis.ancillary_facts.emission_query_source = emission_query.source;
    }
    // Material analysis: classify layering, finalize bindings, and build the
    // selected representation while all source facts remain visibly available.
    ClassifiedLayering layering = LayeringWalker::build_and_classify(
        *resources.graph,
        desc,
        &resources.pixel_stage->getConstantBlock(),
        static_input_query ? &*static_input_query : nullptr,
        analysis.layering_traversal_order
    );
    const bool has_hair = std::any_of(
        layering.bsdfs.begin(),
        layering.bsdfs.end(),
        [](const auto& bsdf)
        {
            return bsdf.kind == BsdfKind::chiang_hair;
        }
    );
    result->is_hair = has_hair;
    if (has_hair) {
        // TODO (MTLX-FUT-041): Extend fiber models and composition beyond a single Chiang closure.
        FALCOR_CHECK(
            layering.bsdfs.size() == 1 && layering.main_layer.is_bsdf() && layering.combiners.empty()
                && layering.root_contextual_weight.kind == RootContextualWeight::Kind::identity,
            "MaterialX hair requires a single Chiang closure; hair mixtures, layers, and closure weights are not "
            "supported."
        );
    }
    analysis.graph_facts.has_surface_root = layering.root_kind == SurfaceRootKind::non_empty;
    analysis.node_emission_facts.combiner_count = layering.combiners.size();
    for (const ClassifiedLayering::BsdfDesc& bsdf : layering.bsdfs) {
        analysis.node_emission_facts.bsdf_types.push_back(bsdf.bsdf_type);
        if (bsdf.microfacet_selection) {
            FALCOR_CHECK(
                bsdf.bsdf_type == microfacet_bsdf_type(*bsdf.microfacet_selection),
                "Structured MaterialX microfacet storage type disagrees with its resolved policy."
            );
            auto [found, inserted] = analysis.node_emission_facts.microfacet_selections.emplace(
                bsdf.shader_node,
                *bsdf.microfacet_selection
            );
            FALCOR_CHECK(
                inserted || found->second == *bsdf.microfacet_selection,
                "One emitted MaterialX microfacet node retains conflicting policy selections."
            );
        }
        if (bsdf.sheen_selection) {
            FALCOR_CHECK(
                bsdf.bsdf_type == sheen_bsdf_type(*bsdf.sheen_selection),
                "Structured MaterialX sheen storage type disagrees with its resolved policy."
            );
            auto [found, inserted]
                = analysis.node_emission_facts.sheen_selections.emplace(bsdf.shader_node, *bsdf.sheen_selection);
            FALCOR_CHECK(
                inserted || found->second == *bsdf.sheen_selection,
                "One emitted MaterialX sheen node retains conflicting policy selections."
            );
        }
    }
    result->codegen_metadata.set_list("materialx_layering_traversal_order", analysis.layering_traversal_order);

    // Resolve texture filenames only after parameter metadata has captured the
    // authored bindings. This may rename bindings, but no later analysis
    // depends on their pre-resolution spelling.
    finalize_filename_bindings(
        *resources.shader_generator,
        resources.document,
        *resources.graph,
        *resources.pixel_stage,
        desc,
        analysis.pre_canonical_input_names,
        *result
    );

    // Validate the progressively built aggregate while ClassifiedLayering is
    // still available as the independent authority for its root and lobe
    // counts. After this point the tree moves into its representation owner and
    // GraphAnalysis remains const and address-stable through graph emission.
    FALCOR_CHECK(
        analysis.static_values.has_value() == needs_static_input_query,
        "Structured MaterialX static-value analysis does not match the selected backend requirements."
    );
    FALCOR_CHECK(
        analysis.graph_facts.has_surface_root == (layering.root_kind == SurfaceRootKind::non_empty),
        "Structured MaterialX surface-root facts disagree with the classified layering description."
    );
    FALCOR_CHECK(
        analysis.node_emission_facts.bsdf_types.size() == layering.bsdfs.size(),
        "Structured MaterialX BSDF emission facts disagree with the layering description."
    );
    FALCOR_CHECK(
        analysis.node_emission_facts.combiner_count == layering.combiners.size(),
        "Structured MaterialX combiner emission facts disagree with the layering description."
    );
    const GraphAnalysis& complete_analysis = analysis;
    write_result_metadata(desc, complete_analysis, *result);

    // Keep the hash as a placeholder until finalization has seen the complete
    // generated source. Every public generated name uses that same eventual
    // hash, while MaterialX receives the graph function name immediately.
    std::string shader_name = resources.graph->getName();
    resources.shader_generator->syntax_ptr()->makeIdentifier(shader_name, resources.graph->getIdentifierMap());
    shader_name += "_$HASH";
    const GeneratedNames names{
        "mtlx_" + shader_name + "_Module",
        "Mx_" + shader_name + "_Graph",
        "Mx_" + shader_name + "_Data",
        "Mx_" + shader_name + "_MaterialInstance",
        "Mx_" + shader_name + "_Material",
    };

    // Include the emitted graph name: without canonicalization, root-only renames
    // change the module even when every port binding stays the same.
    // All instance values, texture bindings and analysis remain fresh on hits.
    const auto cache_key = cache
        ? std::optional(make_cache_key(resources, desc, complete_analysis, layering, *result, names.graph_name))
        : std::nullopt;
    if (cache_key && cache->find(*cache_key, *result))
        return result;

    std::unique_ptr<ILayeringPolicy> policy = has_hair ? create_hair_policy() : create_layering_policy(desc);
    resources.pixel_stage->setFunctionName(names.graph_name);

    // Complex mix may inspect facts that common weighted-layer lowering would
    // erase. Give every non-empty policy that pre-lowering opportunity. A
    // policy that owns a different structural lowering may complete it there;
    // otherwise retain the common lowering exactly once.
    // Empty surfaces deliberately bypass both policy work and lowering.
    if (complete_analysis.graph_facts.has_surface_root) {
        policy->analyze_original_layering(layering);
        if (!layering.weighted_layers_lowered) {
            validate_weighted_layer_top_refs_are_not_shared(layering);
            lower_weighted_layers_once(layering);
        }
        policy->build_representation(std::move(layering), desc);
    } else {
        empty_surface_handler.build_representation(std::move(layering));
    }

    // Storage is chosen only after the representation is complete. The result
    // describes the common graph-emission shape and the one route-specific
    // finalization switch without exposing policy internals to emit_module().
    GraphStoragePlan storage;
    if (complete_analysis.graph_facts.has_surface_root) {
        storage = policy->graph_storage_plan(names);
    } else {
        storage = empty_surface_handler.graph_storage_plan();
    }
    // Code generation: emit one module, then apply the fixed source-finalization waterfall.
    emit_module(resources, desc, complete_analysis, names, storage, *policy, empty_surface_handler, *result);
    finalize_module(*resources.shader_generator, *resources.pixel_stage, names, *result);
    result->module_source = resources.pixel_stage->getSourceCode();
    while (!result->module_source.empty()
           && (result->module_source.back() == '\n' || result->module_source.back() == '\r'
               || result->module_source.back() == ' ' || result->module_source.back() == '\t'))
        result->module_source.pop_back();
    result->module_source.push_back('\n');
    if (cache_key)
        cache->insert(*cache_key, *result);
    return result;
}

} // namespace

std::unique_ptr<CodeGenResult> generate_internal(const CodeGenDesc& desc, CodeGenCache* cache)
{
    // Keep emitted constants consistent with the float values used by static
    // analysis. Scope this over cache keys and emission, restoring caller settings.
    mx::ScopedFloatFormatting formatting(mx::Value::FloatFormatDefault, std::numeric_limits<float>::max_digits10);
    return generate_impl(desc, cache);
}

} // namespace falcor::mtlx::structured
