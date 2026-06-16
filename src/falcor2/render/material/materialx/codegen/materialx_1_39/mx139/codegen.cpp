// SPDX-License-Identifier: Apache-2.0

#include "codegen.h"

#include "codegen_support/opacity_analysis.h"
#include "codegen_support/slang_source_postprocess.h"
#include "codegen_support/shader_input_value.h"
#include "codegen_support/snippet_loader.h"
#include "codegen_support/string_utils.h"
#include "emitter/material_emit.h"
#include "emitter/material_wrapper_helpers.h"
#include "emitter/stack_data_emit.h"
#include "genslangpt/builtin_nodes.h"
#include "genslangpt/microfacet_bsdf.h"
#include "genslangpt/syntax.h"
#include "graph_prepare/graph_prepare.h"
#include "graph_prepare/closure_pruning/closure_pruning_report.h"
#include "layering_analysis.h"
#include "../materialx_test_geomprop_ids.h"
#include "policy.h"
#include "codegen_user_data.h"
#include "prepared_codegen_graph.h"
#include "profile_policy.h"
#include "graph_prepare/closure_pruning/run.h"
#include "graph_prepare/static_input_query/static_input_query.h"
#include "graph_prepare/static_input_query/static_value_analysis.h"
#include "root_prepare.h"

#include "falcor2/core/enum.h"
#include "falcor2/core/error.h"

#include <MaterialXCore/Types.h>
#include <MaterialXCore/Node.h>
#include <MaterialXCore/Unit.h>
#include <MaterialXCore/Value.h>
#include <MaterialXFormat/File.h>
#include <MaterialXGenShader/DefaultColorManagementSystem.h>
#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/GenUserData.h>
#include <MaterialXGenShader/Shader.h>
#include <MaterialXGenShader/ShaderGraph.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <MaterialXGenShader/ShaderStage.h>
#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/UnitSystem.h>
#include <MaterialXGenShader/Util.h>
#include <MaterialXGenHw/HwConstants.h>
#include <MaterialXGenSlang/SlangShaderGenerator.h>
#include <MaterialXGenSlang/SlangSyntax.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <sgl/core/crypto.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {

namespace mx = MaterialX;

namespace {

using emitter::graph_output_can_emit;
using namespace codegen_support;
using namespace genslangpt;

const std::string k_public_uniforms = mx::HW::PUBLIC_UNIFORMS;
const std::string k_private_uniforms = mx::HW::PRIVATE_UNIFORMS;
const std::string k_inputs = "i";
const std::string k_outputs = "o";
const char* k_layering_traversal_order_property = "materialx_layering_traversal_order";
const char* k_required_geometry_streams_property = "required_geometry_streams";

float4 texture_border_color_from_default(const std::string& value, const mx::TypeDesc& output_type)
{
    std::vector<float> values = parse_float_list(value);
    auto component = [&](size_t index)
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

void register_source_path_with_parent(mx::GenContext& context, const std::filesystem::path& path)
{
    if (path.empty())
        return;

    context.registerSourceCodeSearchPath(mx::FilePath(path.string()));
    if (!path.parent_path().empty())
        context.registerSourceCodeSearchPath(mx::FilePath(path.parent_path().string()));
}

void install_unit_system(mx::ShaderGenerator& shader_generator, const mx::DocumentPtr& document)
{
    mx::UnitSystemPtr unit_system = mx::UnitSystem::create(shader_generator.getTarget());
    unit_system->loadLibrary(document);
    unit_system->setUnitConverterRegistry(mx::UnitConverterRegistry::create());

    if (mx::UnitTypeDefPtr distance_type_def = document->getUnitTypeDef("distance"))
        unit_system->getUnitConverterRegistry()->addUnitConverter(
            distance_type_def,
            mx::LinearUnitConverter::create(distance_type_def)
        );

    if (mx::UnitTypeDefPtr angle_type_def = document->getUnitTypeDef("angle"))
        unit_system->getUnitConverterRegistry()->addUnitConverter(
            angle_type_def,
            mx::LinearUnitConverter::create(angle_type_def)
        );

    shader_generator.setUnitSystem(unit_system);
}

void register_mx139_source_paths(mx::GenContext& context)
{
#if defined(MATERIALX_1_39_STD_LIBRARY_PATH)
    register_source_path_with_parent(context, MATERIALX_1_39_STD_LIBRARY_PATH);
#endif
#if defined(MX139_MTLX_LIBRARY_PATH)
    std::filesystem::path mx_library_path = MX139_MTLX_LIBRARY_PATH;
    register_source_path_with_parent(context, mx_library_path);
    register_source_path_with_parent(context, mx_library_path / "pbrlib" / "genslangpt");
    register_source_path_with_parent(context, mx_library_path / "stdlib" / "genslangpt");
#endif
#if defined(FALCOR_PROJECT_DIR)
    register_source_path_with_parent(
        context,
        std::filesystem::path(FALCOR_PROJECT_DIR) / "external" / "MaterialX" / "libraries"
    );
#endif
}

class GenslangPtShaderGenerator : public mx::SlangShaderGenerator {
public:
    static constexpr const char* TARGET = "genslangpt";

    explicit GenslangPtShaderGenerator(mx::TypeSystemPtr type_system)
        : mx::SlangShaderGenerator(type_system ? type_system : mx::TypeSystem::create())
    {
        _syntax = std::make_shared<GenslangPtSyntax>(_typeSystem);
        registerImplementation("IM_texcoord_vector2_" + std::string(TARGET), genslangpt::create_texcoord_node);
        registerImplementation("IM_texcoord_vector3_" + std::string(TARGET), genslangpt::create_texcoord_node);
        registerImplementation("IM_geomcolor_float_" + std::string(TARGET), genslangpt::create_geomcolor_node);
        registerImplementation("IM_geomcolor_color3_" + std::string(TARGET), genslangpt::create_geomcolor_node);
        registerImplementation("IM_geomcolor_color4_" + std::string(TARGET), genslangpt::create_geomcolor_node);
        registerImplementation(
            "IM_geompropvalue_integer_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_geompropvalue_boolean_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation("IM_geompropvalue_float_" + std::string(TARGET), genslangpt::create_geomprop_value_node);
        registerImplementation(
            "IM_geompropvalue_color3_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_geompropvalue_color4_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_geompropvalue_vector2_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_geompropvalue_vector3_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_geompropvalue_vector4_" + std::string(TARGET),
            genslangpt::create_geomprop_value_node
        );
        registerImplementation(
            "IM_dielectric_bsdf_" + std::string(TARGET),
            genslangpt::create_microfacet_dielectric_node
        );
        registerImplementation(
            "IM_conductor_bsdf_" + std::string(TARGET),
            genslangpt::create_microfacet_conductor_node
        );
        registerImplementation(
            "IM_generalized_schlick_bsdf_" + std::string(TARGET),
            genslangpt::create_microfacet_generalized_schlick_node
        );
    }

    static mx::ShaderGeneratorPtr create(mx::TypeSystemPtr type_system = nullptr)
    {
        return std::make_shared<GenslangPtShaderGenerator>(type_system ? type_system : mx::TypeSystem::create());
    }

    const std::string& getTarget() const override
    {
        static const std::string k_target = TARGET;
        return k_target;
    }

    mx::ShaderPtr generate(const std::string& name, mx::ElementPtr element, mx::GenContext& context) const override
    {
        auto user_data = context.getUserData<CodegenUserData>("mx139");
        FALCOR_CHECK(
            user_data && user_data->inputs.result && user_data->inputs.desc,
            "MX139 generator missing user data."
        );
        const ProfileDesc& profile = profile_desc();

        mx::ShaderPtr shader = create_shader(name, element, context);
        mx::ShaderGraph& graph = shader->getGraph();
        mx::ShaderStage& stage = shader->getStage(mx::Stage::PIXEL);
        const LayeringDesc& layering = user_data->analysis.layering;

        std::string shader_name = sanitize_identifier(shader->getName());
        _syntax->makeIdentifier(shader_name, graph.getIdentifierMap());
        shader_name += "_$HASH";
        const std::string module_name = profile.generated_module_prefix + "_" + shader_name + "_Module";
        const std::string graph_name = profile.generated_prefix + "_" + shader_name + "_Graph";
        const std::string material_data_name = profile.generated_prefix + "_" + shader_name + "_Data";
        const std::string material_instance_name = profile.generated_prefix + "_" + shader_name + "_MaterialInstance";
        const std::string material_name = profile.generated_prefix + "_" + shader_name + "_Material";

        setFunctionName(graph_name, stage);

        emitter::emit_module_header(*this, module_name, profile.library_import, stage);

        emitLine("namespace mx139", stage, false);
        emitScopeBegin(stage);
        emitLineBreak(stage);
        if (graph_has_surface_unlit(graph))
            ensure_snippet_emitted(stage, user_data->emit_state, "surface_unlit_transparent.slang");
        emitter::emit_material_data(
            *this,
            [this](const mx::VariableBlock& inputs, mx::ShaderStage& s, bool public_members, bool as_property)
            {
                emit_shader_inputs(inputs, s, public_members, as_property);
            },
            [this](const mx::VariableBlock& inputs)
            {
                return calculate_byte_size(inputs);
            },
            material_data_name,
            layering,
            context,
            stage
        );

        emitTypeDefinitions(context, stage);
        emitLine("struct ClosureData { }", stage, false);
        emit_geomprop_id_constants(*user_data->inputs.result, stage);
        // Advanced modules used to get these constants from the 1.38 umbrella import. Emit the
        // local MX139 snippet whenever generated source references mx_space_* symbols.
        if (graph_uses_space_constants(graph))
            ensure_snippet_emitted(stage, user_data->emit_state, "mx_space_constants.slang");
        if (graph_uses_space_helpers(graph))
            ensure_snippet_emitted(stage, user_data->emit_state, "mx_space_helpers.slang");
        emitLibraryInclude("stdlib/genslang/lib/mx_math.slang", context, stage);
        emitLineBreak(stage);

        emit_graph_struct(graph_name, material_data_name, graph, context, stage);
        emit_material_structs(
            material_name,
            material_instance_name,
            material_data_name,
            graph_name,
            layering,
            user_data->analysis.graph_has_emission,
            *user_data->inputs.desc,
            user_data->emit_state,
            stage
        );
        emitScopeEnd(stage, false);
        emitLineBreak(stage);

        SlangSyntaxFromGlsl(stage);

        std::string code = stage.getSourceCode();
        mark_generated_void_methods_mutating(code);
        replace_glsl_compatibility_tokens(code);
        stage.setSourceCode(code);

        sgl::SHA1 source_code_hash;
        source_code_hash.update(shader->getSourceCode());
        std::string hash = source_code_hash.hex_digest();
        hash.resize(12);

        std::string final_source = shader->getSourceCode();
        replace_hw_geom_tokens(final_source);
        final_source = std::regex_replace(final_source, std::regex(R"(\bclosureData\b)"), "closure_data");
        stage.setSourceCode(final_source);
        replaceTokens(_tokenSubstitutions, stage);
        final_source = stage.getSourceCode();
        replace_all(final_source, "$HASH", hash);
        stage.setSourceCode(final_source);

        CodeGenResult& result = *user_data->inputs.result;
        result.module_source = trim_trailing_whitespace_lines(stage.getSourceCode());
        result.module_name = module_name;
        result.material_name = "mx139::" + material_name;
        result.material_instance_name = "mx139::" + material_instance_name;
        result.material_data_name = "mx139::" + material_data_name;
        replace_all(result.module_name, "$HASH", hash);
        replace_all(result.material_name, "$HASH", hash);
        replace_all(result.material_instance_name, "$HASH", hash);
        replace_all(result.material_data_name, "$HASH", hash);
        result.material_instance_byte_size = 0;
        result.has_entry_point_volume_properties = false;
        result.needs_mx139_lut_scene_globals = true;
        result.codegen_metadata.set("bsdf_profile", std::string("mx139"));
        result.codegen_metadata.set("target", std::string(TARGET));
        result.codegen_metadata.set("graph_emission", true);
        graph_prepare::write_closure_pruning_report_fields(
            result.codegen_metadata,
            user_data->analysis.closure_pruning
        );
        const ClosureRef layering_root = layering.main_layer;
        result.codegen_metadata.set("requested_layering_mode", user_data->inputs.desc->layering_mode);
        result.codegen_metadata.set("compensation_mode", user_data->inputs.desc->compensation_mode);
        result.codegen_metadata.set("layering_root_valid", !layering_root.is_none());
        result.codegen_metadata.set("layering_combiner_count", static_cast<int>(layering.combiners.size()));
        result.codegen_metadata.set("layering_bsdf_count", static_cast<int>(layering.bsdfs.size()));

        return shader;
    }

    void emitFunctionCalls(
        const mx::ShaderGraph& graph,
        mx::GenContext& context,
        mx::ShaderStage& stage,
        uint32_t classification = 0u
    ) const override
    {
        if ((classification & mx::ShaderNode::Classification::CLOSURE) != 0) {
            for (mx::ShaderGraphOutputSocket* output_socket : graph.getOutputSockets()) {
                const mx::ShaderNode* upstream
                    = output_socket->getConnection() ? output_socket->getConnection()->getNode() : nullptr;
                if (upstream && upstream->hasClassification(classification))
                    emitFunctionCall(*upstream, context, stage);
            }
            return;
        }

        mx::ShaderGenerator::emitFunctionCalls(graph, context, stage, classification);
    }

    void emitFunctionBodyBegin(
        const mx::ShaderNode&,
        mx::GenContext&,
        mx::ShaderStage& stage,
        mx::Syntax::Punctuation punc = mx::Syntax::CURLY_BRACKETS
    ) const override
    {
        emitScopeBegin(stage, punc);
    }

private:
    void emit_geomprop_id_constants(const CodeGenResult& result, mx::ShaderStage& stage) const
    {
        std::map<std::string, uint32_t> constants;
        for (const CodeGenResult::GeometryStreamRequest& request : result.required_geomprop_streams) {
            const std::string const_name = geomprop_id_const_name(request.name);
            auto [it, inserted] = constants.emplace(const_name, request.id);
            FALCOR_CHECK(
                inserted || it->second == request.id,
                "MX139 geomprop streams produce conflicting shader constant `{}`.",
                const_name
            );
        }

        for (const auto& [const_name, id] : constants)
            emitLine(fmt::format("static const uint {} = {}u", const_name, id), stage);
        if (!constants.empty())
            emitLineBreak(stage);
    }

    mx::ShaderPtr create_shader(const std::string& name, mx::ElementPtr element, mx::GenContext& context) const
    {
        auto user_data = context.getUserData<CodegenUserData>("mx139");
        CodeGenResult& result = *user_data->inputs.result;

        mx::ShaderGraphPtr graph = mx::ShaderGraph::create(nullptr, name, element, context);
        run_closure_pruning(*graph, context, *user_data);
        mx::ShaderPtr shader = std::make_shared<mx::Shader>(name, graph);

        mx::ShaderStagePtr vertex_stage = createStage(mx::Stage::VERTEX, *shader);
        vertex_stage->createInputBlock(mx::HW::VERTEX_INPUTS, "i_vs");
        vertex_stage->createUniformBlock(mx::HW::PRIVATE_UNIFORMS, "u_prv");
        vertex_stage->createUniformBlock(mx::HW::PUBLIC_UNIFORMS, "u_pub");

        mx::ShaderStagePtr stage = createStage(mx::Stage::PIXEL, *shader);
        stage->createUniformBlock(k_private_uniforms);
        stage->createUniformBlock(k_public_uniforms);
        stage->createInputBlock(k_inputs);
        stage->createOutputBlock(k_outputs);
        addStageConnectorBlock(mx::HW::VERTEX_DATA, mx::HW::T_VERTEX_DATA_INSTANCE, *vertex_stage, *stage);

        createVariables(graph, context, *shader);

        mx::VariableBlock& public_uniforms = stage->getUniformBlock(k_public_uniforms);
        mx::VariableBlock& private_uniforms = stage->getUniformBlock(k_private_uniforms);

        for (mx::ShaderGraphInputSocket* input_socket : graph->getInputSockets()) {
            if (!input_socket->getConnections().empty() && graph->isEditable(*input_socket)) {
                MxParamInfo info = make_param_info(input_socket, user_data);
                if (info.is_editable)
                    public_uniforms.add(input_socket->getSelf());
                else
                    private_uniforms.add(input_socket->getSelf());
                result.all_material_params.add_param_info(std::move(info));
            }
        }

        mx::VariableBlock& outputs = stage->getOutputBlock(k_outputs);
        for (mx::ShaderGraphOutputSocket* output_socket : graph->getOutputSockets())
            outputs.add(output_socket->getSelf());

        std::vector<std::string> traversal_events;
        LayeringDesc layering = LayeringDescBuilder::build(*graph, &traversal_events);
        const OptimizeGraphFlags optimize_graph_flags = effective_optimize_graph_flags(*user_data->inputs.desc);
        const bool needs_static_microfacet_constants = is_set(optimize_graph_flags, OptimizeGraphFlags::fresnel_policy)
            || is_set(optimize_graph_flags, OptimizeGraphFlags::static_scatter_mode);
        const bool needs_static_input_query = is_set(optimize_graph_flags, OptimizeGraphFlags::fresnel_policy);
        StaticValueAnalysisResult static_values;
        std::optional<StaticInputQuery> static_input_query;
        const mx::VariableBlock* microfacet_constants
            = needs_static_microfacet_constants ? &stage->getConstantBlock() : nullptr;
        if (needs_static_input_query) {
            static_values = analyze_static_values_to_fixpoint(*graph);
            static_input_query.emplace(static_values.output_static_values, microfacet_constants);
        }
        classify_layering(
            layering,
            profile_desc(),
            *user_data->inputs.desc,
            microfacet_constants,
            static_input_query ? &*static_input_query : nullptr
        );
        const bool root_opacity_needs_mix
            = graph_output_can_emit(outputs) && !materialx_element_opacity_is_statically_opaque(element);
        if (use_synthetic_opacity_mix(layering, root_opacity_needs_mix))
            append_synthetic_opacity_mix(layering);
        validate_weighted_layer_top_refs_are_not_shared(layering);
        lower_weighted_layers_to_layers_for_emission(layering);
        user_data->analysis.layering = layering;
        user_data->analysis.graph_has_emission = graph_has_emission(*graph);
        result.codegen_metadata.set_list(k_layering_traversal_order_property, std::move(traversal_events));

        std::vector<mx::ShaderGraph*> graph_stack = {graph.get()};
        while (!graph_stack.empty()) {
            mx::ShaderGraph* top_graph = graph_stack.back();
            graph_stack.pop_back();

            for (mx::ShaderNode* node : top_graph->getNodes()) {
                if (node->hasClassification(mx::ShaderNode::Classification::FILETEXTURE)) {
                    for (mx::ShaderInput* input : node->getInputs()) {
                        if (!input->getConnection() && input->getType() == mx::Type::FILENAME) {
                            MxParamInfo info = make_param_info(input, user_data);
                            result.all_material_params.add_param_info(std::move(info));

                            mx::ShaderPort* filename
                                = public_uniforms.add(mx::Type::FILENAME, input->getVariable(), input->getValue());
                            filename->setPath(input->getPath());
                            input->setValue(mx::Value::createValue(input->getVariable()));
                        }
                    }
                }

                if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph())
                    graph_stack.push_back(sub_graph);
            }
        }
        record_required_geometry_streams(*graph, *user_data->inputs.desc, result);

        return shader;
    }

    void
    record_required_geometry_streams(const mx::ShaderGraph& graph, const CodeGenDesc& desc, CodeGenResult& result) const
    {
        std::set<std::string> required_streams;
        std::map<std::string, CodeGenResult::GeometryStreamRequest> required_geomprops;
        std::vector<const mx::ShaderGraph*> graph_stack = {&graph};
        while (!graph_stack.empty()) {
            const mx::ShaderGraph* top_graph = graph_stack.back();
            graph_stack.pop_back();

            for (const mx::ShaderNode* node : top_graph->getNodes()) {
                const std::string& implementation = node->getImplementation().getName();
                if (implementation.starts_with("IM_geomcolor_")) {
                    const std::string index = required_shader_input_value_string(*node, "index", "0");
                    const std::string stream_name = "color_" + index;
                    if (const std::optional<uint32_t> id = resolve_geomprop_id(desc, stream_name, "color4")) {
                        const std::string stream_key = fmt::format("geomprop:{}:color4:{}", stream_name, *id);
                        required_streams.insert(stream_key);
                        required_geomprops.emplace(
                            stream_key,
                            CodeGenResult::GeometryStreamRequest{stream_name, "color4", *id}
                        );
                    }
                } else if (implementation.starts_with("IM_texcoord_")) {
                    const std::string index = required_shader_input_value_string(*node, "index", "0");
                    const std::string output_type = shader_output_type_name(*node);
                    if (index != "0") {
                        const std::string stream_name = "texcoord_" + index;
                        if (const std::optional<uint32_t> id = resolve_geomprop_id(desc, stream_name, output_type)) {
                            const std::string stream_key
                                = fmt::format("geomprop:{}:{}:{}", stream_name, output_type, *id);
                            required_streams.insert(stream_key);
                            required_geomprops.emplace(
                                stream_key,
                                CodeGenResult::GeometryStreamRequest{stream_name, output_type, *id}
                            );
                        }
                    }
                } else if (implementation.starts_with("IM_geompropvalue_")) {
                    const std::string geomprop = required_shader_input_value_string(*node, "geomprop", "");
                    const std::string output_type = shader_output_type_name(*node);
                    if (!geomprop.empty() && !is_builtin_geomprop(geomprop, output_type)) {
                        if (const std::optional<uint32_t> id = resolve_geomprop_id(desc, geomprop, output_type)) {
                            const std::string stream_key = fmt::format("geomprop:{}:{}:{}", geomprop, output_type, *id);
                            required_streams.insert(stream_key);
                            required_geomprops.emplace(
                                stream_key,
                                CodeGenResult::GeometryStreamRequest{geomprop, output_type, *id}
                            );
                        }
                    }
                }

                if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph())
                    graph_stack.push_back(sub_graph);
            }
        }

        result.required_geometry_streams.assign(required_streams.begin(), required_streams.end());
        result.required_geomprop_streams.clear();
        result.required_geomprop_streams.reserve(required_geomprops.size());
        for (const auto& [_, request] : required_geomprops)
            result.required_geomprop_streams.push_back(request);
        result.codegen_metadata.set_list(k_required_geometry_streams_property, result.required_geometry_streams);
    }

    size_t calculate_byte_size(const mx::VariableBlock& inputs) const
    {
        size_t result = 0;
        for (size_t i = 0; i < inputs.size(); ++i) {
            const mx::ShaderPort* input = inputs[i];
            const mx::TypeDesc type_desc = input->getType();
            size_t base_type_size = 0;
            switch (type_desc.getBaseType()) {
            case mx::TypeDesc::BaseType::BASETYPE_INTEGER:
            case mx::TypeDesc::BaseType::BASETYPE_FLOAT:
            case mx::TypeDesc::BaseType::BASETYPE_BOOLEAN:
                base_type_size = 4;
                break;
            case mx::TypeDesc::BaseType::BASETYPE_STRING:
                base_type_size = 4;
                break;
            default:
                FALCOR_THROW("MX139 unsupported material data base type '{}'.", type_desc.getName());
            }

            size_t element_count = type_desc.getSize();
            if (element_count == 3)
                element_count = 4;
            FALCOR_CHECK(element_count > 0, "MX139 unsupported element count for type '{}'.", type_desc.getName());
            result += element_count * base_type_size;
        }
        return result;
    }

    bool graph_has_emission(const mx::ShaderGraph& graph) const
    {
        for (mx::ShaderGraphOutputSocket* output_socket : graph.getOutputSockets()) {
            if (output_socket->getType() == mx::Type::EDF)
                return true;
        }

        for (mx::ShaderNode* node : graph.getNodes()) {
            if (node->getImplementation().getName() == "IM_surface_unlit_genslangpt")
                return true;

            for (mx::ShaderOutput* output : node->getOutputs()) {
                if (output->getType() == mx::Type::EDF)
                    return true;
            }

            if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph()) {
                if (graph_has_emission(*sub_graph))
                    return true;
            }
        }

        return false;
    }

    bool graph_has_surface_unlit(const mx::ShaderGraph& graph) const
    {
        for (mx::ShaderNode* node : graph.getNodes()) {
            if (node->getImplementation().getName() == "IM_surface_unlit_genslangpt")
                return true;

            if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph()) {
                if (graph_has_surface_unlit(*sub_graph))
                    return true;
            }
        }

        return false;
    }

    bool graph_uses_space_helpers(const mx::ShaderGraph& graph) const
    {
        static const std::unordered_set<std::string> k_space_helper_impls = {
            "IM_transformpoint_vector3_genslangpt",
            "IM_transformvector_vector3_genslangpt",
            "IM_transformnormal_vector3_genslangpt",
        };

        for (mx::ShaderNode* node : graph.getNodes()) {
            if (k_space_helper_impls.contains(node->getImplementation().getName()))
                return true;

            if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph()) {
                if (graph_uses_space_helpers(*sub_graph))
                    return true;
            }
        }

        return false;
    }

    bool graph_uses_space_constants(const mx::ShaderGraph& graph) const
    {
        auto is_space_port = [](const mx::ShaderPort* port)
        {
            if (!port || port->getType() != mx::Type::STRING)
                return false;

            const std::string value = port->getValueString();
            return value == "model" || value == "object" || value == "world";
        };

        for (const mx::ShaderGraphInputSocket* input_socket : graph.getInputSockets()) {
            if (is_space_port(input_socket))
                return true;
        }

        for (mx::ShaderNode* node : graph.getNodes()) {
            for (mx::ShaderInput* input : node->getInputs()) {
                if (is_space_port(input))
                    return true;
            }

            if (mx::ShaderGraph* sub_graph = node->getImplementation().getGraph()) {
                if (graph_uses_space_constants(*sub_graph))
                    return true;
            }
        }

        return false;
    }

    void emit_shader_inputs(
        const mx::VariableBlock& inputs,
        mx::ShaderStage& stage,
        bool public_members = false,
        bool as_property = false
    ) const
    {
        static const std::unordered_map<std::string, std::string> k_geomprop_definitions = {
            {"Pobject", "mul(object_from_world, si.position_ws)"},
            {"Pworld", "si.position_ws"},
            {"Nobject", "mul(object_from_world_it, si.shading_frame_ws.normal)"},
            {"Nworld", "si.shading_frame_ws.normal"},
            {"Tobject", "mul(object_from_world, si.shading_frame_ws.tangent)"},
            {"Tworld", "si.shading_frame_ws.tangent"},
            {"Bobject", "mul(object_from_world, si.shading_frame_ws.bitangent)"},
            {"Bworld", "si.shading_frame_ws.bitangent"},
            {"UV0", "si.uv"},
            {"UVMap", "si.uv"},
            {"Vworld", "si.wi_ws"},
        };

        for (size_t i = 0; i < inputs.size(); ++i) {
            const mx::ShaderPort* input = inputs[i];
            std::string type = _syntax->getTypeName(input->getType());
            std::string value = _syntax->getValue(input, true);
            const mx::TypeDesc type_desc = input->getType();

            bool is_geomprop = false;
            const std::string& geomprop = input->getGeomProp();
            if (!geomprop.empty()) {
                auto it = k_geomprop_definitions.find(geomprop);
                if (it != k_geomprop_definitions.end()) {
                    is_geomprop = true;
                    value = it->second;
                }
            }

            if (value.empty())
                value = default_value_for_type(input->getType());
            value = format_value_for_type(type_desc, type, value);

            emitLineBegin(stage);
            if (public_members)
                emitString("public ", stage);
            if (as_property && !is_geomprop)
                emitString("static const ", stage);
            if (as_property && is_geomprop)
                emitString("property ", stage);

            emitString(type + " " + input->getVariable(), stage);
            if (as_property && is_geomprop)
                emitString(" { get { return " + value + "; } }", stage);
            else if (input->getType() != mx::Type::FILENAME)
                emitString(" = " + value, stage);
            emitLineEnd(stage);
        }
    }

    std::string
    format_value_for_type(const mx::TypeDesc& type_desc, const std::string& slang_type, std::string value) const
    {
        value = trim(std::move(value));
        if (value.empty())
            return value;

        const bool is_vector_like = type_desc.getSize() > 1 && type_desc.getBaseType() != mx::TypeDesc::BASETYPE_STRING;
        const bool is_bare_tuple
            = value.find(',') != std::string::npos && value.front() != '{' && value.find('(') == std::string::npos;
        if (is_vector_like && is_bare_tuple)
            return slang_type + "(" + value + ")";

        return value;
    }

    void emit_shader_outputs(const mx::VariableBlock& outputs, mx::ShaderStage& stage) const
    {
        for (size_t i = 0; i < outputs.size(); ++i) {
            const mx::ShaderPort* output = outputs[i];
            const mx::TypeDesc type_desc = output->getType();
            const std::string type_name = _syntax->getTypeName(type_desc);
            std::string value = default_value_for_type(output->getType());
            value = format_value_for_type(type_desc, type_name, value);
            emitLine(type_name + " " + output->getVariable() + " = " + value, stage);
        }
    }


    void emit_graph_struct(
        const std::string& graph_name,
        const std::string& material_data_name,
        const mx::ShaderGraph& graph,
        mx::GenContext& context,
        mx::ShaderStage& stage
    ) const
    {
        emitLine("struct " + graph_name + "<TLodSampler : ILodSampler>", stage, false);
        emitScopeBegin(stage);
        emit_shader_inputs(stage.getInputBlock(k_inputs), stage);
        emit_shader_inputs(stage.getUniformBlock(k_public_uniforms), stage);
        emitLine("SurfaceInteraction si", stage);
        emitLine("uint hints", stage);
        emitLine("TLodSampler lod_sampler", stage);
        emitLine("float4x4 object_from_world", stage);
        emitLine("float4x4 world_from_object", stage);
        emitLine("float3x3 object_from_world_it", stage);
        emitLine("float3x3 world_from_object_it", stage);
        emitLine("ClosureData closure_data", stage);
        emitLine("int mx_stack_next_bsdf = 0", stage);
        emitLine("int mx_stack_next_layer = 0", stage);
        emitLine(profile_desc().stack_data_type + " mx_stack = {}", stage);
        emit_shader_outputs(stage.getOutputBlock(k_outputs), stage);
        emitLineBreak(stage);

        emitLine(
            "__init(" + material_data_name
                + " data_, SurfaceInteraction si_, TLodSampler lod_sampler_, const uint hints_)",
            stage,
            false
        );
        emitFunctionBodyBegin(graph, context, stage);
        emitLine("si = si_", stage);
        emitLine("lod_sampler = lod_sampler_", stage);
        emitLine("hints = hints_", stage);
        if (auto user_data = context.getUserData<CodegenUserData>("mx139");
            user_data && user_data->inputs.desc->flip_v_texcoord)
            emitLine("si.uv = float2(si.uv.x, floor(si.uv.y) + 1.0 - frac(si.uv.y))", stage);
        emitLine(
            "object_from_world = GeomPropProvider::get_space_transform(si, GeomPropSpace::object, "
            "GeomPropSpace::world)",
            stage
        );
        emitLine(
            "world_from_object = GeomPropProvider::get_space_transform(si, GeomPropSpace::world, "
            "GeomPropSpace::object)",
            stage
        );
        emitLine("object_from_world_it = (float3x3)transpose(world_from_object)", stage);
        emitLine("world_from_object_it = (float3x3)transpose(object_from_world)", stage);
        for (size_t i = 0; i < stage.getUniformBlock(k_public_uniforms).size(); ++i) {
            const mx::ShaderPort* input = stage.getUniformBlock(k_public_uniforms)[i];
            emitLine(input->getVariable() + " = data_." + input->getVariable(), stage);
        }
        emitLine("mx_stack.init_vdf_transmission_scales()", stage);
        emitLine("eval_shader_graph()", stage);
        emitFunctionBodyEnd(graph, context, stage);

        emitLine("void eval_shader_graph()", stage, false);
        emitFunctionBodyBegin(graph, context, stage);
        const mx::VariableBlock& constants = stage.getConstantBlock();
        if (!constants.empty()) {
            emitVariableDeclarations(constants, _syntax->getConstantQualifier(), mx::Syntax::SEMICOLON, context, stage);
            emitLineBreak(stage);
        }
        emitFunctionCalls(graph, context, stage, mx::ShaderNode::Classification::TEXTURE);
        for (mx::ShaderGraphOutputSocket* socket : graph.getOutputSockets()) {
            if (!socket->getConnection())
                continue;
            const mx::ShaderNode* upstream = socket->getConnection()->getNode();
            const bool is_material_node = upstream->hasClassification(mx::ShaderNode::Classification::MATERIAL);
            const bool is_concrete_surface_material
                = is_material_node && upstream->getInput(mx::ShaderNode::SURFACESHADER);
            if (upstream->getParent() == &graph
                && (upstream->hasClassification(mx::ShaderNode::Classification::CLOSURE)
                    || upstream->hasClassification(mx::ShaderNode::Classification::SHADER)
                    || (is_material_node && !is_concrete_surface_material))) {
                emitFunctionCall(*upstream, context, stage);
            }
        }
        for (size_t i = 0; i < stage.getOutputBlock(k_outputs).size(); ++i) {
            const mx::ShaderGraphOutputSocket* output_socket = graph.getOutputSocket(i);
            const std::string value = output_socket_uses_empty_material(output_socket)
                ? default_value_for_type(output_socket->getType())
                : getUpstreamResult(output_socket, context);
            emitLine(output_socket->getVariable() + " = " + value, stage);
        }
        emitFunctionBodyEnd(graph, context, stage);

        auto graph_struct_user_data = context.getUserData<CodegenUserData>("mx139");
        FALCOR_CHECK(graph_struct_user_data, "MX139 emit_graph_struct missing user data.");
        ensure_snippet_emitted(stage, graph_struct_user_data->emit_state, "mx_normal_adjustment.slang");
        emitFunctionDefinitions(graph, context, stage);
        emit_shader_inputs(stage.getUniformBlock(k_private_uniforms), stage, false, true);
        emitScopeEnd(stage);
        emitLineBreak(stage);
    }

    void emit_material_structs(
        const std::string& material_name,
        const std::string& material_instance_name,
        const std::string& material_data_name,
        const std::string& graph_name,
        const LayeringDesc& layering,
        bool graph_has_emission,
        const CodeGenDesc& desc,
        codegen_support::SnippetEmitState& emit_state,
        mx::ShaderStage& stage
    ) const
    {
        emitter::emit_material_structs(
            *this,
            [&emit_state](mx::ShaderStage& snippet_stage, std::string_view snippet_name)
            {
                ensure_snippet_emitted(snippet_stage, emit_state, snippet_name);
            },
            emitter::MaterialStructsDesc{
                material_name,
                material_instance_name,
                material_data_name,
                graph_name,
                layering,
                desc,
                graph_has_emission,
                calculate_byte_size(stage.getUniformBlock(k_public_uniforms)),
            },
            stage
        );
    }
    std::string default_value_for_type(const mx::TypeDesc& type) const
    {
        if (type == mx::Type::MATERIAL || type == mx::Type::SURFACESHADER)
            return "{}";
        std::string value = _syntax->getDefaultValue(type, true);
        return value.empty() ? "{}" : value;
    }

    bool output_socket_uses_empty_material(const mx::ShaderGraphOutputSocket* output_socket) const
    {
        if (!output_socket || !output_socket->getConnection())
            return false;

        const mx::ShaderNode* upstream = output_socket->getConnection()->getNode();
        if (!upstream)
            return false;

        const mx::ShaderInput* surface_shader = upstream->getInput(mx::ShaderNode::SURFACESHADER);
        if (!surface_shader)
            return false;

        if (!surface_shader->getConnection())
            return true;

        const mx::ShaderNode* surface_source = surface_shader->getConnectedSibling();
        return !surface_source;
    }

    MxParamInfo make_param_info(mx::ShaderPort* input_socket, const std::shared_ptr<CodegenUserData>& user_data) const
    {
        // make_param_info produces UI-facing metadata for editable material
        // parameters: their display name, type, default value, and any
        // logically-grouped sub-state (e.g. sampler config bundled with a
        // FILENAME socket). It is not a code-generation hot path and does
        // not influence emitted shader source; its output drives editors and
        // tooling only.
        using ParamType = params::Type;

        auto get_authored_input_value
            = [](const mx::DocumentPtr& document, const mx::ShaderInput* input) -> std::optional<std::string>
        {
            if (!document || !input || input->getPath().empty())
                return std::nullopt;

            mx::ElementPtr element = document->getDescendant(input->getPath());
            if (!element)
                return std::nullopt;

            mx::ValueElementPtr value_element = element->asA<mx::ValueElement>();
            if (!value_element)
                return std::nullopt;

            const std::string value = value_element->getResolvedValueString();
            if (value.empty())
                return std::nullopt;

            return value;
        };

        auto get_source_sibling_value = [](const mx::DocumentPtr& document,
                                           const mx::ShaderPort* filename_socket,
                                           const std::string& input_name) -> std::optional<std::string>
        {
            if (!document || !filename_socket || filename_socket->getPath().empty())
                return std::nullopt;

            mx::ElementPtr source_file_input = document->getDescendant(filename_socket->getPath());
            if (!source_file_input)
                return std::nullopt;

            mx::ElementPtr source_node = source_file_input->getParent();
            mx::InterfaceElementPtr source_interface = source_node ? source_node->asA<mx::InterfaceElement>() : nullptr;
            if (!source_interface)
                return std::nullopt;

            mx::InputPtr source_input = source_interface->getInput(input_name);
            if (!source_input || !source_input->hasValueString())
                return std::nullopt;

            return source_input->getValueString();
        };

        auto get_input_value
            = [&get_authored_input_value,
               &get_source_sibling_value,
               &document = user_data->inputs.document,
               input_socket](mx::ShaderNode* node, const std::string& input_name, const std::string& fallback)
        {
            const mx::ShaderInput* input = node ? node->getInput(input_name) : nullptr;
            if (std::optional<std::string> authored_value = get_authored_input_value(document, input))
                return *authored_value;
            if (input) {
                const std::string value = input->getValueString();
                if (!value.empty())
                    return value;
            }
            if (std::optional<std::string> source_value = get_source_sibling_value(document, input_socket, input_name))
                return *source_value;
            return fallback;
        };

        MxParamInfo info;
        const mx::TypeDesc type_desc = input_socket->getType();
        info.param_type = ParamType::unsupported;
        info.mx_type_string = _syntax->getTypeName(type_desc);

        if (type_desc.isFloat2())
            info.param_type = ParamType::float2_;
        else if (type_desc.isFloat3())
            info.param_type = ParamType::float3_;
        else if (type_desc.isFloat4())
            info.param_type = ParamType::float4_;
        else if (type_desc.isScalar()) {
            if (type_desc.getBaseType() == mx::TypeDesc::BASETYPE_BOOLEAN)
                info.param_type = ParamType::bool_;
            else if (type_desc.getBaseType() == mx::TypeDesc::BASETYPE_INTEGER)
                info.param_type = ParamType::int_;
            else if (type_desc.getBaseType() == mx::TypeDesc::BASETYPE_FLOAT)
                info.param_type = ParamType::float_;
            else if (type_desc.getSemantic() == mx::TypeDesc::SEMANTIC_FILENAME)
                info.param_type = ParamType::texture;
        }

        info.param_name = input_socket->getName();
        info.interface.name = info.param_name;
        info.interface.folders = {"Params"};

        if (input_socket->getValue() && info.param_type != ParamType::unsupported) {
            if (input_socket->getValue()->isA<int>())
                info.set_value<int>(input_socket->getValue()->asA<int>());
            else if (input_socket->getValue()->isA<bool>())
                info.set_value<bool>(input_socket->getValue()->asA<bool>());
            else if (input_socket->getValue()->isA<float>())
                info.set_value<float>(input_socket->getValue()->asA<float>());
            else if (input_socket->getValue()->isA<mx::Vector2>())
                info.set_value<falcor::float2>(input_socket->getValue()->asA<mx::Vector2>());
            else if (input_socket->getValue()->isA<mx::Vector3>())
                info.set_value<falcor::float3>(input_socket->getValue()->asA<mx::Vector3>());
            else if (input_socket->getValue()->isA<mx::Vector4>())
                info.set_value<falcor::float4>(input_socket->getValue()->asA<mx::Vector4>());
            else if (input_socket->getValue()->isA<mx::Color3>())
                info.set_value<falcor::float3>(input_socket->getValue()->asA<mx::Color3>());
            else if (input_socket->getValue()->isA<mx::Color4>())
                info.set_value<falcor::float4>(input_socket->getValue()->asA<mx::Color4>());
        }

        if (input_socket->getType() == mx::Type::FILENAME) {
            params::TextureInfo texture_info;
            texture_info.mx_input_name = input_socket->getVariable();
            texture_info.file_name = _syntax->getValue(input_socket, true);
            // Filename uniforms are emitted with the generated variable name,
            // which is unique after nested nodegraphs are flattened.
            info.param_name = input_socket->getVariable();
            mx::ShaderNode* node = input_socket->getNode();
            // TODO(mx139): the four get_input_value calls below are applied
            // uniformly to every FILENAME socket so the UI can group sampler
            // state next to the file path, but only the plain image nodedefs
            // (ND_image_*) actually declare uaddressmode / vaddressmode /
            // filtertype / default as inputs. Hex-tiled image
            // (ND_hextiledimage_*) and any future texture node that bakes
            // sampler behaviour into its shader body do not expose them, so
            // the lambda's fallback path silently swallows the missing port
            // and the UI is shown engine-default sampler state that the
            // shader will not actually honour.
            // Mirror upstream MaterialX GLSL: dispatch on node->getNodeDef()
            // and only surface sampler state for nodedefs that actually
            // declare it; for hex-tiled-style nodes, omit those UI fields
            // rather than synthesising values. Until then, the fallback
            // parameters here are UI policy defaults, not nodedef defaults,
            // and must stay.
            texture_info.uaddressmode = get_input_value(node, "uaddressmode", "periodic");
            texture_info.vaddressmode = get_input_value(node, "vaddressmode", "periodic");
            texture_info.filtertype = get_input_value(node, "filtertype", "linear");
            const mx::TypeDesc output_type
                = node && !node->getOutputs().empty() ? node->getOutputs()[0]->getType() : mx::Type::COLOR4;
            texture_info.border_color
                = texture_border_color_from_default(get_input_value(node, "default", "0.0"), output_type);
            if (user_data->inputs.desc->auto_gamma_image && node)
                texture_info.is_color_output_type
                    = node->getOutputs()[0]->getType().getSemantic() == mx::TypeDesc::SEMANTIC_COLOR;
            info.set_value(std::move(texture_info));
            info.is_editable = true;
        } else {
            info.is_editable = user_data->inputs.desc->make_editable;
        }

        if (!info.holds_value())
            info.is_editable = false;

        return info;
    }
};

struct PreparedCodegenContext {
    PreparedCodegenContext(mx::DocumentPtr doc_, CodeGenDesc desc_)
        : desc(std::move(desc_))
        , doc(std::move(doc_))
    {
        const root_prepare::RootPrepareResult root_prepare_result = root_prepare::prepare_root(doc, desc);
        root_element = root_prepare_result.root_element;
        std::string base_name
            = sanitize_identifier(desc.node_name.empty() ? root_element->getNamePath() : desc.node_name);
        shader_name = base_name;

        result = std::make_unique<CodeGenResult>();
        user_data = std::make_shared<CodegenUserData>();
        user_data->inputs.desc = &desc;
        user_data->inputs.document = doc;
        user_data->inputs.result = result.get();

        shader_generator = GenslangPtShaderGenerator::create();
        context = std::make_unique<mx::GenContext>(shader_generator);
        register_mx139_source_paths(*context);
        shader_generator->setColorManagementSystem(
            mx::DefaultColorManagementSystem::create(GenslangPtShaderGenerator::TARGET)
        );
        shader_generator->getColorManagementSystem()->loadLibrary(doc);
        install_unit_system(*shader_generator, doc);
        shader_generator->registerTypeDefs(doc);
        context->getOptions().targetColorSpaceOverride = desc.target_color_space_override;
        context->getOptions().targetDistanceUnit = "meter";
        context->pushUserData("mx139", user_data);

        graph_prepare::prepare_graph(doc, root_element, desc, GenslangPtShaderGenerator::TARGET);
    }

    CodeGenDesc desc;
    mx::DocumentPtr doc;
    mx::ElementPtr root_element;
    std::string shader_name;
    std::unique_ptr<CodeGenResult> result;
    std::shared_ptr<CodegenUserData> user_data;
    mx::ShaderGeneratorPtr shader_generator;
    std::unique_ptr<mx::GenContext> context;
};

} // namespace

struct PreparedCodegenGraph::Impl {
    Impl(mx::DocumentPtr doc, CodeGenDesc desc)
        : prepared(std::move(doc), std::move(desc))
    {
        graph = mx::ShaderGraph::create(nullptr, prepared.shader_name, prepared.root_element, *prepared.context);
    }

    PreparedCodegenContext prepared;
    mx::ShaderGraphPtr graph;
};

PreparedCodegenGraph::PreparedCodegenGraph(mx::DocumentPtr doc, CodeGenDesc desc)
    : m_impl(std::make_unique<Impl>(std::move(doc), std::move(desc)))
{
}

PreparedCodegenGraph::~PreparedCodegenGraph() = default;

PreparedCodegenGraph::PreparedCodegenGraph(PreparedCodegenGraph&&) noexcept = default;

PreparedCodegenGraph& PreparedCodegenGraph::operator=(PreparedCodegenGraph&&) noexcept = default;

mx::ShaderGraph& PreparedCodegenGraph::graph()
{
    return *m_impl->graph;
}

mx::GenContext& PreparedCodegenGraph::context()
{
    return *m_impl->prepared.context;
}

CodegenUserData& PreparedCodegenGraph::user_data()
{
    return *m_impl->prepared.user_data;
}

CodeGenResult& PreparedCodegenGraph::result()
{
    return *m_impl->prepared.result;
}

const std::string& PreparedCodegenGraph::shader_name() const
{
    return m_impl->prepared.shader_name;
}

std::unique_ptr<CodeGenResult> generate_code(mx::DocumentPtr doc, const CodeGenDesc& desc)
{
    PreparedCodegenContext prepared(std::move(doc), desc);
    prepared.shader_generator->generate(prepared.shader_name, prepared.root_element, *prepared.context);
    return std::move(prepared.result);
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
