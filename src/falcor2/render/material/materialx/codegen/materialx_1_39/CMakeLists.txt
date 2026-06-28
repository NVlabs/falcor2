# Keep the 1.39 wrapper inside falcor2.dll, but compile it in an isolated
# object target. falcor2 itself inherits the old MaterialX usage requirements,
# and those cause nested <MaterialX...> includes in MaterialX 1.39 headers to resolve to
# the wrong tree. Building this one translation unit separately lets us give it
# only the MaterialX 1.39 include roots while still linking the resulting object into
# falcor2 rather than shipping a separate runtime library.
add_library(falcor2_materialx_1_39_codegen OBJECT
    codegen_wrapper_1_39.cpp
    codegen_wrapper_1_39.h
    materialx_test_geomprop_ids.h
    mx139/bsdf_mix_layering.cpp
    mx139/bsdf_mix_layering.h
    mx139/codegen_support/opacity_analysis.cpp
    mx139/codegen_support/opacity_analysis.h
    mx139/codegen_support/slang_source_postprocess.cpp
    mx139/codegen_support/slang_source_postprocess.h
    mx139/codegen_support/shader_input_value.cpp
    mx139/codegen_support/shader_input_value.h
    mx139/codegen_support/snippet_loader.cpp
    mx139/codegen_support/snippet_loader.h
    mx139/codegen_support/string_utils.cpp
    mx139/codegen_support/string_utils.h
    mx139/emitter/material_emit.cpp
    mx139/emitter/material_emit_shared.cpp
    mx139/emitter/material_emit.h
    mx139/emitter/material_wrapper_helpers.cpp
    mx139/emitter/material_wrapper_helpers.h
    mx139/emitter/stack_data_emit.cpp
    mx139/emitter/stack_data_emit.h
    mx139/genslangpt/builtin_nodes.cpp
    mx139/genslangpt/builtin_nodes.h
    mx139/genslangpt/height_to_normal_node.cpp
    mx139/genslangpt/height_to_normal_node.h
    mx139/genslangpt/microfacet_bsdf.cpp
    mx139/genslangpt/microfacet_bsdf.h
    mx139/genslangpt/syntax.cpp
    mx139/genslangpt/syntax.h
    mx139/graph_prepare/core_reachability.cpp
    mx139/graph_prepare/core_reachability.h
    mx139/graph_prepare/closure_pruning/closure_pruning_counters.h
    mx139/graph_prepare/closure_pruning/closure_pruning_report.h
    mx139/graph_prepare/closure_pruning/classify.cpp
    mx139/graph_prepare/closure_pruning/classify.h
    mx139/graph_prepare/closure_pruning/closure_simplification.cpp
    mx139/graph_prepare/closure_pruning/closure_simplification.h
    mx139/graph_prepare/closure_pruning/run.cpp
    mx139/graph_prepare/closure_pruning/run.h
    mx139/graph_prepare/static_input_query/call_site_index.cpp
    mx139/graph_prepare/static_input_query/call_site_index.h
    mx139/graph_prepare/static_input_query/input_static_values.h
    mx139/graph_prepare/static_input_query/operator_match.h
    mx139/graph_prepare/static_input_query/static_input_query.cpp
    mx139/graph_prepare/static_input_query/static_input_query.h
    mx139/graph_prepare/static_input_query/static_value_analysis.h
    mx139/graph_prepare/static_input_query/static_values.h
    mx139/graph_prepare/default_geomprop_materialization.cpp
    mx139/graph_prepare/default_geomprop_materialization.h
    mx139/graph_prepare/graph_prepare.cpp
    mx139/graph_prepare/graph_prepare.h
    mx139/layering_analysis.cpp
    mx139/layering_analysis.h
    mx139/codegen.cpp
    mx139/codegen.h
    mx139/policy.cpp
    mx139/policy.h
    mx139/prepared_codegen_graph.h
    mx139/codegen_user_data.h
    mx139/profile_policy.cpp
    mx139/profile_policy.h
    mx139/root_prepare.cpp
    mx139/root_prepare.h
    mx139/slang/module_header_basic.slang
    mx139/slang/mx_space_constants.slang
    mx139/slang/mx_space_helpers.slang
    mx139/slang/mx_normal_adjustment.slang
    mx139/slang/mx139_snippets.slang
    mx139/slang/mx_direct_microfacet_helpers.slang
    mx139/slang/stack_data_basic.slang
    mx139/slang/surface_unlit_transparent.slang
    mx139/slang/texture_sampling_helper.slang
    mx139/slang/methods/basic/basic_material_wrapper.slang
    libraries/custom_nodes.mtlx
    libraries/pbrlib/genslangpt/mx_bsdfs.slang
    libraries/pbrlib/genslangpt/mx_ops.slang
    libraries/pbrlib/genslangpt/mx_pbr_helpers.slang
    libraries/pbrlib/genslangpt/mx_surface_unlit.slang
    libraries/pbrlib/genslangpt/mx_vdf.slang
    libraries/pbrlib/genslangpt/pbrlib_genslangpt_impl.mtlx
    libraries/nprlib/genslangpt/nprlib_genslangpt_impl.mtlx
    libraries/stdlib/genslangpt/mx_hextiledimage.slang
    libraries/stdlib/genslangpt/mx_normalmap.slang
    libraries/stdlib/genslangpt/stdlib_genslangpt_impl.mtlx
    libraries/targets/genslangpt.mtlx
)

add_subdirectory(mx139/emitter/methods/basic)

target_include_directories(falcor2_materialx_1_39_codegen PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/slang
    ${CMAKE_BINARY_DIR}/src/falcor2/${FALCOR_BUILD_TYPE}/include
    ${CMAKE_BINARY_DIR}/external/MaterialX/source
    ${CMAKE_SOURCE_DIR}/external/MaterialX/source
)

target_link_libraries(falcor2_materialx_1_39_codegen PRIVATE
    sgl
    falcor2_header_only
    MaterialXGenSlang
)

target_compile_definitions(falcor2_materialx_1_39_codegen PRIVATE
    FALCOR_DLL
    FALCOR_PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
    MX139_MTLX_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/slang/libraries"
    MX139_SNIPPET_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/mx139/slang"
    MATERIALX_1_39_SLANG_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/slang/libraries"
    MATERIALX_1_39_STD_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/std/libraries"
)

target_sources(falcor2 PRIVATE
    $<TARGET_OBJECTS:falcor2_materialx_1_39_codegen>
)
