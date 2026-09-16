# Compile the MaterialX integration in an isolated object target. This keeps
# the required MaterialX include roots from leaking into unrelated falcor2
# translation units while still linking the objects into falcor2.dll.
set(MTLX_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/libraries" PARENT_SCOPE)
set(MTLX_SNIPPET_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/snippets" PARENT_SCOPE)

add_library(falcor2_mtlx_codegen OBJECT
    codegen.cpp
    codegen.h
    codegen_types.h
    mx_parameters.cpp
    mx_parameters.h
    test_geomprop_ids.h
    bsdf_mix_layering.cpp
    bsdf_mix_layering.h
    codegen_support/emission_analysis.cpp
    codegen_support/emission_analysis.h
    codegen_support/opacity_analysis.cpp
    codegen_support/opacity_analysis.h
    codegen_support/slang_source_postprocess.cpp
    codegen_support/slang_source_postprocess.h
    codegen_support/shader_input_value.cpp
    codegen_support/shader_input_value.h
    codegen_support/snippet_loader.cpp
    codegen_support/snippet_loader.h
    codegen_support/string_utils.cpp
    codegen_support/string_utils.h
    emitter/material_emit.cpp
    emitter/material_emit_shared.cpp
    emitter/material_emit.h
    emitter/material_wrapper_helpers.cpp
    emitter/material_wrapper_helpers.h
    emitter/stack_data_emit.cpp
    emitter/stack_data_emit.h
    genslangpt/builtin_nodes.cpp
    genslangpt/builtin_nodes.h
    genslangpt/height_to_normal_node.cpp
    genslangpt/height_to_normal_node.h
    genslangpt/microfacet_bsdf.cpp
    genslangpt/microfacet_bsdf.h
    genslangpt/syntax.cpp
    genslangpt/syntax.h
    graph_prepare/core_reachability.cpp
    graph_prepare/core_reachability.h
    graph_prepare/closure_pruning/closure_pruning_counters.h
    graph_prepare/closure_pruning/closure_pruning_report.h
    graph_prepare/closure_pruning/classify.cpp
    graph_prepare/closure_pruning/classify.h
    graph_prepare/closure_pruning/closure_simplification.cpp
    graph_prepare/closure_pruning/closure_simplification.h
    graph_prepare/closure_pruning/run.cpp
    graph_prepare/closure_pruning/run.h
    graph_prepare/static_input_query/call_site_index.cpp
    graph_prepare/static_input_query/call_site_index.h
    graph_prepare/static_input_query/input_static_values.h
    graph_prepare/static_input_query/operator_match.h
    graph_prepare/static_input_query/static_input_query.cpp
    graph_prepare/static_input_query/static_input_query.h
    graph_prepare/static_input_query/static_value_analysis.h
    graph_prepare/static_input_query/static_values.h
    graph_prepare/default_geomprop_materialization.cpp
    graph_prepare/default_geomprop_materialization.h
    graph_prepare/graph_prepare.cpp
    graph_prepare/graph_prepare.h
    layering_analysis.cpp
    layering_analysis.h
    generate.cpp
    generate.h
    policy.cpp
    policy.h
    prepared_codegen_graph.h
    codegen_user_data.h
    profile_policy.cpp
    profile_policy.h
    root_prepare.cpp
    root_prepare.h
    snippets/module_header_basic.slang
    snippets/mx_space_constants.slang
    snippets/mx_space_helpers.slang
    snippets/mx_normal_adjustment.slang
    snippets/mtlx_snippets.slang
    snippets/mx_direct_microfacet_helpers.slang
    snippets/stack_data_basic.slang
    snippets/surface_unlit_transparent.slang
    snippets/texture_sampling_helper.slang
    snippets/methods/basic/basic_material_wrapper.slang
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

add_subdirectory(emitter/methods/basic)

target_include_directories(falcor2_mtlx_codegen PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/slang
    ${CMAKE_BINARY_DIR}/src/falcor2/${FALCOR_BUILD_TYPE}/include
    ${CMAKE_BINARY_DIR}/external/MaterialX/source
    ${CMAKE_SOURCE_DIR}/external/MaterialX/source
)

target_link_libraries(falcor2_mtlx_codegen PRIVATE
    sgl
    falcor2_header_only
    MaterialXGenSlang
)

target_compile_definitions(falcor2_mtlx_codegen PRIVATE
    FALCOR_DLL
    FALCOR_PROJECT_DIR="${CMAKE_SOURCE_DIR}/"
    MTLX_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/mtlx/libraries"
    MTLX_SNIPPET_PATH="${FALCOR_OUTPUT_DIRECTORY}/mtlx/snippets"
    MTLX_STANDARD_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/mtlx/standard"
)

target_sources(falcor2 PRIVATE
    $<TARGET_OBJECTS:falcor2_mtlx_codegen>
)

target_link_libraries(falcor2
    PRIVATE
        $<LINK_ONLY:MaterialXCore>
        $<LINK_ONLY:MaterialXFormat>
        $<LINK_ONLY:MaterialXGenSlang>
)
