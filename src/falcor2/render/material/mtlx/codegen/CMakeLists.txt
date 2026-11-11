# Compile the MaterialX integration in an isolated object target. This keeps
# the required MaterialX include roots from leaking into unrelated falcor2
# translation units while still linking the objects into falcor2.dll.
set(MTLX_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/libraries" PARENT_SCOPE)
set(MTLX_SNIPPET_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/snippets" PARENT_SCOPE)

add_library(falcor2_mtlx_codegen OBJECT
    codegen.cpp
    codegen.h
    codegen_cache.cpp
    codegen_cache.h
    codegen_types.h
    mx_parameters.cpp
    mx_parameters.h
    structured/document/core_reachability.cpp
    structured/document/core_reachability.h
    structured/document/loader.cpp
    structured/document/loader.h
    structured/document/transforms.cpp
    structured/document/transforms.h
    structured/document/passes/default_geomprops.cpp
    structured/document/passes/default_geomprops.h
    structured/generator.cpp
    structured/generator.h
    structured/graph/closure_pruning/classify.cpp
    structured/graph/closure_pruning/classify.h
    structured/graph/closure_pruning/closure_pruning_counters.h
    structured/graph/closure_pruning/closure_pruning_report.h
    structured/graph/closure_pruning/closure_simplification.cpp
    structured/graph/closure_pruning/closure_simplification.h
    structured/graph/closure_pruning/run.cpp
    structured/graph/closure_pruning/run.h
    structured/graph/static_input_query/call_site_index.cpp
    structured/graph/static_input_query/call_site_index.h
    structured/graph/static_input_query/input_static_values.h
    structured/graph/static_input_query/static_input_query.cpp
    structured/graph/static_input_query/static_input_query.h
    structured/graph/static_input_query/static_value_analysis.cpp
    structured/graph/static_input_query/static_value_analysis.h
    structured/graph/static_input_query/static_values.h
    structured/layering/basic/material_wrapper.cpp
    structured/layering/basic/material_wrapper.h
    structured/layering/basic/policy.cpp
    structured/layering/hair/policy.cpp
    structured/layering/bsdf_mix/flat_root.cpp
    structured/layering/bsdf_mix/flat_root.h
    structured/layering/empty_surface_handler.cpp
    structured/layering/empty_surface_handler.h
    structured/layering/policy.cpp
    structured/layering/policy.h
    structured/layering/policy_factories.h
    structured/layering/walker.cpp
    structured/layering/walker.h
    structured/material/analysis.cpp
    structured/material/emission_query.cpp
    structured/material/emission_query.h
    structured/material/analysis.h
    structured/material/bsdf_support.cpp
    structured/material/bsdf_support.h
    structured/material/material_aovs.cpp
    structured/material/material_aovs.h
    structured/material/opacity_analysis.cpp
    structured/material/opacity_analysis.h
    structured/material/profile_policy.cpp
    structured/material/profile_policy.h
    structured/pipeline/generation_types.h
    structured/pipeline/request_validation.cpp
    structured/pipeline/request_validation.h
    structured/target/nodes/geometry_nodes.cpp
    structured/target/nodes/geometry_nodes.h
    structured/target/nodes/height_to_normal_node.cpp
    structured/target/nodes/height_to_normal_node.h
    structured/target/nodes/bsdf_emission.cpp
    structured/target/nodes/bsdf_emission.h
    structured/target/nodes/microfacet_bsdf.cpp
    structured/target/nodes/microfacet_bsdf.h
    structured/target/nodes/sheen_bsdf.cpp
    structured/target/nodes/sheen_bsdf.h
    structured/target/nodes/source_code_node.cpp
    structured/target/nodes/source_code_node.h
    structured/target/nodes/formal_node_operation.h
    structured/target/snippet_renderer.cpp
    structured/target/snippet_renderer.h
    structured/target/syntax.cpp
    structured/target/syntax.h
    structured/target/shader_generator.cpp
    structured/target/shader_generator.h
    structured/target/user_data.cpp
    structured/target/user_data.h
    snippets/module_header_basic.slang
    snippets/mx_space_helpers.slang
    snippets/mx_normal_adjustment.slang
    snippets/mtlx_snippets.slang
    snippets/mx_direct_microfacet_helpers.slang
    snippets/stack_data_basic.slang
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

target_compile_definitions(falcor2_mtlx_codegen PRIVATE FALCOR_DLL)

target_sources(falcor2 PRIVATE
    $<TARGET_OBJECTS:falcor2_mtlx_codegen>
)

target_link_libraries(falcor2
    PRIVATE
        $<LINK_ONLY:MaterialXCore>
        $<LINK_ONLY:MaterialXFormat>
        $<LINK_ONLY:MaterialXGenSlang>
)
