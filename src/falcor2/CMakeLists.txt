
add_library(falcor2 SHARED)


target_sources(falcor2 PRIVATE
    falcor2.cpp
    falcor2.h

    core/asset_resolver.cpp
    core/asset_resolver.h
    core/archive.cpp
    core/archive.h
    core/array_view.h
    core/blob_cache.cpp
    core/blob_cache.h
    core/cursor_writer.h
    core/error.h
    core/format.h
    core/fwd.h
    core/logger.h
    core/macros.h
    core/object.h
    core/pack_file.cpp
    core/pack_file.h
    core/signal.h
    core/platform.cpp
    core/platform.h
    core/platform_linux.cpp
    core/platform_macos.cpp
    core/platform_windows.cpp
    core/properties.cpp
    core/properties.h
    core/python_interpreter.cpp
    core/python_interpreter.h
    core/reflected_object.cpp
    core/reflected_object.h
    core/reflection.h
    core/reflection/class_descriptor.cpp
    core/reflection/class_descriptor.h
    core/reflection/dynamic_properties.h
    core/reflection/fwd.h
    core/reflection/metadata.h
    core/reflection/properties_reflector.h
    core/reflection/property_descriptor.h
    core/reflection/property_range.cpp
    core/reflection/property_range.h
    core/reflection/reflected_object_factory.h
    core/reflection/reflector.h
    core/reflection/registry_reflector.h
    core/reflection/type_registry.cpp
    core/reflection/type_registry.h
    core/types.h
    core/ustring.cpp
    core/ustring.h

    streaming/frame_converter.cpp
    streaming/frame_converter.h
    streaming/video_encoder.cpp
    streaming/video_encoder.h
    streaming/video_format.h
    streaming/webrtc.cpp
    streaming/webrtc.h
    streaming/webrtc_internal.h

    ui/camera_controller.cpp
    ui/camera_controller.h
    ui/editor_command.cpp
    ui/editor_command.h
    ui/icon_data.inc
    ui/icon_library.cpp
    ui/icon_library.h
    ui/property_editor.cpp
    ui/property_editor.h
    ui/scene_editor.cpp
    ui/scene_editor.h
    ui/scene_editor_gizmos.cpp
    ui/scene_editor_gizmos.h
    ui/scene_editor_outliner.cpp
    ui/scene_editor_outliner.h
    ui/scene_interaction_controller.cpp
    ui/scene_interaction_controller.h
    ui/scene_picker.cpp
    ui/scene_picker.h
    ui/selection_overlay.cpp
    ui/selection_overlay.h
    ui/widgets.cpp
    ui/widgets.h

    external/imguizmo.cpp
    external/tiny_gltf.cpp
    external/tiny_gltf.h

    denoisers/dlss_g_feature.cpp
    denoisers/dlss_g_feature.h
    denoisers/dlss_g_feature_impl.h
    denoisers/dlss_g_featureimpl_cuda.cpp
    denoisers/dlss_g_featureimpl_d3d12.cpp
    denoisers/dlss_g_featureimpl_vulkan.cpp
    denoisers/dlss_rr_feature.cpp
    denoisers/dlss_rr_feature.h
    denoisers/dlss_rr_feature_impl.h
    denoisers/dlss_rr_featureimpl_cuda.cpp
    denoisers/dlss_rr_featureimpl_d3d12.cpp
    denoisers/dlss_rr_featureimpl_vulkan.cpp
    denoisers/dlss_sr_feature.cpp
    denoisers/dlss_sr_feature.h
    denoisers/dlss_sr_feature_impl.h
    denoisers/dlss_sr_featureimpl_cuda.cpp
    denoisers/dlss_sr_featureimpl_d3d12.cpp
    denoisers/dlss_sr_featureimpl_vulkan.cpp
    denoisers/ngx.cpp
    denoisers/ngx.h
    denoisers/ngx_impl_cuda.cpp
    denoisers/ngx_impl_d3d12.cpp
    denoisers/ngx_impl_vulkan.cpp
    denoisers/ngx_impl.cpp
    denoisers/ngx_impl.h
    denoisers/ngx_types.h
    denoisers/optix_denoiser.cpp
    denoisers/optix_denoiser.h

    importers/importer_area_light_geometry.cpp
    importers/importer_area_light_geometry.h
    importers/importer_collections.cpp
    importers/importer_collections.h
    importers/importer_edits.cpp
    importers/importer_edits.h
    geometry/tangent_transfer.cpp
    geometry/tangent_transfer.h

    importers/importer_types.cpp
    importers/importer_types.h
    importers/importer_utils.h
    importers/light_import.cpp
    importers/light_import.h
    importers/mesh_import.cpp
    importers/mesh_import.h
    importers/vertex_remap.cpp
    importers/vertex_remap.h
    importers/importer.cpp
    importers/importer.h
    importers/mikkt.cpp
    importers/mikkt.h
    importers/mikkt_internal.h

    importers/curve_tessellator/curve_tessellator.cpp
    importers/curve_tessellator/curve_tessellator.h
    importers/curve_tessellator/curve_tessellator_internal.h
    importers/curve_tessellator/curve_tessellator_ribbons.cpp
    importers/usd_importer/usd_importer_curve_adapter.cpp
    importers/usd_importer/usd_importer_curve_adapter.h
    importers/mesh_tessellator/mesh_tessellator.cpp
    importers/mesh_tessellator/mesh_tessellator.h
    importers/mesh_tessellator/mesh_tessellator_bfr.cpp
    importers/mesh_tessellator/mesh_tessellator_common.cpp
    importers/mesh_tessellator/mesh_tessellator_internal.h
    importers/mesh_tessellator/mesh_tessellator_normals.cpp
    importers/mesh_tessellator/mesh_tessellator_triangulate.cpp

    importers/gltf_importer/gltf_importer.cpp
    importers/gltf_importer/gltf_importer.h
    importers/gltf_importer/gltf_conversions.h
    importers/gltf_importer/gltf_to_gltf.cpp
    importers/gltf_importer/gltf_to_mdl.cpp
    importers/gltf_importer/gltf_conversion_utils.h
    importers/gltf_importer/gltf_to_standard.cpp
    importers/gltf_importer/gltf_to_openpbr.cpp
    importers/gltf_importer/gltf_to_mtlx.cpp

    importers/material_conversions.cpp
    importers/material_conversions.h

    importers/usd_importer/usd_importer_context.cpp
    importers/usd_importer/usd_importer_context.h
    importers/usd_importer/usd_importer_macros.h
    importers/usd_importer/material_network.cpp
    importers/usd_importer/material_network.h
    importers/usd_importer/usd_importer_falcor_material.cpp
    importers/usd_importer/usd_importer_falcor_material.h
    importers/usd_importer/usd_importer_material.cpp
    importers/usd_importer/usd_importer_material.h
    importers/usd_importer/usd_importer_mdl.cpp
    importers/usd_importer/usd_importer_mdl.h
    importers/usd_importer/usd_importer_materialx.cpp
    importers/usd_importer/usd_importer_materialx.h
    importers/usd_importer/usdshade_to_mtlx.h
    importers/usd_importer/usd_importer_mesh_adapter.cpp
    importers/usd_importer/usd_importer_mesh_adapter.h
    importers/usd_importer/usd_importer_skinning.cpp
    importers/usd_importer/usd_importer_skinning_animated.cpp
    importers/usd_importer/usd_importer_skinning.h
    importers/usd_importer/usd_importer_utils.cpp
    importers/usd_importer/usd_importer_utils.h
    importers/usd_importer/usd_importer.cpp
    importers/usd_importer/usd_importer.h

    utils/aabb.h
    utils/color.cpp
    utils/color.h
    utils/buffer_handle.cpp
    utils/buffer_handle.h
    utils/buffer_offset_pair.cpp
    utils/buffer_offset_pair.h
    utils/buffer_suballocator.cpp
    utils/buffer_suballocator.h
    utils/fnv_hash.h
    utils/indexed_vector.h
    utils/managed_vector.h
    utils/offset_allocator.cpp
    utils/offset_allocator.h
    utils/string_utils.h

    utils/algorithm/parallel_reduction.cpp
    utils/algorithm/parallel_reduction.h
    utils/algorithm/prefix_sum.cpp
    utils/algorithm/prefix_sum.h

    utils/io/mesh_writer.cpp
    utils/io/mesh_writer.h

    utils/math/bitfield.h
    utils/math/packing.h
    utils/math/warp.h

    utils/sampling/distribution_1d.cpp
    utils/sampling/distribution_1d.h

    render/component.cpp
    render/component.h
    render/emissive_geometry_system.cpp
    render/emissive_geometry_system.h
    render/light_sampler.cpp
    render/light_sampler.h
    render/reconstruction_filter.cpp
    render/reconstruction_filter.h
    render/emissive_triangle_tree.cpp
    render/emissive_triangle_tree.h
    render/shared_emissive_triangle_tree_types.h
    render/entity.cpp
    render/entity.h
    render/animation.cpp
    render/animation.h
    render/animation_timing.h
    render/geometry.cpp
    render/geometry.h
    render/ray_tracing_policy.h
    render/light_system.cpp
    render/light_system.h
    render/material_aov_registry.cpp
    render/material_aov_registry.h
    render/material_system.cpp
    render/material_system.h
    render/material.cpp
    render/material.h
    render/scene_types.h
    render/ray_tracing_setup.cpp
    render/ray_tracing_setup.h
    render/render_scene.cpp
    render/render_scene.h
    render/scene_capabilities.h
    render/scene_config.h
    render/scene_globals.h
    render/scene_import.cpp
    render/scene_object.cpp
    render/scene_object.h
    render/scene_render_request.h
    render/scene_requirements.h
    render/scene_system.cpp
    render/scene_system.h
    render/scene_update.cpp
    render/scene_update.h
    render/scene.cpp
    render/scene.h
    render/source_asset_ref.h
    render/texture_manager.cpp
    render/texture_manager.h
    render/transform.cpp
    render/transform.h

    render/component/camera.cpp
    render/component/camera.h
    render/component/geometry_instance.cpp
    render/component/geometry_instance.h
    render/component/light.cpp
    render/component/light.h
    render/component/transform_animation.cpp
    render/component/transform_animation.h

    render/geometry/geometry_group.cpp
    render/geometry/geometry_group.h
    render/geometry/geometry_attribute_registry.cpp
    render/geometry/geometry_attribute_registry.h
    render/geometry/geometry_vertex_attribute.cpp
    render/geometry/geometry_vertex_attribute.h
    render/geometry/static_mesh_geometry.cpp
    render/geometry/static_mesh_geometry.h
    render/geometry/static_curve_geometry.cpp
    render/geometry/static_curve_geometry.h

    render/material/mtlx/mtlx_material.cpp
    render/material/mtlx/mtlx_material.h

    render/material/mdl/mdl_context.cpp
    render/material/mdl/mdl_context.h
    render/material/mdl/mdl_discovery.cpp
    render/material/mdl/mdl_discovery.h
    render/material/mdl/mdl_material.cpp
    render/material/mdl/mdl_material.h
    render/material/mdl/mdl_sdk_wrapper.h
    render/material/mdl/mdl_utils.cpp
    render/material/mdl/mdl_utils.h

    render/material/openpbr_material.cpp
    render/material/openpbr_material.h
    render/material/mtlx/dielectric_lut_globals.cpp
    render/material/mtlx/dielectric_lut_globals.h
    render/material/mtlx/lut_globals.cpp
    render/material/mtlx/lut_globals.h
    render/material/emission_only_material.cpp
    render/material/emission_only_material.h
    render/material/material_texture_utils.h
    render/material/gltf_material.cpp
    render/material/gltf_material.h
    render/material/gltf_specgloss_material.cpp
    render/material/gltf_specgloss_material.h
    render/material/gltf_texture_transform.h
    render/material/shared_gltf_texture_selectors.h
    render/material/standard_material.cpp
    render/material/standard_material.h
    render/material/standard_specgloss_material.cpp
    render/material/standard_specgloss_material.h
    render/material/unlit_material.cpp
    render/material/unlit_material.h
)

if(FALCOR_BUILD_TESTS)
    target_sources(falcor2 PRIVATE importers/parallel_mikktspace.cpp)
    target_compile_definitions(falcor2 PRIVATE FALCOR_ENABLE_MIKKT_COMPARISON)
    if(MSVC)
        # The parallel implementation's direct weld uses small checked stream accessors in its inner loops. Allow their
        # explicit inline expansion in Debug while retaining /Od for the surrounding algorithm.
        set_source_files_properties(importers/parallel_mikktspace.cpp PROPERTIES COMPILE_OPTIONS "$<$<CONFIG:Debug>:/Ob1>")
    endif()
endif()

# ImGuizmo is compiled as part of falcor2 but references ImGui symbols from sgl.
# On Windows, mark IMGUI_API as dllimport so the linker resolves them from sgl.dll.
set_source_files_properties(external/imguizmo.cpp PROPERTIES COMPILE_DEFINITIONS
    $<$<PLATFORM_ID:Windows>:IMGUI_API=__declspec\(dllimport\)>
)

if(FALCOR_ENABLE_PCH)
    target_precompile_headers(falcor2
        PRIVATE
            falcor2_pch.h
    )
    # tiny_gltf.cpp defines TINYGLTF_IMPLEMENTATION, skip PCH for it.
    set_source_files_properties(external/tiny_gltf.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
    # imguizmo.cpp has special per-file COMPILE_DEFINITIONS, skip PCH for it.
    set_source_files_properties(external/imguizmo.cpp PROPERTIES SKIP_PRECOMPILE_HEADERS ON)
endif()

add_subdirectory(render/material/mtlx/codegen "${CMAKE_BINARY_DIR}/mtlx")

cmrc_add_resource_library(falcor2_ui_data
    ui/assets/icon_atlas.png
)

target_include_directories(falcor2 PUBLIC ../)
target_include_directories(falcor2 PUBLIC ../../slang)
target_link_libraries(falcor2 PUBLIC sgl)
target_link_libraries(falcor2 PRIVATE falcor2_ui_data)
if(WIN32)
    target_link_libraries(falcor2 PRIVATE Dbghelp)
endif()

if(FALCOR_ENABLE_NGX)
    target_link_libraries(falcor2 PRIVATE falcor2_ngx)
endif()

if(FALCOR_ENABLE_STREAMING)
    target_link_libraries(falcor2 PRIVATE falcor2_openh264 LibDataChannel::LibDataChannel)
endif()

# Link against Python for embedded interpreter
if(FALCOR_BUILD_PYTHON)
    find_package(Python 3.10 COMPONENTS Interpreter Development.Embed REQUIRED)
    target_link_libraries(falcor2 PRIVATE Python::Python)
    # Query the Python prefix so the embedded interpreter can find the standard library.
    execute_process(
        COMMAND ${Python_EXECUTABLE} -c "import sys; print(sys.prefix)"
        OUTPUT_VARIABLE _python_prefix
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    cmake_path(NORMAL_PATH _python_prefix)
    set(FALCOR_PYTHON_HOME "${_python_prefix}" CACHE PATH "Python home for the embedded interpreter")
    if(FALCOR_PYTHON_HOME)
        target_compile_definitions(falcor2 PRIVATE "FALCOR_PYTHON_HOME=\"${FALCOR_PYTHON_HOME}\"")
    endif()
endif()

target_link_libraries(falcor2
    PRIVATE
        git_version_falcor2
        mdl-sdk
        # usd_importer requirements
        falcor2_usd
        falcor2_header_only
        OpenSubdiv::osdCPU_static
        mikktspace::mikktspace)

# Keep MTLX MaterialX libraries and snippets in sync even when only resource files change.
file(GLOB_RECURSE FALCOR_MTLX_RESOURCE_FILES CONFIGURE_DEPENDS
    "${MTLX_LIBRARY}/*.mtlx"
    "${MTLX_LIBRARY}/*.slang"
    "${MTLX_SNIPPET_LIBRARY}/*.slang"
)

set(FALCOR_MTLX_COPY_STAMP "${CMAKE_CURRENT_BINARY_DIR}/mtlx_copy.stamp")
add_custom_command(
    OUTPUT "${FALCOR_MTLX_COPY_STAMP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/mtlx/libraries"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/mtlx/snippets"
    COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY}
        "${MTLX_LIBRARY}"
        "${FALCOR_OUTPUT_DIRECTORY}/mtlx/libraries"
    COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY}
        "${MTLX_SNIPPET_LIBRARY}"
        "${FALCOR_OUTPUT_DIRECTORY}/mtlx/snippets"
    COMMAND ${CMAKE_COMMAND} -E touch "${FALCOR_MTLX_COPY_STAMP}"
    DEPENDS ${FALCOR_MTLX_RESOURCE_FILES}
    COMMENT "Sync MTLX codegen resources"
    VERBATIM
)

add_custom_target(falcor2_sync_mtlx_resources ALL
    DEPENDS "${FALCOR_MTLX_COPY_STAMP}"
)
add_dependencies(falcor2 falcor2_sync_mtlx_resources)

# Keep isolated MaterialX 1.39 runtime libraries in sync even when only
# MaterialX 1.39 dependencies rebuild and the falcor2 target itself does not relink.
set(FALCOR_MTLX_RUNTIME_TARGETS
    MaterialXCore
    MaterialXFormat
    MaterialXGenShader
    MaterialXGenHw
    MaterialXGenSlang
)

set(FALCOR_MTLX_RUNTIME_COPY_COMMANDS)
set(FALCOR_MTLX_RUNTIME_COPY_DEPS)
foreach(target IN LISTS FALCOR_MTLX_RUNTIME_TARGETS)
    if(TARGET ${target})
        list(APPEND FALCOR_MTLX_RUNTIME_COPY_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "$<TARGET_FILE:${target}>" "${FALCOR_OUTPUT_DIRECTORY}"
        )
        list(APPEND FALCOR_MTLX_RUNTIME_COPY_DEPS ${target})
    endif()
endforeach()

add_custom_target(falcor2_copy_mtlx_runtime ALL
    COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}"
    ${FALCOR_MTLX_RUNTIME_COPY_COMMANDS}
    COMMENT "Sync MaterialX runtime libraries"
    VERBATIM
)
add_dependencies(falcor2_copy_mtlx_runtime falcor2 ${FALCOR_MTLX_RUNTIME_COPY_DEPS})

add_custom_target(falcor2_copy_mtlx ALL
    COMMENT "Sync MaterialX runtime and standard libraries"
)
add_dependencies(falcor2_copy_mtlx
    falcor2_copy_mtlx_runtime
    falcor2_copy_mtlx_standard_libraries
)

add_dependencies(falcor2 falcor2_usd_deployment)

# Copy openpbr-bsdf shaders files.
add_custom_command(TARGET falcor2
    POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/external/openpbr-bsdf" "${FALCOR_SHADER_OUTPUT_DIRECTORY}/openpbr-bsdf"
)

set_target_properties(falcor2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
)

target_compile_definitions(falcor2 PRIVATE FALCOR_DLL FALCOR_PROJECT_DIR="${FALCOR_PROJECT_DIR}")

target_compile_definitions(falcor2 PRIVATE
    MI_NEURAYLIB_DEPRECATED_15_0
    MI_NEURAYLIB_DEPRECATED_16_0
)

# Suppress warnings from old TBB 2020_U3 headers.
target_compile_definitions(falcor2 PRIVATE
    TBB_SUPPRESS_DEPRECATED_MESSAGES=1              # TBB's own deprecation warnings
    _SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING  # MSVC C4996: TBB uses deprecated std::iterator base class
)


file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${FALCOR_BUILD_TYPE}/include/falcor2/core/config.h CONTENT
"#pragma once

#define FALCOR_BUILD_TYPE \"${FALCOR_BUILD_TYPE}\"
${FALCOR_OPTIONAL_CONFIG_DEFINITIONS}#define FALCOR_ENABLE_NGX $<BOOL:${FALCOR_ENABLE_NGX}>
#define FALCOR_ENABLE_STREAMING $<BOOL:${FALCOR_ENABLE_STREAMING}>

"
)
target_include_directories(falcor2 PUBLIC ${CMAKE_CURRENT_BINARY_DIR}/${FALCOR_BUILD_TYPE}/include)

add_library(falcor2_mtlx_usd_importer OBJECT
    importers/usd_importer/usdshade_to_mtlx.cpp
)
target_include_directories(falcor2_mtlx_usd_importer
    PRIVATE
        ../
        ../../slang
        ${CMAKE_CURRENT_BINARY_DIR}/${FALCOR_BUILD_TYPE}/include
)
target_compile_definitions(falcor2_mtlx_usd_importer PRIVATE FALCOR_DLL)
target_link_libraries(falcor2_mtlx_usd_importer
    PRIVATE
        sgl
        falcor2_header_only
        MaterialXCore
        MaterialXFormat
)
set_target_properties(falcor2_mtlx_usd_importer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(falcor2 PRIVATE falcor2_mtlx_usd_importer)

install(
    TARGETS falcor2 ${FALCOR_MTLX_RUNTIME_TARGETS}
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/slang/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/slang
)
install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/external/MaterialX/libraries/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mtlx/standard
)
install(
    DIRECTORY "${MTLX_LIBRARY}/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mtlx/libraries
)
install(
    DIRECTORY "${MTLX_SNIPPET_LIBRARY}/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mtlx/snippets
)
install(
    FILES "${CMAKE_SOURCE_DIR}/external/MaterialX/resources/Lights/san_giuseppe_bridge.hdr"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/mtlx/resources/Lights
)
install(
    DIRECTORY "${CMAKE_SOURCE_DIR}/external/openpbr-bsdf/"
    DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/shaders/openpbr-bsdf
    PATTERN ".git" EXCLUDE
)

target_compile_options(falcor2
    PUBLIC
        # MSVC flags.
        $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:
            /MP                             # enable multi-processor compilation
            /Zi                             # generate debug symbols
            # Configure warnings
            /WX                             # warnings as errors
            /W4                             # increase warning level
            /wd4251                         # 'type' : class 'type1' needs to have dll-interface to be used by clients of class 'type2'
            /wd4201                         # nonstandard extension used: nameless struct/union
            /wd4189                         # local variable is initialized but not referenced
            /wd4275                         # non dll-interface class 'type' used as base for dll-interface class 'type2'
            /wd4505                         # unreferenced local function has been removed
            /external:templates-            # Allows warnings from external headers when they occur in a template that's instantiated in your code.
            # Enable warnings that are disabled with /W4
            /we4062                         # enumerator 'identifier' in switch of enum 'enumeration' is not handled
            /we4263                         # 'function': member function does not override any base class virtual member function
            /we4264                         # 'virtual_function': no override available for virtual member function from base 'class'; function is hidden
            /we5038                         # data member 'member1' will be initialized after data member 'member2'
            /we5259                         # 'specialized-type': explicit specialization requires 'template <>'
            /we5263                         # calling 'std::move' on a temporary object prevents copy elision
        >
        # Clang/GCC flags.
        $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:GNU>>:
            -fms-extensions                 # enable MS extensions (among other things allow anonymous structs)
            -fvisibility=hidden             # hide symbols by default
            -Wall                           # set warning level
            -Wextra                         # enable extra warnings
            -Wno-unused-function
            -Wno-unused-variable
            -Wno-unused-but-set-variable
            -Wno-sign-compare
            -Wno-missing-field-initializers
            -Wno-parentheses

            # this is a workaround for warnings in the nanobind binding code
            -Wno-odr
        >
        # Clang flags.
        $<$<CXX_COMPILER_ID:Clang>:
            -Wno-unused-private-field
            -Wno-braced-scalar-init
            -Wno-self-assign-overloaded
            -Wno-deprecated-copy
            -Wno-missing-braces
        >
        # GCC flags.
        $<$<CXX_COMPILER_ID:GNU>:
            -Wno-literal-suffix
            -Wno-class-memaccess
            -Wno-strict-aliasing
            -Wno-maybe-uninitialized
            -Wno-stringop-truncation
            -Wno-deprecated
        >
)
