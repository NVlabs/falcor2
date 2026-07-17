# -----------------------------------------------------------------------------
# falcor2 unit tests
# -----------------------------------------------------------------------------

add_executable(falcor2_tests)
target_sources(falcor2_tests PRIVATE
    native/falcor2_tests.cpp
    native/testing.cpp

    native/core/test_archive.cpp
    native/core/test_asset_resolver.cpp
    native/core/test_macros.cpp
    native/core/test_blob_cache.cpp
    native/core/test_pack_file.cpp
    native/core/test_platform.cpp
    native/core/test_properties.cpp
    native/core/test_python_interpreter.cpp
    native/core/test_signal.cpp
    native/core/reflection/test_dynamic_properties.cpp
    native/core/reflection/test_metadata.cpp
    native/core/reflection/test_reflection.cpp
    native/core/reflection/test_type_registry.cpp
    native/denoisers/test_denoiser.cpp
    native/denoisers/test_ngx.cpp
    native/device/test_device.cpp
    native/gltf_importer/test_gltf_importer.cpp
    native/gltf_importer/test_sampler_info.cpp
    native/importers/test_importer_animation.cpp
    native/importers/test_importer_edit_stream.cpp
    native/importers/test_mikkt_cache.cpp
    native/render/test_scene.cpp
    native/render/test_animation.cpp
    native/render/test_materials.cpp
    native/render/test_scene_object.cpp
    native/render/test_static_curve_geometry.cpp
    native/ui/test_property_layout.cpp
    native/usd_importer/test_usd_importer.cpp
    native/utils/test_offset_allocator.cpp
)

target_include_directories(falcor2_tests PRIVATE ./native)
target_link_libraries(falcor2_tests PRIVATE falcor2 falcor2_header_only usd)
add_dependencies(falcor2_tests
    falcor2_ext
    falcor2_copy_materialx_1_39
)
target_compile_definitions(falcor2_tests
    PRIVATE
        SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
        FALCOR_PROJECT_DIR="${FALCOR_PROJECT_DIR}"
)

set_target_properties(falcor2_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
    LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
)

add_test(NAME falcor2_tests COMMAND $<TARGET_FILE:falcor2_tests>)
