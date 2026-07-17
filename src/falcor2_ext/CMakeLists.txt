set(FALCOR_PACKAGE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../falcor2)

set(falcor2_ext_sources
    falcor2_ext.cpp
    nbdictionary.h
    core/any.cpp
    core/any.h
    core/asset_resolver.cpp
    core/platform.cpp
    core/properties.cpp
    core/property_type_map.cpp
    core/property_type_map.h
    core/reflected_object.cpp
    core/reflection.cpp
    core/signal.cpp
    core/signal.h
    core/reflection/python_class_reflection.h
    core/reflection/python_class_reflection.cpp
    core/reflection/python_property_descriptor.h
    core/reflection/python_property_descriptor.cpp
    denoisers/ngx.cpp
    denoisers/ngx_features.cpp
    denoisers/ngx_types.cpp
    denoisers/optix_denoiser.cpp
    denoisers/optix_types.cpp
    importers/gltf_importer.cpp
    importers/importer.cpp
    importers/importer_types.cpp
    importers/usd_importer.cpp
    render/mesh.cpp
    render/ray_tracing_setup.cpp
    render/mdl_discovery.cpp
    render/materialx_codegen.cpp
    render/python_scene_object.h
    render/scene.cpp
    render/texture_manager.cpp
    render/transform.cpp
    ui/camera_controller.cpp
    ui/scene_editor.cpp
    ui/scene_interaction_controller.cpp
    ui/scene_picker.cpp
    ui/selection_overlay.cpp
    ui/widgets.cpp
    utils/aabb.cpp
    utils/managed_buffer.cpp
    utils/algorithm/parallel_reduction.cpp
    utils/algorithm/prefix_sum.cpp
    utils/sampling/distribution_1d.cpp
)

if(FALCOR_BUILD_TESTS)
    list(APPEND falcor2_ext_sources
        testing/nanobind_reflector.cpp
    )
endif()

if(FALCOR_ENABLE_PCH)
    target_precompile_headers(falcor2_ext
        PRIVATE
            falcor2_ext_pch.h
    )
endif()

nanobind_add_module(falcor2_ext NB_STATIC LTO
    ${falcor2_ext_sources}
)

target_include_directories(falcor2_ext PRIVATE .)
target_link_libraries(falcor2_ext PRIVATE falcor2)
target_compile_definitions(falcor2_ext PRIVATE FALCOR_BUILD_TESTS=$<BOOL:${FALCOR_BUILD_TESTS}>)

set_target_properties(falcor2_ext PROPERTIES
    OUTPUT_NAME falcor2_ext
    ARCHIVE_OUTPUT_NAME falcor2_ext
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FALCOR_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELEASE ${FALCOR_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_RELEASE ${FALCOR_PACKAGE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${FALCOR_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_DEBUG ${FALCOR_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_DEBUG ${FALCOR_PACKAGE_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${FALCOR_PACKAGE_DIR}
    LIBRARY_OUTPUT_DIRECTORY_RELWITHDEBINFO ${FALCOR_PACKAGE_DIR}
    PDB_OUTPUT_DIRECTORY_RELWITHDEBINFO ${FALCOR_PACKAGE_DIR}
)

# Write a .build_dir file to the package directory.
# This is used by __init__.py to setup the correct DLL path in development mode.
add_custom_target(
    falcor2_ext_build_dir
    COMMAND echo "${FALCOR_OUTPUT_DIRECTORY}" > ${FALCOR_PACKAGE_DIR}/.build_dir
)

add_dependencies(falcor2_ext
    falcor2_ext_build_dir
    falcor2_copy_materialx_1_39
)

# Generate main stub file.
nanobind_add_stub(
    falcor2_stub
    MODULE falcor2
    PYTHON_PATH ${FALCOR_PACKAGE_DIR}/..
    OUTPUT ${FALCOR_PACKAGE_DIR}/__init__.pyi
    DEPENDS falcor2_ext ${FALCOR_PACKAGE_DIR}/__init__.py slangpy_stub
    INCLUDE_PRIVATE # allow us to have functions/variables that start with _
)

# Post-process the main stub file (using SlangPy's postprocess_stub.py).
postprocess_stub(${FALCOR_PACKAGE_DIR}/__init__.pyi  "--submodule")

# Generate submodule stub files.
foreach(submodule IN ITEMS ngx platform reflection ui)
    string(REPLACE "." "/" submodule_path ${submodule})
    nanobind_add_stub(
        falcor2_${submodule}_stub
        MODULE falcor2.${submodule}
        PYTHON_PATH ${FALCOR_PACKAGE_DIR}/..
        OUTPUT ${FALCOR_PACKAGE_DIR}/${submodule_path}/__init__.pyi
        DEPENDS falcor2_ext ${FALCOR_PACKAGE_DIR}/${submodule_path}/__init__.py slangpy_stub
        INCLUDE_PRIVATE # allow us to have functions/variables that start with _
    )

    # Post-process the submodule stub file.
    postprocess_stub(${FALCOR_PACKAGE_DIR}/${submodule_path}/__init__.pyi "--submodule")

endforeach()

# Recursively collect include directories from a target and all its transitive dependencies.
# Results are accumulated in the parent scope variable specified by OUT_VAR.
# VISITED_VAR is used to track already-visited targets and avoid infinite recursion.
function(collect_include_dirs_recursive TARGET OUT_VAR VISITED_VAR)
    # Check if target exists and hasn't been visited
    if(NOT TARGET ${TARGET})
        return()
    endif()
    list(FIND ${VISITED_VAR} ${TARGET} _index)
    if(NOT _index EQUAL -1)
        return()
    endif()

    # Mark as visited
    list(APPEND ${VISITED_VAR} ${TARGET})
    set(${VISITED_VAR} ${${VISITED_VAR}} PARENT_SCOPE)

    # Collect include directories from this target
    foreach(_prop INCLUDE_DIRECTORIES INTERFACE_INCLUDE_DIRECTORIES)
        get_target_property(_dirs ${TARGET} ${_prop})
        if(_dirs)
            foreach(_dir ${_dirs})
                # Unwrap BUILD_INTERFACE generator expression
                string(REGEX REPLACE "\\$<BUILD_INTERFACE:([^>]+)>" "\\1" _dir "${_dir}")
                # Skip remaining generator expressions
                if(_dir MATCHES "^\\$<.*>$")
                    continue()
                endif()
                list(APPEND ${OUT_VAR} ${_dir})
            endforeach()
        endif()
    endforeach()

    # Recurse into linked libraries
    get_target_property(_libs ${TARGET} INTERFACE_LINK_LIBRARIES)
    if(_libs)
        foreach(_lib ${_libs})
            # Skip generator expressions
            if(_lib MATCHES "^\\$<.*>$")
                continue()
            endif()
            # Recurse
            collect_include_dirs_recursive(${_lib} ${OUT_VAR} ${VISITED_VAR})
            # Propagate visited list back up
            set(${VISITED_VAR} ${${VISITED_VAR}} PARENT_SCOPE)
        endforeach()
    endif()

    # Propagate results to parent scope
    set(${OUT_VAR} ${${OUT_VAR}} PARENT_SCOPE)
endfunction()

set(MKDOC_ARGS "")

# Add system include directories.
foreach(value ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
    list(APPEND MKDOC_ARGS -I${value})
endforeach()

# Recursively collect all include directories from falcor2 and its dependencies.
set(_all_include_dirs "")
set(_visited_targets "")
collect_include_dirs_recursive(falcor2 _all_include_dirs _visited_targets)

# Remove duplicates and add to MKDOC_ARGS
list(REMOVE_DUPLICATES _all_include_dirs)
foreach(_dir ${_all_include_dirs})
    list(APPEND MKDOC_ARGS -I${_dir})
endforeach()

list(APPEND MKDOC_ARGS -DNOMINMAX)

file(GLOB_RECURSE FALCOR_HEADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/../falcor2/*.h")
# Exclude fwd.h files, they generate duplicate pydoc entries!
list(FILTER FALCOR_HEADER_FILES EXCLUDE REGEX ".*/fwd\.h")
# Exclude mdl headers, they generate a lot of entries not needed!
list(FILTER FALCOR_HEADER_FILES EXCLUDE REGEX ".*/mdl/.*\.h")

# Note: Older versions of python packages pybind11_mkdoc and clang
# may not support C++20. If you see the following error:
#   ValueError: Unknown template argument kind 604
# Upgrade to:
#   pybind_mkdoc >= 2.6.2
#   clang >= 20.1.5
# Or manually change the cindex.py file in your
# clang python package to add the following lines:
# > # A concept declaration
# > CursorKind.CONCEPT_DECL = CursorKind(604)
add_custom_target(falcor2_pydoc COMMAND
    COMMAND ${Python_EXECUTABLE} -m pybind11_mkdoc -std=c++20 -stdlib=libc++
    ${MKDOC_ARGS}
    -I${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/include
    ${FALCOR_HEADER_FILES}
    -o ${CMAKE_CURRENT_SOURCE_DIR}/py_doc.h
)

install(
    TARGETS falcor2_ext
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
