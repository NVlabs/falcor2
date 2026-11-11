# USD plugins do not produce .lib
set(VCPKG_POLICY_DLLS_WITHOUT_LIBS enabled)

# Proper support for a true static usd build is left as a future port improvement.
vcpkg_check_linkage(ONLY_DYNAMIC_LIBRARY)

string(REGEX REPLACE "^([0-9]+)[.]([0-9])\$" "\\1.0\\2" USD_VERSION "${VERSION}")
string(REPLACE "." "_" FALCOR2_USD_NAMESPACE_VERSION "${VERSION}")

if(NOT DEFINED ENV{FALCOR_PYTHON_EXECUTABLE} OR "$ENV{FALCOR_PYTHON_EXECUTABLE}" STREQUAL "")
    message(FATAL_ERROR "The usd port requires FALCOR_PYTHON_EXECUTABLE to select Falcor's Python ABI")
endif()
file(TO_CMAKE_PATH "$ENV{FALCOR_PYTHON_EXECUTABLE}" FALCOR_PYTHON_EXECUTABLE)

if(VCPKG_HOST_IS_WINDOWS AND HOST_TRIPLET MATCHES "^x64-")
    vcpkg_download_distfile(DOXYGEN_ARCHIVE
        URLS "https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0.windows.x64.bin.zip"
        FILENAME "doxygen-1.18.0.windows.x64.bin.zip"
        SHA512 c4f7b45a4ae5f49b9d232036ec200033be7a44c41dfa717f121c05a06a9377838795ef9e931267dcb4d73be2d3bb34c97d79f46f464c8a4d62ef4664210e5491
    )
    vcpkg_extract_source_archive(DOXYGEN_ROOT
        ARCHIVE "${DOXYGEN_ARCHIVE}"
        NO_REMOVE_ONE_LEVEL
    )
    set(DOXYGEN_EXECUTABLE "${DOXYGEN_ROOT}/doxygen.exe")
elseif(VCPKG_HOST_IS_LINUX AND HOST_TRIPLET MATCHES "^x64-")
    vcpkg_download_distfile(DOXYGEN_ARCHIVE
        URLS "https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0.linux.bin.tar.gz"
        FILENAME "doxygen-1.18.0.linux.bin.tar.gz"
        SHA512 fd8a26b73dbb29e18ad1966ea44ce09409e23ea6a91b218dcb65d36fb118fcc385ef5b92be93856a54e2746630a2c4c9a4940d0df940e494f918d2a290ed8f96
    )
    vcpkg_extract_source_archive(DOXYGEN_ROOT
        ARCHIVE "${DOXYGEN_ARCHIVE}"
    )
    set(DOXYGEN_EXECUTABLE "${DOXYGEN_ROOT}/bin/doxygen")
elseif(VCPKG_HOST_IS_OSX AND HOST_TRIPLET MATCHES "^arm64-")
    vcpkg_download_distfile(DOXYGEN_ARCHIVE
        URLS "https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0-mac-arm.zip"
        FILENAME "doxygen-1.18.0-mac-arm.zip"
        SHA512 f56f65f46bb8b8e184677997f63ca6fabcc80da3339e28de7249bac5ce851fd91171447037e1750ef952e5192036fc74593eb57d3c0b39a656624105d63ed530
    )
    vcpkg_extract_source_archive(DOXYGEN_ROOT
        ARCHIVE "${DOXYGEN_ARCHIVE}"
    )
    set(DOXYGEN_EXECUTABLE "${DOXYGEN_ROOT}/doxygen")
elseif(VCPKG_HOST_IS_OSX AND HOST_TRIPLET MATCHES "^x64-")
    vcpkg_download_distfile(DOXYGEN_ARCHIVE
        URLS "https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0-mac-intel.zip"
        FILENAME "doxygen-1.18.0-mac-intel.zip"
        SHA512 70fb5e7c55681193d9a5f27e684e61187c3af6b793dd2527c20fec23548de1540d0f8a9045f62f161c68853879b3a6233a1d372c5938d505cf3edd035b5f77a1
    )
    vcpkg_extract_source_archive(DOXYGEN_ROOT
        ARCHIVE "${DOXYGEN_ARCHIVE}"
    )
    set(DOXYGEN_EXECUTABLE "${DOXYGEN_ROOT}/doxygen")
else()
    find_program(DOXYGEN_EXECUTABLE NAMES doxygen)
endif()

if(DOXYGEN_EXECUTABLE)
    set(USD_DOCUMENTATION_OPTIONS
        "-DDOXYGEN_EXECUTABLE:FILEPATH=${DOXYGEN_EXECUTABLE}"
        -DPXR_BUILD_DOCUMENTATION:BOOL=ON
        -DPXR_BUILD_HTML_DOCUMENTATION:BOOL=OFF
    )
else()
    message(STATUS "Doxygen is unavailable for host triplet ${HOST_TRIPLET}; USD documentation and stubs will not be generated")
    set(USD_DOCUMENTATION_OPTIONS -DPXR_BUILD_DOCUMENTATION:BOOL=OFF)
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO PixarAnimationStudios/OpenUSD
    REF "v${USD_VERSION}"
    SHA512 82f7bb4f77b295be79cd36f591f242276d36c0d589b0fca383344c27e70daf7e29151bf587528bb2fc2b2a8b974387a253993ace3caf5f79ae70f9ebbc71cf1a
    HEAD_REF release
    PATCHES
        003-fix-dep.patch
        004-fix_cmake_package.patch
        007-fix_cmake_hgi_interop.patch
        008-fix_clang8_compiler_error.patch
        009-vcpkg_install_folder_conventions.patch
        010-cmake_export_plugin_as_modules.patch
        014-monolithic-private-dependencies.patch
        015-private-python-runtime-rpath.patch
        016-private-python-package.patch # Keep wrapper links and Python names private to Falcor2.
        017-python-docs-installed-package.patch # Prefer the package being built over any host pxr package.
        018-resolver-asset-include.patch # OpenAsset bindings require a complete ArAsset without precompiled headers.
        019-monolithic-plugin-registry.patch # Extend vcpkg PR #53664 to the monolithic resource layout.
)

# Changes accompanying upstream 003-fix-dep.patch.
file(REMOVE
    "${SOURCE_PATH}/cmake/modules/FindOpenColorIO.cmake"
    "${SOURCE_PATH}/pxr/imaging/hgiVulkan/vk_mem_alloc.cpp"
    "${SOURCE_PATH}/pxr/imaging/hgiVulkan/vk_mem_alloc.h"
)

vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        materialx      PXR_ENABLE_MATERIALX_SUPPORT
        metal          PXR_ENABLE_METAL_SUPPORT
        openimageio    PXR_BUILD_OPENIMAGEIO_PLUGIN
        vulkan         PXR_ENABLE_VULKAN_SUPPORT
)

if (PXR_ENABLE_MATERIALX_SUPPORT)
    list(APPEND FEATURE_OPTIONS "-DMaterialX_DIR=${CURRENT_INSTALLED_DIR}/share/materialx")
endif()

# hgiInterop Metal and Vulkan backend requires garch which is only enabled if PXR_ENABLE_GL_SUPPORT is ON
if(PXR_ENABLE_VULKAN_SUPPORT OR PXR_ENABLE_METAL_SUPPORT)
    list(APPEND FEATURE_OPTIONS "-DPXR_ENABLE_GL_SUPPORT:BOOL=ON")
else()
    list(APPEND FEATURE_OPTIONS "-DPXR_ENABLE_GL_SUPPORT:BOOL=OFF")
endif()

vcpkg_cmake_configure(
    SOURCE_PATH ${SOURCE_PATH}
    OPTIONS ${FEATURE_OPTIONS}
        -DPXR_BUILD_PYTHON_DOCUMENTATION:BOOL=OFF
        -DPXR_BUILD_EXAMPLES:BOOL=OFF
        -DPXR_BUILD_TESTS:BOOL=OFF
        -DPXR_BUILD_TUTORIALS:BOOL=OFF
        -DPXR_BUILD_USD_TOOLS:BOOL=OFF

        -DPXR_BUILD_MONOLITHIC:BOOL=ON
        -DPXR_BUILD_EXEC:BOOL=OFF
        -DPXR_LIB_PREFIX:STRING=falcor2_
        -DPXR_SET_INTERNAL_NAMESPACE:STRING=falcor2UsdInternal_v0_${FALCOR2_USD_NAMESPACE_VERSION}
        -DPXR_OVERRIDE_PLUGINPATH_NAME:STRING=FALCOR2_USD_PLUGIN_PATH

        -DPXR_BUILD_ALEMBIC_PLUGIN:BOOL=OFF
        -DPXR_BUILD_DRACO_PLUGIN:BOOL=OFF
        -DPXR_BUILD_EMBREE_PLUGIN:BOOL=OFF
        -DPXR_BUILD_PRMAN_PLUGIN:BOOL=OFF

        -DPXR_BUILD_IMAGING:BOOL=OFF
        -DPXR_BUILD_USD_IMAGING:BOOL=OFF

        -DPXR_ENABLE_OPENVDB_SUPPORT:BOOL=OFF

        -DPXR_PREFER_SAFETY_OVER_SPEED:BOOL=ON 

        -DPXR_ENABLE_PRECOMPILED_HEADERS:BOOL=OFF

        -DCMAKE_INSTALL_BINDIR:STRING=bin
        -DCMAKE_INSTALL_LIBDIR:STRING=lib
        -DPXR_PYTHON_INSTALL_DIR:STRING=lib/python
        -DPXR_ENABLE_PYTHON_SUPPORT:BOOL=ON
        -DPXR_USE_DEBUG_PYTHON:BOOL=OFF
        -DPython3_EXECUTABLE:FILEPATH=${FALCOR_PYTHON_EXECUTABLE}
    OPTIONS_RELEASE
        ${USD_DOCUMENTATION_OPTIONS}
    OPTIONS_DEBUG
        -DPXR_BUILD_DOCUMENTATION:BOOL=OFF
    MAYBE_UNUSED_VARIABLES
        PXR_USE_PYTHON_3
        PYTHON_EXECUTABLE
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()

# Install OpenUSD's Python package below lib/python/pxr.
# Relocate it to a vcpkg-private staging tree that mirrors the final Python
# package, then rewrite absolute imports to Falcor2's private namespace. Keep
# the native module names and relative imports intact.
function(falcor2_rewrite_usd_python_imports python_root)
    file(GLOB_RECURSE _python_files "${python_root}/*.py")
    foreach(_python_file IN LISTS _python_files)
        file(READ "${_python_file}" _python_contents)
        string(REPLACE "from pxr" "from falcor2.pxr" _python_contents "${_python_contents}")
        string(REPLACE "import pxr" "import falcor2.pxr" _python_contents "${_python_contents}")
        if(_python_contents MATCHES "(^|[\r\n])[ \t]*(from|import)[ \t]+pxr")
            message(FATAL_ERROR "Unisolated pxr import remains after rewriting ${_python_file}")
        endif()
        file(WRITE "${_python_file}" "${_python_contents}")
    endforeach()
endfunction()

foreach(_package_prefix
    "${CURRENT_PACKAGES_DIR}"
    "${CURRENT_PACKAGES_DIR}/debug"
)
    set(_python_source "${_package_prefix}/lib/python/pxr")
    set(_python_root "${_package_prefix}/python/falcor2/pxr")
    if(EXISTS "${_python_source}")
        file(MAKE_DIRECTORY "${_package_prefix}/python/falcor2")
        file(RENAME "${_python_source}" "${_python_root}")
        falcor2_rewrite_usd_python_imports("${_python_root}")
        file(REMOVE_RECURSE "${_package_prefix}/lib/python")
    elseif(NOT VCPKG_BUILD_TYPE OR _package_prefix STREQUAL CURRENT_PACKAGES_DIR)
        message(FATAL_ERROR "Missing OpenUSD Python bindings: ${_python_source}")
    endif()
endforeach()

# Handle debug path for USD plugins
if(NOT DEFINED VCPKG_BUILD_TYPE OR VCPKG_BUILD_TYPE STREQUAL "debug")
    file(GLOB_RECURSE debug_targets
        "${CURRENT_PACKAGES_DIR}/debug/share/pxr/*-debug.cmake"
        )
    foreach(debug_target IN LISTS debug_targets)
        file(READ "${debug_target}" contents)
        string(REPLACE "\${_IMPORT_PREFIX}/usd" "\${_IMPORT_PREFIX}/debug/usd" contents "${contents}")
        string(REPLACE "\${_IMPORT_PREFIX}/plugin" "\${_IMPORT_PREFIX}/debug/plugin" contents "${contents}")
        file(WRITE "${debug_target}" "${contents}")
    endforeach()
endif()

vcpkg_cmake_config_fixup(PACKAGE_NAME "pxr")

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

if(VCPKG_TARGET_IS_WINDOWS)
    # Move all dlls to bin
    file(GLOB RELEASE_DLL ${CURRENT_PACKAGES_DIR}/lib/*.dll)
    file(MAKE_DIRECTORY ${CURRENT_PACKAGES_DIR}/bin)
    if(NOT VCPKG_BUILD_TYPE)
      file(GLOB DEBUG_DLL ${CURRENT_PACKAGES_DIR}/debug/lib/*.dll)
      file(MAKE_DIRECTORY ${CURRENT_PACKAGES_DIR}/debug/bin)
    endif()
    foreach(CURRENT_FROM ${RELEASE_DLL} ${DEBUG_DLL})
        string(REPLACE "/lib/" "/bin/" CURRENT_TO ${CURRENT_FROM})
        file(RENAME ${CURRENT_FROM} ${CURRENT_TO})
    endforeach()

    function(file_replace_regex filename match_string replace_string)
        file(READ ${filename} _contents)
        string(REGEX REPLACE "${match_string}" "${replace_string}" _contents "${_contents}")
        file(WRITE ${filename} "${_contents}")
    endfunction()

    # fix dll path for cmake
    if(NOT VCPKG_BUILD_TYPE)
      file_replace_regex(${CURRENT_PACKAGES_DIR}/share/pxr/pxrTargets-debug.cmake "debug/lib/([a-zA-Z0-9_]+)\\.dll" "debug/bin/\\1.dll")
    endif()
    file_replace_regex(${CURRENT_PACKAGES_DIR}/share/pxr/pxrTargets-release.cmake "lib/([a-zA-Z0-9_]+)\\.dll" "bin/\\1.dll")

    # fix plugInfo.json for runtime
    file(GLOB_RECURSE PLUGINFO_FILES ${CURRENT_PACKAGES_DIR}/lib/usd/*/resources/plugInfo.json)
    file(GLOB_RECURSE PLUGINFO_FILES_DEBUG ${CURRENT_PACKAGES_DIR}/debug/lib/usd/*/resources/plugInfo.json)
    foreach(PLUGINFO ${PLUGINFO_FILES} ${PLUGINFO_FILES_DEBUG})
        file_replace_regex(${PLUGINFO} [=["LibraryPath": "../../([a-zA-Z0-9_]+).dll"]=] [=["LibraryPath": "../../../bin/\1.dll"]=])
    endforeach()
endif()

function(falcor2_generate_usd_python_docs package_prefix dependency_bin)
    set(_python_root "${package_prefix}/python/falcor2/pxr")
    set(_xml_index "${package_prefix}/docs/doxy_xml/index.xml")
    if(NOT EXISTS "${_python_root}" OR NOT EXISTS "${_xml_index}")
        return()
    endif()

    set(_xml_index_for_python "${_xml_index}")
    if(VCPKG_HOST_IS_WINDOWS)
        # OpenUSD's converter opens entity XML beside the index, where names can exceed MAX_PATH.
        file(TO_NATIVE_PATH "${_xml_index}" _xml_index_native)
        string(SUBSTRING "${_xml_index_native}" 0 2 _path_prefix)
        if(_path_prefix STREQUAL "\\\\")
            string(SUBSTRING "${_xml_index_native}" 2 -1 _unc_path)
            string(CONCAT _xml_index_for_python "\\\\?\\UNC\\" "${_unc_path}")
        else()
            string(CONCAT _xml_index_for_python "\\\\?\\" "${_xml_index_native}")
        endif()
    endif()

    file(GLOB _module_init_files "${_python_root}/*/__init__.py")
    set(_module_names)
    foreach(_module_init_file IN LISTS _module_init_files)
        get_filename_component(_module_dir "${_module_init_file}" DIRECTORY)
        get_filename_component(_module_name "${_module_dir}" NAME)
        list(APPEND _module_names "${_module_name}")
    endforeach()
    list(SORT _module_names)
    string(JOIN "," _modules ${_module_names})
    set(_stubgen_module_args)
    foreach(_module_name IN LISTS _module_names)
        list(APPEND _stubgen_module_args --module "${_module_name}")
    endforeach()

    set(_saved_path "$ENV{PATH}")
    set(_saved_ld_library_path "$ENV{LD_LIBRARY_PATH}")
    set(_saved_dyld_library_path "$ENV{DYLD_LIBRARY_PATH}")
    set(_dll_args)
    set(_stubgen_dll_args)
    if(VCPKG_HOST_IS_WINDOWS)
        vcpkg_add_to_path(PREPEND "${package_prefix}/bin")
        vcpkg_add_to_path(PREPEND "${dependency_bin}")
        set(_dll_args --dllPath "${package_prefix}/bin")
        set(_stubgen_dll_args
            --dll-path "${package_prefix}/bin"
            --dll-path "${dependency_bin}"
        )
    elseif(VCPKG_HOST_IS_LINUX)
        set(ENV{LD_LIBRARY_PATH} "${package_prefix}/lib:${CURRENT_INSTALLED_DIR}/lib:$ENV{LD_LIBRARY_PATH}")
    elseif(VCPKG_HOST_IS_OSX)
        set(ENV{DYLD_LIBRARY_PATH} "${package_prefix}/lib:${CURRENT_INSTALLED_DIR}/lib:$ENV{DYLD_LIBRARY_PATH}")
    endif()
    vcpkg_execute_required_process(
        COMMAND
            "${FALCOR_PYTHON_EXECUTABLE}"
            -X utf8 # USD's converter uses the default encoding for Doxygen XML and generated Python.
            "${SOURCE_PATH}/docs/python/convertDoxygen.py"
            --package falcor2.pxr
            --module "${_modules}"
            --inputIndex "${_xml_index_for_python}"
            --pythonPath "${package_prefix}/python"
            ${_dll_args}
            --output "${_python_root}"
        WORKING_DIRECTORY "${SOURCE_PATH}/cmake"
        LOGNAME "generate-python-docs"
    )
    falcor2_rewrite_usd_python_imports("${_python_root}")

    vcpkg_execute_required_process(
        COMMAND
            "${FALCOR_PYTHON_EXECUTABLE}"
            "${CURRENT_PORT_DIR}/stubgen_usd.py"
            --package falcor2.pxr
            ${_stubgen_module_args}
            --output "${_python_root}"
            --python-path "${package_prefix}/python"
            ${_stubgen_dll_args}
        WORKING_DIRECTORY "${SOURCE_PATH}/cmake"
        LOGNAME "generate-python-stubs"
    )
    set(ENV{PATH} "${_saved_path}")
    set(ENV{LD_LIBRARY_PATH} "${_saved_ld_library_path}")
    set(ENV{DYLD_LIBRARY_PATH} "${_saved_dyld_library_path}")

    file(REMOVE_RECURSE "${package_prefix}/docs")
endfunction()

falcor2_generate_usd_python_docs(
    "${CURRENT_PACKAGES_DIR}"
    "${CURRENT_INSTALLED_DIR}/bin"
)

# Python docstrings are configuration-independent. Generate them once from the
# Release bindings and share the results with the Debug package.
set(_release_python_root "${CURRENT_PACKAGES_DIR}/python/falcor2/pxr")
set(_debug_python_root "${CURRENT_PACKAGES_DIR}/debug/python/falcor2/pxr")
if(EXISTS "${_debug_python_root}")
    file(GLOB_RECURSE _generated_python_files
        RELATIVE "${_release_python_root}"
        "${_release_python_root}/*/__DOC.py"
        "${_release_python_root}/*.pyi"
    )
    foreach(_generated_python_file IN LISTS _generated_python_files)
        get_filename_component(_generated_python_dir "${_generated_python_file}" DIRECTORY)
        file(MAKE_DIRECTORY "${_debug_python_root}/${_generated_python_dir}")
        file(COPY_FILE
            "${_release_python_root}/${_generated_python_file}"
            "${_debug_python_root}/${_generated_python_file}"
        )
    endforeach()
endif()

# Handle copyright
vcpkg_install_copyright(FILE_LIST ${SOURCE_PATH}/LICENSE.txt)
