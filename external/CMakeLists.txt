set(SGL_BUILD_EXAMPLES OFF)
set(SGL_BUILD_TESTS OFF)
set(SGL_GENERATE_SETPATH_SCRIPTS OFF)

add_subdirectory(slangpy)

# -----------------------------------------------------------------------------
# MDL SDK
# -----------------------------------------------------------------------------

if (NOT DEFINED FALCOR_MDL_URL)
    set(MDL_YEAR "2025")
    set(MDL_VERSION "2025.0.0")
    set(MDL_PATCH "387700.1252")
    if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(MDL_OS "nt")
        set(MDL_EXTENSION "zip")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(MDL_OS "linux")
        set(MDL_EXTENSION "tgz")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(MDL_OS "macosx")
        set(MDL_EXTENSION "tgz")
    endif()
    if(FALCOR_ARCHITECTURE MATCHES "x86_64")
        set(MDL_ARCH "x86-64")
    elseif(FALCOR_ARCHITECTURE MATCHES "aarch64|arm64")
        set(MDL_ARCH "aarch64")
    else()
        message(FATAL_ERROR "Unsupported processor architecture")
    endif()

    set(FALCOR_MDL_URL "https://github.com/NVIDIA/MDL-SDK/releases/download/${MDL_YEAR}/MDL-SDK-${MDL_VERSION}-${MDL_PATCH}-${MDL_OS}-${MDL_ARCH}.${MDL_EXTENSION}")
endif()

FetchPackage(mdl-sdk URL ${FALCOR_MDL_URL})
add_library(mdl-sdk INTERFACE)
target_include_directories(mdl-sdk INTERFACE ${mdl-sdk_SOURCE_DIR}/include)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/bin/dds.dll .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/bin/libmdl_core.dll .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/bin/libmdl_sdk.dll .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/bin/mdl_distiller.dll .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/bin/nv_openimageio.dll .)
else()
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/lib/dds.so .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/lib/libmdl_core.so .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/lib/libmdl_sdk.so .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/lib/mdl_distiller.so .)
    falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/lib/nv_openimageio.so .)
endif()
falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/examples/mdl/nvidia/core_definitions.mdl ./mdl/nvidia)
falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/examples/mdl/nvidia/core_definitions_de.xlf ./mdl/nvidia)
falcor2_copy_binary(${mdl-sdk_SOURCE_DIR}/examples/mdl/nvidia/core_definitions_fr.xlf ./mdl/nvidia)

# -----------------------------------------------------------------------------
# NVIDIA NGX / DLSS SDK
# -----------------------------------------------------------------------------

set(FALCOR_DLSS_DLL_VERSION "310.6.0" CACHE STRING "NVIDIA DLSS DLL version")
set(FALCOR_DLSS_LOCAL_SDK_DIR "" CACHE PATH "Optional local NVIDIA DLSS SDK path. Leave empty to fetch from GitHub releases.")

if(FALCOR_ENABLE_NGX)
    if(FALCOR_DLSS_LOCAL_SDK_DIR)
        set(FALCOR_DLSS_SDK_DIR "${FALCOR_DLSS_LOCAL_SDK_DIR}")
    else()
        set(_falcor_dlss_url "https://github.com/NVIDIA/DLSS/archive/refs/tags/v${FALCOR_DLSS_DLL_VERSION}.zip")
        message(STATUS "NGX enabled; fetching DLSS SDK if needed from ${_falcor_dlss_url}")
        FetchPackage(falcor_dlss_sdk URL ${_falcor_dlss_url})
        set(FALCOR_DLSS_SDK_DIR "${falcor_dlss_sdk_SOURCE_DIR}")
    endif()

    if(FALCOR_WINDOWS)
        if(NOT FALCOR_ARCHITECTURE MATCHES "x86_64")
            message(FATAL_ERROR "NGX is only wired for Windows x86_64 SDK binaries in this integration.")
        endif()
        set(FALCOR_DLSS_NGX_LIBRARY_DEBUG "${FALCOR_DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d_dbg.lib")
        set(FALCOR_DLSS_NGX_LIBRARY_RELEASE "${FALCOR_DLSS_SDK_DIR}/lib/Windows_x86_64/x64/nvsdk_ngx_d.lib")
        set(FALCOR_DLSS_RUNTIME_BINARIES
            "${FALCOR_DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlss.dll"
            "${FALCOR_DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlssd.dll"
            "${FALCOR_DLSS_SDK_DIR}/lib/Windows_x86_64/rel/nvngx_dlssg.dll"
        )
    elseif(FALCOR_LINUX)
        if(NOT FALCOR_ARCHITECTURE MATCHES "x86_64")
            message(FATAL_ERROR "NGX is only wired for Linux x86_64 SDK binaries in this integration.")
        endif()
        set(FALCOR_DLSS_NGX_LIBRARY_DEBUG "${FALCOR_DLSS_SDK_DIR}/lib/Linux_x86_64/libnvsdk_ngx.a")
        set(FALCOR_DLSS_NGX_LIBRARY_RELEASE "${FALCOR_DLSS_NGX_LIBRARY_DEBUG}")
        set(FALCOR_DLSS_RUNTIME_BINARIES
            "${FALCOR_DLSS_SDK_DIR}/lib/Linux_x86_64/rel/libnvidia-ngx-dlss.so.${FALCOR_DLSS_DLL_VERSION}"
            "${FALCOR_DLSS_SDK_DIR}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssd.so.${FALCOR_DLSS_DLL_VERSION}"
            "${FALCOR_DLSS_SDK_DIR}/lib/Linux_x86_64/rel/libnvidia-ngx-dlssg.so.${FALCOR_DLSS_DLL_VERSION}"
        )
    else()
        message(FATAL_ERROR "NGX is not supported on ${FALCOR_PLATFORM}.")
    endif()

    foreach(FALCOR_DLSS_RUNTIME_BINARY IN LISTS FALCOR_DLSS_RUNTIME_BINARIES)
        falcor2_copy_binary("${FALCOR_DLSS_RUNTIME_BINARY}" .)
        install(FILES "${FALCOR_DLSS_RUNTIME_BINARY}" DESTINATION ${CMAKE_INSTALL_BINDIR})
    endforeach()
    message(STATUS "NGX enabled using DLSS SDK: ${FALCOR_DLSS_SDK_DIR}")

    add_library(falcor2_ngx INTERFACE)
    target_include_directories(falcor2_ngx SYSTEM INTERFACE "${FALCOR_DLSS_SDK_DIR}/include")
    target_link_libraries(falcor2_ngx INTERFACE
        debug "${FALCOR_DLSS_NGX_LIBRARY_DEBUG}"
        optimized "${FALCOR_DLSS_NGX_LIBRARY_RELEASE}"
    )
endif()

# -----------------------------------------------------------------------------
# MaterialX 1.39
# -----------------------------------------------------------------------------

# MaterialX headers use C++ exception handlers. Keep this scoped to the
# MaterialX tree instead of relying on global CMAKE_CXX_FLAGS defaults.
if(MSVC AND NOT CMAKE_CXX_FLAGS MATCHES "(^| )/EH")
    add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/EHsc>)
endif()

# MaterialX 1.39
message(STATUS "Configure MaterialX 1.39")
cmake_policy(PUSH)
set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
set(MATERIALX_BUILD_PYTHON OFF) # off by default in MaterialX as well
set(MATERIALX_BUILD_SHARED_LIBS ON)
set(MATERIALX_BUILD_GEN_GLSL OFF)
set(MATERIALX_BUILD_GEN_OSL OFF)
set(MATERIALX_BUILD_GEN_MDL OFF)
set(MATERIALX_BUILD_GEN_MSL OFF)
set(MATERIALX_BUILD_RENDER OFF)
set(MATERIALX_BUILD_RENDER_PLATFORMS OFF)
set(MATERIALX_BUILD_GEN_SLANG ON)
set(MATERIALX_BUILD_OIIO OFF)
set(MATERIALX_BUILD_TESTS OFF)
set(MATERIALX_TEST_RENDER OFF)
set(MATERIALX_WARNINGS_AS_ERRORS ON)
mark_as_advanced(MATERIALX_BUILD_PYTHON OFF) # off by default in MaterialX as well
mark_as_advanced(MATERIALX_BUILD_SHARED_LIBS ON)
mark_as_advanced(MATERIALX_BUILD_GEN_GLSL OFF)
mark_as_advanced(MATERIALX_BUILD_GEN_OSL OFF)
mark_as_advanced(MATERIALX_BUILD_GEN_MDL OFF)
mark_as_advanced(MATERIALX_BUILD_GEN_MSL OFF)
mark_as_advanced(MATERIALX_BUILD_RENDER OFF)
mark_as_advanced(MATERIALX_BUILD_RENDER_PLATFORMS OFF)
mark_as_advanced(MATERIALX_BUILD_GEN_SLANG ON)
mark_as_advanced(MATERIALX_BUILD_OIIO OFF)
mark_as_advanced(MATERIALX_BUILD_TESTS OFF)
mark_as_advanced(MATERIALX_TEST_RENDER OFF)
mark_as_advanced(MATERIALX_WARNINGS_AS_ERRORS ON)

foreach(var IN ITEMS
	MATERIALX_LIBNAME_SUFFIX
	MATERIALX_NAMESPACE_SUFFIX
	MATERIALX_TARGET_PREFIX
	MATERIALX_TARGET_NAMESPACE
	MATERIALX_EXPORT_SET_NAME
	MATERIALX_PACKAGE_NAME
)
	unset(${var})
	unset(${var} CACHE)
endforeach()

add_subdirectory(MaterialX EXCLUDE_FROM_ALL)

add_custom_target(falcor2_copy_materialx_1_39_std_libraries ALL
	COMMAND ${CMAKE_COMMAND} -E make_directory ${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/std/libraries
	COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY}
		${CMAKE_CURRENT_SOURCE_DIR}/MaterialX/libraries
		${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/std/libraries
	VERBATIM
)

cmake_policy(POP)

# Override MaterialX output directories so shared libraries are placed directly
# in the falcor2 output directory, avoiding blind copy_directory operations.
# NOTE: Update this list if MaterialX adds new shared library targets.
foreach(_mtlx_target MaterialXCore MaterialXFormat MaterialXGenShader)
    if(TARGET ${_mtlx_target})
        set_target_properties(${_mtlx_target} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}
            LIBRARY_OUTPUT_DIRECTORY ${FALCOR_LIBRARY_OUTPUT_DIRECTORY}
        )
    endif()
endforeach()

# -----------------------------------------------------------------------------
# USD plugins
# Copy USD plugins as described here: https://github.com/microsoft/vcpkg/issues/37947
# -----------------------------------------------------------------------------

# Validate USD plugin paths before copying.
set(_pxr_usd_lib_dir "${pxr_DIR}/../../lib/usd")
get_filename_component(_pxr_usd_lib_dir "${_pxr_usd_lib_dir}" ABSOLUTE)
if(NOT EXISTS "${_pxr_usd_lib_dir}")
    message(FATAL_ERROR "USD plugin lib directory not found: ${_pxr_usd_lib_dir}\n"
        "The pxr package layout may have changed. pxr_DIR is: ${pxr_DIR}")
endif()

# These targets always run but cost is mitigated by copy_directory_if_different
# on CMake >= 3.26.
if(FALCOR_LINUX OR FALCOR_MACOS)
    add_custom_target(falcor2_copy_usd_plugins ALL
        COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY} "${_pxr_usd_lib_dir}" "${FALCOR_OUTPUT_DIRECTORY}/usd"
        VERBATIM
    )
else()
    set(_pxr_usd_bin_dir "${pxr_DIR}/../../bin/usd")
    get_filename_component(_pxr_usd_bin_dir "${_pxr_usd_bin_dir}" ABSOLUTE)
    if(NOT EXISTS "${_pxr_usd_bin_dir}")
        message(FATAL_ERROR "USD plugin bin directory not found: ${_pxr_usd_bin_dir}\n"
            "The pxr package layout may have changed. pxr_DIR is: ${pxr_DIR}")
    endif()
    add_custom_target(falcor2_copy_usd_plugins ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/usd/"
        COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY} "${_pxr_usd_lib_dir}" "${FALCOR_OUTPUT_DIRECTORY}/usd"
        COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY} "${_pxr_usd_bin_dir}" "${FALCOR_OUTPUT_DIRECTORY}/usd"
        VERBATIM
    )
endif()

# -----------------------------------------------------------------------------
# OpenPBR shaders
# -----------------------------------------------------------------------------

file(GLOB_RECURSE _openpbr_files CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/external/openpbr-bsdf/*")
set(_openpbr_stamp "${CMAKE_CURRENT_BINARY_DIR}/openpbr_copy.stamp")
add_custom_command(
    OUTPUT "${_openpbr_stamp}"
    COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY} "${CMAKE_SOURCE_DIR}/external/openpbr-bsdf" "${FALCOR_SHADER_OUTPUT_DIRECTORY}/openpbr-bsdf"
    COMMAND ${CMAKE_COMMAND} -E touch "${_openpbr_stamp}"
    DEPENDS ${_openpbr_files}
    COMMENT "Sync OpenPBR shaders"
    VERBATIM
)
add_custom_target(falcor2_copy_openpbr ALL
    DEPENDS "${_openpbr_stamp}"
)

# header only
add_library(falcor2_header_only INTERFACE)
target_include_directories(falcor2_header_only INTERFACE include)
