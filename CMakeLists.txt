cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0077 NEW)

# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
    cmake_policy(SET CMP0135 NEW)
endif()

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/external/slangpy/cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/external/slangpy/external/slang-rhi/cmake)
include(utils)

set(VCPKG_INSTALL_OPTIONS "--no-print-usage" "--allow-unsupported" "--x-buildtrees-root=${CMAKE_CURRENT_SOURCE_DIR}/build/buildtrees")

project(falcor2)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CTEST_OUTPUT_ON_FAILURE TRUE)
enable_testing()

# Support use of falcor2 cache directory in fully offline environment.
if("$ENV{FALCOR2_FETCHCONTENT_FULLY_DISCONNECTED}" STREQUAL "1")
    set(FETCHCONTENT_BASE_DIR "/opt/falcor2/cache/cmake-deps" CACHE PATH "Base directory for FetchContent dependencies")
    set(FETCHCONTENT_FULLY_DISCONNECTED ON CACHE BOOL "Use already-populated FetchContent dependencies")
endif()

include(DetermineTargetArchitecture)
include(GNUInstallDirs)
include(CMakeRC)
include(FetchContent)
include(FetchPackage)

set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")

if (APPLE)
set(FALCOR_ORIGIN "@loader_path")
elseif(UNIX)
  set(FALCOR_ORIGIN "$ORIGIN")
endif()

# TODO we should probably set install RPATH on individual targets instead
set(CMAKE_INSTALL_RPATH "${FALCOR_ORIGIN};${FALCOR_ORIGIN}/../lib")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

option(FALCOR_BUILD_PYTHON "Build Python extension" ON)
#option(FALCOR_BUILD_EXAMPLES "Build example applications" ON)
option(FALCOR_BUILD_TESTS "Build tests" ON)
#option(FALCOR_BUILD_DOC "Build documentation" OFF)

# Enable/disable precompiled headers.
option(FALCOR_ENABLE_PCH "Enable precompiled headers" OFF)

# Enable/disable Crashpad integration through SlangPy/SGL.
option(FALCOR_ENABLE_CRASHPAD "Enable Crashpad integration" OFF)
set(SGL_ENABLE_CRASHPAD ${FALCOR_ENABLE_CRASHPAD})

set(FALCOR_PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE STRING "Project directory (defaults to CMAKE_CURRENT_SOURCE_DIR)")

set(FALCOR_BINARY_DIR ${CMAKE_BINARY_DIR} CACHE STRING "Binary directory (defaults to CMAKE_BINARY_DIR)")

# -----------------------------------------------------------------------------
# Check platform
# -----------------------------------------------------------------------------

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(FALCOR_PLATFORM "Windows")
    set(FALCOR_WINDOWS TRUE)
    set(FALCOR_HAS_D3D12 ON)
    set(FALCOR_HAS_VULKAN ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    set(FALCOR_PLATFORM "Linux")
    set(FALCOR_LINUX TRUE)
    set(FALCOR_HAS_D3D12 OFF)
    set(FALCOR_HAS_VULKAN ON)
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(FALCOR_PLATFORM "macOS")
    set(FALCOR_MACOS TRUE)
    set(FALCOR_HAS_D3D12 OFF)
    set(FALCOR_HAS_VULKAN ON)
else()
    message(FATAL_ERROR "Unsupported platform!")
endif()
message(STATUS "Platform: ${FALCOR_PLATFORM}")

# Determine the target architecture we build for.
# CMAKE_SYSTEM_PROCESSOR is not a reliable way to determine the target architecture.
determine_target_architecture(FALCOR_ARCHITECTURE)
message(STATUS "Architecture: ${FALCOR_ARCHITECTURE}")

if((FALCOR_WINDOWS OR FALCOR_LINUX) AND FALCOR_ARCHITECTURE MATCHES "x86_64")
    set(FALCOR_ENABLE_NGX_DEFAULT ON)
else()
    set(FALCOR_ENABLE_NGX_DEFAULT OFF)
endif()
option(FALCOR_ENABLE_NGX "Enable NVIDIA NGX support for DLSS" ${FALCOR_ENABLE_NGX_DEFAULT})

# -----------------------------------------------------------------------------
# Python
# -----------------------------------------------------------------------------

if(FALCOR_BUILD_PYTHON)
    find_package(Python 3.9 COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# -----------------------------------------------------------------------------
# Global setup
# -----------------------------------------------------------------------------

# Require builds to be outside of source tree.
file(TO_CMAKE_PATH "${PROJECT_BINARY_DIR}/CMakeLists.txt" LOC_PATH)
if(EXISTS "${LOC_PATH}")
    message(FATAL_ERROR "You cannot build in a source directory (or any directory with a CMakeLists.txt file). Please use a build directory instead.")
endif()

# Enable folders (for Visual Studio).
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Use copy_directory_if_different when available (CMake >= 3.26) for better
# incremental build performance.
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.26")
    set(FALCOR_COPY_DIRECTORY copy_directory_if_different)
else()
    set(FALCOR_COPY_DIRECTORY copy_directory)
endif()

# Setup available build configurations.
if(NOT SETUP_CONFIGURATION_TYPES)
    set(SETUP_CONFIGURATION_TYPES 1)

    if(CMAKE_CONFIGURATION_TYPES)
        # multi config generator
        set(CMAKE_CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo" CACHE STRING "" FORCE)
    else()
        # single config generator
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY HELPSTRING "Choose the type of build")
        set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release;RelWithDebInfo")
    endif()
endif()

set(FALCOR_RUNTIME_OUTPUT_DIRECTORY ${FALCOR_BINARY_DIR})
set(FALCOR_LIBRARY_OUTPUT_DIRECTORY ${FALCOR_BINARY_DIR})

if(CMAKE_CONFIGURATION_TYPES)
    set(FALCOR_BUILD_TYPE $<CONFIG>)
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>)
else()
    set(FALCOR_BUILD_TYPE ${CMAKE_BUILD_TYPE})
    set(FALCOR_OUTPUT_DIRECTORY ${FALCOR_RUNTIME_OUTPUT_DIRECTORY})
endif()

set(FALCOR_SHADER_OUTPUT_DIRECTORY ${FALCOR_OUTPUT_DIRECTORY}/shaders)

# -----------------------------------------------------------------------------
# Version
# -----------------------------------------------------------------------------

# Embed git information.
include(git_version_2)
git_version_setup_2(falcor2)

# -----------------------------------------------------------------------------
# Binary files
# -----------------------------------------------------------------------------

set_property(GLOBAL PROPERTY FALCOR_BINARY_FILES_SRC "")
set_property(GLOBAL PROPERTY FALCOR_BINARY_FILES_DST "")

function(falcor2_copy_binary src dst_dir)
   get_filename_component(filename ${src} NAME)
   set(dst ${FALCOR_OUTPUT_DIRECTORY}/${dst_dir}/${filename})
   set_property(GLOBAL APPEND PROPERTY FALCOR_BINARY_FILES_SRC ${src})
   set_property(GLOBAL APPEND PROPERTY FALCOR_BINARY_FILES_DST ${dst})
   ## Complains about:
   # [cmake] CMake Warning (dev) at CMakeLists.txt:186 (install):
   # [cmake]   Policy CMP0177 is not set: install() DESTINATION paths are normalized.  Run
   # [cmake]   "cmake --help-policy CMP0177" for policy details.  Use the cmake_policy
   # [cmake]   command to set the policy and suppress this warning.
   #install(FILES ${src} DESTINATION ${CMAKE_INSTALL_BINDIR}/${dst_dir})
endfunction()

function(falcor2_copy_library target)
   get_property(src TARGET ${target} PROPERTY IMPORTED_LOCATION)
   falcor2_copy_binary(${src} .)
endfunction()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

# Require for Usd importer
find_package(pxr REQUIRED)
# Required to add tangents to meshes
find_package(mikktspace REQUIRED)
# Required to tessellate subdivs
find_package(OpenSubdiv REQUIRED)

# -----------------------------------------------------------------------------
# Sources
# -----------------------------------------------------------------------------

add_subdirectory(external)
add_subdirectory(src)

if(FALCOR_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# -----------------------------------------------------------------------------
# copy binary files
# -----------------------------------------------------------------------------

get_property(binary_files_src GLOBAL PROPERTY FALCOR_BINARY_FILES_SRC)
get_property(binary_files_dst GLOBAL PROPERTY FALCOR_BINARY_FILES_DST)
foreach(file IN ZIP_LISTS binary_files_src binary_files_dst)
   get_filename_component(filename ${file_0} NAME)
   add_custom_command(
       OUTPUT ${file_1}
       DEPENDS ${file_0}
       COMMAND ${CMAKE_COMMAND} -E copy ${file_0} ${file_1}
       COMMENT "Copying binary file ${filename} to ${file_1}"
   )
endforeach()

add_custom_target(falcor2_copy_binary_files ALL DEPENDS ${binary_files_dst})
add_dependencies(falcor2 falcor2_copy_binary_files)

# -----------------------------------------------------------------------------
# misc
# -----------------------------------------------------------------------------

# Generate a merged natvis file.
include(merge_natvis)
set(natvis_files
    "${CMAKE_CURRENT_SOURCE_DIR}/src/falcor2/falcor2.natvis"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/slangpy/src/sgl/sgl.natvis"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/slangpy/external/slang-rhi/src/core.natvis"
)
merge_natvis_files("${natvis_files}" "all.natvis")

# -----------------------------------------------------------------------------
# install
# -----------------------------------------------------------------------------

# Install binaries.
install(
    TARGETS sgl
    LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install header files.
install(
    DIRECTORY src/sgl
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h"
)
