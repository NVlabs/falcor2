cmake_minimum_required(VERSION 3.24)
project(usdFileFormats LANGUAGES C CXX)

option(USD_FILEFORMATS_ENABLE_FBX "Build the FBX file-format plugin" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(GNUInstallDirs)
include(CMakeDependentOption)
include(cmake/compiler_config.cmake)
include(cmake/register_plugins.cmake)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pxr CONFIG REQUIRED)
find_package(OpenImageIO CONFIG REQUIRED)
find_package(ZLIB REQUIRED)
if(NOT TARGET usd_m)
    message(FATAL_ERROR "This overlay requires Falcor2's monolithic OpenUSD package")
endif()

if(USD_FILEFORMATS_ENABLE_FBX)
    find_package(LibXml2 REQUIRED)
    set(FBXSDK_DEPENDENCIES ZLIB::ZLIB LibXml2::LibXml2)
    if(UNIX)
        find_package(Threads REQUIRED)
        list(APPEND FBXSDK_DEPENDENCIES Threads::Threads ${CMAKE_DL_LIBS})
    endif()
    if(APPLE)
        find_package(Iconv REQUIRED)
        find_library(COREFOUNDATION_LIBRARY CoreFoundation REQUIRED)
        list(APPEND FBXSDK_DEPENDENCIES Iconv::Iconv "${COREFOUNDATION_LIBRARY}")
    endif()
    # The upstream finder returns immediately when this target already exists.
    add_library(fbxsdk::fbxsdk STATIC IMPORTED)
    set_target_properties(fbxsdk::fbxsdk PROPERTIES
        IMPORTED_CONFIGURATIONS "Debug;Release"
        IMPORTED_LOCATION_DEBUG "${FBXSDK_LIBRARY_DEBUG}"
        IMPORTED_LOCATION_RELEASE "${FBXSDK_LIBRARY_RELEASE}"
        MAP_IMPORTED_CONFIG_RELWITHDEBINFO Release
        MAP_IMPORTED_CONFIG_MINSIZEREL Release
        INTERFACE_INCLUDE_DIRECTORIES "${FBXSDK_ROOT}/include"
        INTERFACE_LINK_LIBRARIES "${FBXSDK_DEPENDENCIES}"
    )
endif()
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(USD_FILEFORMATS_BUILD_TESTS OFF)
set(USD_FILEFORMATS_ENABLE_INSTALL OFF)
set(USD_FILEFORMATS_ENABLE_PLY OFF)
set(USD_FILEFORMATS_ENABLE_SPZ OFF)
set(USD_BOOST_VERSION 0)
set(VERSION "2026.07")
configure_file(version.h.in version.h)
add_subdirectory(utils)
set(PLUGIN_TARGETS fileformatUtils)
if(USD_FILEFORMATS_ENABLE_FBX)
    add_subdirectory(fbx)
    list(APPEND PLUGIN_TARGETS usdFbx)
endif()

foreach(target IN LISTS PLUGIN_TARGETS)
    # The USD component interface targets do not link the private monolith.
    target_link_libraries(${target} PRIVATE usd_m TBB::tbb Python3::Python)
    set_target_properties(${target} PROPERTIES
        OUTPUT_NAME "falcor2_adobe_${target}"
        RESOURCE ""
    )
    if(APPLE)
        # The toolchain selects the architecture; upstream's -m64 is x86-specific.
        get_target_property(compile_options ${target} COMPILE_OPTIONS)
        list(REMOVE_ITEM compile_options -m64)
        set_target_properties(${target} PROPERTIES
            COMPILE_OPTIONS "${compile_options}"
            INSTALL_NAME_DIR "@rpath"
            INSTALL_RPATH "@loader_path"
        )
    elseif(UNIX)
        set_target_properties(${target} PROPERTIES INSTALL_RPATH "$ORIGIN")
    endif()
    install(TARGETS ${target}
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endforeach()

if(USD_FILEFORMATS_ENABLE_FBX)
    if(WIN32)
        set(plugin_library_dir bin)
    else()
        set(plugin_library_dir lib)
    endif()
    set(PLUG_INFO_LIBRARY_PATH "../../../../${plugin_library_dir}/${CMAKE_SHARED_LIBRARY_PREFIX}falcor2_adobe_usdFbx${CMAKE_SHARED_LIBRARY_SUFFIX}")
    configure_file(fbx/src/plugInfo.json.in falcor2-plugInfo.json @ONLY)
    install(FILES "${CMAKE_CURRENT_BINARY_DIR}/falcor2-plugInfo.json"
        DESTINATION share/usd-fileformat-plugins/plugin/usdFbx/resources
        RENAME plugInfo.json
    )
    install(FILES fbx/tests/SanityCube.fbx DESTINATION share/usd-fileformat-plugins/tests)
endif()
install(DIRECTORY utils/include/fileformatutils DESTINATION include/usd-fileformat-plugins)
