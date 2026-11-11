# Acquire only when the fbx feature is selected. Every archive and static library
# is pinned so local overrides and downloaded SDKs use the same binaries.
if(VCPKG_TARGET_IS_WINDOWS)
    # The VS2019 SDK binaries also link with VS2022 17.9. Autodesk's VS2022
    # binaries require newer STL helpers absent from that toolset.
    set(sdk_variant vs2019)
    set(sdk_filename fbx2020311_fbxsdk_vs2019_win.exe)
    set(sdk_archive_hash 7366de2210c5a3b29cc89a34f23f040b5b65926da0e8775045176171a4bcf99843fa81fa9eab9804d3e355550b8e90d878cd4a338a211e3e11b81e099a790e4e)
    set(sdk_library_dir lib/x64)
    set(sdk_library_name libfbxsdk-md.lib)
    set(sdk_license_name License.rtf)
    set(sdk_release_hash da2acd2994ecfd122652c9397935476f8722f21c6da48fff5fddb80433c67c2f6af5121698e30b22e852f77b0d3ce75b92fdfae0ca2ab8f302daf94d8e825555)
    set(sdk_debug_hash 2494078d4dc36dd26c8bdc3dd662c9c9018a068f0d9b779f460918382ec6be9e9417070a2fce9769e4f0001bced62f92ef666f23ac0a04ace31fc858da631b9c)
elseif(VCPKG_TARGET_IS_LINUX)
    set(sdk_variant gcc)
    set(sdk_filename fbx2020311_fbxsdk_gcc_linux.tar.gz)
    set(sdk_archive_hash 42de32a80c1b4354c73088dd1b08e13dc5bf559eb51ca04cfb2a00333069c83636692abf385611e1ae73aedf6ac70b3059ac3e1f3ad48d80bd618b37041ff6da)
    set(sdk_library_dir lib)
    set(sdk_library_name libfbxsdk.a)
    set(sdk_license_name License.txt)
    set(sdk_release_hash a7cf0621468e0a37431b292c9bf2d8fdbd73ed980e97cd12b73f6a5f42c538d1449872901d5117e144542ac40ad2b6a6664ae828d6fa0f4b8e522694e1d3420b)
    set(sdk_debug_hash a10e42b8a0ff6ff240bd7af2a8124fa9373f2ac34b931c7ba78a3b9bf78ecd1ab83d796c7b60a5830cd13e6e8cec48c35401b7ebbec68cd3e533c89135297d20)
elseif(VCPKG_TARGET_IS_OSX)
    set(sdk_variant clang)
    set(sdk_filename fbx2020311_fbxsdk_clang_mac.pkg.tgz)
    set(sdk_archive_hash 37cf2f63cb50dbc9540df7925a657034a4807f9738453c3218f8b612173a9c24ab9b1c7d3e423cf044ea64dadead75cad27a713a7ed47b8a07c651b95c77a99f)
    set(sdk_library_dir lib/clang)
    set(sdk_library_name libfbxsdk.a)
    set(sdk_license_name License.rtf)
    set(sdk_release_hash d78bf122cb3936b792e4f828520e280d1b779a9dd9554909099a1400033f974cc7e59ef9a73a58d44372d674e54931867a81868760ef93d49e28fba54af24929)
    set(sdk_debug_hash ae4bda1834179615de546a9acb9a5f407b2ab09aa7859e241feb7c29c05609e68a0d3f578d3af661bbca990b241aaac58b7e1851aa51b25e2bdf3c010ebe3a5d)
else()
    message(FATAL_ERROR "No pinned FBX SDK for this platform")
endif()

if(DEFINED ENV{FBXSDK_ROOT} AND NOT "$ENV{FBXSDK_ROOT}" STREQUAL "")
    file(TO_CMAKE_PATH "$ENV{FBXSDK_ROOT}" FBXSDK_ROOT)
else()
    vcpkg_download_distfile(sdk_archive
        URLS "https://damassets.autodesk.net/content/dam/autodesk/www/files/${sdk_filename}"
        FILENAME "${sdk_filename}"
        SHA512 "${sdk_archive_hash}"
    )
    set(sdk_staging "${CURRENT_BUILDTREES_DIR}/fbxsdk-2020.3.11-${sdk_variant}")
    file(REMOVE_RECURSE "${sdk_staging}")
    # EXE extraction uses 7-Zip. None of these operations execute the vendor
    # installer or write to system locations.
    vcpkg_extract_archive(ARCHIVE "${sdk_archive}" DESTINATION "${sdk_staging}")
    if(VCPKG_TARGET_IS_WINDOWS)
        set(FBXSDK_ROOT "${sdk_staging}")
    elseif(VCPKG_TARGET_IS_LINUX)
        vcpkg_execute_required_process(
            COMMAND "$ENV{FALCOR_PYTHON_EXECUTABLE}" "${CURRENT_PORT_DIR}/unwrap_linux_sdk.py"
                "${sdk_staging}/fbx2020311_fbxsdk_linux" "${sdk_staging}/sdk.tar.gz"
            WORKING_DIRECTORY "${sdk_staging}"
            LOGNAME unwrap-fbxsdk
        )
        set(FBXSDK_ROOT "${sdk_staging}/sdk")
        vcpkg_extract_archive(ARCHIVE "${sdk_staging}/sdk.tar.gz" DESTINATION "${FBXSDK_ROOT}")
    else()
        # macOS's system tar supports XAR and CPIO; CMake's bundled libarchive
        # does not consistently include XAR support.
        find_program(FBXSDK_TAR NAMES tar HINTS /usr/bin REQUIRED)
        file(MAKE_DIRECTORY "${sdk_staging}/package" "${sdk_staging}/payload")
        vcpkg_execute_required_process(
            COMMAND "${FBXSDK_TAR}" -xf "${sdk_staging}/fbx2020311_fbxsdk_clang_macos.pkg"
                -C "${sdk_staging}/package"
            WORKING_DIRECTORY "${sdk_staging}"
            LOGNAME extract-fbxsdk-package
        )
        vcpkg_execute_required_process(
            COMMAND "${FBXSDK_TAR}" -xf "${sdk_staging}/package/Root.pkg/Payload"
                -C "${sdk_staging}/payload"
            WORKING_DIRECTORY "${sdk_staging}"
            LOGNAME extract-fbxsdk-payload
        )
        set(FBXSDK_ROOT "${sdk_staging}/payload/Applications/Autodesk/FBX SDK/2020.3.11")
    endif()
endif()

foreach(config IN ITEMS release debug)
    set(library "${FBXSDK_ROOT}/${sdk_library_dir}/${config}/${sdk_library_name}")
    if(NOT EXISTS "${library}" OR NOT EXISTS "${FBXSDK_ROOT}/include/fbxsdk.h")
        message(FATAL_ERROR "Missing FBX SDK 2020.3.11 ${sdk_variant} files at ${FBXSDK_ROOT}. Unset FBXSDK_ROOT to download the pinned SDK.")
    endif()
    file(SHA512 "${library}" actual_hash)
    if(NOT actual_hash STREQUAL sdk_${config}_hash)
        message(FATAL_ERROR "Unexpected FBX SDK 2020.3.11 ${sdk_variant} ${config} library hash: ${actual_hash}")
    endif()
    string(TOUPPER "${config}" upper_config)
    set(FBXSDK_LIBRARY_${upper_config} "${library}")
endforeach()
set(FBXSDK_LICENSE "${FBXSDK_ROOT}/${sdk_license_name}")
if(NOT EXISTS "${FBXSDK_LICENSE}")
    message(FATAL_ERROR "Missing Autodesk FBX SDK license at ${FBXSDK_LICENSE}")
endif()
