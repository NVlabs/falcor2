# Acquire only when the fbx feature is selected. Every archive and static library
# is pinned so local overrides and downloaded SDKs use the same binaries.
if(VCPKG_TARGET_IS_WINDOWS)
    set(sdk_variant vs2022)
    set(sdk_filename fbx2020311_fbxsdk_vs2022_win.exe)
    set(sdk_archive_hash 60ed2d3656073bd7258fde6f33d9ef606f4d8c606f36e590abf6c221ae4ae7cbb40e25f219854dd27a60fb37f8d69668d61406f807dbd46d7fc441d1a4d20e04)
    set(sdk_library_dir lib/x64)
    set(sdk_library_name libfbxsdk-md.lib)
    set(sdk_license_name License.rtf)
    set(sdk_release_hash b879bc9283f65bc7225159c3c25818eed62cc63ddd6cb94d50518283e90b3f2b02af3ec8621a959e00cf5921010115ecf62f703cf866d72c8d35030dd1371c1a)
    set(sdk_debug_hash e0c4a04715065a7480a2b8862b900772a7eb8cafb131db50c593bdd2c31223a2d01acc914a26592a4e524742587f46195560fd7332d4ac8617d30042075bd7ab)
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
