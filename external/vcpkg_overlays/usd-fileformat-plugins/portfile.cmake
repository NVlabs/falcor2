# USD loads file-format plugins as shared libraries, even on a static triplet.
set(VCPKG_LIBRARY_LINKAGE dynamic)
# Plugin metadata contains paths to configuration-matching DLLs, so Debug needs
# its own registry rather than sharing Release's LibraryPath entries.
set(VCPKG_POLICY_ALLOW_DEBUG_SHARE enabled)

set(FBXSDK_OPTIONS)
if("fbx" IN_LIST FEATURES)
    include("${CURRENT_PORT_DIR}/fbxsdk.cmake")
    list(APPEND FBXSDK_OPTIONS
        "-DFBXSDK_ROOT:PATH=${FBXSDK_ROOT}"
        "-DFBXSDK_LIBRARY_DEBUG:FILEPATH=${FBXSDK_LIBRARY_DEBUG}"
        "-DFBXSDK_LIBRARY_RELEASE:FILEPATH=${FBXSDK_LIBRARY_RELEASE}"
    )
endif()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO skallweitNV/USD-Fileformat-plugins
    REF 4997b5f3b6f0e4797a91c9a971becbae9704c54f
    SHA512 6407d4582d614dd3a87fb5d871bd89026ca06329321f62c5dae7b976b6a2912ae81fbb7bb411bb8662a390ec4ee8e79e2f097d67cca59ca7fbaad43a13404a0f
)

# Use a small feature-selected build instead of the upstream standalone dependency
# finders/fetchers. It still consumes the upstream source lists.
configure_file("${CURRENT_PORT_DIR}/CMakeLists.txt" "${SOURCE_PATH}/CMakeLists.txt" COPYONLY)
vcpkg_check_features(OUT_FEATURE_OPTIONS FEATURE_OPTIONS
    FEATURES
        fbx USD_FILEFORMATS_ENABLE_FBX
)
vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${FEATURE_OPTIONS}
        ${FBXSDK_OPTIONS}
        "-DPython3_EXECUTABLE:FILEPATH=$ENV{FALCOR_PYTHON_EXECUTABLE}"
)
vcpkg_cmake_install()
vcpkg_copy_pdbs()

# Debug has its own registry because its DLLs must match Debug USD and the CRT.
file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include")
set(COPYRIGHT_FILES "${SOURCE_PATH}/LICENSE-2.0.txt")
if("fbx" IN_LIST FEATURES)
    list(APPEND COPYRIGHT_FILES "${FBXSDK_LICENSE}")
endif()
vcpkg_install_copyright(FILE_LIST ${COPYRIGHT_FILES})
file(INSTALL "${CURRENT_PORT_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}")
