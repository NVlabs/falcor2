# USD's Python extension modules must share one privately named scheduler.
set(VCPKG_LIBRARY_LINKAGE dynamic)

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO oneapi-src/oneTBB
    REF "v${VERSION}"
    SHA512 1f536b98c4eb331c3ba21e4b36362c932d7852b974ea7237b5f7bffb0e996d76f688c7ac534cb56f5866cfdffd5d3f015521b4c2ca21e584503070782ca82a2f
    HEAD_REF master
    PATCHES private-runtime-names.patch
)

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        -DTBB_TEST:BOOL=OFF
        -DTBB_STRICT:BOOL=OFF
        -DTBBMALLOC_PROXY_BUILD:BOOL=OFF
        # This only disables build-time discovery. The runtime-name patch also
        # keeps optional tbbbind lookups private if another helper is nearby.
        -DTBB_DISABLE_HWLOC_AUTOMATIC_SEARCH:BOOL=ON
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(CONFIG_PATH "lib/cmake/TBB")
vcpkg_copy_pdbs()
vcpkg_fixup_pkgconfig()

file(REMOVE_RECURSE
    "${CURRENT_PACKAGES_DIR}/share/doc"
    "${CURRENT_PACKAGES_DIR}/debug/include"
    "${CURRENT_PACKAGES_DIR}/debug/share"
)

file(READ "${CURRENT_PACKAGES_DIR}/share/tbb/TBBConfig.cmake" _contents)
file(WRITE "${CURRENT_PACKAGES_DIR}/share/tbb/TBBConfig.cmake" "include(CMakeFindDependencyMacro)
find_dependency(Threads)
${_contents}")

file(COPY "${CMAKE_CURRENT_LIST_DIR}/usage" DESTINATION "${CURRENT_PACKAGES_DIR}/share/tbb")
vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE.txt")
