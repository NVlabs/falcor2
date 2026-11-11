include_guard(GLOBAL)

function(_falcor2_configure_usd_deployment)
    if(NOT TARGET falcor2_usd OR NOT TARGET usd_m OR NOT TARGET TBB::tbb OR NOT TARGET TBB::tbbmalloc)
        message(FATAL_ERROR "Falcor2's private OpenUSD/TBB targets are unavailable")
    endif()

    # A monolithic USD build embeds plugin code in usd_m. Only its metadata
    # and resources are installed under lib/usd.
    set(_pxr_usd_lib_dir "${pxr_DIR}/../../lib/usd")
    get_filename_component(_pxr_usd_lib_dir "${_pxr_usd_lib_dir}" ABSOLUTE)
    if(NOT EXISTS "${_pxr_usd_lib_dir}/plugInfo.json")
        message(FATAL_ERROR "USD plugin registry not found: ${_pxr_usd_lib_dir}/plugInfo.json\n"
            "The pxr package layout may have changed. pxr_DIR is: ${pxr_DIR}")
    endif()

    file(GLOB_RECURSE _falcor2_usd_plugin_files "${_pxr_usd_lib_dir}/*")
    set(_falcor2_usd_plugin_stamp "${FALCOR_OUTPUT_DIRECTORY}/usd/.falcor2_usd_plugins.stamp")
    add_custom_command(
        OUTPUT "${_falcor2_usd_plugin_stamp}"
        COMMAND ${CMAKE_COMMAND} -E remove_directory "${FALCOR_OUTPUT_DIRECTORY}/usd"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/usd"
        COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY} "${_pxr_usd_lib_dir}" "${FALCOR_OUTPUT_DIRECTORY}/usd"
        COMMAND ${CMAKE_COMMAND} -E touch "${_falcor2_usd_plugin_stamp}"
        DEPENDS ${_falcor2_usd_plugin_files}
        COMMENT "Sync Falcor2 OpenUSD plugin metadata and resources"
        VERBATIM
    )
    add_custom_target(falcor2_copy_usd_plugins DEPENDS "${_falcor2_usd_plugin_stamp}")
    install(
        DIRECTORY "${_pxr_usd_lib_dir}/"
        DESTINATION "${CMAKE_INSTALL_BINDIR}/usd"
    )

    # Falcor and its Python wrappers share the OpenUSD monolith and one private
    # TBB scheduler/allocator pair. Stage and install them even in native-only
    # builds.
    set(_falcor2_tbb_name "$<TARGET_FILE_NAME:TBB::tbb>")
    set(_falcor2_tbbmalloc_name "$<TARGET_FILE_NAME:TBB::tbbmalloc>")
    if(UNIX)
        # oneTBB's physical filename includes its minor ABI version. The
        # loader and allocator lookup require the major-version SONAME.
        set(_falcor2_tbb_name "$<TARGET_SONAME_FILE_NAME:TBB::tbb>")
        set(_falcor2_tbbmalloc_name "$<TARGET_SONAME_FILE_NAME:TBB::tbbmalloc>")
    endif()
    set(_falcor2_usd_runtime_stamp "${FALCOR_OUTPUT_DIRECTORY}/.falcor2_usd_runtime.stamp")
    set(_falcor2_usd_runtime_commands
        COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:usd_m>"
            "${FALCOR_OUTPUT_DIRECTORY}/$<TARGET_FILE_NAME:usd_m>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:TBB::tbb>"
            "${FALCOR_OUTPUT_DIRECTORY}/${_falcor2_tbb_name}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "$<TARGET_FILE:TBB::tbbmalloc>"
            "${FALCOR_OUTPUT_DIRECTORY}/${_falcor2_tbbmalloc_name}"
    )

    # Native Python wrappers search two directories up from falcor2/pxr. Keep
    # a matching runtime set in that configuration-specific package root.
    if(FALCOR_BUILD_PYTHON)
        list(APPEND _falcor2_usd_runtime_commands
            COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:usd_m>"
                "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2/$<TARGET_FILE_NAME:usd_m>"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:TBB::tbb>"
                "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2/${_falcor2_tbb_name}"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "$<TARGET_FILE:TBB::tbbmalloc>"
                "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2/${_falcor2_tbbmalloc_name}"
        )
    endif()

    add_custom_command(
        OUTPUT "${_falcor2_usd_runtime_stamp}"
        ${_falcor2_usd_runtime_commands}
        COMMAND ${CMAKE_COMMAND} -E touch "${_falcor2_usd_runtime_stamp}"
        DEPENDS "$<TARGET_FILE:usd_m>" "$<TARGET_FILE:TBB::tbb>" "$<TARGET_FILE:TBB::tbbmalloc>"
        COMMENT "Sync Falcor2-private OpenUSD and TBB runtimes"
        VERBATIM
    )
    add_custom_target(falcor2_copy_usd_runtime DEPENDS "${_falcor2_usd_runtime_stamp}")

    install(
        IMPORTED_RUNTIME_ARTIFACTS usd_m TBB::tbb TBB::tbbmalloc
        LIBRARY DESTINATION ${CMAKE_INSTALL_BINDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )

    # Stage the configuration-matching Falcor-owned OpenUSD Python package in
    # the build tree. Repository imports extend falcor2.__path__ to this path.
    if(FALCOR_BUILD_PYTHON)
        set(_falcor2_pxr_release_source "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/python/falcor2/pxr")
        set(_falcor2_pxr_debug_source "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/debug/python/falcor2/pxr")
        if(NOT EXISTS "${_falcor2_pxr_release_source}")
            message(FATAL_ERROR "Falcor2 OpenUSD Release Python package not found: ${_falcor2_pxr_release_source}")
        endif()
        if((CMAKE_CONFIGURATION_TYPES AND "Debug" IN_LIST CMAKE_CONFIGURATION_TYPES)
            OR CMAKE_BUILD_TYPE STREQUAL "Debug")
            if(NOT EXISTS "${_falcor2_pxr_debug_source}")
                message(FATAL_ERROR "Falcor2 OpenUSD Debug Python package not found: ${_falcor2_pxr_debug_source}")
            endif()
        endif()

        set(_falcor2_pxr_source "$<IF:$<CONFIG:Debug>,${_falcor2_pxr_debug_source},${_falcor2_pxr_release_source}>")
        file(GLOB_RECURSE _falcor2_pxr_files
            "${_falcor2_pxr_release_source}/*"
            "${_falcor2_pxr_debug_source}/*"
        )
        file(GLOB_RECURSE _falcor2_pxr_stub_files
            RELATIVE "${_falcor2_pxr_release_source}"
            "${_falcor2_pxr_release_source}/*.pyi"
        )
        if(_falcor2_pxr_stub_files)
            set(_falcor2_pxr_stub_commands)
            set(_falcor2_pxr_stub_sources)
            set(_falcor2_pxr_editor_stubs)
            foreach(_falcor2_pxr_stub_file IN LISTS _falcor2_pxr_stub_files)
                get_filename_component(_falcor2_pxr_stub_dir "${_falcor2_pxr_stub_file}" DIRECTORY)
                list(APPEND _falcor2_pxr_stub_sources
                    "${_falcor2_pxr_release_source}/${_falcor2_pxr_stub_file}"
                )
                list(APPEND _falcor2_pxr_editor_stubs
                    "${CMAKE_SOURCE_DIR}/falcor2/pxr/${_falcor2_pxr_stub_file}"
                )
                list(APPEND _falcor2_pxr_stub_commands
                    COMMAND ${CMAKE_COMMAND} -E make_directory
                        "${CMAKE_SOURCE_DIR}/falcor2/pxr/${_falcor2_pxr_stub_dir}"
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${_falcor2_pxr_release_source}/${_falcor2_pxr_stub_file}"
                        "${CMAKE_SOURCE_DIR}/falcor2/pxr/${_falcor2_pxr_stub_file}"
                )
            endforeach()
            add_custom_command(
                OUTPUT ${_falcor2_pxr_editor_stubs}
                ${_falcor2_pxr_stub_commands}
                DEPENDS ${_falcor2_pxr_stub_sources}
                COMMENT "Sync Falcor2 OpenUSD editor stubs"
                VERBATIM
            )
            add_custom_target(falcor2_copy_usd_stubs DEPENDS ${_falcor2_pxr_editor_stubs})
        else()
            message(STATUS "Falcor2 OpenUSD Python stubs are unavailable; skipping editor stub deployment")
        endif()

        set(_falcor2_pxr_stamp "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2/.falcor2_usd_python.stamp")
        add_custom_command(
            OUTPUT "${_falcor2_pxr_stamp}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2"
            COMMAND ${CMAKE_COMMAND} -E ${FALCOR_COPY_DIRECTORY}
                "${_falcor2_pxr_source}"
                "${FALCOR_OUTPUT_DIRECTORY}/python/falcor2/pxr"
            COMMAND ${CMAKE_COMMAND} -E touch "${_falcor2_pxr_stamp}"
            DEPENDS ${_falcor2_pxr_files}
            COMMENT "Sync configuration-matching Falcor2 OpenUSD Python bindings"
            VERBATIM
        )
        add_custom_target(falcor2_copy_usd_python DEPENDS "${_falcor2_pxr_stamp}")

        install(
            DIRECTORY "${_falcor2_pxr_release_source}/"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/pxr"
            CONFIGURATIONS Release RelWithDebInfo MinSizeRel
        )
        install(
            DIRECTORY "${_falcor2_pxr_debug_source}/"
            DESTINATION "${CMAKE_INSTALL_BINDIR}/pxr"
            CONFIGURATIONS Debug
        )
    endif()

    # Keep the implementation targets private to this module. Falcor and the
    # default build depend only on this one logical deployment target.
    add_custom_target(falcor2_usd_deployment ALL)
    add_dependencies(falcor2_usd_deployment
        falcor2_copy_usd_plugins
        falcor2_copy_usd_runtime
    )
    if(TARGET falcor2_copy_usd_python)
        add_dependencies(falcor2_usd_deployment falcor2_copy_usd_python)
    endif()
    if(TARGET falcor2_copy_usd_stubs)
        add_dependencies(falcor2_usd_deployment falcor2_copy_usd_stubs)
    endif()
endfunction()

_falcor2_configure_usd_deployment()
