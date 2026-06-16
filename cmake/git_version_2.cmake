find_package(Git QUIET)

set(GIT_VERSION_LIST_DIR ${CMAKE_CURRENT_LIST_DIR})

function(git_version_write_state file hash)
    file(WRITE ${file} ${hash})
endfunction()

function(git_version_read_state file hash)
    if(EXISTS ${file})
        file(READ ${file} content)
        set(${hash} ${content} PARENT_SCOPE)
    endif()
endfunction()

macro(run_git)
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" ${ARGV}
        WORKING_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE output
        ERROR_VARIABLE stderr
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endmacro()

function(git_version_check_2 pre_configure_file post_configure_file state_file)

    set(GIT_AVAILABLE "0")
    set(GIT_COMMIT "unknown")
    set(GIT_BRANCH "unknown")
    set(GIT_DIRTY "0")

    if(Git_FOUND)
        set(GIT_AVAILABLE "1")

        # Get commit
        run_git(log -1 --format=%h)
        if(exit_code EQUAL 0)
            set(GIT_COMMIT ${output})
        endif()

        # Get branch
        run_git(symbolic-ref --short -q HEAD)
        if(exit_code EQUAL 0)
            set(GIT_BRANCH ${output})
        endif()

        # Check for uncommitted changes
        run_git(status --porcelain -uno)
        if(exit_code EQUAL 0)
            if(NOT "${output}" STREQUAL "")
                set(GIT_DIRTY "1")
            endif()
        endif()
    endif()

    string(SHA256 hash "${GIT_AVAILABLE},${GIT_COMMIT},${GIT_BRANCH},${GIT_DIRTY}")
    git_version_read_state(${state_file} old_hash)

    # Only update git_version.h and state cache if anything changed.
    if(NOT EXISTS ${post_configure_file}
        OR NOT DEFINED old_hash
        OR NOT "${hash}" STREQUAL "${old_hash}")
        git_version_write_state(${state_file} ${hash})
        get_filename_component(parent_dir ${post_configure_file} DIRECTORY)
        file(MAKE_DIRECTORY ${parent_dir})
        configure_file(${pre_configure_file} ${post_configure_file} @ONLY)
    endif()

endfunction()

function(git_version_setup_2 suffix)
    set(pre_configure_file ${GIT_VERSION_LIST_DIR}/git_version.h.in)
    set(post_configure_file ${CMAKE_CURRENT_BINARY_DIR}/git_version/git_version_${suffix}.h)
    set(state_file ${CMAKE_CURRENT_BINARY_DIR}/git_version/git_state_${suffix})
    add_custom_target(git_version_run_check_${suffix} COMMAND ${CMAKE_COMMAND}
        -DGIT_VERSION_RUN_CHECK=1
        -DPRE_CONFIGURE_FILE=${pre_configure_file}
        -DPOST_CONFIGURE_FILE=${post_configure_file}
        -DSTATE_FILE=${state_file}
        -P ${GIT_VERSION_LIST_DIR}/git_version_2.cmake
        BYPRODUCTS ${post_configure_file} ${state_file}
    )
    set_target_properties(git_version_run_check_${suffix} PROPERTIES FOLDER "misc")

    add_library(git_version_${suffix} INTERFACE)
    target_include_directories(git_version_${suffix} INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/git_version)
    add_dependencies(git_version_${suffix} git_version_run_check_${suffix})

    git_version_check_2(${pre_configure_file} ${post_configure_file} ${state_file})
endfunction()

# Used to run from external cmake process.
if(GIT_VERSION_RUN_CHECK)
    git_version_check_2(${PRE_CONFIGURE_FILE} ${POST_CONFIGURE_FILE} ${STATE_FILE})
endif()
