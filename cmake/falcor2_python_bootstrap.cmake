include_guard(GLOBAL)

# OpenUSD's vcpkg port is configured while project() loads the toolchain. Find
# only the interpreter executable first so direct CMake invocations and build
# frontends select the same Python ABI before vcpkg computes package ABIs.
# Calling FindPython here would initialize platform-dependent variables such as
# Python_SOABI before project() has established the target platform.
if(DEFINED Python_EXECUTABLE AND NOT Python_EXECUTABLE STREQUAL "")
    set(_falcor2_python_executable "${Python_EXECUTABLE}")
else()
    set(_falcor2_python_hints)
    if(DEFINED Python_ROOT_DIR AND NOT Python_ROOT_DIR STREQUAL "")
        list(APPEND _falcor2_python_hints
            "${Python_ROOT_DIR}"
            "${Python_ROOT_DIR}/bin"
            "${Python_ROOT_DIR}/Scripts"
        )
    endif()
    find_program(
        _falcor2_python_executable
        NAMES python3 python
        HINTS ${_falcor2_python_hints}
        REQUIRED
    )
    set(Python_EXECUTABLE "${_falcor2_python_executable}" CACHE FILEPATH "Python interpreter used by Falcor2")
endif()

execute_process(
    COMMAND "${_falcor2_python_executable}" -c "import sys; print(sys.implementation.cache_tag)"
    RESULT_VARIABLE _falcor2_python_abi_result
    OUTPUT_VARIABLE _falcor2_python_abi
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT _falcor2_python_abi_result EQUAL 0 OR _falcor2_python_abi STREQUAL "")
    message(FATAL_ERROR "Failed to determine the Python ABI for '${_falcor2_python_executable}'")
endif()

set(FALCOR_PYTHON_ABI "${_falcor2_python_abi}" CACHE STRING "Python ABI used to build bundled OpenUSD" FORCE)
set(ENV{FALCOR_PYTHON_EXECUTABLE} "${_falcor2_python_executable}")
set(ENV{FALCOR_PYTHON_ABI} "${FALCOR_PYTHON_ABI}")

# vcpkg otherwise removes unrelated environment variables before evaluating a
# port. Preserve the two values consumed by the private OpenUSD overlay.
set(_falcor2_vcpkg_keep_env_vars "$ENV{VCPKG_KEEP_ENV_VARS}")
list(APPEND _falcor2_vcpkg_keep_env_vars FALCOR_PYTHON_EXECUTABLE FALCOR_PYTHON_ABI)
list(REMOVE_DUPLICATES _falcor2_vcpkg_keep_env_vars)
set(ENV{VCPKG_KEEP_ENV_VARS} "${_falcor2_vcpkg_keep_env_vars}")

unset(_falcor2_python_abi)
unset(_falcor2_python_abi_result)
unset(_falcor2_python_executable)
unset(_falcor2_python_executable CACHE)
unset(_falcor2_python_hints)
unset(_falcor2_vcpkg_keep_env_vars)
