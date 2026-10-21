set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

# _DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR avoids the use of constexpr mutex
# constructors in vcpkg packages, which can lead to binary incompatibilities.
set(VCPKG_C_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")
set(VCPKG_CXX_FLAGS "-D_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR")

# Falcor2's OpenUSD Python modules are built for the interpreter driving the
# outer build. Track the ABI in vcpkg's package identity, while leaving the
# interpreter path untracked so compatible virtual environments share binaries.
if(PORT STREQUAL "usd")
    set(VCPKG_ENV_PASSTHROUGH FALCOR_PYTHON_ABI)
    set(VCPKG_ENV_PASSTHROUGH_UNTRACKED FALCOR_PYTHON_EXECUTABLE)
endif()
