include_guard(GLOBAL)

# Falcor2 consumes its bundled OpenUSD only through this target. The overlay
# provides a private monolith and private TBB loader identities; deployment is
# handled separately by falcor2_usd_deployment.cmake.
if(DEFINED Python_EXECUTABLE)
    # OpenUSD uses FindPython3 while Falcor2 uses FindPython. Override the
    # interpreter recorded when the ABI-compatible vcpkg package was built so
    # both packages resolve the active environment.
    set(Python3_EXECUTABLE "${Python_EXECUTABLE}")
endif()
# Resolve development files together with the selected interpreter. Looking
# for Development alone can miss a Python environment that is not activated.
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pxr CONFIG REQUIRED)

if(NOT TARGET usd_m)
    message(FATAL_ERROR "Falcor2's bundled OpenUSD package does not provide the expected usd_m target")
endif()

add_library(falcor2_usd INTERFACE)
# USD's public headers call oneTBB directly. The monolith's private dependency
# does not propagate the link library or compile definitions to consumers.
target_link_libraries(falcor2_usd INTERFACE usd_m TBB::tbb Python3::Python)
