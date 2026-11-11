# Select the optional plugin before project() invokes the vcpkg toolchain.
option(FALCOR_ENABLE_USD_FBX "Build the USD FBX plugin (downloads Autodesk FBX SDK 2020.3.11)" OFF)
if(FALCOR_ENABLE_USD_FBX)
    list(APPEND VCPKG_MANIFEST_FEATURES usd-fbx)
endif()

# The overlay's supports expression defines the available SDK platforms.

function(falcor2_generate_usd_fbx_config)
    set(_prefix null)
    if(FALCOR_ENABLE_USD_FBX OR "usd-fbx" IN_LIST VCPKG_MANIFEST_FEATURES)
        set(_prefix "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}$<$<CONFIG:Debug>:/debug>")
        string(REPLACE "\\" "/" _prefix "${_prefix}")
        string(REPLACE "\"" "\\\"" _prefix "${_prefix}")
        set(_prefix "\"${_prefix}\"")
    endif()
    # Always regenerate, including when disabled, so stale packages cannot enable tests.
    file(GENERATE OUTPUT "${FALCOR_OUTPUT_DIRECTORY}/usd_fbx_config.json"
        CONTENT "{\n  \"usd_fbx_prefix\": ${_prefix}\n}\n")
endfunction()
