target_sources(falcor2 PRIVATE
    codegen.cpp
    codegen.h
    codegen_types.h

    mx_parameters.cpp
    mx_parameters.h
)

add_subdirectory(materialx_1_39)

set(MATERIALX_1_39_SLANG_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/materialx_1_39/libraries" PARENT_SCOPE)
set(MATERIALX_1_39_MX139_SNIPPET_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/materialx_1_39/mx139/slang" PARENT_SCOPE)
target_compile_definitions(falcor2 PRIVATE
    MATERIALX_1_39_SLANG_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/slang/libraries"
    MATERIALX_1_39_STD_LIBRARY_PATH="${FALCOR_OUTPUT_DIRECTORY}/materialx/1_39/std/libraries")

target_link_libraries(falcor2
    PRIVATE
        $<LINK_ONLY:MaterialXCore>
        $<LINK_ONLY:MaterialXFormat>
        $<LINK_ONLY:MaterialXGenSlang>
)
