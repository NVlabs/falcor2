# Basic-family emitter sources. Contributes sources to the parent OBJECT target
# `falcor2_materialx_1_39_codegen` -- this directory does not create a new target.
target_sources(falcor2_materialx_1_39_codegen PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/composition_aliases.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/composition_aliases.h
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf_mix_emit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/bsdf_mix_emit.h
    ${CMAKE_CURRENT_SOURCE_DIR}/closure_tree_emit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/closure_tree_emit.h
    ${CMAKE_CURRENT_SOURCE_DIR}/root_strategy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/root_strategy.h
    ${CMAKE_CURRENT_SOURCE_DIR}/stack_data_basic_emit.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/stack_data_basic_emit.h
)
