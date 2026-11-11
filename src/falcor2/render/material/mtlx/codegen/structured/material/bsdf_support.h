// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

#include <cstdint>
#include <string_view>

namespace falcor::mtlx::structured {

/// Stable identity of a BSDF implementation after resolving its MaterialX
/// implementation name. Keep implementation-name matching in bsdf_support.cpp;
/// downstream analysis and policies reason about this typed identity instead.
enum class BsdfKind {
    unknown,
    oren_nayar_diffuse,
    burley_diffuse,
    translucent,
    subsurface,
    beer,
    scratch_conductor,
    dielectric,
    conductor,
    generalized_schlick,
    sheen,
    chiang_hair,
    surface_unlit_transparent,
};

/// Target-independent BSDF lobe properties carried by the structured host
/// model. Convert these to Slang BSDFFlags only at an emission boundary.
enum class BsdfLobeFlags : uint32_t {
    none = 0,
    diffuse_reflection = 1 << 0,
    diffuse_transmission = 1 << 1,
    glossy_reflection = 1 << 2,
    glossy_transmission = 1 << 3,
    glossy_curve = 1 << 4,
    delta_transmission = 1 << 5,
};
FALCOR_ENUM_CLASS_OPERATORS(BsdfLobeFlags);

/// Construction and lobe properties for one BSDF implementation supported by
/// the structured target. This is the sole support authority used by both the
/// layering walker and conservative zero-weight pruning.
struct BsdfInfo {
    BsdfKind kind;
    std::string_view type;
    BsdfLobeFlags lobe_flags;
    bool transmissive = false;
    bool preserve_tangent_frame = false;
};

const BsdfInfo* find_bsdf_info(std::string_view implementation);

} // namespace falcor::mtlx::structured
