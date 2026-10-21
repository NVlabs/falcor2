// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace falcor::mtlx::structured {

/// Formal MaterialX operation used by conservative structured graph analysis.
///
/// Structured keeps this as one complete authority derived from the resolved
/// NodeDef. MaterialX classification overlaps only a subset and serves other
/// generator decisions; consulting both would create semantic paths that can
/// drift. CombinerMode is produced later and describes the target closure
/// representation, not this formal node kind.
enum class FormalNodeOperation {
    unknown,
    constant,
    convert,
    multiply,
    add,
    subtract,
    divide,
    mix,
    layer,
    weighted_layer,
    clamp,
    minimum,
    maximum,
    absolute_value,
    invert,
    dot,
    luminance,
    extract,
    uniform_edf,
    generalized_schlick_edf,
    surface,
};

} // namespace falcor::mtlx::structured
