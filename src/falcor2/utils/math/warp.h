// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"

namespace falcor {

// ----------------------------------------------------------------------------
// Octahedral mapping
//
// The center of the map represents the +z axis and its corners -z.
// The rotated inner square is the xy-plane projected onto the upper hemi-
// sphere, the outer four triangles folds down over the lower hemisphere.
// There are versions for equal-area and non-equal area (slightly faster).
//
// For details refer to:
// - Clarberg 2008, "Fast Equal-Area Mapping of the (Hemi)Sphere using SIMD".
// - Cigolle et al. 2014, "Survey of Efficient Representations for Independent Unit Vectors".
//
// ----------------------------------------------------------------------------

/// Helper function to reflect the folds of the lower hemisphere
/// over the diagonals in the octahedral map.
template<typename T>
inline sgl::math::vector<T, 2> oct_wrap(sgl::math::vector<T, 2> v)
{
    return (T(1) - abs(v.yx())) * select(gt(v.xy(), T(0)), sgl::math::vector<T, 2>(1), sgl::math::vector<T, 2>(-1));
}

/// Converts normalized direction to the octahedral map (non-equal area, signed normalized).
/// @param n Normalized direction.
/// @return Position in octahedral map in [-1,1] for each component.
template<typename T>
inline sgl::math::vector<T, 2> ndir_to_oct_snorm(sgl::math::vector<T, 3> n)
{
    // Project the sphere onto the octahedron (|x|+|y|+|z| = 1) and then onto the xy-plane.
    sgl::math::vector<T, 2> p = n.xy() * (T(1) / (sgl::math::abs(n.x) + sgl::math::abs(n.y) + sgl::math::abs(n.z)));
    p = (n.z < T(0)) ? oct_wrap(p) : p;
    return p;
}

/// Converts normalized direction to the octahedral map (non-equal area, unsigned normalized).
/// @param n Normalized direction.
/// @return Position in octahedral map in [0,1] for each component.
template<typename T>
inline sgl::math::vector<T, 2> ndir_to_oct_unorm(sgl::math::vector<T, 3> n)
{
    return ndir_to_oct_snorm(n) * T(0.5) + T(0.5);
}

/// Converts point in the octahedral map to normalized direction (non-equal area, signed normalized).
/// @param p Position in octahedral map in [-1,1] for each component.
/// @return Normalized direction.
template<typename T>
inline sgl::math::vector<T, 3> oct_to_ndir_snorm(sgl::math::vector<T, 2> p)
{
    sgl::math::vector<T, 3> n = sgl::math::vector<T, 3>(p.xy(), T(1) - sgl::math::abs(p.x) - sgl::math::abs(p.y));
    n = sgl::math::vector<T, 3>((n.z < T(0)) ? oct_wrap(n.xy()) : n.xy(), n.z);
    return normalize(n);
}

/// Converts point in the octahedral map to normalized direction (non-equal area, unsigned normalized).
/// @param p Position in octahedral map in [0,1] for each component.
/// @return Normalized direction.
template<typename T>
inline sgl::math::vector<T, 3> oct_to_ndir_unorm(sgl::math::vector<T, 2> p)
{
    return oct_to_ndir_snorm(p * T(2) - T(1));
}

} // namespace falcor
