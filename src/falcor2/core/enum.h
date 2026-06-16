// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/enum.h>

// -------------------------------------------------------------------------------------------------
// Enum flags
// -------------------------------------------------------------------------------------------------

// clang-format off
/// Implement logical operators on a class enum for making it usable as a flags enum.
#define FALCOR_ENUM_CLASS_OPERATORS(e_) \
    inline constexpr e_ operator& (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)& static_cast<int>(b)); } \
    inline constexpr e_ operator| (e_ a, e_ b) { return static_cast<e_>(static_cast<int>(a)| static_cast<int>(b)); } \
    inline constexpr e_& operator|= (e_& a, e_ b) { a = a | b; return a; }; \
    inline constexpr e_& operator&= (e_& a, e_ b) { a = a & b; return a; }; \
    inline constexpr e_  operator~ (e_ a) { return static_cast<e_>(~static_cast<int>(a)); } \
    inline constexpr bool is_set(e_ val, e_ flag) { return (val & flag) != static_cast<e_>(0); } \
    inline void flip_bit(e_& val, e_ flag) { val = is_set(val, flag) ? (val & (~flag)) : (val | flag); }
// clang-format on
