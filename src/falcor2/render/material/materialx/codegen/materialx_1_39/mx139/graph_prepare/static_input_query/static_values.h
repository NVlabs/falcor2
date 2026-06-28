// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>

namespace falcor {
namespace materialx {
namespace mx139 {

// Static closure states.
//
//   KnownEmpty    :: closure is statically zero contribution
//   KnownNonEmpty :: closure is statically known to carry contribution
//   Unknown       :: could be either KnownEmpty or KnownNonEmpty at runtime
//
// Merge: disagreement becomes Unknown.
enum class ClosureStaticValue : uint8_t {
    KnownEmpty = 0,
    KnownNonEmpty = 1,
    Unknown = 2,
};

inline ClosureStaticValue merge_closure_values(ClosureStaticValue a, ClosureStaticValue b)
{
    return a == b ? a : ClosureStaticValue::Unknown;
}

inline bool is_empty_closure(ClosureStaticValue v)
{
    return v == ClosureStaticValue::KnownEmpty;
}
inline bool is_known_non_empty(ClosureStaticValue v)
{
    return v == ClosureStaticValue::KnownNonEmpty;
}

// Static scalar / color states. KnownConstant carries 1..4 floats (scalar..color4).
// Comparison is bit-exact; near-zero is NOT zero.
struct ScalarStaticConstant {
    std::array<float, 4> values{};
    uint8_t size = 0; // 1..4

    bool operator==(const ScalarStaticConstant& other) const
    {
        if (size != other.size)
            return false;
        for (uint8_t i = 0; i < size; ++i) {
            if (values[i] != other.values[i])
                return false;
        }
        return true;
    }
    bool operator!=(const ScalarStaticConstant& other) const { return !(*this == other); }
};

enum class ScalarStaticValueKind : uint8_t {
    KnownZero = 0,
    KnownOne = 1,
    KnownConstant = 2,
    Unknown = 3,
};

struct ScalarStaticValue {
    ScalarStaticValueKind kind = ScalarStaticValueKind::Unknown;
    ScalarStaticConstant value{}; // valid only when kind == KnownConstant

    static ScalarStaticValue zero() { return ScalarStaticValue{ScalarStaticValueKind::KnownZero, {}}; }
    static ScalarStaticValue one() { return ScalarStaticValue{ScalarStaticValueKind::KnownOne, {}}; }
    static ScalarStaticValue unknown() { return ScalarStaticValue{ScalarStaticValueKind::Unknown, {}}; }
    static ScalarStaticValue constant(const ScalarStaticConstant& v)
    {
        ScalarStaticValue s;
        s.kind = ScalarStaticValueKind::KnownConstant;
        s.value = v;
        return s;
    }

    bool operator==(const ScalarStaticValue& other) const
    {
        if (kind != other.kind)
            return false;
        if (kind == ScalarStaticValueKind::KnownConstant)
            return value == other.value;
        return true;
    }
    bool operator!=(const ScalarStaticValue& other) const { return !(*this == other); }
};

inline ScalarStaticValue merge_scalar_values(const ScalarStaticValue& a, const ScalarStaticValue& b)
{
    return a == b ? a : ScalarStaticValue::unknown();
}

inline bool is_zero_static_value(const ScalarStaticValue& v)
{
    return v.kind == ScalarStaticValueKind::KnownZero;
}
inline bool is_one_static_value(const ScalarStaticValue& v)
{
    return v.kind == ScalarStaticValueKind::KnownOne;
}
inline bool is_unknown_static_value(const ScalarStaticValue& v)
{
    return v.kind == ScalarStaticValueKind::Unknown;
}

} // namespace mx139
} // namespace materialx
} // namespace falcor
