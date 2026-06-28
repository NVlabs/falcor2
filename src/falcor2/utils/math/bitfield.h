// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace falcor {

/// Utility for packing and unpacking bitfields within an unsigned integer type.
/// @tparam T Unsigned integer type used as the storage.
/// @tparam BitOffset Bit position of the field's least significant bit.
/// @tparam BitSize Number of bits in the field.
template<typename T, size_t BitOffset, size_t BitSize>
struct BitField {
    static_assert(std::is_unsigned_v<T>, "BitField requires an unsigned integer type.");
    static_assert(BitSize + BitOffset <= sizeof(T) * 8, "Size and offset exceed base type size.");

    /// Maximum value representable by this field.
    static constexpr T MAX_VALUE = (T(1) << BitSize) - T(1);
    /// Bitmask for this field, shifted to its position.
    static constexpr T MASK = ((T(1) << BitSize) - T(1)) << BitOffset;

    /// Insert a value into the field, preserving other bits.
    /// @param base The integer to modify.
    /// @param field The value to insert (must not exceed MAX_VALUE).
    static void insert(T& base, T field) { base = (base & ~MASK) | (field << BitOffset); }

    /// Extract the field value from an integer.
    /// @param base The integer to extract from.
    /// @return The extracted field value.
    static T extract(T base) { return (base & MASK) >> BitOffset; }
};

} // namespace falcor
