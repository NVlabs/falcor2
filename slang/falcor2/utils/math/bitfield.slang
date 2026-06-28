// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Utility for packing and unpacking bitfields within an integer type.
/// @tparam T Integer type used as the storage.
/// @tparam BitOffset Bit position of the field's least significant bit.
/// @tparam BitSize Number of bits in the field.
public __generic<T : __BuiltinIntegerType, let BitOffset : uint, let BitSize : uint> struct BitField
{
    // static_assert(BitSize + BitOffset < sizeof(T) * 8, "Size and offset exceed base type size.");

    /// Maximum value representable by this field.
    public static const T MAX_VALUE = (T(1) << BitSize) - T(1);
    /// Bitmask for this field, shifted to its position.
    public static const T MASK = ((T(1) << BitSize) - T(1)) << BitOffset;

    /// Insert a value into the field, preserving other bits.
    /// @param base The integer to modify.
    /// @param field The value to insert (must not exceed MAX_VALUE).
    public static void insert(inout T base, T field) { base = (base & ~MASK) | (field << BitOffset); }

    /// Extract the field value from an integer.
    /// @param base The integer to extract from.
    /// @return The extracted field value.
    public static T extract(T base) { return (base & MASK) >> BitOffset; }
};
