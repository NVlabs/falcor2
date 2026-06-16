// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Return true if value contains any bit from flag.
public bool is_set<T : __EnumType>(T value, T flag)
{
    return value & flag;
}

/// Toggle flag in value.
public void flip_bit<T : __EnumType>(inout T value, T flag)
{
    value = is_set(value, flag) ? (value & ~flag) : (value | flag);
}
