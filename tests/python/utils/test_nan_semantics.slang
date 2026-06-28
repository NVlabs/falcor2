// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/testing/shader_test.slangh"
import falcor2.utils;


void check_nan_semantics<T : IFloat>(T fNAN)
{
    T val0 = fNAN;
    T val1 = val0 + T(5);

    // == must always be false for NaN
    CHECK_FALSE(fNAN == fNAN);
    CHECK_FALSE(val0 == fNAN);
    CHECK_FALSE(val0 == val0);
    CHECK_FALSE(val0 == val1);

    // >= must always be false for NaN
    CHECK_FALSE(fNAN >= fNAN);
    CHECK_FALSE(val0 >= fNAN);
    CHECK_FALSE(val0 >= val0);
    CHECK_FALSE(val0 >= val1);

    // <= must always be false for NaN
    CHECK_FALSE(fNAN <= fNAN);
    CHECK_FALSE(val0 <= fNAN);
    CHECK_FALSE(val0 <= val0);
    CHECK_FALSE(val0 <= val1);

    // > must always be false for NaN
    CHECK_FALSE(fNAN > fNAN);
    CHECK_FALSE(val0 > fNAN);
    CHECK_FALSE(val0 > val0);
    CHECK_FALSE(val0 > val1);

    // < must always be false for NaN
    CHECK_FALSE(fNAN < fNAN);
    CHECK_FALSE(val0 < fNAN);
    CHECK_FALSE(val0 < val0);
    CHECK_FALSE(val0 < val1);

    // != must always be true for NaN
    CHECK(val0 != fNAN);
    CHECK(val0 != val0);
    CHECK(val0 != val1);
}

TEST_CASE(nan_semantics_half)
{
    check_nan_semantics(half.NAN);
}

TEST_CASE(nan_semantics_float)
{
    check_nan_semantics(float.NAN);
}

TEST_CASE(nan_semantics_double)
{
    check_nan_semantics(double.NAN);
}
