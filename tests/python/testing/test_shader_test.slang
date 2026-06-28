// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Self-test for the shader test framework.

#include "falcor2/testing/shader_test.slangh"

TEST_CASE(shader_test_pass)
{
    // Scalar checks.
    CHECK(true);
    CHECK_FALSE(false);
    CHECK_EQ(1, 1);
    CHECK_NE(1, 2);
    CHECK_APPROX(1.0f, 1.0f, 1e-6f);

    // Vector checks.
    CHECK_EQ(float3(1, 2, 3), float3(1, 2, 3));
    CHECK_NE(float3(1, 2, 3), float3(4, 5, 6));
    CHECK_APPROX(float3(1, 2, 3), float3(1, 2, 3), 1e-6f);
    CHECK_APPROX(float2(0.1f, 0.2f), float2(0.1f + 1e-7f, 0.2f - 1e-7f), 1e-6f);
    CHECK_APPROX(int2(10, 20), int2(11, 21), 1);

    // Matrix checks.
    // TODO: Disabled due to Slang compiler hang during generic overload resolution
    // with matrix types. File a Slang issue and re-enable once fixed.
    // float3x3 m = float3x3(1, 2, 3, 4, 5, 6, 7, 8, 9);
    // CHECK_EQ(m, float3x3(1, 2, 3, 4, 5, 6, 7, 8, 9));
    // CHECK_NE(m, float3x3(0));
    // CHECK_APPROX(m, m, 1e-6f);
    // CHECK_APPROX(float2x2(1, 2, 3, 4), float2x2(1, 2, 3, 4), 1e-6f);
}

// This entry point deliberately fails and is marked as expected-fail via PYTEST_MARK below.
// PYTEST_MARK(pytest.mark.xfail(reason="Intentional failing self-test.", strict=True, run=True))
TEST_CASE(shader_test_fail)
{
    CHECK(false);
}
