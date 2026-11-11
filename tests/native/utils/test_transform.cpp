// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/transform.h"
#include "falcor2/utils/math/transform.h"

#include <limits>

using namespace falcor;

TEST_CASE("axially symmetric transform")
{
    float4x4 transform = float4x4::identity();
    CHECK(is_axially_symmetric_transform(transform, 0));

    // Axial scale may differ from the two radial scales.
    transform.set_col(0, float4(2.f, 0.f, 0.f, 0.f));
    transform.set_col(1, float4(0.f, 3.f, 0.f, 0.f));
    transform.set_col(2, float4(0.f, 0.f, -3.f, 0.f));
    transform.set_col(3, float4(4.f, 5.f, 6.f, 1.f));
    CHECK(is_axially_symmetric_transform(transform, 0));
    CHECK_FALSE(is_axially_symmetric_transform(transform, 1));

    // Radial non-uniform scaling and shear do not preserve axial symmetry.
    transform.set_col(2, float4(0.f, 0.f, 4.f, 0.f));
    CHECK_FALSE(is_axially_symmetric_transform(transform, 0));
    transform.set_col(2, float4(0.f, 1.f, 3.f, 0.f));
    CHECK_FALSE(is_axially_symmetric_transform(transform, 0));

    // Degenerate, non-finite, and invalid-axis inputs are unsupported.
    transform = float4x4::identity();
    transform.set_col(0, float4(0.f));
    CHECK_FALSE(is_axially_symmetric_transform(transform, 0));
    transform.set_col(0, float4(std::numeric_limits<float>::infinity(), 0.f, 0.f, 0.f));
    CHECK_FALSE(is_axially_symmetric_transform(transform, 0));
    CHECK_FALSE(is_axially_symmetric_transform(float4x4::identity(), 3));
}


TEST_CASE("Transform cached decomposition identity flags")
{
    using Flags = Transform::IdentityFlags;
    for (int mask = 0; mask < 8; ++mask) {
        float4x4 matrix = float4x4::identity();
        Flags expected = Flags::none;
        if (mask & 1)
            matrix[0][3] = 3.f;
        else
            expected |= Flags::translation;
        if (mask & 2) {
            matrix[1][1] = matrix[2][2] = 0.f;
            matrix[1][2] = -1.f;
            matrix[2][1] = 1.f;
        } else {
            expected |= Flags::rotation;
        }
        if (mask & 4)
            matrix[0][0] = 2.f;
        else
            expected |= Flags::scale;
        Transform transform(matrix);
        CHECK(transform.identity_flags() == expected);
        CHECK(transform.matrix() == matrix);
        const Transform copied(transform);
        CHECK(copied.identity_flags() == expected);
        CHECK(copied.matrix() == matrix);
        transform.reset();
        CHECK(transform.identity_flags() == Flags::all);
        CHECK(transform.matrix() == float4x4::identity());
    }
    auto perturbed = float4x4::identity();
    perturbed[0][1] = 4.f * std::numeric_limits<float>::epsilon();
    const Transform corrected(perturbed);
    CHECK(corrected.identity_flags() == Flags::all);
    CHECK(corrected.matrix() == float4x4::identity());
}
