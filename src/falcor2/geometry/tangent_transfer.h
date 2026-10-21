// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/macros.h"
#include <span>
#include <vector>

namespace falcor {

/// Transfer tangents through a fixed incident triangle's geometric frame, then project onto the target normal.
/// Reference arrays are borrowed for this object's lifetime. Normals and tangents must be unit and perpendicular.
/// Degenerate triangles use shortest-arc normal alignment. The caller preserves tangent handedness.
class FALCOR_API TangentTransfer {
public:
    TangentTransfer(
        std::span<const uint3> triangles,
        std::span<const float3> positions,
        std::span<const float3> normals,
        std::span<const float3> tangents
    );

    /// Evaluate one pose independently of previous results. Arrays must match the reference vertex count.
    void evaluate(std::span<const float3> positions, std::span<const float3> normals, std::span<float3> tangents) const;

private:
    std::span<const float3> m_normals;
    std::span<const float3> m_tangents;
    std::vector<uint2> m_neighbors;
    std::vector<float3> m_local_tangents;
};

} // namespace falcor
