// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

public uint2 emissive_triangle_counts()
{
    return uint2(g_scene.emissive_geometry.triangle_count, g_scene.emissive_geometry.active_triangle_count);
}
