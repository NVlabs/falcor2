// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;
import falcor2.ui.kernels.scene_picker;

public uint probe_backface()
{
    RayTracingSceneIntersector intersector;
    return uint(intersector.intersect(Ray(float3(0, 0, -2), float3(0, 0, 1))).valid);
}
