// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

public enum class AreaLightGeometryFlags : uint {
    none = 0,
    visible = 1 << 0,
};
#ifdef FALCOR_SHARED_HOST
FALCOR_ENUM_CLASS_OPERATORS(AreaLightGeometryFlags);
#endif

public struct AreaLightGeometryInstanceDesc
{
    public GeometryInstanceDescHeader header;
    public LightID light_id;
    public AreaLightGeometryFlags flags;
    public uint reserved[2];
};
