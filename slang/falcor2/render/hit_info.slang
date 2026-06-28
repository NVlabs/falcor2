// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public interface IHitInfo
{
    /// The type of geometry that was hit.
    public property GeometryType geometry_type;
    /// The ID of the geometry instance that was hit.
    public property GeometryInstanceID geometry_instance_id;
    /// The index of the primitive that was hit.
    public property uint primitive_index;

    /// True if the hit is valid.
    public property bool valid { get; }
};

public struct HitData
{
    uint data[2] = { 0, 0 };
};

public struct HitInfo : IHitInfo
{
    public GeometryType geometry_type = GeometryType::invalid;
    public GeometryInstanceID geometry_instance_id = GeometryInstanceID::invalid;
    public uint primitive_index = 0;
    public HitData _hit_data;

    public property bool valid
    {
        get {
            return geometry_type != GeometryType::invalid;
        }
    }
};

public interface IDeferredHitInfo
{
    public HitInfo create();
}
