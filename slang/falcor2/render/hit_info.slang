// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public interface IHitInfo
{
    /// The type of geometry that was hit.
    public property GeometryType geometry_type { get; }
    /// The ID of the geometry instance that was hit.
    public property GeometryInstanceID geometry_instance_id { get; }
    /// The index of the primitive that was hit.
    public property uint primitive_index { get; }

    /// True if the hit is valid.
    public property bool valid { get; }

    public PackedHitInfo pack();
};

public struct PackedHitInfo
{
    public GeometryType geometry_type = GeometryType::invalid;
    /// Instance identity is meaningful only for a valid hit.
    public GeometryInstanceID geometry_instance_id = GeometryInstanceID(0);
    /// Geometry-owned encoding; shared readers must not interpret these words.
    public uint payload[3] = { 0, 0, 0 };

    public IHitInfo unpack()
    {
        if (geometry_type == GeometryType::triangle) {
            return TriangleHitInfo(this);
        } else if (geometry_type == GeometryType::lss) {
            return LSSHitInfo(this);
        } else if (geometry_type == GeometryType::area_light) {
            return AreaLightHitInfo(this);
        } else {
            return InvalidHitInfo();
        }
    }
};

public struct InvalidHitInfo : IHitInfo
{
    public property GeometryType geometry_type
    {
        get {
            return GeometryType::invalid;
        }
    }

    public property GeometryInstanceID geometry_instance_id
    {
        get {
            return GeometryInstanceID::invalid;
        }
    }

    public property uint primitive_index
    {
        get {
            return 0;
        }
    }

    public property bool valid
    {
        get {
            return false;
        }
    }

    public PackedHitInfo pack() { return {}; }
};

public interface IDeferredHitInfo
{
    public IHitInfo create();
};
