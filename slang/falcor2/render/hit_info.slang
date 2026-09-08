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
    public uint data[5] = { uint(GeometryType::invalid), 0, 0, 0, 0 };

    public IHitInfo unpack()
    {
        if (data[0] == uint(GeometryType::triangle)) {
            return TriangleHitInfo(this);
        } else if (data[0] == uint(GeometryType::lss)) {
            return LSSHitInfo(this);
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
