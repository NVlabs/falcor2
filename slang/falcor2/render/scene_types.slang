// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

// ----------------------------------------------------------------------------
// Material types
// ----------------------------------------------------------------------------

/// Material ID.
public enum class MaterialID : uint { invalid = 0 };

public enum class MaterialFlags : uint {
    none = 0,
    double_sided = 1 << 0,
    thin_walled = 1 << 1,
};
#ifdef FALCOR_SHARED_HOST
FALCOR_ENUM_CLASS_OPERATORS(MaterialFlags);
#endif

public struct MaterialHeader
{
    public uint type_id;
    public MaterialFlags flags;
    public uint _pad1;
    public uint _pad2;
};

public struct MaterialPayload
{
    public uint data[28];
};

public struct MaterialData
{
    public MaterialHeader header;
    public MaterialPayload payload;
};

// ----------------------------------------------------------------------------
// Geometry types
// ----------------------------------------------------------------------------

public enum class GeometryType : uint {
    invalid = 0xffffffff,
    triangle = 0,
    lss = 1,
};

/// Geometry ID.
public enum class GeometryID : uint { invalid = 0xffffffff };

/// Geometry group ID.
public enum class GeometryGroupID : uint { invalid = 0xffffffff };

/// Geometry group instance ID.
/// Corresponds to InstanceIndex() system value.
public enum class GeometryGroupInstanceID : uint { invalid = 0xffffffff };

/// Geometry instance ID.
/// Corresponds to InstanceID() + GeometryIndex() system value.
public enum class GeometryInstanceID : uint { invalid = 0xffffffff };

public enum class TransformID : uint { invalid = 0xffffffff };

public struct GeometryInstanceDescHeader
{
    public GeometryType type;
    public GeometryID geometry_id;
    public MaterialID material_id;
    public TransformID transform_id;
};

public struct GeometryInstanceDescPayload
{
    public uint data[4];
};

public struct GeometryInstanceDesc
{
    public GeometryInstanceDescHeader header;
    public GeometryInstanceDescPayload payload;
};

public struct InstanceData
{
    public TransformID transform_id;
};

// ----------------------------------------------------------------------------
// Light types
// ----------------------------------------------------------------------------

public enum class LightID : uint { invalid = 0xffffffff };

public enum class LightType : uint {
    constant_light,
    distant_light,
    env_map_light,
    point_light,
    spot_light,
};

public enum class LightFlags : uint {
    none = 0,
    environment = 1 << 0,
    delta_position = 1 << 1,
    delta_direction = 1 << 2,
};
#ifdef FALCOR_SHARED_HOST
FALCOR_ENUM_CLASS_OPERATORS(LightFlags);
#endif

public struct LightHeader
{
    LightType type;
};

public struct LightPayload
{
    uint data[31];
};

public struct LightData
{
    public LightHeader header;
    public LightPayload payload;
};
