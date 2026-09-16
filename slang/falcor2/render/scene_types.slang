// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef FALCOR_SHARED_HOST
implementing render;
import falcor2.utils;
#endif

// ----------------------------------------------------------------------------
// Material types
// ----------------------------------------------------------------------------

/// Material ID.
public enum class MaterialID : uint { invalid = 0 };

public enum class MaterialFlags : uint {
    none = 0,
    /// The material can interact with rays on both the front- and back-facing sides of a surface.
    two_sided = 1 << 0,
    /// The material represents an infinitely thin boundary and is never added to or removed from
    /// the path's interior material list.
    thin_walled = 1 << 1,
    /// The material contributes terminal emitted radiance when hit; it has no scattering and is
    /// not sampled as a light.
    unlit = 1 << 2,
};
#ifdef FALCOR_SHARED_HOST
FALCOR_ENUM_CLASS_OPERATORS(MaterialFlags);
#endif

/// Controls how material opacity is evaluated during intersection traversal.
public enum class OpacityFlags : uint {
    none = 0,
    /// Evaluate the material opacity instead of accepting every hit.
    enabled = 1 << 0,
    /// Apply a binary threshold instead of stochastic hit acceptance.
    use_threshold = 1 << 1,
    /// Evaluate opacity through IMaterial instead of using the packed factor and texture.
    evaluate_material = 1 << 2,
};
#ifdef FALCOR_SHARED_HOST
FALCOR_ENUM_CLASS_OPERATORS(OpacityFlags);
#endif

public struct MaterialHeader
{
    public uint type_id;
    public uint material_data;
    public uint opacity_texture;
    public uint opacity_data;
};

namespace detail {

struct MaterialHeaderDataFields
{
    typedef BitField<uint, 0, 8> flags;
    typedef BitField<uint, 8, 8> nested_priority;
};

struct MaterialOpacityDataFields
{
    /// Opacity flags.
    typedef BitField<uint, 0, 6> flags;
    /// RGBA texture channel.
    typedef BitField<uint, 6, 2> texture_channel;
    /// UNORM8 opacity threshold.
    typedef BitField<uint, 8, 8> threshold;
    /// UNORM16 opacity factor.
    typedef BitField<uint, 16, 16> factor;
};

#ifdef FALCOR_SHARED_HOST
#define FALCOR_SHARED_FUNCTION inline
#else
#define FALCOR_SHARED_FUNCTION public static
#endif

FALCOR_SHARED_FUNCTION MaterialFlags get_material_flags(MaterialHeader header)
{
    return MaterialFlags(MaterialHeaderDataFields::flags::extract(header.material_data));
}

FALCOR_SHARED_FUNCTION uint get_nested_priority(MaterialHeader header)
{
    return MaterialHeaderDataFields::nested_priority::extract(header.material_data);
}

FALCOR_SHARED_FUNCTION OpacityFlags get_opacity_flags(MaterialHeader header)
{
    return OpacityFlags(MaterialOpacityDataFields::flags::extract(header.opacity_data));
}

FALCOR_SHARED_FUNCTION uint get_opacity_texture_channel(MaterialHeader header)
{
    return MaterialOpacityDataFields::texture_channel::extract(header.opacity_data);
}

FALCOR_SHARED_FUNCTION float get_opacity_threshold(MaterialHeader header)
{
    return unpack_unorm8(MaterialOpacityDataFields::threshold::extract(header.opacity_data));
}

FALCOR_SHARED_FUNCTION float get_opacity_factor(MaterialHeader header)
{
    return unpack_unorm16(MaterialOpacityDataFields::factor::extract(header.opacity_data));
}

#ifdef FALCOR_SHARED_HOST
inline uint pack_material_data(MaterialFlags flags, uint nested_priority)
{
    uint data = 0;
    MaterialHeaderDataFields::flags::insert(data, uint(flags));
    MaterialHeaderDataFields::nested_priority::insert(data, nested_priority);
    return data;
}

inline uint pack_opacity_data(OpacityFlags flags, uint texture_channel, float threshold, float factor)
{
    uint data = 0;
    MaterialOpacityDataFields::flags::insert(data, uint(flags));
    MaterialOpacityDataFields::texture_channel::insert(data, texture_channel);
    MaterialOpacityDataFields::threshold::insert(data, pack_unorm8(threshold, PackOptions::safe));
    MaterialOpacityDataFields::factor::insert(data, pack_unorm16(factor, PackOptions::safe));
    return data;
}
#endif

#undef FALCOR_SHARED_FUNCTION

} // namespace detail

#ifndef FALCOR_SHARED_HOST
public extension MaterialHeader
{
    public property MaterialFlags flags
    {
        get {
            return detail::get_material_flags(this);
        }
    }

    public property uint nested_priority
    {
        get {
            return detail::get_nested_priority(this);
        }
    }

    public property OpacityFlags opacity_flags
    {
        get {
            return detail::get_opacity_flags(this);
        }
    }

    public property uint opacity_texture_channel
    {
        get {
            return detail::get_opacity_texture_channel(this);
        }
    }

    public property float opacity_threshold
    {
        get {
            return detail::get_opacity_threshold(this);
        }
    }

    public property float opacity_factor
    {
        get {
            return detail::get_opacity_factor(this);
        }
    }
}
#endif

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
    sphere_light,
    disk_light,
    rect_light,
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
