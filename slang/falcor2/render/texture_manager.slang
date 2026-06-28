// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;

/// DescriptorHandle cannot be used in a dynamic dispatch context for now.
/// This wrapper is a workaround for that. Once Slang is fixed we can replace all the wrapped
/// handles and remove the `.unwrap()` call to turn them into actual descriptor handles.
/// https://github.com/shader-slang/slang/issues/9651
public struct DescriptorHandleWAR<T : IOpaqueDescriptor>
{
#ifdef __TARGET_CUDA__
    public uint64_t handle;
#else
    public uint2 handle;
#endif
    DescriptorHandle<T> unwrap() { return DescriptorHandle<T>(handle); }
};


public interface ILodSampler
{
    public __generic<T : ITexelElement> T sample(Texture2D<T> texture, SamplerState sampler, float2 uv);
    public __generic<T : ITexelElement> T sample(Texture3D<T> texture, SamplerState sampler, float3 uvw);
}

public struct ImplicitLodSampler : ILodSampler
{
    public __init() { }

    public __generic<T : ITexelElement> T sample(Texture2D<T> texture, SamplerState sampler, float2 uv)
    {
        return texture.Sample(sampler, uv);
    }

    public __generic<T : ITexelElement> T sample(Texture3D<T> texture, SamplerState sampler, float3 uvw)
    {
        return texture.Sample(sampler, uvw);
    }
}

public struct ExplicitLodSampler : ILodSampler
{
    public float lod = 0;

    public __init() { }
    public __init(float lod) { this.lod = lod; }

    public __generic<T : ITexelElement> T sample(Texture2D<T> texture, SamplerState sampler, float2 uv)
    {
        return texture.SampleLevel(sampler, uv, lod);
    }

    public __generic<T : ITexelElement> T sample(Texture3D<T> texture, SamplerState sampler, float3 uvw)
    {
        return texture.SampleLevel(sampler, uvw, lod);
    }
}

public extern static const bool FALCOR_TEXTURE_COORDINATE_ORIGIN_LOWER_LEFT = false;

public enum class TextureHandleFlags : uint {
    none = 0,
    udim = 0x1,
    uv_origin_correction = 0x2,
    all = 0x3,
};

// TODO(TCO-FUT-001): Use TextureHandleFlags directly once SlangPy treats enum-valued
// generic arguments as concrete value arguments rather than unknown type arguments.
// `uint(TextureHandleFlags::all)` is the desired spelling, but it currently fails
// in this generic default before SlangPy binding gets involved.
__generic<T : ITexelElement = float4, uint FlagsAsUint = 0x3 /*TextureHandleFlags::all*/> public struct TextureHandle
{
    public static const TextureHandleFlags Flags = TextureHandleFlags(FlagsAsUint);

    public static const bool ENABLE_UDIM = is_set(Flags, TextureHandleFlags::udim);
    public static const bool FLIP_V_COORDINATE
        = is_set(Flags, TextureHandleFlags::uv_origin_correction) && FALCOR_TEXTURE_COORDINATE_ORIGIN_LOWER_LEFT;

    public uint data = detail::INVALID_TEXTURE_HANDLE;

    public __init() { }

    public __init(uint data) { this.data = data; }

    public __generic<uint OtherFlagsAsUint> __init(TextureHandle<T, OtherFlagsAsUint> other) { this.data = other.data; }

    public property uint sampler_index
    {
        get {
            return detail::TextureHandleFields::sampler::extract(data);
        }
    }

    public property uint texture_index
    {
        get {
            return detail::TextureHandleFields::texture::extract(data);
        }
    }

    public bool is_valid() { return data != detail::INVALID_TEXTURE_HANDLE; }

    public bool is_udim() { return ENABLE_UDIM && (detail::TextureHandleFields::udim::extract(data) != 0); }

    public __generic<LodSampler : ILodSampler>
        T sample(LodSampler lod_sampler, float2 uv, const T default_value = T(T.Element(0)))
    {
        if (!is_valid())
            return default_value;
        This resolved_handle = resolve_udim(uv);
        if (!resolved_handle.is_valid())
            return default_value;
        if (FLIP_V_COORDINATE)
            uv.y = 1.0 - uv.y;
#ifdef __TARGET_CUDA__
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint64_t(resolved_handle.texture_index));
        SamplerState sampler;
#else
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint2(resolved_handle.texture_index, 0));
        SamplerState sampler = DescriptorHandle<SamplerState>(uint2(resolved_handle.sampler_index, 0));
#endif
        return lod_sampler.sample(texture, sampler, uv);
    }

    public __generic<LodSampler : ILodSampler>
        T sample(LodSampler lod_sampler, float3 uvw, const T default_value = T(T.Element(0)))
    {
        if (!is_valid())
            return default_value;
#ifdef __TARGET_CUDA__
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint64_t(texture_index));
        SamplerState sampler;
#else
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint2(texture_index, 0));
        SamplerState sampler = DescriptorHandle<SamplerState>(uint2(sampler_index, 0));
#endif
        return lod_sampler.sample(texture, sampler, uvw);
    }

    public T load(int2 coord, int miplevel = 0, const T default_value = T(T.Element(0)))
    {
        if (!is_valid() || is_udim())
            return default_value;
#ifdef __TARGET_CUDA__
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint64_t(texture_index));
#else
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint2(texture_index, 0));
#endif
        return texture.Load(int3(coord, miplevel));
    }

    public T load(int3 coord, int miplevel = 0, const T default_value = T(T.Element(0)))
    {
        if (!is_valid() || is_udim())
            return default_value;
#ifdef __TARGET_CUDA__
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint64_t(texture_index));
#else
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint2(texture_index, 0));
#endif
        return texture.Load(int4(coord, miplevel));
    }

    public bool get_dimensions(float2 uv, uint miplevel, out uint width, out uint height, out uint mip_levels)
    {
        width = {};
        height = {};
        mip_levels = {};

        if (!is_valid())
            return false;
        This resolved_handle = resolve_udim(uv);
        if (!resolved_handle.is_valid())
            return false;

        return get_dimensions(miplevel, width, height, mip_levels);
    }

    public bool get_dimensions(uint miplevel, out uint width, out uint height, out uint mip_levels)
    {
        width = {};
        height = {};
        mip_levels = {};

        if (!is_valid() || is_udim())
            return false;

#ifdef __TARGET_CUDA__
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint64_t(texture_index));
#else
        Texture2D<T> texture = DescriptorHandle<Texture2D<T>>(uint2(texture_index, 0));
#endif
        texture.GetDimensions(miplevel, width, height, mip_levels);
        return true;
    }

    public bool get_dimensions(out uint width, out uint height, out uint depth)
    {
        width = {};
        height = {};
        depth = {};

        if (!is_valid())
            return false;

#ifdef __TARGET_CUDA__
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint64_t(texture_index));
#else
        Texture3D<T> texture = DescriptorHandle<Texture3D<T>>(uint2(texture_index, 0));
#endif

        texture.GetDimensions(width, height, depth);

        return true;
    }

    public This resolve_udim(inout float2 uv)
    {
        if (ENABLE_UDIM && is_udim()) {
            int2 tile = int2(floor(uv));
#ifdef __TARGET_CUDA__
            Texture2D<uint> indirection_texture = DescriptorHandle<Texture2D<uint>>(uint64_t(texture_index));
#else
            Texture2D<uint> indirection_texture = DescriptorHandle<Texture2D<uint>>(uint2(texture_index, 0));
#endif
            int2 dim;
            indirection_texture.GetDimensions(dim.x, dim.y);
            if (any(tile < 0) || any(tile >= dim))
                return This();
            uv = frac(uv);
            return This(indirection_texture[tile]);
        } else {
            return this;
        }
    }
};
