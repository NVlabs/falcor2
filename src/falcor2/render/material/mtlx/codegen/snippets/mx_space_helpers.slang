// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing mtlx_snippets;

// TODO(mtlx-cleanup): Decide whether these transform helpers should live in a shared runtime Slang library once
// the rewrite path is no longer mirroring legacy source emission.
float3 mx_transform_point(float3 p, GeomPropSpace from_space, GeomPropSpace to_space, SurfaceInteraction si)
{
    if ((from_space == GeomPropSpace::model || from_space == GeomPropSpace::object)
        && to_space == GeomPropSpace::world) {
        let world_from_object = si.get_geomprop_accessor().get_space_transform(to_space, from_space);
        float4 hp = mul(world_from_object, float4(p, 1.0));
        return hp.xyz / hp.w;
    }
    if (from_space == GeomPropSpace::world && (to_space == GeomPropSpace::model || to_space == GeomPropSpace::object)) {
        let object_from_world = si.get_geomprop_accessor().get_space_transform(to_space, from_space);
        float4 hp = mul(object_from_world, float4(p, 1.0));
        return hp.xyz / hp.w;
    }
    return p;
}

float3 mx_transform_vector(float3 v, GeomPropSpace from_space, GeomPropSpace to_space, SurfaceInteraction si)
{
    if ((from_space == GeomPropSpace::model || from_space == GeomPropSpace::object)
        && to_space == GeomPropSpace::world) {
        let world_from_object = si.get_geomprop_accessor().get_space_transform(to_space, from_space);
        return mul(world_from_object, float4(v, 0.0)).xyz;
    }
    if (from_space == GeomPropSpace::world && (to_space == GeomPropSpace::model || to_space == GeomPropSpace::object)) {
        let object_from_world = si.get_geomprop_accessor().get_space_transform(to_space, from_space);
        return mul(object_from_world, float4(v, 0.0)).xyz;
    }
    return v;
}

float3 mx_transform_normal(float3 n, GeomPropSpace from_space, GeomPropSpace to_space, SurfaceInteraction si)
{
    // Match MaterialX's GLSL reference: normalize even for same-space or unavailable-transform identity paths.
    if ((from_space == GeomPropSpace::model || from_space == GeomPropSpace::object)
        && to_space == GeomPropSpace::world) {
        let world_from_object_it
            = (float3x3)transpose(si.get_geomprop_accessor().get_space_transform(from_space, to_space));
        return normalize(mul(world_from_object_it, n));
    }
    if (from_space == GeomPropSpace::world && (to_space == GeomPropSpace::model || to_space == GeomPropSpace::object)) {
        let object_from_world_it
            = (float3x3)transpose(si.get_geomprop_accessor().get_space_transform(from_space, to_space));
        return normalize(mul(object_from_world_it, n));
    }
    return normalize(n);
}
