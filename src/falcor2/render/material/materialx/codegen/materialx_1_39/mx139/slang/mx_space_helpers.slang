// SPDX-License-Identifier: Apache-2.0

implementing mx139_snippets;

// TODO(mx139-cleanup): Decide whether these transform helpers should live in a shared runtime Slang library once
// the rewrite path is no longer mirroring legacy source emission.
float3
mx_transform_point(float3 p, int from_space, int to_space, float4x4 object_from_world, float4x4 world_from_object)
{
    if ((from_space == mx_space_model || from_space == mx_space_object) && to_space == mx_space_world) {
        float4 hp = mul(world_from_object, float4(p, 1.0));
        return hp.xyz / hp.w;
    }
    if (from_space == mx_space_world && (to_space == mx_space_model || to_space == mx_space_object)) {
        float4 hp = mul(object_from_world, float4(p, 1.0));
        return hp.xyz / hp.w;
    }
    return p;
}

float3
mx_transform_vector(float3 v, int from_space, int to_space, float4x4 object_from_world, float4x4 world_from_object)
{
    if ((from_space == mx_space_model || from_space == mx_space_object) && to_space == mx_space_world) {
        return mul(world_from_object, float4(v, 0.0)).xyz;
    }
    if (from_space == mx_space_world && (to_space == mx_space_model || to_space == mx_space_object)) {
        return mul(object_from_world, float4(v, 0.0)).xyz;
    }
    return v;
}

float3 mx_transform_normal(
    float3 n,
    int from_space,
    int to_space,
    float3x3 object_from_world_it,
    float3x3 world_from_object_it
)
{
    if ((from_space == mx_space_model || from_space == mx_space_object) && to_space == mx_space_world) {
        return mul(world_from_object_it, n);
    }
    if (from_space == mx_space_world && (to_space == mx_space_model || to_space == mx_space_object)) {
        return mul(object_from_world_it, n);
    }
    return n;
}
