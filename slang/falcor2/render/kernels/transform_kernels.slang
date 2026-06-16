// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

[shader("compute")]
[numthreads(32, 1, 1)]
void compute_inverse(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<float4x4> transforms,
    RWStructuredBuffer<float4x4> inverse_transforms,
    uniform uint first,
    uniform uint count
)
{
    if (tid.x >= count)
        return;
    uint index = tid.x + first;
    float4x4 transform = transforms[index];
    inverse_transforms[index] = inverse(transform);
}

#ifdef __TARGET_CUDA__

struct AccelerationStructureInstanceDesc
{
    float3x4 transform;
    uint32_t instance_id;
    uint32_t sbt_offset;
    uint32_t visibility_mask;
    uint32_t flags;
    uint64_t traversable_handle;
    uint32_t _pad[2];
};

#else

struct AccelerationStructureInstanceDesc
{
    float3x4 transform;
    uint32_t instance_id : 24;
    uint32_t instance_mask : 8;
    uint32_t instance_contribution_to_hit_group_index : 24;
    uint32_t flags : 8;
    uint64_t acceleration_structure;
};

#endif


[shader("compute")]
[numthreads(32, 1, 1)]
void update_tlas_instance_desc_transforms(
    uint3 tid: SV_DispatchThreadID,
    StructuredBuffer<InstanceData> instance_data,
    StructuredBuffer<float4x4> world_from_object,
    RWStructuredBuffer<AccelerationStructureInstanceDesc> as_instance_descs,
    uniform uint first,
    uniform uint count
)
{
    if (tid.x >= count)
        return;
    uint index = tid.x + first;
    float4x4 transform = world_from_object[uint(instance_data[index].transform_id)];
    as_instance_descs[index].transform = float3x4(transform);
}
