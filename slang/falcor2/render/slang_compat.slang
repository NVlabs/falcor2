// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

// This module implements a compatibility layer for things that should ideally be fixed in Slang itself.

// Define a GeometryIndex2() which works the same as GeometryIndex() but is also implemented for OptiX.
// This is needed because Slang stdlib doesn't implement GeometryIndex() for OptiX yet.
#ifdef __TARGET_CUDA__
[require(cuda)]
public uint GeometryIndex2()
{
    // OptiX returns the GAS index (==GeometryIndex) if sbtIndexOffsetBuffer is null for all geometries in an AS.
    __intrinsic_asm "optixGetSbtGASIndex()";
}
#else  // __TARGET_CUDA__
public uint GeometryIndex2()
{
    return GeometryIndex();
}
#endif // __TARGET_CUDA__

// clang-format off
/// Extend RayQuery with API for querying linear swept sphere (LSS) geometry.
/// These are not yet available as built-in intrinsics in Slang, so we define them here using inline assembly.
__generic<let RayFlags : RAY_FLAG> extension RayQuery<RayFlags> {
    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    bool CandidateIsNonOpaqueLss()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCandidateIsNonOpaqueLss($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 0; // 0 for candidate
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    result:$$bool = OpRayQueryIsLSSHitNV &this $iCandidateOrCommitted;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    bool CommittedIsLss()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCommittedIsLss($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 1; // 1 for committed
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    result:$$bool = OpRayQueryIsLSSHitNV &this $iCandidateOrCommitted;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    float CandidateBuiltinPrimitiveRayT()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCandidateBuiltinPrimitiveRayT($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 0; // 0 for candidate
                return spirv_asm
                {
                    result:$$float = OpRayQueryGetIntersectionTKHR &this $iCandidateOrCommitted;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    float2x4 CandidateLssPositionsAndRadii()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCandidateLssObjectPositionsAndRadii($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 0; // 0 for candidate
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    OpCapability RayTracingSpheresGeometryNV;
                    %positions:$$float3[2] = OpRayQueryGetIntersectionLSSPositionsNV &this $iCandidateOrCommitted;
                    %radii:$$float[2] = OpRayQueryGetIntersectionLSSRadiiNV &this $iCandidateOrCommitted;
                    %r0:$$float = OpCompositeExtract %radii 0;
                    %r1:$$float = OpCompositeExtract %radii 1;
                    %p0:$$float3 = OpCompositeExtract %positions 0;
                    %p1:$$float3 = OpCompositeExtract %positions 1;
                    %a:$$float4 = OpCompositeConstruct %p0 %r0;
                    %b:$$float4 = OpCompositeConstruct %p1 %r1;
                    result:$$float2x4 = OpCompositeConstruct %a %b;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    float2x4 CommittedLssPositionsAndRadii()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCommittedLssObjectPositionsAndRadii($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 1; // 1 for committed
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    OpCapability RayTracingSpheresGeometryNV;
                    %positions:$$float3[2] = OpRayQueryGetIntersectionLSSPositionsNV &this $iCandidateOrCommitted;
                    %radii:$$float[2] = OpRayQueryGetIntersectionLSSRadiiNV &this $iCandidateOrCommitted;
                    %r0:$$float = OpCompositeExtract %radii 0;
                    %r1:$$float = OpCompositeExtract %radii 1;
                    %p0:$$float3 = OpCompositeExtract %positions 0;
                    %p1:$$float3 = OpCompositeExtract %positions 1;
                    %a:$$float4 = OpCompositeConstruct %p0 %r0;
                    %b:$$float4 = OpCompositeConstruct %p1 %r1;
                    result:$$float2x4 = OpCompositeConstruct %a %b;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    float CandidateLssHitParameter()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCandidateLssHitParameter($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 0; // 0 for candidate
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    result:$$float = OpRayQueryGetIntersectionLSSHitValueNV &this $iCandidateOrCommitted;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    float CommittedLssHitParameter()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCommittedLssHitParameter($0.RayFlags())";
            case spirv:
                uint iCandidateOrCommitted = 1; // 1 for committed
                return spirv_asm
                {
                    OpExtension "SPV_NV_linear_swept_spheres";
                    OpCapability RayTracingLinearSweptSpheresGeometryNV;
                    result:$$float = OpRayQueryGetIntersectionLSSHitValueNV &this $iCandidateOrCommitted;
                };
        }
    }

    [__requiresNVAPI]
    [__NoSideEffect]
    [NonUniformReturn]
    [require(hlsl_spirv, rayquery)]
    void CommitNonOpaqueBuiltinPrimitiveHit()
    {
        __target_switch
        {
            case hlsl : __intrinsic_asm "__NvRtCommitNonOpaqueBuiltinPrimitiveHit($0.RayFlags())";
            case spirv: spirv_asm { OpRayQueryConfirmIntersectionKHR &this };
        }
    }

};
// clang-format on
