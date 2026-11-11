// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_skinning.h"
#include "usd_importer_utils.h"
#include "falcor2/core/error.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdSkel/animMapper.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/utils.h>
END_DISABLE_USD_WARNINGS

#include <vector>

namespace falcor::usd_importer {
namespace {

using namespace pxr;

void bake_mesh_animated(const UsdSkelSkeletonQuery& skeleton, const UsdSkelSkinningQuery& skinning)
{
    const UsdPrim prim = skinning.GetPrim();
    const std::string path = prim.GetPath().GetString();
    UsdGeomMesh mesh(prim);
    FALCOR_CHECK(mesh, "Animated USD skinning requires a mesh at {}", path);
    FALCOR_CHECK(!skinning.HasBlendShapes(), "Animated USD skinning does not support blend shapes at {}", path);

    TfToken scheme;
    mesh.GetSubdivisionSchemeAttr().Get(&scheme);
    const UsdGeomPrimvar normal_primvar = UsdGeomPrimvarsAPI(mesh).GetPrimvar(TfToken("normals"));
    const bool use_primvar = normal_primvar && normal_primvar.HasValue();
    const UsdAttribute normal_attribute = scheme == UsdGeomTokens->none
        ? (use_primvar ? normal_primvar.GetAttr() : mesh.GetNormalsAttr())
        : UsdAttribute();
    const bool has_normals = normal_attribute && normal_attribute.HasValue();
    const TfToken interpolation = use_primvar ? normal_primvar.GetInterpolation() : mesh.GetNormalsInterpolation();
    if (has_normals) {
        FALCOR_CHECK(
            interpolation == UsdGeomTokens->vertex || interpolation == UsdGeomTokens->varying
                || interpolation == UsdGeomTokens->faceVarying,
            "Cannot bake USD normals with interpolation {} at {}",
            interpolation.GetString(),
            path
        );
        FALCOR_CHECK(!use_primvar || normal_primvar.GetElementSize() == 1, "Invalid normal element size at {}", path);
    }

    // Geometry owns the schedule. Skeleton keys and stage playback frames must not add samples.
    std::vector<UsdAttribute> attributes = {mesh.GetPointsAttr()};
    if (has_normals)
        attributes.push_back(normal_attribute);
    std::vector<double> times;
    UsdAttribute::GetUnionedTimeSamples(attributes, &times);
    const bool has_samples = !times.empty();
    if (!has_samples)
        times.push_back(0);

    // Flattening normals must not hide changing primvar indices from the mesh importer.
    if (has_normals && use_primvar && normal_primvar.IsIndexed()) {
        VtIntArray indices;
        read_fixed_indices(normal_primvar.GetIndicesAttr(), indices, times.front(), true);
    }

    // Read and deform all samples before authoring: a stronger baked sample would otherwise
    // replace the source time samples used by subsequent reads.
    std::vector<VtVec3fArray> positions(times.size()), normals(has_normals ? times.size() : 0);
    for (size_t sample = 0; sample < times.size(); ++sample) {
        const UsdTimeCode time(times[sample]);
        VtMatrix4dArray transforms;
        FALCOR_CHECK(skeleton.ComputeSkinningTransforms(&transforms, time), "Cannot evaluate USD skeleton at {}", path);
        if (const auto& mapper = skinning.GetJointMapper()) {
            VtMatrix4dArray ordered_transforms;
            FALCOR_CHECK(
                mapper->RemapTransforms(transforms, &ordered_transforms),
                "Cannot remap USD joints at {}",
                path
            );
            transforms.swap(ordered_transforms);
        }
        auto& points = positions[sample];
        FALCOR_CHECK(mesh.GetPointsAttr().Get(&points, time), "Cannot read USD points at {}", path);
        VtIntArray joint_indices;
        VtFloatArray joint_weights;
        FALCOR_CHECK(
            skinning.ComputeVaryingJointInfluences(points.size(), &joint_indices, &joint_weights, time),
            "Cannot read USD joint influences at {}",
            path
        );
        const GfMatrix4d bind_transform = skinning.GetGeomBindTransform(time);
        FALCOR_CHECK(
            UsdSkelSkinPoints(
                skinning.GetSkinningMethod(),
                bind_transform,
                transforms,
                joint_indices,
                joint_weights,
                skinning.GetNumInfluencesPerComponent(),
                points
            ),
            "Cannot skin USD points at {}",
            path
        );

        // USD skinning produces skeleton-space values. Keep the mesh's existing transform.
        UsdGeomXformCache xforms(time);
        const GfMatrix4d skeleton_to_mesh
            = xforms.GetLocalToWorldTransform(skeleton.GetPrim()) * xforms.GetLocalToWorldTransform(prim).GetInverse();
        for (GfVec3f& point : points)
            point = GfVec3f(skeleton_to_mesh.Transform(point));

        if (has_normals) {
            auto& output_normals = normals[sample];
            FALCOR_CHECK(
                use_primvar ? normal_primvar.ComputeFlattened(&output_normals, time)
                            : normal_attribute.Get(&output_normals, time),
                "Cannot read USD normals at {}",
                path
            );
            VtMatrix3dArray normal_transforms(transforms.size());
            for (size_t joint = 0; joint < transforms.size(); ++joint)
                normal_transforms[joint] = transforms[joint].ExtractRotationMatrix().GetInverse().GetTranspose();
            const GfMatrix3d normal_bind = bind_transform.ExtractRotationMatrix().GetInverse().GetTranspose();
            bool success;
            if (interpolation == UsdGeomTokens->faceVarying) {
                VtIntArray face_indices;
                FALCOR_CHECK(
                    mesh.GetFaceVertexIndicesAttr().Get(&face_indices, time),
                    "Cannot read USD face indices at {}",
                    path
                );
                success = UsdSkelSkinFaceVaryingNormals(
                    skinning.GetSkinningMethod(),
                    normal_bind,
                    normal_transforms,
                    joint_indices,
                    joint_weights,
                    skinning.GetNumInfluencesPerComponent(),
                    face_indices,
                    output_normals
                );
            } else {
                success = UsdSkelSkinNormals(
                    skinning.GetSkinningMethod(),
                    normal_bind,
                    normal_transforms,
                    joint_indices,
                    joint_weights,
                    skinning.GetNumInfluencesPerComponent(),
                    output_normals
                );
            }
            FALCOR_CHECK(success, "Cannot skin USD normals at {}", path);
            const GfMatrix3d normal_to_mesh = skeleton_to_mesh.ExtractRotationMatrix().GetInverse().GetTranspose();
            for (GfVec3f& normal : output_normals)
                normal = GfVec3f(normal * normal_to_mesh);
        }
    }

    for (size_t sample = 0; sample < times.size(); ++sample) {
        const UsdTimeCode time = has_samples ? UsdTimeCode(times[sample]) : UsdTimeCode::Default();
        FALCOR_CHECK(mesh.GetPointsAttr().Set(positions[sample], time), "Cannot write baked USD points at {}", path);
        if (has_normals)
            FALCOR_CHECK(
                mesh.CreateNormalsAttr().Set(normals[sample], time),
                "Cannot write baked USD normals at {}",
                path
            );
    }
    if (has_normals) {
        FALCOR_CHECK(mesh.SetNormalsInterpolation(interpolation), "Cannot set USD normal interpolation at {}", path);
        if (use_primvar)
            normal_primvar.GetAttr().Block();
    }
    FALCOR_CHECK(
        prim.CreateAttribute(TfToken("refinementEnableOverride"), SdfValueTypeNames->Bool).Set(true)
            && prim.CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Int).Set(0),
        "Cannot disable refinement for baked USD mesh {}",
        path
    );
}

} // namespace

void bake_skinning_animated(const pxr::UsdStageRefPtr& stage)
{
    using namespace pxr;
    // Samples authored in the private layer must use the stage's time-code units.
    stage->GetEditTarget().GetLayer()->SetTimeCodesPerSecond(stage->GetTimeCodesPerSecond());
    UsdSkelCache cache;
    const auto predicate = UsdTraverseInstanceProxies();
    for (const UsdPrim& prim : stage->Traverse(predicate)) {
        if (auto root = UsdSkelRoot(prim)) {
            FALCOR_CHECK(
                cache.Populate(root, predicate),
                "Cannot populate USD skeleton {}",
                prim.GetPath().GetString()
            );
            std::vector<UsdSkelBinding> bindings;
            FALCOR_CHECK(
                cache.ComputeSkelBindings(root, &bindings, predicate),
                "Cannot resolve USD skeleton bindings for {}",
                prim.GetPath().GetString()
            );
            for (const UsdSkelBinding& binding : bindings) {
                const UsdSkelSkeletonQuery skeleton = cache.GetSkelQuery(binding.GetSkeleton());
                for (const UsdSkelSkinningQuery& target : binding.GetSkinningTargets()) {
                    FALCOR_CHECK(
                        !target.GetPrim().IsInstanceProxy() && !prim.IsInstance(),
                        "Cannot bake instanced USD skinning at {}",
                        target.GetPrim().GetPath().GetString()
                    );
                    bake_mesh_animated(skeleton, target);
                }
            }
        }
    }
}

} // namespace falcor::usd_importer
