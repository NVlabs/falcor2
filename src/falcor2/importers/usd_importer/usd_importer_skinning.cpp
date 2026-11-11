// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_skinning.h"
#include "falcor2/core/error.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/interval.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/bakeSkinning.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
END_DISABLE_USD_WARNINGS

#include <utility>
#include <vector>

namespace falcor::usd_importer {

using namespace pxr;

void bake_skinning(const UsdStageRefPtr& stage, UsdTimeCode time)
{
    UsdSkelCache cache;
    bool has_skinning = false;
    SdfPathVector skinned_meshes;
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
                for (const UsdSkelSkinningQuery& target : binding.GetSkinningTargets()) {
                    // OpenUSD cannot author baked values on instance proxies.
                    FALCOR_CHECK(
                        !target.GetPrim().IsInstanceProxy() && !prim.IsInstance(),
                        "Cannot bake instanced USD skinning at {}",
                        target.GetPrim().GetPath().GetString()
                    );
                    has_skinning = true;
                    if (target.HasJointInfluences() && target.GetPrim().IsA<UsdGeomMesh>())
                        skinned_meshes.push_back(target.GetPrim().GetPath());
                }
            }
        }
    }
    if (!has_skinning)
        return;
    cache.Clear();

    // Static instances stay shared. Freezing animated prototype children would require
    // expanding instances, since their attributes cannot be overridden individually.
    for (const UsdPrim& prototype : stage->GetPrototypes()) {
        for (const UsdPrim& prim : UsdPrimRange(prototype)) {
            for (const UsdAttribute& attr : prim.GetAttributes())
                FALCOR_CHECK(
                    attr.GetNumTimeSamples() == 0,
                    "Cannot freeze animated USD prototype attribute {}",
                    attr.GetPath().GetString()
                );
        }
    }

    // Strong default values hide weaker animation samples. Clear only values in our private
    // layer, keeping cached geometry and transforms at the same time as the baked skin.
    // Sample before baking: BakeSkinning alone can use the rest pose when the requested
    // interval contains no authored animation samples.
    for (const UsdPrim& prim : stage->Traverse()) {
        for (const UsdAttribute& attr : prim.GetAttributes()) {
            if (attr.GetNumTimeSamples() == 0)
                continue;
            VtValue value;
            if (!attr.Get(&value, time))
                continue;
            if (value.IsHolding<SdfAssetPath>()) {
                const auto& asset = value.UncheckedGet<SdfAssetPath>();
                if (!asset.GetResolvedPath().empty())
                    value = SdfAssetPath(asset.GetResolvedPath());
            } else if (value.IsHolding<VtArray<SdfAssetPath>>()) {
                auto assets = value.UncheckedGet<VtArray<SdfAssetPath>>();
                for (SdfAssetPath& asset : assets) {
                    if (!asset.GetResolvedPath().empty())
                        asset = SdfAssetPath(asset.GetResolvedPath());
                }
                value = std::move(assets);
            }
            FALCOR_CHECK(attr.Clear() && attr.Set(value), "Cannot freeze USD attribute {}", attr.GetPath().GetString());
        }
    }
    // OpenUSD bakes built-in normals, while the importer prefers primvars:normals.
    // Preserve authored polygon normals by adapting them in the private layer first.
    for (const SdfPath& path : skinned_meshes) {
        UsdGeomMesh mesh(stage->GetPrimAtPath(path));
        TfToken scheme;
        mesh.GetSubdivisionSchemeAttr().Get(&scheme);
        if (scheme != UsdGeomTokens->none)
            continue;
        const UsdGeomPrimvar primvar = UsdGeomPrimvarsAPI(mesh).GetPrimvar(TfToken("normals"));
        if (!primvar || !primvar.HasValue())
            continue;
        const TfToken interpolation = primvar.GetInterpolation();
        FALCOR_CHECK(
            interpolation == UsdGeomTokens->vertex || interpolation == UsdGeomTokens->varying
                || interpolation == UsdGeomTokens->faceVarying,
            "Cannot bake USD normal primvar {} with interpolation {}",
            primvar.GetAttr().GetPath().GetString(),
            interpolation.GetString()
        );
        VtVec3fArray normals;
        FALCOR_CHECK(
            primvar.GetElementSize() == 1 && primvar.ComputeFlattened(&normals) && mesh.CreateNormalsAttr().Set(normals)
                && mesh.SetNormalsInterpolation(interpolation),
            "Cannot adapt USD normal primvar {} for skinning",
            primvar.GetAttr().GetPath().GetString()
        );
        primvar.GetAttr().Block();
    }
    // This overload writes to the current edit layer without saving it. OpenUSD owns
    // bind transforms, joint remapping, blend shapes and deformation before tessellation.
    FALCOR_CHECK(
        UsdSkelBakeSkinning(stage->Traverse(), GfInterval(time.GetValue(), time.GetValue())),
        "Cannot bake USD skinning at time {}",
        time.GetValue()
    );
    for (const SdfPath& path : skinned_meshes) {
        const UsdPrim prim = stage->GetPrimAtPath(path);
        FALCOR_CHECK(
            prim.CreateAttribute(TfToken("refinementEnableOverride"), SdfValueTypeNames->Bool).Set(true)
                && prim.CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Int).Set(0),
            "Cannot disable refinement for baked USD mesh {}",
            path.GetString()
        );
    }
}

} // namespace falcor::usd_importer
