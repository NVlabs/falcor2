// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_context.h"
#include "usd_importer_mesh_tesselator.h"
#include "usd_importer_curve_tesselator.h"
#include "usd_importer_utils.h"
#include "usd_importer_macros.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/usd_importer/usd_importer_material.h"
#include "falcor2/render/transform.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/primRange.h>

#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/scope.h>

#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/blackbody.h>

#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/animation.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace falcor {
namespace usd_importer {
namespace {
template<typename TUsdLuxLightType>
float3 get_light_intensity(const TUsdLuxLightType& light, const pxr::UsdPrim& prim)
{
    using namespace pxr;

    float exposure = get_authored_attribute(light.GetExposureAttr(), prim.GetAttribute(TfToken("exposure")), 0.f);
    float intensity = get_authored_attribute(light.GetIntensityAttr(), prim.GetAttribute(TfToken("intensity")), 1.f);
    GfVec3f blackbody_RGB(1.f, 1.f, 1.f);
    if (get_authored_attribute(
            light.GetEnableColorTemperatureAttr(),
            prim.GetAttribute(TfToken("enableColorTemperature")),
            false
        )) {
        float temperature = get_authored_attribute(
            light.GetColorTemperatureAttr(),
            prim.GetAttribute(TfToken("colorTemperature")),
            6500.f
        );

        blackbody_RGB = UsdLuxBlackbodyTemperatureAsRgb(temperature);
    }
    GfVec3f color
        = get_authored_attribute(light.GetColorAttr(), prim.GetAttribute(TfToken("color")), GfVec3f(1.f, 1.f, 1.f));
    return std::exp2(exposure) * intensity * to_falcor(blackbody_RGB) * to_falcor(color);
}

float distant_light_size_factor(float degree_angular_diameter)
{
    // UsdLux normalization divides emitted radiance by this projected solid-angle factor.
    // See https://openusd.org/release/user_guides/schemas/usdLux/LightAPI.html#inputs-normalize
    // Distance lights normalization.
    const float pi = static_cast<float>(M_PI);
    const float half_angle = std::clamp(sgl::math::radians(degree_angular_diameter) * 0.5f, 0.f, pi);
    if (half_angle == 0.f)
        return 1.f;

    const float sin_half_angle = std::sin(half_angle);
    const float sin_squared = sin_half_angle * sin_half_angle;
    return pi * (half_angle <= 0.5f * pi ? sin_squared : 2.f - sin_squared);
}

struct LightWorldAxes {
    float3 x;
    float3 y;
    float3 z;
};

LightWorldAxes get_light_world_axes(const pxr::UsdPrim& prim)
{
    const float4x4 world_from_light
        = to_falcor(pxr::UsdGeomXformable(prim).ComputeLocalToWorldTransform(pxr::UsdTimeCode::Default()));
    return {
        sgl::math::transform_vector(world_from_light, float3(1.f, 0.f, 0.f)),
        sgl::math::transform_vector(world_from_light, float3(0.f, 1.f, 0.f)),
        sgl::math::transform_vector(world_from_light, float3(0.f, 0.f, 1.f)),
    };
}

float planar_light_size_factor(const pxr::UsdPrim& prim, float local_area)
{
    const LightWorldAxes axes = get_light_world_axes(prim);
    const float world_area_scale = sgl::math::length(sgl::math::cross(axes.x, axes.y));
    const float world_area = sgl::math::abs(local_area) * world_area_scale;
    return world_area > 0.f && sgl::math::isfinite(world_area) ? world_area : 1.f;
}

float sphere_light_size_factor(const pxr::UsdPrim& prim, float radius)
{
    const LightWorldAxes axes = get_light_world_axes(prim);
    const float x_scale_squared = sgl::math::dot(axes.x, axes.x);
    const float y_scale_squared = sgl::math::dot(axes.y, axes.y);
    const float z_scale_squared = sgl::math::dot(axes.z, axes.z);
    const float max_scale_squared = sgl::math::max(x_scale_squared, sgl::math::max(y_scale_squared, z_scale_squared));
    const float tolerance = max_scale_squared * 1e-6f;
    const bool is_uniform_scale = max_scale_squared > 0.f && sgl::math::isfinite(max_scale_squared)
        && sgl::math::abs(x_scale_squared - y_scale_squared) <= tolerance
        && sgl::math::abs(x_scale_squared - z_scale_squared) <= tolerance
        && sgl::math::abs(sgl::math::dot(axes.x, axes.y)) <= tolerance
        && sgl::math::abs(sgl::math::dot(axes.x, axes.z)) <= tolerance
        && sgl::math::abs(sgl::math::dot(axes.y, axes.z)) <= tolerance;

    float world_area_scale = 1.f;
    if (is_uniform_scale) {
        world_area_scale = (x_scale_squared + y_scale_squared + z_scale_squared) / 3.f;
    } else {
        // OpenUSD recommends using light size attributes instead of non-uniform transforms because the latter
        // complicate light sampling and integration. Preserve the authored spherical area in that case.
        sgl::log_warn(
            "SphereLight '{}' has a non-uniform scale or shear; ignoring transform scaling for light "
            "normalization. Use the radius attribute to size the light.",
            prim.GetPath().GetString()
        );
    }

    const float pi = static_cast<float>(M_PI);
    const float world_area = 4.f * pi * radius * radius * world_area_scale;
    return world_area > 0.f && sgl::math::isfinite(world_area) ? world_area : 1.f;
}

template<typename TUsdLuxLightType>
bool is_light_normalized(const TUsdLuxLightType& light, const pxr::UsdPrim& prim)
{
    return get_authored_attribute(light.GetNormalizeAttr(), prim.GetAttribute(pxr::TfToken("normalize")), false);
}

quatf euler_to_quat(const pxr::GfVec3f& degrees, pxr::UsdGeomXformCommonAPI::RotationOrder order)
{
    float rx = sgl::math::radians(degrees[0]);
    float ry = sgl::math::radians(degrees[1]);
    float rz = sgl::math::radians(degrees[2]);

    quatf qx(std::sin(rx * 0.5f), 0.f, 0.f, std::cos(rx * 0.5f));
    quatf qy(0.f, std::sin(ry * 0.5f), 0.f, std::cos(ry * 0.5f));
    quatf qz(0.f, 0.f, std::sin(rz * 0.5f), std::cos(rz * 0.5f));

    switch (order) {
    case pxr::UsdGeomXformCommonAPI::RotationOrderXYZ:
        return mul(qz, mul(qy, qx));
    case pxr::UsdGeomXformCommonAPI::RotationOrderXZY:
        return mul(qy, mul(qz, qx));
    case pxr::UsdGeomXformCommonAPI::RotationOrderYXZ:
        return mul(qz, mul(qx, qy));
    case pxr::UsdGeomXformCommonAPI::RotationOrderYZX:
        return mul(qx, mul(qz, qy));
    case pxr::UsdGeomXformCommonAPI::RotationOrderZXY:
        return mul(qy, mul(qx, qz));
    case pxr::UsdGeomXformCommonAPI::RotationOrderZYX:
        return mul(qx, mul(qy, qz));
    default:
        return mul(qz, mul(qy, qx));
    }
}

struct DecomposedTransform {
    float3 translation;
    quatf rotation;
    float3 scale;
};

/// Extract TRS from a UsdGeomXformable at a given time code.
/// Prefers UsdGeomXformCommonAPI for direct TRS extraction without lossy matrix decomposition.
/// Falls back to full matrix decomposition for prims with non-common xform ops.
DecomposedTransform get_xform_at_time(
    const pxr::UsdGeomXformCommonAPI& xform_api,
    const pxr::UsdGeomXformable& xformable,
    pxr::UsdTimeCode time_code
)
{
    if (xform_api) {
        pxr::GfVec3d trans;
        pxr::GfVec3f rot;
        pxr::GfVec3f scl;
        pxr::GfVec3f pivot;
        pxr::UsdGeomXformCommonAPI::RotationOrder rot_order;

        xform_api.GetXformVectors(&trans, &rot, &scl, &pivot, &rot_order, time_code);

        return {
            float3(static_cast<float>(trans[0]), static_cast<float>(trans[1]), static_cast<float>(trans[2])),
            euler_to_quat(rot, rot_order),
            to_falcor(scl),
        };
    }

    // Fallback: decompose from matrix for prims with non-common xform ops.
    bool resets = false;
    pxr::GfMatrix4d usd_transform;
    xformable.GetLocalTransformation(&usd_transform, &resets, time_code);
    Transform transform(to_falcor(usd_transform));

    return {
        transform.translation(),
        transform.rotation(),
        transform.scale(),
    };
}


} // namespace
std::string
UsdImporterContext::resolve_material(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name)
{
    FALCOR_UNUSED(mesh_name);
    using namespace pxr;

    auto usd_material = get_bound_material(binding_api);
    std::string material_name = usd_material.GetPath().GetString();
    if (material_name.empty())
        return "";

    {
        std::scoped_lock l(m_scene_mutexes.material);
        auto it = m_material_name_to_index.find(material_name);
        if (it != m_material_name_to_index.end())
            return material_name;
    }

    std::set<TfToken> contexts;
    contexts.insert(TfToken());
    auto outputs = usd_material.GetOutputs();
    for (auto& it : outputs) {
        const auto& split_name = it.GetAttr().SplitName();
        if (split_name.size() == 3)
            contexts.insert(TfToken(split_name[1]));
    }

    ImporterMaterial material;
    material.name = material_name;

    auto make_terminal_name = [](TfToken context, std::string_view output) -> std::string
    {
        if (context.IsEmpty())
            return fmt::format("_terminal:{}", output);
        return fmt::format("_terminal:{}:{}", context.GetText(), output);
    };

    // Read all the nodes into the node structure, and connect the outputs.
    material.params.set("_type", "usd_NodeMaterial");
    for (auto& it : contexts) {
        const TfTokenVector context_vector = {it};
        TfTokenVector shader_source_types = {};

        if (UsdShadeShader surface = usd_material.ComputeSurfaceSource(context_vector)) {
            build_material(
                material,
                surface.GetPrim(),
                make_terminal_name(it, "surface"),
                shader_source_types,
                context_vector
            );
        }

        if (UsdShadeShader displacement = usd_material.ComputeDisplacementSource(context_vector)) {
            build_material(
                material,
                displacement.GetPrim(),
                make_terminal_name(it, "displacement"),
                shader_source_types,
                context_vector
            );
        }

        if (UsdShadeShader volume = usd_material.ComputeVolumeSource(context_vector)) {
            build_material(
                material,
                volume.GetPrim(),
                make_terminal_name(it, "volume"),
                shader_source_types,
                context_vector
            );
        }
    }

    if (auto flattened = usdpreviewsurface_to_flattened(
            material,
            [&](const ImporterTexture& texture)
            {
                return add_texture_to_scene(texture);
            }
        )) {
        material.params = std::move(flattened.value());
    }

    // Store material
    {
        std::scoped_lock l(m_scene_mutexes.material);
        auto it = m_material_name_to_index.find(material_name);
        if (it != m_material_name_to_index.end())
            return material_name;
        int material_id = (int)m_scene->materials.size();
        m_scene->materials.push_back(std::move(material));
        m_material_name_to_index[material_name] = material_id;
        return material_name;
    }
}

void UsdImporterContext::traverse_scene(const pxr::UsdPrim& root)
{
    static constexpr bool DEINSTANCE = false;
    using namespace pxr;

    Usd_PrimFlagsPredicate pred = UsdPrimDefaultPredicate;
    /// Deinstances the scene
    if (DEINSTANCE)
        pred.TraverseInstanceProxies(true);

    UsdPrimRange range = UsdPrimRange::PreAndPostVisit(root, pred);

    for (auto it = range.begin(); it != range.end(); ++it) {
        UsdPrim prim = *it;
        const std::string& prim_name = get_name(prim);

        if (!it.IsPostVisit()) {
            if (prim.IsA<UsdGeomXformable>()) {
                push_transform(UsdGeomXformable(prim));
            }

            /// Prune non-renderable primitives
            if (prim.IsA<UsdGeomImageable>() && !is_renderable(UsdGeomImageable(prim))) {
                sgl::log_debug("Pruning non-renderable prim {}", prim_name);
                it.PruneChildren();
                continue;
            }

            // User-defined primitive drops, drops all subprims as well.
            if (should_drop_prim(prim)) {
                it.PruneChildren();
                continue;
            }

            // Not handled here explicitly.
            if (prim.IsA<UsdShadeMaterial>() || prim.IsA<UsdShadeShader>() || prim.IsA<UsdGeomSubset>()) {
                it.PruneChildren();
                continue;
            }

            if (prim.IsInstance() && !DEINSTANCE) {
                add_instance(prim);
            } else if (prim.IsA<UsdGeomPointInstancer>()) {
                add_point_instancer(prim);
            } else if (prim.IsA<UsdGeomMesh>()) {
                add_mesh(prim);
            } else if (prim.IsA<UsdGeomBasisCurves>()) {
                add_curve(prim);
            } else if (prim.IsA<UsdSkelRoot>()) {
                // TODO(tdavidovic): Add skinning once we support animation.
                continue;
            } else if (prim.IsA<UsdLuxBoundableLightBase>() || prim.IsA<UsdLuxNonboundableLightBase>()) {
                add_light(prim);
            } else if (prim.IsA<UsdGeomCamera>()) {
                add_camera(prim);
            } else if (prim.IsA<UsdGeomXform>() || prim.IsA<UsdGeomScope>()) {
                continue;
            } else if (!prim.GetTypeName().IsEmpty()) {
                sgl::log_warn(
                    "Ignoring prim {} of unknown type: {} and all its children.",
                    prim_name,
                    prim.GetTypeName().GetString()
                );
            }
        } else {
            // Post visits
            if (prim.IsA<UsdGeomXformable>()) {
                pop_transform();
            }
        }
    }
}

void UsdImporterContext::add_mesh(pxr::UsdPrim prim, bool run_immediate)
{
    auto resolve_material_func = [&](const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name)
    {
        return resolve_material(binding_api, mesh_name);
    };
    auto add_mesh_task
        = [prim, parent = parent_node_idx(), &scene = m_scene, &scene_mutexes = m_scene_mutexes, resolve_material_func]
    {
        add_mesh(prim, parent, scene, scene_mutexes, resolve_material_func);
    };

    process_task(add_mesh_task, run_immediate);
}

void UsdImporterContext::add_curve(pxr::UsdPrim prim, bool run_immediate)
{
    auto resolve_material_func = [&](const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view curve_name)
    {
        return resolve_material(binding_api, curve_name);
    };
    auto add_curve_task
        = [prim, parent = parent_node_idx(), &scene = m_scene, &scene_mutexes = m_scene_mutexes, resolve_material_func]
    {
        add_curve(prim, parent, scene, scene_mutexes, resolve_material_func);
    };

    process_task(add_curve_task, run_immediate);
}

void UsdImporterContext::add_light(pxr::UsdPrim prim, bool run_immediate)
{
    auto add_light_task = [prim, parent = parent_node_idx(), &scene = m_scene, &scene_mutexes = m_scene_mutexes]
    {
        add_light(prim, parent, scene, scene_mutexes);
    };

    process_task(add_light_task, run_immediate);
}

void UsdImporterContext::add_camera(pxr::UsdPrim prim, bool run_immediate)
{
    auto add_camera_task = [prim,
                            parent = parent_node_idx(),
                            meters_per_unit = get_meters_per_unit(),
                            &scene = m_scene,
                            &scene_mutexes = m_scene_mutexes]
    {
        add_camera(prim, parent, meters_per_unit, scene, scene_mutexes);
    };

    process_task(add_camera_task, run_immediate);
}

void UsdImporterContext::add_instance(pxr::UsdPrim prim)
{
    const pxr::UsdPrim prototype(prim.GetPrototype());
    if (!prototype.IsValid()) {
        sgl::log_error("No valid prototype prim for instance '{}'.", get_name(prim));
        return;
    }
    if (should_drop_prim(prototype))
        return;

    int new_prototype_idx = add_prototype(prototype);
    int parent = parent_node_idx();
    std::scoped_lock l(m_scene_mutexes.nodes);
    int new_node_idx = static_cast<int>(m_scene->nodes.size());
    m_scene->nodes.push_back(
        ImporterNode{
            .name = get_name(prim),
            .prototype_index = new_prototype_idx,
            .parent = parent,
        }
    );
    m_scene->nodes[parent].children.push_back(new_node_idx);
}

void UsdImporterContext::add_point_instancer(pxr::UsdPrim prim)
{
    using namespace pxr;

    std::string name = get_name(prim);
    UsdGeomPointInstancer instancer(prim);

    SdfPathVector prototype_paths;
    if (!instancer.GetPrototypesRel().GetForwardedTargets(&prototype_paths)) {
        sgl::log_error("Error occurred gathering prototypes for point instancer '{}'.", name);
        return;
    }

    /// Index into prototype_paths[]
    pxr::VtArray<int> prototype_usd_indices;
    UsdAttribute prototype_usd_indices_attr(instancer.GetProtoIndicesAttr());
    if (!prototype_usd_indices_attr.IsDefined()
        || !prototype_usd_indices_attr.Get(&prototype_usd_indices, UsdTimeCode::EarliestTime())) {
        sgl::log_error("Point instancer '{}' has no prototype indices. Ignoring prim.", name);
        return;
    }

    /// Index into m_scene.prototypes[]
    std::vector<int> prototype_indices;
    std::vector<std::string> prototype_names;
    for (auto path : prototype_paths) {
        UsdPrim prototype_prim(m_usd_stage->GetPrimAtPath(path));
        prototype_names.push_back(get_name(prototype_prim));

        if (prototype_prim.IsDefined()) {
            prototype_indices.push_back(add_prototype(prototype_prim));
        } else {
            sgl::log_error("Point instancer '{}' references nonexistent prim '{}'. Ignoring.", name, path.GetString());
            prototype_indices.push_back(-1);
        }
    }

    pxr::VtArray<pxr::GfMatrix4d> static_xforms;
    /// No need to create animation keyframes
    // Compute 4x4 transforms for each instance at the earliest time sample.
    // The prototype xform is included in its definition, so we exclude it in the computed instance xforms.
    if (!instancer.ComputeInstanceTransformsAtTime(
            &static_xforms,
            UsdTimeCode::EarliestTime(),
            UsdTimeCode::EarliestTime(),
            UsdGeomPointInstancer::ProtoXformInclusion::ExcludeProtoXform
        )) {
        sgl::log_error("Error occurred computing point instancer transforms for '{}'. Ignoring prim.", name);
        return;
    }
    if (prototype_usd_indices.size() != static_xforms.size()) {
        sgl::log_error(
            "Point instancer '{}' has {} prototype indices but {} transforms.",
            name,
            prototype_usd_indices.size(),
            static_xforms.size()
        );
        return;
    }

    int parent = parent_node_idx();
    std::scoped_lock l(m_scene_mutexes.nodes);

    for (size_t i = 0; i < prototype_usd_indices.size(); ++i) {
        int prototype_index = prototype_indices[prototype_usd_indices[i]];

        if (prototype_index < 0)
            continue;

        int new_node_idx = static_cast<int>(m_scene->nodes.size());
        m_scene->nodes.push_back(
            ImporterNode{
                .name = fmt::format("{}_{}", name, i),
                .prototype_index = prototype_index,
                .parent = parent,
            }
        );
        m_scene->nodes[parent].children.push_back(new_node_idx);
    }
}

int UsdImporterContext::add_prototype(pxr::UsdPrim prototype)
{
    if (auto it = m_prototype_to_index.find(prototype); it != m_prototype_to_index.end())
        return it->second;

    push_prototype_stack();

    int new_prototype_idx = static_cast<int>(m_scene->prototypes.size());
    ImporterPrototype importer_prototype;
    importer_prototype.name = get_name(prototype);
    importer_prototype.root_node = push_transform(float4x4::identity(), get_name(prototype) + "_root");
    m_prototype_to_index[prototype] = new_prototype_idx;
    m_scene->prototypes.push_back(std::move(importer_prototype));

    traverse_scene(prototype);

    pop_prototype_stack();

    return new_prototype_idx;
}

void UsdImporterContext::add_mesh(
    pxr::UsdPrim prim,
    int parent,
    ImporterScene* scene,
    UsdSceneMutexes& scene_mutexes,
    std::function<std::string(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view)>
        resolve_material_func
)
{
    using namespace pxr;

    UsdGeomMesh usd_mesh(prim);
    UsdGeomPrimvarsAPI primvar_API(usd_mesh);
    std::string prim_name = usd_mesh.GetPath().GetString();

    UsdGeomPointBased geom_point_based(prim);

    TesselatorInputMesh tess_meshes;
    tess_meshes.name = prim_name;

    tess_meshes.subdiv_scheme = get_attribute(usd_mesh.GetSubdivisionSchemeAttr(), UsdGeomTokens->catmullClark);
    tess_meshes.orientation = get_attribute(usd_mesh.GetOrientationAttr(), UsdGeomTokens->rightHanded);
    tess_meshes.facevarying_interpolation
        = get_attribute(usd_mesh.GetFaceVaryingLinearInterpolationAttr(), UsdGeomTokens->cornersPlus1);
    tess_meshes.vertex_boundary_interpolation
        = get_attribute(usd_mesh.GetInterpolateBoundaryAttr(), UsdGeomTokens->edgeAndCorner);

    usd_mesh.GetPointsAttr().Get(&tess_meshes.positions);
    usd_mesh.GetFaceVertexCountsAttr().Get(&tess_meshes.face_vertex_counts);
    usd_mesh.GetFaceVertexIndicesAttr().Get(&tess_meshes.face_indices);

    if (auto attr = usd_mesh.GetHoleIndicesAttr(); attr)
        attr.Get(&tess_meshes.hole_indices);

    bool apply_override = true;
    if (auto attr = usd_mesh.GetPrim().GetAttribute(TfToken("refinementEnableOverride")); attr)
        attr.Get<bool>(&apply_override);

    if (apply_override) {
        if (auto attr = usd_mesh.GetPrim().GetAttribute(TfToken("refinementLevel")); attr)
            attr.Get<int>(&tess_meshes.refinement_level);
    }

    if (UsdGeomPrimvar uv_primvar = get_uv_primvar(usd_mesh); uv_primvar && uv_primvar.HasValue()) {
        tess_meshes.uv_interpolation = uv_primvar.GetInterpolation();
        uv_primvar.ComputeFlattened(&tess_meshes.uvs);
    }

    if (UsdGeomPrimvar normals_primvar = primvar_API.GetPrimvar(TfToken("primvars:normals"));
        normals_primvar && normals_primvar.HasValue()) {
        tess_meshes.normals_interpolation = normals_primvar.GetInterpolation();
        normals_primvar.ComputeFlattened(&tess_meshes.normals);
    } else if (usd_mesh.GetNormalsAttr().IsAuthored()) {
        // Normals specified via the attribute cannot be indexed, so there is no need to flatten them.
        tess_meshes.normals_interpolation = usd_mesh.GetNormalsInterpolation();
        usd_mesh.GetNormalsAttr().Get(&tess_meshes.normals);
    }

    std::vector<UsdGeomSubset> geom_subsets = UsdGeomSubset::GetAllGeomSubsets(usd_mesh);

    for (auto& geom_subset : geom_subsets) {
        UsdAttribute indices_attr = geom_subset.GetIndicesAttr();
        if (indices_attr) {
            TesselatorInputMesh::Subgeometry subgeometry;
            indices_attr.Get(&subgeometry.face_indices);
            subgeometry.name = geom_subset.GetPath().GetString();
            subgeometry.material_name = resolve_material_func(UsdShadeMaterialBindingAPI(geom_subset), prim_name);
            tess_meshes.subgeometries.push_back(std::move(subgeometry));
        }
    }

    // Assign all unassigned indices (all indices if there are no subsets at all)
    pxr::VtArray<int> unassigned_indices
        = UsdGeomSubset::GetUnassignedIndices(geom_subsets, tess_meshes.face_vertex_counts.size());
    if (!unassigned_indices.empty()) {
        TesselatorInputMesh::Subgeometry subgeometry;
        subgeometry.face_indices = unassigned_indices;
        subgeometry.name = prim_name;
        subgeometry.material_name = resolve_material_func(UsdShadeMaterialBindingAPI(usd_mesh), prim_name);
        tess_meshes.subgeometries.push_back(std::move(subgeometry));
    }

    bool tangents_from_uvs = !tess_meshes.uvs.empty();

    /// If we have UVs, we generate tangents via MikkTSpace and then deduplicate vertices.
    /// If we do not have UVs, those are set to 0, and we first deduplicate and then generate tangents from normals.
    ImporterMesh result = tessellate(tess_meshes);
    /// Some USDs contain invalid meshes as part of their hierarchy (e.g., Casa_GP), we discard those.
    if (!result.position_stream().valid())
        return;

    if (tangents_from_uvs)
        result.add_tangents_from_uvs();
    result.deduplicate_vertices();
    if (!tangents_from_uvs)
        result.add_tangents_from_normals();

    std::scoped_lock l(scene_mutexes.nodes, scene_mutexes.mesh);
    int new_node_idx = static_cast<int>(scene->nodes.size());
    int new_mesh_idx = static_cast<int>(scene->meshes.size());
    scene->meshes.push_back(std::move(result));
    scene->nodes.push_back(
        ImporterNode{
            .name = prim_name,
            .mesh_index = new_mesh_idx,
            .parent = parent,
        }
    );
    scene->nodes[parent].children.push_back(new_node_idx);
}

void UsdImporterContext::add_curve(
    pxr::UsdPrim prim,
    int parent,
    ImporterScene* scene,
    UsdSceneMutexes& scene_mutexes,
    std::function<std::string(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view)>
        resolve_material_func
)
{
    using namespace pxr;

    UsdGeomBasisCurves usd_curves(prim);
    std::string prim_name = prim.GetPath().GetString();

    // Read curve type (linear or cubic).
    TfToken type_token;
    usd_curves.GetTypeAttr().Get(&type_token);

    bool is_linear = (type_token == UsdGeomTokens->linear);
    bool is_cubic = (type_token == UsdGeomTokens->cubic);

    if (!is_linear && !is_cubic) {
        sgl::log_warn("Skipping BasisCurves prim '{}' with unsupported type '{}'", prim_name, type_token.GetString());
        return;
    }

    // Read points.
    VtArray<GfVec3f> usd_points;
    usd_curves.GetPointsAttr().Get(&usd_points);
    if (usd_points.empty()) {
        sgl::log_warn("Skipping BasisCurves prim '{}' with no points", prim_name);
        return;
    }

    // Read curve vertex counts.
    VtArray<int> usd_vertex_counts;
    usd_curves.GetCurveVertexCountsAttr().Get(&usd_vertex_counts);
    if (usd_vertex_counts.empty()) {
        sgl::log_warn("Skipping BasisCurves prim '{}' with no vertex counts", prim_name);
        return;
    }

    // Read widths (USD widths are diameters, convert to radii).
    VtArray<float> usd_widths;
    usd_curves.GetWidthsAttr().Get(&usd_widths);

    // Build tessellation input.
    TesselatorInputCurve input;
    input.name = prim_name;
    input.points = std::move(usd_points);
    input.widths = std::move(usd_widths);
    input.vertex_counts = std::move(usd_vertex_counts);

    if (is_linear) {
        input.type = TesselatorInputCurve::Type::linear;
    } else {
        input.type = TesselatorInputCurve::Type::cubic;

        TfToken basis_token;
        usd_curves.GetBasisAttr().Get(&basis_token);

        if (basis_token == UsdGeomTokens->bezier) {
            input.basis = TesselatorInputCurve::Basis::bezier;
        } else if (basis_token == UsdGeomTokens->bspline) {
            input.basis = TesselatorInputCurve::Basis::bspline;
        } else if (basis_token == UsdGeomTokens->catmullRom) {
            input.basis = TesselatorInputCurve::Basis::catmull_rom;
        } else {
            sgl::log_warn(
                "Skipping cubic BasisCurves prim '{}' with unsupported basis '{}'",
                prim_name,
                basis_token.GetString()
            );
            return;
        }

        TfToken wrap_token;
        usd_curves.GetWrapAttr().Get(&wrap_token);
        if (wrap_token == UsdGeomTokens->periodic) {
            sgl::log_warn(
                "BasisCurves prim '{}' uses periodic wrap mode which is not yet supported, treating as nonperiodic",
                prim_name
            );
        }
    }

    ImporterCurve curve = tessellate_curves(input);

    if (curve.indices.empty())
        return;

    curve.material_name = resolve_material_func(pxr::UsdShadeMaterialBindingAPI(prim), prim_name);

    std::scoped_lock l(scene_mutexes.nodes, scene_mutexes.curve);
    int new_node_idx = static_cast<int>(scene->nodes.size());
    int new_curve_idx = static_cast<int>(scene->curves.size());
    scene->curves.push_back(std::move(curve));
    scene->nodes.push_back(
        ImporterNode{
            .name = prim_name,
            .curve_index = new_curve_idx,
            .parent = parent,
        }
    );
    scene->nodes[parent].children.push_back(new_node_idx);
}

void UsdImporterContext::add_light(pxr::UsdPrim prim, int parent, ImporterScene* scene, UsdSceneMutexes& scene_mutexes)
{
    ImporterLight light = create_light(prim);

    std::scoped_lock l(scene_mutexes.nodes, scene_mutexes.light);
    int new_node_idx = static_cast<int>(scene->nodes.size());
    int new_light_idx = static_cast<int>(scene->lights.size());
    scene->lights.push_back(std::move(light));
    scene->nodes.push_back(
        ImporterNode{
            .name = prim.GetPath().GetString(),
            .light_index = new_light_idx,
            .parent = parent,
        }
    );
    scene->nodes[parent].children.push_back(new_node_idx);
}

ImporterLight UsdImporterContext::create_light(pxr::UsdPrim prim)
{
    /// We do not scale any of the dimensions with get_meters_per_unit(),
    /// because that is already baked into the Scene's root transform.
    ///
    /// Light intensities are adjusted for lights marked as normalize, as per
    /// https://openusd.org/release/user_guides/schemas/usdLux/LightAPI.html#inputs-normalize
    ///
    /// Scaling is applied during normalize, however none-uniform scaling of sphere lights results
    /// in a warning, and scaling is general is not recommended by the spec:
    /// https://openusd.org/release/api/usd_lux_page_front.html#usdLux_Geometry

    using namespace pxr;

    ImporterLight result;
    result.name = prim.GetPath().GetString();

    if (auto light = UsdLuxDistantLight(prim)) {
        result.type = ImporterLight::Type::distant;
        result.intensity = get_light_intensity(light, prim);
        result.degree_angular_diameter = get_authored_attribute(
            light.GetAngleAttr(),
            prim.GetAttribute(TfToken("angle")),
            result.degree_angular_diameter
        );
        if (is_light_normalized(light, prim))
            result.intensity /= distant_light_size_factor(result.degree_angular_diameter);

        return result;
    }

    if (auto light = UsdLuxRectLight(prim)) {
        result.type = ImporterLight::Type::rectangular;
        result.intensity = get_light_intensity(light, prim);
        result.width = get_authored_attribute(light.GetWidthAttr(), prim.GetAttribute(TfToken("width")), result.width);
        result.height
            = get_authored_attribute(light.GetHeightAttr(), prim.GetAttribute(TfToken("height")), result.height);
        if (is_light_normalized(light, prim))
            result.intensity /= planar_light_size_factor(prim, result.width * result.height);

        return result;
    }

    if (auto light = UsdLuxSphereLight(prim)) {
        result.type = ImporterLight::Type::sphere;
        result.intensity = get_light_intensity(light, prim);
        result.radius
            = get_authored_attribute(light.GetRadiusAttr(), prim.GetAttribute(TfToken("radius")), result.radius);
        if (is_light_normalized(light, prim))
            result.intensity /= sphere_light_size_factor(prim, result.radius);

        if (get_authored_attribute(light.GetTreatAsPointAttr(), prim.GetAttribute(TfToken("treatAsPoint")), false)) {
            result.radius = 0;
        }
        if (result.radius == 0)
            result.type = ImporterLight::Type::point;

        return result;
    }

    if (auto light = UsdLuxDiskLight(prim)) {
        result.type = ImporterLight::Type::disk;
        result.intensity = get_light_intensity(light, prim);
        result.radius
            = get_authored_attribute(light.GetRadiusAttr(), prim.GetAttribute(TfToken("radius")), result.radius);
        if (is_light_normalized(light, prim)) {
            const float pi = static_cast<float>(M_PI);
            result.intensity /= planar_light_size_factor(prim, pi * result.radius * result.radius);
        }

        return result;
    }

    if (auto light = UsdLuxDomeLight(prim)) {
        result.type = ImporterLight::Type::dome;
        result.intensity = get_light_intensity(light, prim);

        SdfAssetPath texture_path = get_authored_attribute(
            light.GetTextureFileAttr(),
            prim.GetAttribute(TfToken("texture:file")),
            SdfAssetPath()
        );
        result.env_map_path = texture_path.GetResolvedPath();
        if (result.env_map_path.empty()) {
            result.env_map_path = texture_path.GetAssetPath();
            if (!result.env_map_path.empty())
                sgl::log_error("Cannot find texture on dome light: " + result.name);
        }

        return result;
    }
    return result;
}

void UsdImporterContext::add_camera(
    pxr::UsdPrim prim,
    int parent,
    float meters_per_unit,
    ImporterScene* scene,
    UsdSceneMutexes& scene_mutexes
)
{
    using namespace pxr;

    GfCamera usd_camera = UsdGeomCamera(prim).GetCamera(0.f);

    ImporterCamera result;
    result.name = prim.GetName().GetString();
    result.focus_distance = usd_camera.GetFocusDistance() * meters_per_unit;

    // value is tenths of a world unit, so *10 to get meters and then /1000 to go to mm
    // (Many scenes have cm as their scene unit, so this becomes a no-op)

    // Focal length, per the USD spec, is supposed to be specified in tenths of a USD world unit.
    // However, it seems to always be specified in mm in OV assets.
    result.focal_length = usd_camera.GetFocalLength();
    result.fstop = usd_camera.GetFStop();
    result.depth_range
        = float2(usd_camera.GetClippingRange().GetMin(), usd_camera.GetClippingRange().GetMax()) * meters_per_unit;

    switch (usd_camera.GetProjection()) {
    case GfCamera::Projection::Perspective:
        result.projection = ImporterCamera::Projection::perspective;
        break;
    case GfCamera::Projection::Orthographic:
        result.projection = ImporterCamera::Projection::orthographic;
        break;
    }

    const float vertical_aperture = usd_camera.GetVerticalAperture();
    const float horizontal_aperture = usd_camera.GetHorizontalAperture();
    const bool has_valid_vertical_aperture = prim.GetAttribute(UsdGeomTokens->verticalAperture).IsAuthored()
        && std::isfinite(vertical_aperture) && vertical_aperture > 0.f;
    const bool has_valid_horizontal_aperture = prim.GetAttribute(UsdGeomTokens->horizontalAperture).IsAuthored()
        && std::isfinite(horizontal_aperture) && horizontal_aperture > 0.f;

    if (has_valid_vertical_aperture) {
        result.fov_direction = ImporterCamera::FOVDirection::vertical;
        result.sensor_size_mm = vertical_aperture;
    } else if (has_valid_horizontal_aperture) {
        result.fov_direction = ImporterCamera::FOVDirection::horizontal;
        result.sensor_size_mm = horizontal_aperture;
    } else if (std::isfinite(vertical_aperture) && vertical_aperture > 0.f) {
        // Neither aperture is authored. Preserve the composed USD vertical aperture,
        // including the schema fallback, rather than inventing an aspect ratio.
        result.fov_direction = ImporterCamera::FOVDirection::vertical;
        result.sensor_size_mm = vertical_aperture;
    }

    std::scoped_lock l(scene_mutexes.nodes, scene_mutexes.camera);
    int new_node_idx = static_cast<int>(scene->nodes.size());
    int new_camera_idx = static_cast<int>(scene->cameras.size());
    scene->cameras.push_back(std::move(result));
    scene->nodes.push_back(
        ImporterNode{
            .name = prim.GetPath().GetString(),
            .camera_index = new_camera_idx,
            .parent = parent,
        }
    );
    scene->nodes[parent].children.push_back(new_node_idx);
}

int UsdImporterContext::add_texture_to_scene(const ImporterTexture& texture)
{
    {
        std::scoped_lock l(m_scene_mutexes.texture);
        auto it = m_texture_to_index.find(texture);
        if (it != m_texture_to_index.end()) {
            return it->second; // Texture already exists, return its index
        }

        int texture_index = (int)m_scene->textures.size();
        m_scene->textures.push_back(texture);
        m_texture_to_index[texture] = texture_index;
        return texture_index;
    }
}

void UsdImporterContext::extract_animations()
{
    if (m_prim_path_to_node_idx.empty())
        return;

    double time_codes_per_second = m_usd_stage->GetTimeCodesPerSecond();
    if (time_codes_per_second <= 0.0)
        time_codes_per_second = 24.0;

    bool has_animation = false;

    for (const auto& [prim_path, node_idx] : m_prim_path_to_node_idx) {
        pxr::UsdPrim prim = m_usd_stage->GetPrimAtPath(prim_path);
        if (!prim.IsValid())
            continue;

        pxr::UsdGeomXformable xformable(prim);
        if (!xformable)
            continue;

        // Query time samples for the xform ops.
        std::vector<double> time_samples;
        if (!xformable.GetTimeSamples(&time_samples))
            continue;

        // Need at least 2 time samples for animation.
        if (time_samples.size() < 2)
            continue;

        if (!has_animation) {
            m_scene->animation.name = "UsdAnimation";
            has_animation = true;
        }

        size_t keyframe_count = time_samples.size();

        // Use XformCommonAPI to get TRS directly without lossy matrix decomposition.
        auto xform_api = pxr::UsdGeomXformCommonAPI(xformable);

        // Evaluate the local transform at each time sample.
        std::vector<float> times(keyframe_count);
        std::vector<float> translations(keyframe_count * 3);
        std::vector<float> rotations(keyframe_count * 4);
        std::vector<float> scales(keyframe_count * 3);

        for (size_t i = 0; i < keyframe_count; ++i) {
            times[i] = static_cast<float>(time_samples[i] / time_codes_per_second);

            auto [translation, rotation, scale]
                = get_xform_at_time(xform_api, xformable, pxr::UsdTimeCode(time_samples[i]));

            translations[i * 3 + 0] = translation.x;
            translations[i * 3 + 1] = translation.y;
            translations[i * 3 + 2] = translation.z;

            // quatf is stored as (x, y, z, w)
            rotations[i * 4 + 0] = rotation.x;
            rotations[i * 4 + 1] = rotation.y;
            rotations[i * 4 + 2] = rotation.z;
            rotations[i * 4 + 3] = rotation.w;

            scales[i * 3 + 0] = scale.x;
            scales[i * 3 + 1] = scale.y;
            scales[i * 3 + 2] = scale.z;
        }

        // Check if translation is animated (not all same).
        bool has_translation_anim = false;
        for (size_t i = 1; i < keyframe_count && !has_translation_anim; ++i) {
            for (int c = 0; c < 3; ++c) {
                if (translations[i * 3 + c] != translations[c]) {
                    has_translation_anim = true;
                    break;
                }
            }
        }

        // Check if rotation is animated.
        bool has_rotation_anim = false;
        for (size_t i = 1; i < keyframe_count && !has_rotation_anim; ++i) {
            for (int c = 0; c < 4; ++c) {
                if (rotations[i * 4 + c] != rotations[c]) {
                    has_rotation_anim = true;
                    break;
                }
            }
        }

        // Check if scale is animated.
        bool has_scale_anim = false;
        for (size_t i = 1; i < keyframe_count && !has_scale_anim; ++i) {
            for (int c = 0; c < 3; ++c) {
                if (scales[i * 3 + c] != scales[c]) {
                    has_scale_anim = true;
                    break;
                }
            }
        }

        ImporterNode& node = m_scene->nodes[node_idx];

        if (has_translation_anim) {
            ImporterAnimationChannel channel;
            channel.value_type = AnimationValueType::float3;
            channel.interpolation_mode = AnimationInterpolationMode::linear;
            channel.times = times;
            channel.values = std::move(translations);
            node.animation_translation = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }

        if (has_rotation_anim) {
            ImporterAnimationChannel channel;
            channel.value_type = AnimationValueType::quatf;
            channel.interpolation_mode = AnimationInterpolationMode::linear;
            channel.times = times;
            channel.values = std::move(rotations);
            node.animation_rotation = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }

        if (has_scale_anim) {
            ImporterAnimationChannel channel;
            channel.value_type = AnimationValueType::float3;
            channel.interpolation_mode = AnimationInterpolationMode::linear;
            channel.times = times;
            channel.values = std::move(scales);
            node.animation_scale = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }
    }
}

} // namespace usd_importer
} // namespace falcor
