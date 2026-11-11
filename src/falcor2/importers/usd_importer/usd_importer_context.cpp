// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_context.h"
#include "usd_importer_mesh_adapter.h"
#include "usd_importer_curve_adapter.h"
#include "usd_importer_utils.h"
#include "usd_importer_macros.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/usd_importer/usd_importer_material.h"
#include "falcor2/render/transform.h"
#include "falcor2/utils/math/transform.h"

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
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/xformCommonAPI.h>
#include <pxr/usd/usdGeom/xformable.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/scope.h>

#include <pxr/usd/usdLux/cylinderLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/shapingAPI.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/domeLight_1.h>

#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/animation.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string_view>

namespace falcor {
namespace usd_importer {
namespace {

pxr::UsdPrim get_prototype_root(pxr::UsdPrim prim)
{
    while (!prim.IsPrototype())
        prim = prim.GetParent();
    return prim;
}

std::map<pxr::SdfPath, pxr::SdfPath> build_prototype_source_paths(const pxr::UsdStageRefPtr& stage)
{
    // Prototype names are stage-local. Choose a deterministic authored instance as the
    // representative for shared provenance, resolving enclosing prototypes for nested instances.
    // Runtime instances retain this shared reference. Changing the instance set may change the representative.
    std::map<pxr::SdfPath, pxr::SdfPath> paths;
    auto resolve = [&](const pxr::UsdPrim& prim, auto&& recurse) -> pxr::SdfPath
    {
        if (!prim.IsInPrototype())
            return prim.GetPath();

        const pxr::UsdPrim prototype = get_prototype_root(prim);
        auto it = paths.find(prototype.GetPath());
        if (it == paths.end()) {
            pxr::SdfPath instance_path;
            for (const auto& instance : prototype.GetInstances()) {
                const pxr::SdfPath candidate = recurse(instance, recurse);
                if (instance_path.IsEmpty() || candidate < instance_path)
                    instance_path = candidate;
            }
            FALCOR_CHECK(
                !instance_path.IsEmpty(),
                "USD prototype '{}' has no instances",
                prototype.GetPath().GetString()
            );
            it = paths.emplace(prototype.GetPath(), instance_path).first;
        }
        return prim.GetPath().ReplacePrefix(prototype.GetPath(), it->second);
    };
    for (const auto& prototype : stage->GetPrototypes())
        resolve(prototype, resolve);
    return paths;
}

template<typename T>
T get_light_attribute(const pxr::UsdAttribute& attribute, const pxr::UsdAttribute& legacy_attribute, const T& fallback)
{
    T value = fallback;
    if (attribute && attribute.IsAuthored())
        attribute.Get(&value);
    else if (legacy_attribute && legacy_attribute.IsAuthored())
        legacy_attribute.Get(&value);
    else if (attribute)
        attribute.Get(&value);
    return value;
}

template<typename TUsdLuxDomeLightType>
void read_dome_light_properties(const TUsdLuxDomeLightType& light, const pxr::UsdPrim& prim, ImporterLight& result)
{
    using namespace pxr;

    result.type = ImporterLight::Type::dome;

    const SdfAssetPath texture_path = get_authored_attribute(
        light.GetTextureFileAttr(),
        prim.GetAttribute(TfToken("texture:file")),
        SdfAssetPath()
    );
    result.env_map_path = texture_path.GetResolvedPath();
}

EnvMapPoleAxis get_dome_light_pole_axis(const pxr::UsdLuxDomeLight_1& light)
{
    using namespace pxr;

    const UsdPrim prim = light.GetPrim();
    TfToken pole_axis = UsdLuxTokens->scene;
    light.GetPoleAxisAttr().Get(&pole_axis);
    if (pole_axis == UsdLuxTokens->scene)
        pole_axis = UsdGeomGetStageUpAxis(prim.GetStage()) == UsdGeomTokens->z ? UsdLuxTokens->Z : UsdLuxTokens->Y;

    // Resolve the USD stage-dependent value before passing the light to the runtime.
    if (pole_axis == UsdLuxTokens->Y)
        return EnvMapPoleAxis::y;
    if (pole_axis == UsdLuxTokens->Z)
        return EnvMapPoleAxis::z;

    sgl::log_warn(
        "DomeLight_1 '{}' has unsupported poleAxis '{}'; treating it as Y.",
        prim.GetPath().GetString(),
        pole_axis.GetString()
    );
    return EnvMapPoleAxis::y;
}

float4x4 sanitize_planar_light_transform(const pxr::UsdPrim& prim, float4x4 transform)
{
    if (!prim.IsA<pxr::UsdLuxRectLight>() && !prim.IsA<pxr::UsdLuxDiskLight>())
        return transform;

    const float3 normal_axis = transform.get_col(2).xyz();
    if (sgl::math::dot(normal_axis, normal_axis) > 0.f)
        return transform;

    const float3 area_vector = sgl::math::cross(transform.get_col(0).xyz(), transform.get_col(1).xyz());
    const float area_scale = sgl::math::length(area_vector);
    if (area_scale > 0.f && sgl::math::isfinite(area_scale))
        transform.set_col(2, float4(area_vector / area_scale, 0.f));
    return transform;
}

bool read_light_shaping(const pxr::UsdPrim& prim, ImporterLight& light)
{
    using namespace pxr;

    if (!prim.HasAPI<UsdLuxShapingAPI>())
        return false;

    const UsdLuxShapingAPI shaping(prim);
    light.shaping_cone_angle = get_authored_attribute(
        shaping.GetShapingConeAngleAttr(),
        prim.GetAttribute(TfToken("inputs:shaping:cone:angle")),
        light.shaping_cone_angle
    );
    light.shaping_cone_softness = std::clamp(
        get_authored_attribute(
            shaping.GetShapingConeSoftnessAttr(),
            prim.GetAttribute(TfToken("inputs:shaping:cone:softness")),
            light.shaping_cone_softness
        ),
        0.f,
        1.f
    );
    light.shaping_focus = std::max(
        get_authored_attribute(
            shaping.GetShapingFocusAttr(),
            prim.GetAttribute(TfToken("inputs:shaping:focus")),
            light.shaping_focus
        ),
        0.f
    );
    return true;
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

bool has_unreal_exporter_provenance(const pxr::UsdStageRefPtr& stage)
{
    using namespace pxr;

    static const TfToken UNREAL_MATERIAL("unrealMaterial");
    static const TfToken UNREAL_SURFACE_OUTPUT("outputs:unreal:surface");
    for (const UsdPrim& prim : stage->Traverse()) {
        if (prim.GetAttribute(UNREAL_MATERIAL) || prim.GetAttribute(UNREAL_SURFACE_OUTPUT))
            return true;
    }
    return false;
}

} // namespace

UsdImporterContext::UsdImporterContext(
    const pxr::UsdStageRefPtr& stage,
    const ImportOptions& import_options,
    ustring source_asset_path,
    bool parallel_adds
)
    : m_usd_stage(stage)
    , m_source_asset_path(source_asset_path)
    , m_prototype_source_paths(build_prototype_source_paths(stage))
    , m_import_options(import_options)
    , m_parallel_adds(parallel_adds)
{
    m_node_idx_stacks.resize(1);
    m_meters_per_unit = static_cast<float>(pxr::UsdGeomGetStageMetersPerUnit(m_usd_stage));
    m_enable_virtual_sphere_shrinking = has_unreal_exporter_provenance(m_usd_stage);
    m_scene = make_ref<ImporterScene>();
}

int UsdImporterContext::push_transform(const pxr::UsdGeomXformable& prim)
{
    bool resets = false;
    pxr::GfMatrix4d usd_transform;
    prim.GetLocalTransformation(&usd_transform, &resets, pxr::UsdTimeCode::EarliestTime());

    const pxr::UsdPrim usd_prim = prim.GetPrim();
    const float4x4 transform = sanitize_planar_light_transform(usd_prim, to_falcor(usd_transform));
    const int node_idx = push_transform(transform, get_name(usd_prim), resets ? root_node_idx() : current_node_idx());
    m_scene->nodes[node_idx].source_asset_ref = source_asset_ref_for_prim(usd_prim);
    // Track prim path -> node index for animation extraction.
    m_prim_path_to_node_idx[usd_prim.GetPath()] = node_idx;
    return node_idx;
}

int UsdImporterContext::resolve_material(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name)
{
    FALCOR_UNUSED(mesh_name);
    using namespace pxr;

    // Subsets use the containing gprim's sidedness. Promote a shared material if any binding needs it.
    UsdPrim geometry_prim = binding_api.GetPrim();
    if (geometry_prim.IsA<UsdGeomSubset>())
        geometry_prim = geometry_prim.GetParent();
    bool double_sided = false;
    if (UsdGeomGprim gprim{geometry_prim})
        gprim.GetDoubleSidedAttr().Get(&double_sided);

    auto usd_material = get_bound_material(binding_api);
    std::string material_name = usd_material.GetPath().GetString();
    if (material_name.empty())
        return -1;

    {
        std::scoped_lock l(m_scene_mutexes.material);
        auto it = m_material_name_to_index.find(material_name);
        if (it != m_material_name_to_index.end()) {
            if (double_sided)
                m_scene->materials[it->second].params.set("double_sided", true);
            return it->second;
        }
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
    material.source_asset_ref = source_asset_ref_for_prim(usd_material.GetPrim());
    auto register_asset = [this](const std::filesystem::path& path)
    {
        add_asset_to_scene(path);
    };

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
                context_vector,
                register_asset
            );
        }

        if (UsdShadeShader displacement = usd_material.ComputeDisplacementSource(context_vector)) {
            build_material(
                material,
                displacement.GetPrim(),
                make_terminal_name(it, "displacement"),
                shader_source_types,
                context_vector,
                register_asset
            );
        }

        if (UsdShadeShader volume = usd_material.ComputeVolumeSource(context_vector)) {
            build_material(
                material,
                volume.GetPrim(),
                make_terminal_name(it, "volume"),
                shader_source_types,
                context_vector,
                register_asset
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
        if (it != m_material_name_to_index.end()) {
            if (double_sided)
                m_scene->materials[it->second].params.set("double_sided", true);
            return it->second;
        }
        if (double_sided)
            material.params.set("double_sided", true);
        int material_id = (int)m_scene->materials.size();
        m_scene->materials.push_back(std::move(material));
        m_material_name_to_index[material_name] = material_id;
        return material_id;
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
        const UsdGeomXformable xformable(prim);
        const bool is_instance = prim.IsInstance() && !DEINSTANCE;
        // Use the same ownership condition on entry and exit, including pruned or invisible prims.
        const bool owns_node = xformable || is_instance || prim.IsA<UsdGeomScope>();

        if (!it.IsPostVisit()) {
            int node_idx = -1;
            if (xformable)
                node_idx = push_transform(xformable);
            else if (owns_node) {
                node_idx = push_transform(float4x4::identity(), prim_name);
                m_scene->nodes[node_idx].source_asset_ref = source_asset_ref_for_prim(prim);
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

            auto is_renderable_prim = [&]()
            {
                UsdGeomImageable imageable(prim);
                return !imageable || is_renderable(imageable);
            };

            if (is_instance) {
                if (is_renderable_prim())
                    add_instance(prim, node_idx);
            } else if (prim.IsA<UsdGeomPointInstancer>()) {
                if (is_renderable_prim())
                    add_point_instancer(prim, node_idx);
            } else if (prim.IsA<UsdGeomMesh>()) {
                if (is_renderable_prim())
                    add_mesh(prim, node_idx);
            } else if (prim.IsA<UsdGeomBasisCurves>()) {
                if (is_renderable_prim())
                    add_curve(prim, node_idx);
            } else if (prim.IsA<UsdSkelRoot>()) {
                // TODO(tdavidovic): Add skinning once we support animation.
                continue;
            } else if (prim.IsA<UsdLuxBoundableLightBase>() || prim.IsA<UsdLuxNonboundableLightBase>()) {
                if (is_renderable_prim())
                    add_light(prim, node_idx);
            } else if (prim.IsA<UsdGeomCamera>()) {
                add_camera(prim, node_idx);
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
            if (owns_node) {
                pop_transform();
            }
        }
    }
}

void UsdImporterContext::add_mesh(pxr::UsdPrim prim, int node_idx, bool run_immediate)
{
    process_task(
        [this, prim, node_idx]
        {
            extract_mesh(prim, node_idx);
        },
        run_immediate
    );
}

void UsdImporterContext::add_curve(pxr::UsdPrim prim, int node_idx, bool run_immediate)
{
    process_task(
        [this, prim, node_idx]
        {
            extract_curve(prim, node_idx);
        },
        run_immediate
    );
}

void UsdImporterContext::add_light(pxr::UsdPrim prim, int node_idx, bool run_immediate)
{
    process_task(
        [this, prim, node_idx]
        {
            extract_light(prim, node_idx);
        },
        run_immediate
    );
}

void UsdImporterContext::add_camera(pxr::UsdPrim prim, int node_idx, bool run_immediate)
{
    process_task(
        [this, prim, node_idx]
        {
            extract_camera(prim, node_idx);
        },
        run_immediate
    );
}

void UsdImporterContext::add_instance(pxr::UsdPrim prim, int node_idx)
{
    const pxr::UsdPrim prototype(prim.GetPrototype());
    if (!prototype.IsValid()) {
        sgl::log_error("No valid prototype prim for instance '{}'.", get_name(prim));
        return;
    }
    if (should_drop_prim(prototype))
        return;

    int new_prototype_idx = add_prototype(prototype);
    std::scoped_lock l(m_scene_mutexes.nodes);
    FALCOR_ASSERT(node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size()));
    m_scene->nodes[node_idx].prototype_index = new_prototype_idx;
}

void UsdImporterContext::add_point_instancer(pxr::UsdPrim prim, int node_idx)
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

    const SourceAssetRef source_asset_ref = source_asset_ref_for_prim(prim);
    std::scoped_lock l(m_scene_mutexes.nodes);
    FALCOR_ASSERT(node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size()));

    for (size_t i = 0; i < prototype_usd_indices.size(); ++i) {
        int prototype_index = prototype_indices[prototype_usd_indices[i]];

        if (prototype_index < 0)
            continue;

        int new_node_idx = static_cast<int>(m_scene->nodes.size());
        m_scene->nodes.push_back(
            ImporterNode{
                .name = fmt::format("{}_{}", name, i),
                .source_asset_ref = source_asset_ref,
                .prototype_index = prototype_index,
                .parent = node_idx,
            }
        );
        m_scene->nodes[node_idx].children.push_back(new_node_idx);
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
    m_scene->nodes[importer_prototype.root_node].source_asset_ref = source_asset_ref_for_prim(prototype);
    m_prototype_to_index[prototype] = new_prototype_idx;
    m_scene->prototypes.push_back(std::move(importer_prototype));

    traverse_scene(prototype);

    pop_prototype_stack();

    return new_prototype_idx;
}

void UsdImporterContext::extract_mesh(pxr::UsdPrim prim, int node_idx)
{
    auto resolve_material_func = [this](const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name)
    {
        return resolve_material(binding_api, mesh_name);
    };
    pxr::UsdGeomMesh usd_mesh(prim);
    ImporterMesh result = convert_usd_mesh(usd_mesh, resolve_material_func, m_import_options);
    result.source_asset_ref = source_asset_ref_for_prim(prim);

    /// Some USDs contain invalid meshes as part of their hierarchy (e.g., Casa_GP), we discard those.
    if (!result.position_stream().valid())
        return;

    std::scoped_lock l(m_scene_mutexes.nodes, m_scene_mutexes.mesh);
    int new_mesh_idx = static_cast<int>(m_scene->meshes.size());
    m_scene->meshes.push_back(std::move(result));
    FALCOR_ASSERT(node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size()));
    m_scene->nodes[node_idx].mesh_index = new_mesh_idx;
}

void UsdImporterContext::extract_curve(pxr::UsdPrim prim, int node_idx)
{
    using namespace pxr;

    auto geometry = convert_usd_curves(
        UsdGeomBasisCurves(prim),
        [this](const UsdShadeMaterialBindingAPI& binding, std::string_view name)
        {
            return resolve_material(binding, name);
        },
        m_import_options
    );
    if (!geometry)
        return;
    if (auto* ribbon = std::get_if<ImporterMesh>(&*geometry)) {
        if (!ribbon->position_stream().valid())
            return;
        ribbon->source_asset_ref = source_asset_ref_for_prim(prim);
        std::scoped_lock lock(m_scene_mutexes.nodes, m_scene_mutexes.mesh);
        const int index = static_cast<int>(m_scene->meshes.size());
        m_scene->meshes.push_back(std::move(*ribbon));
        m_scene->nodes[node_idx].mesh_index = index;
    } else {
        auto& curve = std::get<ImporterCurve>(*geometry);
        curve.source_asset_ref = source_asset_ref_for_prim(prim);
        std::scoped_lock lock(m_scene_mutexes.nodes, m_scene_mutexes.curve);
        const int index = static_cast<int>(m_scene->curves.size());
        m_scene->curves.push_back(std::move(curve));
        m_scene->nodes[node_idx].curve_index = index;
    }
}

void UsdImporterContext::extract_light(pxr::UsdPrim prim, int node_idx)
{
    /// We do not scale any of the dimensions with get_meters_per_unit(),
    /// because that is already baked into the Scene's root transform.
    ///
    /// Authored radiometric controls are preserved independently. Normalization is
    /// evaluated later from the represented light's final world-space dimensions.

    using namespace pxr;

    ImporterLight result;
    result.name = prim.GetPath().GetString();
    result.source_asset_ref = source_asset_ref_for_prim(prim);
    const UsdLuxLightAPI light_api(prim);
    result.intensity
        = get_light_attribute(light_api.GetIntensityAttr(), prim.GetAttribute(TfToken("intensity")), result.intensity);
    result.color = to_falcor(get_light_attribute(
        light_api.GetColorAttr(),
        prim.GetAttribute(TfToken("color")),
        GfVec3f(result.color.x, result.color.y, result.color.z)
    ));
    result.exposure
        = get_light_attribute(light_api.GetExposureAttr(), prim.GetAttribute(TfToken("exposure")), result.exposure);
    result.normalize
        = get_light_attribute(light_api.GetNormalizeAttr(), prim.GetAttribute(TfToken("normalize")), result.normalize);
    result.enable_color_temperature = get_light_attribute(
        light_api.GetEnableColorTemperatureAttr(),
        prim.GetAttribute(TfToken("enableColorTemperature")),
        result.enable_color_temperature
    );
    result.color_temperature = get_light_attribute(
        light_api.GetColorTemperatureAttr(),
        prim.GetAttribute(TfToken("colorTemperature")),
        result.color_temperature
    );

    result.geometry_visible
        = get_attribute(prim.GetAttribute(TfToken("inputs:enableInPrimaryRay")), result.geometry_visible);
    result.geometry_visible
        = get_attribute(prim.GetAttribute(TfToken("karma:light:renderlightgeo")), result.geometry_visible);
    result.geometry_visible
        = get_attribute(prim.GetAttribute(TfToken("falcor:geometryVisible")), result.geometry_visible);

    if (auto distant_light = UsdLuxDistantLight(prim)) {
        result.type = ImporterLight::Type::distant;
        result.degree_angular_diameter = get_light_attribute(
            distant_light.GetAngleAttr(),
            prim.GetAttribute(TfToken("angle")),
            result.degree_angular_diameter
        );
    } else if (auto rect_light = UsdLuxRectLight(prim)) {
        result.type = ImporterLight::Type::rectangular;
        result.enable_shaping = read_light_shaping(prim, result);
        result.width
            = get_light_attribute(rect_light.GetWidthAttr(), prim.GetAttribute(TfToken("width")), result.width);
        result.height
            = get_light_attribute(rect_light.GetHeightAttr(), prim.GetAttribute(TfToken("height")), result.height);
    } else if (auto sphere_light = UsdLuxSphereLight(prim)) {
        result.type = ImporterLight::Type::sphere;
        const float4x4 world_from_light
            = to_falcor(UsdGeomXformable(prim).ComputeLocalToWorldTransform(UsdTimeCode::Default()));
        if (!is_similarity_transform(world_from_light)) {
            sgl::log_warn(
                "SphereLight '{}' has non-uniform world scaling or shear; using volume-equivalent uniform scale {}.",
                prim.GetPath().GetString(),
                volume_equivalent_uniform_scale(world_from_light)
            );
        }
        result.radius
            = get_light_attribute(sphere_light.GetRadiusAttr(), prim.GetAttribute(TfToken("radius")), result.radius);
        result.enable_shaping = read_light_shaping(prim, result);
        result.treat_as_point = get_light_attribute(
            sphere_light.GetTreatAsPointAttr(),
            prim.GetAttribute(TfToken("treatAsPoint")),
            result.treat_as_point
        );

        result.enable_virtual_sphere_shrinking
            = m_enable_virtual_sphere_shrinking && result.type == ImporterLight::Type::sphere && !result.treat_as_point;
    } else if (auto cylinder_light = UsdLuxCylinderLight(prim)) {
        result.type = ImporterLight::Type::cylinder;
        result.enable_shaping = read_light_shaping(prim, result);
        result.radius
            = get_light_attribute(cylinder_light.GetRadiusAttr(), prim.GetAttribute(TfToken("radius")), result.radius);
        result.length
            = get_light_attribute(cylinder_light.GetLengthAttr(), prim.GetAttribute(TfToken("length")), result.length);
        result.treat_as_line = get_light_attribute(
            cylinder_light.GetTreatAsLineAttr(),
            prim.GetAttribute(TfToken("treatAsLine")),
            result.treat_as_line
        );

        const float4x4 world_from_light
            = to_falcor(UsdGeomXformable(prim).ComputeLocalToWorldTransform(UsdTimeCode::Default()));
        if (!is_axially_symmetric_transform(world_from_light, 0)) {
            sgl::log_warn(
                "CylinderLight '{}' has radial non-uniform scaling or shear; using an area-equivalent circular "
                "radius.",
                prim.GetPath().GetString()
            );
        }
    } else if (auto disk_light = UsdLuxDiskLight(prim)) {
        result.type = ImporterLight::Type::disk;
        result.enable_shaping = read_light_shaping(prim, result);
        result.radius
            = get_light_attribute(disk_light.GetRadiusAttr(), prim.GetAttribute(TfToken("radius")), result.radius);
    } else if (auto dome_light = UsdLuxDomeLight(prim)) {
        read_dome_light_properties(dome_light, prim, result);
    } else if (auto dome_light_1 = UsdLuxDomeLight_1(prim)) {
        read_dome_light_properties(dome_light_1, prim, result);
        result.pole_axis = get_dome_light_pole_axis(dome_light_1);
    }

    result.properties = ImporterLight::make_properties(m_import_options, result.type);
    read_import_property<bool>(prim, "falcor:convertToMesh", result.properties, "convert_to_mesh");

    if (!result.env_map_path.empty())
        add_asset_to_scene(result.env_map_path);

    std::scoped_lock l(m_scene_mutexes.nodes, m_scene_mutexes.light);
    int new_light_idx = static_cast<int>(m_scene->lights.size());
    m_scene->lights.push_back(std::move(result));
    FALCOR_ASSERT(node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size()));
    m_scene->nodes[node_idx].light_index = new_light_idx;
}

void UsdImporterContext::extract_camera(pxr::UsdPrim prim, int node_idx)
{
    using namespace pxr;

    GfCamera usd_camera = UsdGeomCamera(prim).GetCamera(0.f);

    ImporterCamera result;
    result.name = prim.GetName().GetString();
    result.source_asset_ref = source_asset_ref_for_prim(prim);

    // value is tenths of a world unit, so *10 to get meters and then /1000 to go to mm
    // (Many scenes have cm as their scene unit, so this becomes a no-op)

    // Focal length, per the USD spec, is supposed to be specified in tenths of a USD world unit.
    // However, it seems to always be specified in mm in OV assets.
    result.focal_length = usd_camera.GetFocalLength();
    const float usd_fstop = usd_camera.GetFStop();
    if (usd_fstop > 0.f)
        result.fstop = usd_fstop;

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

    result.enable_depth_of_field = usd_fstop > 0.f;
    result.focus_distance = usd_camera.GetFocusDistance() * m_meters_per_unit;
    result.depth_range
        = float2(usd_camera.GetClippingRange().GetMin(), usd_camera.GetClippingRange().GetMax()) * m_meters_per_unit;

    switch (usd_camera.GetProjection()) {
    case GfCamera::Projection::Perspective:
        result.projection = ImporterCamera::Projection::perspective;
        break;
    case GfCamera::Projection::Orthographic:
        result.projection = ImporterCamera::Projection::orthographic;
        break;
    }

    std::scoped_lock l(m_scene_mutexes.nodes, m_scene_mutexes.camera);
    int new_camera_idx = static_cast<int>(m_scene->cameras.size());
    m_scene->cameras.push_back(std::move(result));
    FALCOR_ASSERT(node_idx >= 0 && node_idx < static_cast<int>(m_scene->nodes.size()));
    m_scene->nodes[node_idx].camera_index = new_camera_idx;
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

void UsdImporterContext::add_asset_to_scene(const std::filesystem::path& path)
{
    if (!is_package_asset(path))
        return;

    {
        std::scoped_lock l(m_scene_mutexes.asset);
        if (!m_asset_paths.insert(path).second)
            return;
    }

    std::vector<uint8_t> data = read_package_asset(path).value();
    std::scoped_lock l(m_scene_mutexes.asset);
    m_scene->assets.push_back(ImporterAsset{.path = path, .data = std::move(data)});
}

void UsdImporterContext::extract_animations()
{
    if (m_prim_path_to_node_idx.empty())
        return;

    double time_codes_per_second = m_usd_stage->GetTimeCodesPerSecond();
    if (time_codes_per_second <= 0.0)
        time_codes_per_second = 24.0;

    const auto interpolation = m_usd_stage->GetInterpolationType() == pxr::UsdInterpolationTypeHeld
        ? AnimationInterpolationMode::step
        : AnimationInterpolationMode::linear;
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
            channel.interpolation_mode = interpolation;
            channel.times = times;
            channel.values = std::move(translations);
            node.animation_translation = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }

        if (has_rotation_anim) {
            ImporterAnimationChannel channel;
            channel.value_type = AnimationValueType::quatf;
            channel.interpolation_mode = interpolation;
            channel.times = times;
            channel.values = std::move(rotations);
            node.animation_rotation = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }

        if (has_scale_anim) {
            ImporterAnimationChannel channel;
            channel.value_type = AnimationValueType::float3;
            channel.interpolation_mode = interpolation;
            channel.times = times;
            channel.values = std::move(scales);
            node.animation_scale = static_cast<int>(m_scene->animation.channels.size());
            m_scene->animation.channels.push_back(std::move(channel));
        }
    }
}

pxr::SdfPath UsdImporterContext::source_path_for_prim(const pxr::UsdPrim& prim) const
{
    if (!prim.IsInPrototype())
        return prim.GetPath();
    const pxr::SdfPath prototype_path = get_prototype_root(prim).GetPath();
    return prim.GetPath().ReplacePrefix(prototype_path, m_prototype_source_paths.at(prototype_path));
}

SourceAssetRef UsdImporterContext::source_asset_ref_for_object(std::string_view object_path) const
{
    static const ustring format("usd");
    return SourceAssetRef(format, m_source_asset_path, ustring(object_path));
}

SourceAssetRef UsdImporterContext::source_asset_ref_for_prim(const pxr::UsdPrim& prim) const
{
    return source_asset_ref_for_object(source_path_for_prim(prim).GetString());
}

} // namespace usd_importer
} // namespace falcor
