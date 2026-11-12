// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"
#include "falcor2/core/ustring.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/usd_importer/usd_importer_utils.h"
#include "falcor2/importers/usd_importer/usd_importer_tasks.h"

#include "sgl/core/thread.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/ar/resolverContextBinder.h>
#include <pxr/usd/ar/resolverScopedCache.h>
#include <pxr/usd/usd/prim.h>

#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdSkel/skeletonQuery.h>
#include <pxr/usd/usdSkel/skinningQuery.h>
END_DISABLE_USD_WARNINGS

#include <filesystem>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <utility>
#include <vector>

namespace falcor {
namespace usd_importer {

class UsdImporterContext {
public:
    UsdImporterContext(
        const pxr::UsdStageRefPtr& stage,
        pxr::UsdTimeCode initial_time,
        const ImportOptions& import_options,
        ustring source_asset_path,
        bool parallel_adds = true
    );

    /// How many meters is a single scene unit.
    /// Multiply scene units with this to get meters.
    float get_meters_per_unit() const { return m_meters_per_unit; }

    bool should_drop_prim(const pxr::UsdPrim&)
    {
        return false; /// We do not implement filtering yet.
    }

    int push_transform(const float4x4& parent_from_local, std::string_view name = "")
    {
        return push_transform(parent_from_local, name, current_node_idx());
    }

    int push_transform(const pxr::UsdGeomXformable& prim);

    void pop_transform() { node_idx_stack().pop_back(); }

    /// Attach prim contents to the node created by traversal; asynchronous tasks capture its index by value.
    void add_mesh(pxr::UsdPrim prim, int node_idx, bool run_immediate = false);
    void add_curve(pxr::UsdPrim prim, int node_idx, bool run_immediate = false);
    void add_light(pxr::UsdPrim prim, int node_idx, bool run_immediate = false);
    void add_camera(pxr::UsdPrim prim, int node_idx, bool run_immediate = false);
    void add_instance(pxr::UsdPrim prim, int node_idx);
    void add_point_instancer(pxr::UsdPrim prim, int node_idx);

    void run(const pxr::UsdPrim& root)
    {
        prepare_skinning(root);
        run_traversal_and_wait(
            m_task_group,
            [&]
            {
                traverse_scene(root);
            }
        );
        extract_animations();
        extract_skinning();
    }

    ref<ImporterScene> get_scene() { return m_scene; }

    std::string get_name(const pxr::UsdPrim& prim) const { return prim.GetPath().GetString(); }

private:
    /// Usually only authored is valid and holds the complete local transform.
    /// For potentially animated transforms with an effective pivot operation,
    /// pivot and pivot_inverse are also valid: they surround authored with
    /// T(pivot) and T(-pivot), while authored holds the translation/rotation/scale.
    struct TransformNodes {
        int authored;
        int pivot = -1;
        int pivot_inverse = -1;
    };

    void traverse_scene(const pxr::UsdPrim& root);

    TransformNodes
    add_transform_nodes(const float4x4& transform, std::string_view name, int parent, std::optional<float3> pivot);

    std::vector<int>& node_idx_stack() { return m_node_idx_stacks.back(); }
    const std::vector<int>& node_idx_stack() const { return m_node_idx_stacks.back(); }
    void push_prototype_stack() { m_node_idx_stacks.push_back({}); }
    void pop_prototype_stack() { m_node_idx_stacks.pop_back(); }
    bool is_prototype() const { return m_node_idx_stacks.size() > 1; }

    /// Get the current node index, -1 if none is present.
    int current_node_idx() const { return node_idx_stack().empty() ? -1 : node_idx_stack().back(); }

    /// Returns the current root node, used when `GetLocalTransformation` asks for reset.
    /// The reset ignores inheritance in the Usd hierarchy, but we still want to inherit from the
    /// UsdImporter provided transform that scales and rotates the scene.
    int root_node_idx() const { return node_idx_stack().empty() ? -1 : node_idx_stack().front(); }

    int push_transform(const float4x4& parent_from_local, std::string_view name, int parent)
    {
        int node_idx = add_transform(parent_from_local, name, parent);
        node_idx_stack().push_back(node_idx);
        return node_idx;
    }

    int add_transform(const float4x4& parent_from_local, std::string_view name, int parent)
    {
        ImporterNode node;
        node.name = name;
        node.parent = parent;
        node.transform = parent_from_local;

        std::scoped_lock l(m_scene_mutexes.nodes);
        int new_node_idx = static_cast<int>(m_scene->nodes.size());
        if (parent >= 0) {
            m_scene->nodes[parent].children.push_back(new_node_idx);
        } else if (!is_prototype()) {
            add_root_node(new_node_idx);
        }
        m_scene->nodes.push_back(std::move(node));
        return new_node_idx;
    }

    /// Adds root node to the current scope (scene or prototype in the future).
    void add_root_node(int root_idx) { m_scene->root_nodes.push_back(root_idx); }

    /// Resolve the bound material inside USD without exposing its TBB-backed cache types.
    pxr::UsdShadeMaterial get_bound_material(const pxr::UsdShadeMaterialBindingAPI& binding_api)
    {
        return binding_api.ComputeBoundMaterial();
    }

    /// Thread safe, locks access to m_material_name_to_index and m_scene.materials.
    int resolve_material(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name = "");

    /// Extract mesh data and attach it to the node. Shared scene access is locked.
    void extract_mesh(pxr::UsdPrim prim, int node_idx);

    /// Extract curve data and attach it to the node. Shared scene access is locked.
    void extract_curve(pxr::UsdPrim prim, int node_idx);

    /// Extract light data and attach it to the node. Shared scene access is locked.
    void extract_light(pxr::UsdPrim prim, int node_idx);

    /// Extract camera data and attach it to the node. Shared scene access is locked.
    void extract_camera(pxr::UsdPrim prim, int node_idx);

    /// Returns an index of prototype, creating it if necessary.
    int add_prototype(pxr::UsdPrim prototype);

    /// Extract animation data from USD time samples and populate ImporterAnimation.
    void extract_animations();

    void prepare_skinning(const pxr::UsdPrim& root);
    ImporterSkin import_skeleton(const pxr::UsdSkelSkeletonQuery& query);
    void extract_skinning();

    struct SkinBinding {
        pxr::UsdSkelSkinningQuery mesh;
        pxr::UsdSkelSkeletonQuery skeleton;
    };

    /// Run or enqeue tasks, depending on whether we do parallism.
    template<typename F>
    void process_task(F&& task, bool run_immediate)
    {
        // Resolver bindings are thread-local. A pool worker must use this import's context,
        // including when the waiting caller helps execute work from another import.
        auto bound_task = [context = m_resolver_context, cache = &m_resolver_cache, task = std::forward<F>(task)]
        {
            pxr::ArResolverContextBinder binder(context);
            // Cache scopes are also thread-local. Share package mappings and resolutions
            // with this import rather than reopening USDZ archives for every asset query.
            pxr::ArResolverScopedCache cache_scope(cache);
            task();
        };
        if (run_immediate || !m_parallel_adds)
            bound_task();
        else
            m_task_group.do_async(std::move(bound_task));
    }

    /// Add a texture to the scene and return its index
    int add_texture_to_scene(const ImporterTexture& texture);

    /// Add a package-contained asset to the scene.
    void add_asset_to_scene(const std::filesystem::path& path);

    pxr::SdfPath source_path_for_prim(const pxr::UsdPrim& prim) const;
    SourceAssetRef source_asset_ref_for_object(std::string_view object_path) const;
    SourceAssetRef source_asset_ref_for_prim(const pxr::UsdPrim& prim) const;

private:
    pxr::UsdStageRefPtr m_usd_stage;
    const pxr::UsdTimeCode m_initial_time;
    const pxr::ArResolverContext m_resolver_context;
    // Construct and destroy on the importing thread; run() drains all children first.
    pxr::ArResolverScopedCache m_resolver_cache;
    const ustring m_source_asset_path;
    /// Representative paths resolved once, before asynchronous extraction starts.
    const std::map<pxr::SdfPath, pxr::SdfPath> m_prototype_source_paths;
    ImportOptions m_import_options;

    float m_meters_per_unit;
    std::vector<std::vector<int>> m_node_idx_stacks;
    ref<ImporterScene> m_scene;

    /// Maps Usd prototype prims to their index in the ImporterScene.
    std::map<pxr::UsdPrim, int> m_prototype_to_index;

    /// A USD prim can have both an ordinary scene node and a separately imported prototype node.
    /// Animation targets the authored TRS node and optional pivot helpers.
    std::map<pxr::SdfPath, std::vector<TransformNodes>> m_prim_path_to_nodes;

    // Populated before parallel extraction; queries and their source stage remain immutable during import.
    std::map<pxr::SdfPath, SkinBinding> m_skin_bindings;

    struct UsdSceneMutexes {
        std::mutex nodes;
        std::mutex material;
        std::mutex asset;
        std::mutex texture;
        std::mutex camera;
        std::mutex light;
        std::mutex mesh;
        std::mutex curve;
    };

    UsdSceneMutexes m_scene_mutexes;
    std::map<std::string, int> m_material_name_to_index;
    std::map<ImporterTexture, int> m_texture_to_index;
    std::set<std::filesystem::path> m_asset_paths;
    bool m_enable_virtual_sphere_shrinking = false;
    bool m_parallel_adds = false;
    USDTaskGroup m_task_group;
};

} // namespace usd_importer
} // namespace falcor
