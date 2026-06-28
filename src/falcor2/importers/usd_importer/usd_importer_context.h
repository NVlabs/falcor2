// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/usd_importer/usd_importer_utils.h"

#include "sgl/core/thread.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/prim.h>

#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/metrics.h>
END_DISABLE_USD_WARNINGS

#include <vector>
#include <mutex>


namespace falcor {
namespace usd_importer {

class UsdImporterContext {
public:
    UsdImporterContext(const pxr::UsdStageRefPtr& stage, bool parallel_adds = true)
        : m_usd_stage(stage)
        , m_parallel_adds(parallel_adds)
    {
        m_node_idx_stacks.resize(1);
        m_meters_per_unit = (float)pxr::UsdGeomGetStageMetersPerUnit(m_usd_stage);
        m_scene = make_ref<ImporterScene>();
    }

    ~UsdImporterContext() { m_task_group.wait(); }

    void traverse_scene(const pxr::UsdPrim& root);

    /// How many meters is a single scene unit.
    /// Multiply scene units with this to get meters.
    float get_meters_per_unit() const { return m_meters_per_unit; }

    bool should_drop_prim(const pxr::UsdPrim&)
    {
        return false; /// We do not implement filtering yet.
    }

    int push_transform(const float4x4& parent_from_local, std::string_view name = "")
    {
        return push_transform(parent_from_local, name, parent_node_idx());
    }

    int push_transform(const pxr::UsdGeomXformable& prim)
    {
        bool resets = false;
        pxr::GfMatrix4d usd_transform;
        prim.GetLocalTransformation(&usd_transform, &resets, pxr::UsdTimeCode::EarliestTime());
        int node_idx = push_transform(
            to_falcor(usd_transform),
            get_name(prim.GetPrim()),
            resets ? root_node_idx() : parent_node_idx()
        );
        // Track prim path -> node index for animation extraction.
        m_prim_path_to_node_idx[prim.GetPrim().GetPath()] = node_idx;
        return node_idx;
    }

    void pop_transform() { node_idx_stack().pop_back(); }

    void add_mesh(pxr::UsdPrim prim, bool run_immediate = false);
    void add_curve(pxr::UsdPrim prim, bool run_immediate = false);
    void add_light(pxr::UsdPrim prim, bool run_immediate = false);
    void add_camera(pxr::UsdPrim prim, bool run_immediate = false);
    void add_instance(pxr::UsdPrim prim);
    void add_point_instancer(pxr::UsdPrim prim);

    void finalize()
    {
        m_task_group.wait();
        extract_animations();
    }

    ref<ImporterScene> get_scene() { return m_scene; }

    std::string get_name(const pxr::UsdPrim& prim) const { return prim.GetPath().GetString(); }

private:
    struct UsdSceneMutexes;

    std::vector<int>& node_idx_stack() { return m_node_idx_stacks.back(); }
    const std::vector<int>& node_idx_stack() const { return m_node_idx_stacks.back(); }
    void push_prototype_stack() { m_node_idx_stacks.push_back({}); }
    void pop_prototype_stack() { m_node_idx_stacks.pop_back(); }
    bool is_prototype() const { return m_node_idx_stacks.size() > 1; }

    /// Get the current parent node index, -1 if none is present.
    int parent_node_idx() const { return node_idx_stack().empty() ? -1 : node_idx_stack().back(); }

    /// Returns the current root node, used when `GetLocalTransformation` asks for reset.
    /// The reset ignores inheritance in the Usd hierarchy, but we still want to inherit from the
    /// UsdImporter provided transform that scales and rotates the scene.
    int root_node_idx() const { return node_idx_stack().empty() ? -1 : node_idx_stack().front(); }

    int push_transform(const float4x4& parent_from_local, std::string_view name, int parent)
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
        node_idx_stack().push_back(new_node_idx);
        return new_node_idx;
    }

    /// Adds root node to the current scope (scene or prototype in the future).
    void add_root_node(int root_idx) { m_scene->root_nodes.push_back(root_idx); }

    /// Thread safe, locks access to caches.
    pxr::UsdShadeMaterial get_bound_material(const pxr::UsdShadeMaterialBindingAPI& binding_api)
    {
        std::lock_guard l(m_scene_mutexes.material);
        return binding_api.ComputeBoundMaterial(&m_bindings_cache, &m_coll_query_cache);
    }

    /// Thread safe, locks access to m_material_name_to_index and m_scene.materials.
    std::string resolve_material(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view mesh_name = "");

    /// Thread safe, scene access is locked, resolve_material_func is resolve_material.
    /// Separated out to be easy to package into a task.
    /// The `ws_from_local` must be captured by value, rather than referene to m_scene_mutexes.
    static void add_mesh(
        pxr::UsdPrim prim,
        int parent,
        ImporterScene* scene,
        UsdSceneMutexes& scene_mutexes,
        std::function<std::string(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view)>
            resolve_material_func
    );

    /// Thread safe, scene access is locked.
    static void add_curve(
        pxr::UsdPrim prim,
        int parent,
        ImporterScene* scene,
        UsdSceneMutexes& scene_mutexes,
        std::function<std::string(const pxr::UsdShadeMaterialBindingAPI& binding_api, std::string_view)>
            resolve_material_func
    );

    /// Thread safe, scene access is locked.
    /// Separated out to be easy to package into a task.
    /// The `ws_from_local` must be captured by value, rather than referene to m_scene_mutexes.
    static void add_light(pxr::UsdPrim prim, int parent, ImporterScene* scene, UsdSceneMutexes& scene_mutexes);
    static ImporterLight create_light(pxr::UsdPrim prim);

    /// Thread safe, scene access is locked.
    /// Separated out to be easy to package into a task.
    /// The `ws_from_local` must be captured by value, rather than referene to m_scene_mutexes.
    static void add_camera(
        pxr::UsdPrim prim,
        int parent,
        float meters_per_unit,
        ImporterScene* scene,
        UsdSceneMutexes& scene_mutexes
    );

    /// Returns an index of prototype, creating it if necessary.
    int add_prototype(pxr::UsdPrim prototype);

    /// Extract animation data from USD time samples and populate ImporterAnimation.
    void extract_animations();

    /// Run or enqeue tasks, depending on whether we do parallism.
    template<typename F>
    void process_task(F&& task, bool run_immediate)
    {
        if (run_immediate || !m_parallel_adds)
            task();
        else
            m_task_group.do_async(task);
    }

    /// Add a texture to the scene and return its index
    int add_texture_to_scene(const ImporterTexture& texture);

private:
    pxr::UsdStageRefPtr m_usd_stage;

    float m_meters_per_unit;
    pxr::UsdShadeMaterialBindingAPI::CollectionQueryCache m_coll_query_cache; ///< Material collection binding cache
    pxr::UsdShadeMaterialBindingAPI::BindingsCache m_bindings_cache;          ///< Material binding cache

    std::vector<std::vector<int>> m_node_idx_stacks;
    ref<ImporterScene> m_scene;

    /// Maps Usd prototype prims to their index in the ImporterScene.
    std::map<pxr::UsdPrim, int> m_prototype_to_index;

    /// Maps Usd prim paths to their node index in the ImporterScene (for animation).
    std::map<pxr::SdfPath, int> m_prim_path_to_node_idx;

    struct UsdSceneMutexes {
        std::mutex nodes;
        std::mutex material;
        std::mutex texture;
        std::mutex camera;
        std::mutex light;
        std::mutex mesh;
        std::mutex curve;
    };

    UsdSceneMutexes m_scene_mutexes;
    std::map<std::string, int> m_material_name_to_index;
    std::map<ImporterTexture, int> m_texture_to_index;
    bool m_parallel_adds = false;
    sgl::thread::TaskGroup m_task_group;
};

} // namespace usd_importer
} // namespace falcor
