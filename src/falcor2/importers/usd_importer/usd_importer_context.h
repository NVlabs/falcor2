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
#include <pxr/usd/usd/prim.h>

#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/metrics.h>
END_DISABLE_USD_WARNINGS

#include <filesystem>
#include <map>
#include <mutex>
#include <set>
#include <vector>

namespace falcor {
namespace usd_importer {

class UsdImporterContext {
public:
    UsdImporterContext(
        const pxr::UsdStageRefPtr& stage,
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
        run_traversal_and_wait(
            m_task_group,
            [&]
            {
                traverse_scene(root);
            }
        );
        extract_animations();
    }

    ref<ImporterScene> get_scene() { return m_scene; }

    std::string get_name(const pxr::UsdPrim& prim) const { return prim.GetPath().GetString(); }

private:
    void traverse_scene(const pxr::UsdPrim& root);

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

    /// Add a package-contained asset to the scene.
    void add_asset_to_scene(const std::filesystem::path& path);

    pxr::SdfPath source_path_for_prim(const pxr::UsdPrim& prim) const;
    SourceAssetRef source_asset_ref_for_object(std::string_view object_path) const;
    SourceAssetRef source_asset_ref_for_prim(const pxr::UsdPrim& prim) const;

private:
    pxr::UsdStageRefPtr m_usd_stage;
    const ustring m_source_asset_path;
    /// Representative paths resolved once, before asynchronous extraction starts.
    const std::map<pxr::SdfPath, pxr::SdfPath> m_prototype_source_paths;
    ImportOptions m_import_options;

    float m_meters_per_unit;
    std::vector<std::vector<int>> m_node_idx_stacks;
    ref<ImporterScene> m_scene;

    /// Maps Usd prototype prims to their index in the ImporterScene.
    std::map<pxr::UsdPrim, int> m_prototype_to_index;

    /// Maps Usd prim paths to their node index in the ImporterScene (for animation).
    std::map<pxr::SdfPath, int> m_prim_path_to_node_idx;

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
    sgl::thread::TaskGroup m_task_group;
};

} // namespace usd_importer
} // namespace falcor
