// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_edits.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"
#include "falcor2/core/asset_resolver.h"
#include "falcor2/core/blob_cache.h"

#include <sgl/core/logger.h>
#include <sgl/core/string.h>
#include <sgl/core/thread.h>

#include <exception>
#include <mutex>
#include <utility>

namespace falcor {

namespace {

std::mutex s_importer_cache_mutex;
ref<BlobCache> s_importer_cache;
thread_local ref<Importer> s_current_importer;

ref<BlobCache> ensure_importer_cache()
{
    std::lock_guard lock(s_importer_cache_mutex);
    if (s_importer_cache)
        return s_importer_cache;

    try {
        s_importer_cache = make_ref<BlobCache>();
    } catch (const std::exception& e) {
        sgl::log_warn("Failed to create importer blob cache: {}", e.what());
    }

    return s_importer_cache;
}

} // namespace

Importer::Importer(ImportOptions default_import_options)
    : m_default_import_options(default_import_options)
    , m_env(*this)
    , m_cameras(*this)
    , m_nodes(*this)
    , m_materials(*this)
{
}

Importer::~Importer() = default;

ref<Importer> Importer::create(ImportOptions default_import_options)
{
    return make_ref<Importer>(default_import_options);
}

ref<Importer> Importer::get()
{
    if (!s_current_importer) {
        s_current_importer = Importer::create();
    }
    return s_current_importer;
}

void Importer::set_current(ref<Importer> importer)
{
    s_current_importer = std::move(importer);
}

void Importer::clear_current()
{
    s_current_importer.reset();
}

ImporterCameraCollection& Importer::cameras()
{
    return m_cameras;
}

ImporterNodeCollection& Importer::nodes()
{
    return m_nodes;
}

ImporterMaterialCollection& Importer::materials()
{
    return m_materials;
}

ImporterEnvCollection& Importer::env()
{
    return m_env;
}

void Importer::set_source_path(const std::filesystem::path& path)
{
    m_source_path = path.empty() ? std::filesystem::path{} : std::filesystem::absolute(path).lexically_normal();
}

void Importer::add_search_path(const std::filesystem::path& path)
{
    m_search_paths.push_back(path);
}

void Importer::add_search_paths(std::span<const std::filesystem::path> paths)
{
    m_search_paths.insert(m_search_paths.end(), paths.begin(), paths.end());
}

void Importer::import_asset(const std::filesystem::path& path)
{
    add_edit(std::make_unique<ImportAssetEdit>(path, m_default_import_options));
}

void Importer::on_scene_created(SceneCreatedCallback callback)
{
    m_scene_created_callbacks.push_back(std::move(callback));
}

void Importer::run_scene_created_callbacks(ref<Scene> scene) const
{
    for (const auto& callback : m_scene_created_callbacks) {
        callback(scene);
    }
}

ref<ImporterScene> Importer::build_importer_scene() const
{
    ImporterBuildContext context;
    context.scene = make_ref<ImporterScene>();

    std::vector<std::filesystem::path> search_paths;
    search_paths.reserve(m_search_paths.size() + (m_source_path.empty() ? 0 : 1));
    if (!m_source_path.empty()) {
        context.source_directory = m_source_path.parent_path();
        search_paths.push_back(context.source_directory);
    }
    for (const auto& search_path : m_search_paths) {
        search_paths.push_back(
            !context.source_directory.empty() && search_path.is_relative() ? context.source_directory / search_path
                                                                           : search_path
        );
    }
    context.asset_resolver = make_ref<AssetResolver>(std::span<const std::filesystem::path>(search_paths));

    for (const auto& edit : m_edits) {
        edit->apply(context);
    }

    context.scene->calculate_aabbs();
    return context.scene;
}

uint32_t Importer::next_selector_id()
{
    return m_next_selector_id++;
}

void Importer::add_edit(std::unique_ptr<ImporterEdit> edit)
{
    m_edits.push_back(std::move(edit));
}

ScopedCurrentImporter::ScopedCurrentImporter(ref<Importer> importer)
    : m_previous(s_current_importer)
{
    Importer::set_current(std::move(importer));
}

ScopedCurrentImporter::~ScopedCurrentImporter()
{
    Importer::set_current(std::move(m_previous));
}

void set_importer_cache(ref<BlobCache> cache)
{
    std::lock_guard lock(s_importer_cache_mutex);
    s_importer_cache = std::move(cache);
}

ref<BlobCache> importer_cache()
{
    std::lock_guard lock(s_importer_cache_mutex);
    return s_importer_cache;
}

ref<ImporterScene> import_scene(const std::filesystem::path& path, const ImportOptions& import_options)
{
    // Ensure the global importer cache is set up.
    ensure_importer_cache();

    ref<ImporterScene> importer_scene;
    std::string extension = sgl::string::to_lower(path.extension().string());
    if (extension == ".usd" || extension == ".usda" || extension == ".usdc") {
        importer_scene = UsdImporter().load_scene(path);
    } else if (extension == ".gltf" || extension == ".glb") {
        importer_scene = GltfImporter().load_scene(path);
    } else {
        FALCOR_THROW("Unknown scene file extension \"{}\"", extension);
    }

    if (import_options.recompute_normals) {
        sgl::thread::parallel_for(
            sgl::thread::blocked_range<size_t>(0, importer_scene->meshes.size()),
            [&](auto range)
            {
                for (auto i : range) {
                    auto& mesh = importer_scene->meshes[i];
                    mesh.add_normals_from_faces();
                    mesh.add_tangents_from_normals();
                }
            }
        );
    }

    return importer_scene;
}

} // namespace falcor
