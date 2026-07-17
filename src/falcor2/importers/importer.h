// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/fwd.h"
#include "falcor2/core/types.h"
#include "falcor2/core/object.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer_collections.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <vector>

namespace falcor {

class Scene;

/// Options used when importing source scene assets.
struct ImportOptions {
    /// Recompute mesh normals and tangents after asset import.
    bool recompute_normals = false;
};

/// Native importer edit stream used to build ImporterScene instances.
///
/// Importer owns contextual collections and a list of deferred edits. The
/// collections append edits; they do not store created importer objects. Calling
/// build_importer_scene creates a fresh ImporterScene and applies the recorded
/// edits in order.
class FALCOR_API Importer : public Object {
public:
    /// Callback invoked after a Python scene creates the live render Scene.
    using SceneCreatedCallback = std::function<void(ref<Scene>)>;

    /// Create an empty importer with the specified default import options.
    explicit Importer(ImportOptions default_import_options = {});
    ~Importer() override;

    /// Create a reference-counted importer.
    static ref<Importer> create(ImportOptions default_import_options = {});
    /// Return the current importer for this thread, creating one if needed.
    static ref<Importer> get();
    /// Set the current importer for this thread.
    static void set_current(ref<Importer> importer);
    /// Clear the current importer for this thread.
    static void clear_current();

    /// Camera edit-recording collection.
    ImporterCameraCollection& cameras();
    /// Node edit-recording collection.
    ImporterNodeCollection& nodes();
    /// Material edit-recording collection.
    ImporterMaterialCollection& materials();
    /// Environment edit-recording collection.
    ImporterEnvCollection& env();

    /// Source Python file used to resolve scene-relative paths, or an empty path when unset.
    const std::filesystem::path& source_path() const { return m_source_path; }
    /// Set the source Python file, normalizing non-empty paths to absolute lexical form immediately.
    void set_source_path(const std::filesystem::path& path);

    /// Record one additional build-time asset search root without accessing the filesystem.
    void add_search_path(const std::filesystem::path& path);
    /// Record additional build-time asset search roots without accessing the filesystem.
    void add_search_paths(std::span<const std::filesystem::path> paths);

    /// Record a deferred asset import edit.
    ///
    /// The asset is not loaded until build_importer_scene applies the edit.
    void import_asset(const std::filesystem::path& path);

    /// Register a callback to run after a Python scene has created the live render Scene.
    void on_scene_created(SceneCreatedCallback callback);
    /// Run scene-created callbacks in registration order.
    void run_scene_created_callbacks(ref<Scene> scene) const;

    /// Build a fresh ImporterScene by applying all recorded edits in order.
    ref<ImporterScene> build_importer_scene() const;

private:
    friend class ImporterCameraCollection;
    friend class ImporterNodeCollection;
    friend class ImporterEnvCollection;
    friend class ImporterMaterialCollection;
    friend class ImporterMaterialSelector;

    uint32_t next_selector_id();
    void add_edit(std::unique_ptr<ImporterEdit> edit);

    ImportOptions m_default_import_options;
    std::filesystem::path m_source_path;
    std::vector<std::filesystem::path> m_search_paths;
    ImporterEnvCollection m_env;
    ImporterCameraCollection m_cameras;
    ImporterNodeCollection m_nodes;
    ImporterMaterialCollection m_materials;
    std::vector<std::unique_ptr<ImporterEdit>> m_edits;
    std::vector<SceneCreatedCallback> m_scene_created_callbacks;
    uint32_t m_next_selector_id = 1;

    FALCOR_NON_COPYABLE_AND_MOVABLE(Importer);
};

/// Scoped helper that temporarily installs a current importer for this thread.
///
/// The previous current importer is restored when the scope exits.
class FALCOR_API ScopedCurrentImporter {
public:
    /// Install importer as the current importer until this object is destroyed.
    explicit ScopedCurrentImporter(ref<Importer> importer);
    ~ScopedCurrentImporter();

private:
    ref<Importer> m_previous;

    FALCOR_NON_COPYABLE_AND_MOVABLE(ScopedCurrentImporter);
};

ref<ImporterScene> FALCOR_API import_scene(const std::filesystem::path& path, const ImportOptions& import_options = {});

/// Set the global blob cache used by importers (e.g. for MikkTSpace caching).
FALCOR_API void set_importer_cache(ref<BlobCache> cache);

/// Get the global blob cache used by importers. May return nullptr if not set.
FALCOR_API ref<BlobCache> importer_cache();

} // namespace falcor
