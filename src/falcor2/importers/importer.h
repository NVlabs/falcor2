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

#include <array>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace falcor {

class Scene;

/// Selects the runtime material used for glTF PBR materials. Unlit materials keep their native type.
enum class GLTFMaterialMode {
    /// Historical name for the dedicated native GLTFMaterial/GLTFSpecGlossMaterial route.
    standard,
    /// Use native glTF for supported combinations, otherwise use the MDL SDK route.
    automatic,
    /// Use the MDL SDK glTF material for all PBR materials.
    mdl,
    /// Explicitly select dedicated native glTF materials.
    gltf,
    /// Convert to StandardMaterial/StandardSpecGlossMaterial as an approximate compatibility target.
    standard_material,
    /// Generate a MaterialX graph using its existing glTF material definition.
    materialx,
    /// Convert supported inputs to OpenPBR without extending its GPU implementation.
    openpbr,
};
SGL_ENUM_INFO(
    GLTFMaterialMode,
    {
        {GLTFMaterialMode::standard, "standard"},
        {GLTFMaterialMode::automatic, "automatic"},
        {GLTFMaterialMode::mdl, "mdl"},
        {GLTFMaterialMode::gltf, "gltf"},
        {GLTFMaterialMode::standard_material, "standard_material"},
        {GLTFMaterialMode::materialx, "materialx"},
        {GLTFMaterialMode::openpbr, "openpbr"},
    }
);
SGL_ENUM_REGISTER(GLTFMaterialMode);

/// Options used when importing source scene assets.
/// USD static values use the authored playback start, or numeric zero without a complete playback range.
struct ImportOptions {
    /// Import USD mesh position and normal time samples. Scene loading uses sample zero; curves are unaffected.
    bool import_mesh_motion = false;

    /// Import USD curve position, width, and ribbon normal time samples. Scene loading uses sample zero.
    /// Oriented ribbons store these as mesh samples; this option also controls their import.
    bool import_curve_motion = false;

    /// Retain USD linear-blend skin bindings and joint animation for explicit ImporterScene::bake_pose() calls.
    /// bake_poses also enables this import. Runtime skinning is not enabled.
    bool usd_import_skinning = false;

    /// Attempt to detect, report and fix invalid geometry.
    bool repair_broken_geometry = true;
    /// Default for per-mesh normal and tangent recomputation after asset import.
    bool recompute_normals = false;
    /// Bake imported skin bindings into static geometry. USD evaluates imported animation at its initial viewing time;
    /// glTF uses the stored pose. Ordinary node transforms and animation remain unchanged.
    /// Skinned USD meshes retain their control cage. Combined mesh motion and skinning, blend shapes,
    /// dual-quaternion skinning and instanced USD skinning are unsupported.
    bool bake_poses = false;
    /// Default mesh refinement level, in [0, 256]. Zero selects the control cage; positive N uses N+1 edge segments.
    int mesh_refinement_level = 0;
    /// Default curve refinement level, in [0, 256]. Zero selects CVs; positive N uses N+1 segments per cubic span.
    int curve_refinement_level = 0;
    /// Default for converting rectangular analytic lights to emissive geometry regardless of source visibility.
    bool force_rectangle_lights_to_geometry = false;
    /// Default for converting disk analytic lights to emissive geometry regardless of source visibility.
    bool force_disk_lights_to_geometry = false;
    /// Default for converting finite sphere analytic lights to emissive geometry regardless of source visibility.
    bool force_sphere_lights_to_geometry = false;
    /// Exact, case-sensitive USD primvar names imported as texture coordinate sets 1 through 7.
    /// One leading "primvars:" prefix is optional; an empty name disables that set.
    std::array<std::string, 7> usd_texcoord_primvar_names = {"st1", "st2", "st3", "st4", "st5", "st6", "st7"};
    /// Select the runtime material when importing glTF PBR materials. Standard is the default;
    /// the choice is retained with each imported material, including deferred imports and append.
    GLTFMaterialMode gltf_material_mode = GLTFMaterialMode::standard;
};

/// Native importer edit stream used to build ImporterScene instances.
///
/// Importer owns contextual collections and a list of deferred edits. The
/// collections append edits; they do not store created importer objects. Calling
/// build_importer_scene creates a fresh ImporterScene and applies the recorded
/// edits in order.
class FALCOR_API Importer : public Object {
public:
    /// Callback invoked after importer data is loaded into a live render Scene.
    using SceneLoadedCallback = std::function<void(ref<Scene>)>;

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

    /// Register a callback to run after importer data is loaded into a live render Scene.
    void on_scene_loaded(SceneLoadedCallback callback);
    /// Run scene-loaded callbacks in registration order.
    void run_scene_loaded_callbacks(ref<Scene> scene) const;

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
    std::vector<SceneLoadedCallback> m_scene_loaded_callbacks;
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

namespace detail {

/// Apply generic import options to imported CPU-side scene data.
void apply_import_options(
    ImporterScene& scene,
    const ImportOptions& options,
    std::optional<double> bake_time_seconds = {}
);

} // namespace detail

} // namespace falcor
