// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/asset_resolver.h"
#include "falcor2/core/object.h"
#include "falcor2/core/macros.h"
#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_collections.h"
#include "falcor2/importers/importer_types.h"

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>

namespace falcor {

/// Transient state used while applying an Importer edit list into a new ImporterScene.
class FALCOR_API ImporterBuildContext {
public:
    /// Resolve one required edit path or throw a build-time diagnostic.
    std::filesystem::path resolve_required_asset(const std::filesystem::path& path, std::string_view kind) const;

    /// Scene currently being built by applying importer edits.
    ref<ImporterScene> scene;
    /// Immutable resolver shared by all path-bearing edits in this build.
    ref<AssetResolver> asset_resolver = make_ref<AssetResolver>();
    /// Normalized source directory, or an empty path when source context is unset.
    std::filesystem::path source_directory;
    /// Mapping from ImporterCameraSelector ids to ImporterScene::cameras indices.
    std::unordered_map<uint32_t, int> cameras;
    /// Mapping from ImporterNodeSelector ids to ImporterScene::nodes indices.
    std::unordered_map<uint32_t, int> nodes;
};

/// Base class for deferred importer edits.
///
/// Edits are recorded by contextual collections and applied later, in order, to
/// construct a fresh ImporterScene.
class FALCOR_API ImporterEdit {
public:
    virtual ~ImporterEdit() = default;
    /// Apply this edit to the importer build context.
    virtual void apply(ImporterBuildContext& context) const = 0;
};

/// Deferred edit that imports an external scene asset during build.
///
/// ImportAssetEdit records the requested path and import options but does not
/// load the asset until apply is called.
class FALCOR_API ImportAssetEdit : public ImporterEdit {
public:
    /// Create an asset import edit for the given source path and options.
    ImportAssetEdit(std::filesystem::path path, ImportOptions import_options);

    void apply(ImporterBuildContext& context) const override;

private:
    std::filesystem::path m_path;
    ImportOptions m_import_options;
};

/// Deferred edit that replaces every material matching an exact name.
///
/// Replacement preserves each material name, overwrites its parameters, and
/// clears any imported material network. No matches is a valid no-op.
class FALCOR_API ReplaceMaterialEdit : public ImporterEdit {
public:
    /// Create an exact-name material replacement edit.
    ReplaceMaterialEdit(std::string name, std::string material_type, Properties props);

    /// Create an exact-name material replacement edit backed by a live constructor.
    ReplaceMaterialEdit(std::string name, ImporterMaterial::Constructor constructor, Properties props);

    void apply(ImporterBuildContext& context) const override;

private:
    std::string m_name;
    std::string m_material_type;
    ImporterMaterial::Constructor m_constructor;
    Properties m_props;
};

/// Deferred edit that sets the scene environment map.
///
/// The edit overwrites the first existing dome light when applied. If the scene
/// has no dome light, it creates a dome light and a root node for it.
class FALCOR_API SetEnvironmentEdit : public ImporterEdit {
public:
    /// Create an environment setup edit.
    SetEnvironmentEdit(std::filesystem::path path, float exposure, std::string name);

    void apply(ImporterBuildContext& context) const override;

private:
    std::filesystem::path m_path;
    float m_exposure = 0.f;
    std::string m_name;
};

/// Deferred edit that appends a camera to the ImporterScene.
class FALCOR_API CreateCameraEdit : public ImporterEdit {
public:
    /// Create an edit that satisfies the given camera selector id.
    CreateCameraEdit(uint32_t selector_id, ImporterCamera camera);

    void apply(ImporterBuildContext& context) const override;

private:
    uint32_t m_selector_id = 0;
    ImporterCamera m_camera;
};

/// Deferred edit that appends a node to the ImporterScene.
class FALCOR_API CreateNodeEdit : public ImporterEdit {
public:
    /// Create an edit that satisfies the given node selector id.
    ///
    /// Optional camera and parent selectors are resolved against earlier edits
    /// when apply is called.
    CreateNodeEdit(
        uint32_t selector_id,
        std::string name,
        float4x4 transform,
        std::optional<ImporterCameraSelector> camera,
        std::optional<ImporterNodeSelector> parent
    );

    void apply(ImporterBuildContext& context) const override;

private:
    uint32_t m_selector_id = 0;
    std::string m_name;
    float4x4 m_transform = float4x4::identity();
    std::optional<ImporterCameraSelector> m_camera;
    std::optional<ImporterNodeSelector> m_parent;
};

} // namespace falcor
