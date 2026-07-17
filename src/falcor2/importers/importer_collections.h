// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/macros.h"
#include "falcor2/core/properties.h"
#include "falcor2/importers/fwd.h"
#include "falcor2/importers/importer_types.h"

#include <filesystem>
#include <optional>
#include <string>

namespace falcor {

/// Opaque reference to an object that will be created when importer edits are applied.
struct ImporterSelector {
    /// Importer-local selector id assigned when the edit is recorded.
    uint32_t id = 0;
};

/// Selector for a camera created by ImporterCameraCollection.
struct ImporterCameraSelector : ImporterSelector { };

/// Selector for a node created by ImporterNodeCollection.
struct ImporterNodeSelector : ImporterSelector { };

/// Deferred exact-name selection over materials in an ImporterScene.
///
/// The selector does not resolve materials while edits are recorded. Operations
/// resolve the name when their edit is applied and affect every matching
/// material. The selector is valid only while its owning Importer is alive.
class FALCOR_API ImporterMaterialSelector {
public:
    /// Record an edit that replaces the type and properties of all selected materials.
    ///
    /// Material names are preserved. A selector with no matches is a no-op.
    void replace(std::string material_type, Properties props = {}) const;

    /// Record an edit that replaces all selected materials using a live constructor.
    ///
    /// Material names are preserved. A selector with no matches is a no-op.
    void replace(ImporterMaterial::Constructor constructor, Properties props = {}) const;

private:
    ImporterMaterialSelector(Importer& importer, std::string name);

    Importer* m_importer;
    std::string m_name;

    friend class ImporterMaterialCollection;
};

/// Contextual interface for selecting and editing materials on an Importer.
///
/// This collection does not store materials. Name selectors are resolved later
/// against the ImporterScene produced by preceding edits.
class FALCOR_API ImporterMaterialCollection {
public:
    /// Create a material collection bound to the specified importer.
    explicit ImporterMaterialCollection(Importer& importer);

    /// Return a deferred selector for every material with the exact name.
    ImporterMaterialSelector find(std::string name);

    /// Return the same deferred exact-name selector as find().
    ImporterMaterialSelector operator[](std::string name);

private:
    Importer& m_importer;
};

/// Contextual interface for recording environment setup edits on an Importer.
///
/// This collection does not store lights. Each set call appends a deferred edit
/// to the owning Importer that updates the first dome light or creates one if
/// none exists.
class FALCOR_API ImporterEnvCollection {
public:
    /// Create an environment collection bound to the specified importer.
    explicit ImporterEnvCollection(Importer& importer);

    /// Record an edit that configures the environment map.
    ///
    /// Applying the edit overwrites the first existing dome light, or creates a
    /// new dome light and root node if the scene does not contain one.
    void set(std::filesystem::path path, float exposure = 0.f, std::string name = "Environment Map");

private:
    Importer& m_importer;
};

/// Contextual interface for recording camera creation edits on an Importer.
///
/// This collection does not store cameras. Each create call appends a deferred
/// edit to the owning Importer and returns a selector for the future camera.
class FALCOR_API ImporterCameraCollection {
public:
    /// Create a camera collection bound to the specified importer.
    explicit ImporterCameraCollection(Importer& importer);

    /// Record an edit that creates a camera from the supplied properties.
    ///
    /// The returned selector can be used by later edits, such as creating a node
    /// that references this camera.
    ImporterCameraSelector create(
        std::string name,
        float focus_distance = 1.f,
        float focal_length = 50.f,
        float fstop = 8.f,
        float2 depth_range = float2(0.01f, 10000.f),
        ImporterCamera::Projection projection = ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection fov_direction = ImporterCamera::FOVDirection::vertical,
        float sensor_size_mm = 24.f
    );

    /// Record an edit that creates a camera from a field of view.
    ///
    /// This is a convenience wrapper around create(). It stores the derived
    /// focal length and sensor size in the deferred camera edit.
    ImporterCameraSelector create_fov(
        std::string name,
        float fov_degrees = 70.f,
        float focus_distance = 1.f,
        float fstop = 8.f,
        float2 depth_range = float2(0.01f, 10000.f),
        ImporterCamera::Projection projection = ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection fov_direction = ImporterCamera::FOVDirection::vertical,
        float sensor_size_mm = 24.f
    );

private:
    Importer& m_importer;
};

/// Contextual interface for recording node creation edits on an Importer.
///
/// This collection does not store nodes. Each create call appends a deferred
/// edit to the owning Importer and returns a selector for the future node.
class FALCOR_API ImporterNodeCollection {
public:
    /// Create a node collection bound to the specified importer.
    explicit ImporterNodeCollection(Importer& importer);

    /// Record an edit that creates a node.
    ///
    /// Optional camera and parent selectors are resolved when the importer edit
    /// list is built into an ImporterScene.
    ImporterNodeSelector create(
        std::string name,
        float4x4 transform = float4x4::identity(),
        std::optional<ImporterCameraSelector> camera = {},
        std::optional<ImporterNodeSelector> parent = {}
    );

    /// Record edits that create a camera and its node from the supplied properties.
    ///
    /// The camera and node use the same name. The returned selector references
    /// the node rather than the camera.
    ImporterNodeSelector create_camera(
        std::string name = "Camera",
        float4x4 transform = float4x4::identity(),
        float focus_distance = 1.f,
        float focal_length = 50.f,
        float fstop = 8.f,
        float2 depth_range = float2(0.01f, 10000.f),
        ImporterCamera::Projection projection = ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection fov_direction = ImporterCamera::FOVDirection::vertical,
        float sensor_size_mm = 24.f
    );

    /// Record edits that create a field-of-view camera and its node.
    ///
    /// The camera and node use the same name. The returned selector references
    /// the node rather than the camera.
    ImporterNodeSelector create_camera_fov(
        std::string name = "Camera",
        float4x4 transform = float4x4::identity(),
        float fov_degrees = 70.f,
        float focus_distance = 1.f,
        float fstop = 8.f,
        float2 depth_range = float2(0.01f, 10000.f),
        ImporterCamera::Projection projection = ImporterCamera::Projection::perspective,
        ImporterCamera::FOVDirection fov_direction = ImporterCamera::FOVDirection::vertical,
        float sensor_size_mm = 24.f
    );

private:
    Importer& m_importer;
};

} // namespace falcor
