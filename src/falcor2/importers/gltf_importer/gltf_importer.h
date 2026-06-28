// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/importers/importer_types.h"

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <filesystem>

// Forward declaration to avoid including the tinygltf header here
namespace tinygltf {
class Model;
} // namespace tinygltf

namespace falcor {

/// GLTF importer.
class FALCOR_API GltfImporter : public Object {
    FALCOR_OBJECT(GltfImporter)
public:
    GltfImporter();
    ~GltfImporter();

    /// Load a GLTF scene from the specified file path.
    /// @param path Path to the GLTF file.
    /// @return The imported scene.
    ref<ImporterScene> load_scene(const std::filesystem::path& path);

private:
    bool load_gltf_file(const std::filesystem::path& path);
    void extract_meshes();
    void extract_cameras();
    void extract_nodes();
    void build_node_hierarchy();
    void extract_textures();
    void extract_materials();
    void extract_animations();

    std::filesystem::path m_path;
    std::unique_ptr<tinygltf::Model> m_model;
    ref<ImporterScene> m_scene;
};

} // namespace falcor
