// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include <string>
#include <string_view>
#include <vector>
#include <filesystem>

namespace falcor {

/// MeshWriter provides a simple interface for constructing and exporting triangle meshes to glTF format.
///
/// Usage:
///   1. Call begin_submesh() to start a new submesh.
///   2. Add triangles using triangle().
///   3. Call end_submesh() to finalize the submesh.
///   4. Repeat steps 1-3 for additional submeshes.
///   5. Call write() to export all submeshes to a .gltf or .glb file.
///
/// Each submesh becomes a separate mesh/node in the resulting glTF scene.
class FALCOR_API MeshWriter : Object {
    FALCOR_OBJECT(MeshWriter)
public:
    /// Write all submeshes to a glTF file.
    /// @param path Output file path. Use .gltf for JSON format or .glb for binary format.
    void write(const std::filesystem::path& path);

    /// Begin defining a new submesh.
    /// @param name Name for the submesh (used as mesh and node name in glTF).
    void begin_submesh(std::string_view name);

    /// End the current submesh definition.
    void end_submesh();

    /// Add a triangle to the current submesh.
    /// @param v0 First vertex position.
    /// @param v1 Second vertex position.
    /// @param v2 Third vertex position.
    void triangle(float3 v0, float3 v1, float3 v2);

private:
    struct Submesh {
        std::string name;
        std::vector<float3> positions;
        std::vector<uint32_t> indices;
    };

    /// Currently active submesh, or nullptr if none.
    Submesh* m_current_submesh = nullptr;
    std::vector<Submesh> m_submeshes;
};

} // namespace falcor
