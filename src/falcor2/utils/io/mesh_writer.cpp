// SPDX-License-Identifier: Apache-2.0

#include "mesh_writer.h"

#include "falcor2/core/error.h"

#include "falcor2/external/tiny_gltf.h"

#include <cstring>
#include <limits>
#include <algorithm>

namespace falcor {

void MeshWriter::begin_submesh(std::string_view name)
{
    FALCOR_CHECK(!m_current_submesh, "Cannot begin a submesh while another is active. Call end_submesh() first.");
    m_submeshes.push_back({std::string(name), {}, {}});
    m_current_submesh = &m_submeshes.back();
}

void MeshWriter::end_submesh()
{
    FALCOR_CHECK(m_current_submesh, "No active submesh to end.");
    m_current_submesh = nullptr;
}

void MeshWriter::triangle(float3 v0, float3 v1, float3 v2)
{
    FALCOR_CHECK(m_current_submesh, "No active submesh. Call begin_submesh() first.");
    uint32_t base = static_cast<uint32_t>(m_current_submesh->positions.size());
    m_current_submesh->positions.push_back(v0);
    m_current_submesh->positions.push_back(v1);
    m_current_submesh->positions.push_back(v2);
    m_current_submesh->indices.push_back(base + 0);
    m_current_submesh->indices.push_back(base + 1);
    m_current_submesh->indices.push_back(base + 2);
}

void MeshWriter::write(const std::filesystem::path& path)
{
    FALCOR_CHECK(!m_current_submesh, "Cannot write while a submesh is active. Call end_submesh() first.");
    FALCOR_CHECK(!m_submeshes.empty(), "No submeshes to write.");

    tinygltf::Model model;
    model.asset.version = "2.0";
    model.asset.generator = "falcor2::MeshWriter";

    // All vertex and index data is packed into a single buffer.
    // Each submesh's data is appended sequentially with proper alignment.
    tinygltf::Buffer buffer;

    // Helper to append raw data to buffer and return the byte offset.
    auto append_data = [&buffer](const void* data, size_t size) -> size_t
    {
        size_t offset = buffer.data.size();
        buffer.data.resize(offset + size);
        std::memcpy(buffer.data.data() + offset, data, size);
        return offset;
    };

    // Helper to pad buffer to alignment boundary (required by glTF spec for accessors).
    auto pad_buffer = [&buffer](size_t alignment)
    {
        size_t remainder = buffer.data.size() % alignment;
        if (remainder != 0)
            buffer.data.resize(buffer.data.size() + (alignment - remainder), 0);
    };

    // Create a single scene that will contain all submeshes as child nodes.
    tinygltf::Scene scene;
    scene.name = "scene";

    // Process each submesh, creating the necessary glTF structures:
    // - BufferViews for position and index data
    // - Accessors to describe the data format
    // - Mesh with a single triangle primitive
    // - Node to place the mesh in the scene
    for (size_t i = 0; i < m_submeshes.size(); ++i) {
        const Submesh& submesh = m_submeshes[i];
        FALCOR_CHECK(!submesh.positions.empty(), "Submesh \"{}\" has no triangles.", submesh.name);

        // Compute bounding box for positions (required by glTF for POSITION accessor).
        float3 pos_min(std::numeric_limits<float>::max());
        float3 pos_max(std::numeric_limits<float>::lowest());
        for (const float3& p : submesh.positions) {
            pos_min = min(pos_min, p);
            pos_max = max(pos_max, p);
        }

        // Write position data.
        pad_buffer(sizeof(float));
        size_t pos_offset = append_data(submesh.positions.data(), submesh.positions.size() * sizeof(float3));
        size_t pos_size = submesh.positions.size() * sizeof(float3);

        // Write index data.
        pad_buffer(sizeof(uint32_t));
        size_t idx_offset = append_data(submesh.indices.data(), submesh.indices.size() * sizeof(uint32_t));
        size_t idx_size = submesh.indices.size() * sizeof(uint32_t);

        // Buffer views.
        int pos_bv_idx = static_cast<int>(model.bufferViews.size());
        {
            tinygltf::BufferView bv;
            bv.name = submesh.name + "_positions";
            bv.buffer = 0;
            bv.byteOffset = pos_offset;
            bv.byteLength = pos_size;
            bv.target = TINYGLTF_TARGET_ARRAY_BUFFER;
            model.bufferViews.push_back(bv);
        }

        int idx_bv_idx = static_cast<int>(model.bufferViews.size());
        {
            tinygltf::BufferView bv;
            bv.name = submesh.name + "_indices";
            bv.buffer = 0;
            bv.byteOffset = idx_offset;
            bv.byteLength = idx_size;
            bv.target = TINYGLTF_TARGET_ELEMENT_ARRAY_BUFFER;
            model.bufferViews.push_back(bv);
        }

        // Accessors.
        int pos_acc_idx = static_cast<int>(model.accessors.size());
        {
            tinygltf::Accessor acc;
            acc.name = submesh.name + "_positions";
            acc.bufferView = pos_bv_idx;
            acc.byteOffset = 0;
            acc.componentType = TINYGLTF_COMPONENT_TYPE_FLOAT;
            acc.count = submesh.positions.size();
            acc.type = TINYGLTF_TYPE_VEC3;
            acc.minValues = {pos_min.x, pos_min.y, pos_min.z};
            acc.maxValues = {pos_max.x, pos_max.y, pos_max.z};
            model.accessors.push_back(acc);
        }

        int idx_acc_idx = static_cast<int>(model.accessors.size());
        {
            tinygltf::Accessor acc;
            acc.name = submesh.name + "_indices";
            acc.bufferView = idx_bv_idx;
            acc.byteOffset = 0;
            acc.componentType = TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT;
            acc.count = submesh.indices.size();
            acc.type = TINYGLTF_TYPE_SCALAR;
            model.accessors.push_back(acc);
        }

        // Mesh with a single primitive.
        tinygltf::Mesh mesh;
        mesh.name = submesh.name;
        {
            tinygltf::Primitive prim;
            prim.attributes["POSITION"] = pos_acc_idx;
            prim.indices = idx_acc_idx;
            prim.mode = TINYGLTF_MODE_TRIANGLES;
            mesh.primitives.push_back(prim);
        }
        int mesh_idx = static_cast<int>(model.meshes.size());
        model.meshes.push_back(mesh);

        // Node referencing the mesh.
        tinygltf::Node node;
        node.name = submesh.name;
        node.mesh = mesh_idx;
        int node_idx = static_cast<int>(model.nodes.size());
        model.nodes.push_back(node);

        scene.nodes.push_back(node_idx);
    }

    model.scenes.push_back(scene);
    model.defaultScene = 0;
    model.buffers.push_back(buffer);

    // Determine format from extension.
    std::string ext = path.extension().string();
    std::transform(
        ext.begin(),
        ext.end(),
        ext.begin(),
        [](char c)
        {
            return static_cast<char>(std::tolower(c));
        }
    );
    bool write_binary = (ext == ".glb");

    tinygltf::TinyGLTF writer;
    bool ok = writer.WriteGltfSceneToFile(
        &model,
        path.string(),
        true, // embedImages
        true, // embedBuffers
        true, // prettyPrint
        write_binary
    );

    FALCOR_CHECK(ok, "Failed to write glTF file \"{}\".", path.string());
}

} // namespace falcor
