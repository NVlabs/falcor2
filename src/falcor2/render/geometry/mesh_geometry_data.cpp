// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_geometry_data.h"
#include "static_mesh_geometry.h"
#include "geometry_attribute_registry.h"
#include <cstring>

namespace falcor {
namespace {

const MeshVertexAttributeDesc*
find_attribute(std::span<const MeshVertexAttributeDesc> attributes, MeshAttributeSemantic semantic, uint32_t index)
{
    for (const auto& attribute : attributes) {
        if (attribute.semantic == semantic && attribute.index == index)
            return &attribute;
    }
    return nullptr;
}

void validate_attribute(const MeshVertexAttributeDesc& attribute, uint32_t vertex_count)
{
    validate_geometry_attribute_storage(attribute, vertex_count, "Static mesh");

    bool valid = false;
    switch (attribute.semantic) {
    case MeshAttributeSemantic::position:
    case MeshAttributeSemantic::normal:
    case MeshAttributeSemantic::tangent:
        valid = attribute.index == 0 && attribute.type == sgl::DataType::float32 && attribute.component_count == 3;
        break;
    case MeshAttributeSemantic::handedness:
        valid = attribute.index == 0 && attribute.type == sgl::DataType::float32 && attribute.component_count == 1;
        break;
    case MeshAttributeSemantic::tex_coord:
        // Sets outside the fixed shader slots remain available as named attributes.
        valid = attribute.component_count >= 2 && attribute.component_count <= 4;
        break;
    case MeshAttributeSemantic::color:
        // Only COLOR_0 has a fixed slot; higher sets use the named attribute registry.
        valid = attribute.component_count >= 3 && attribute.component_count <= 4;
        break;
    default:
        valid = true;
        break;
    }
    SGL_CHECK(valid, "Unsupported static mesh vertex attribute semantic, index, or component count");
}

float4 read_attribute_float4(const MeshVertexAttributeDesc* attribute, uint32_t vertex_index)
{
    float4 result = float4(0.f);
    if (!attribute)
        return result;
    std::memcpy(
        &result,
        reinterpret_cast<const uint8_t*>(attribute->data) + vertex_index * attribute->stride,
        attribute->component_count * sizeof(float)
    );
    return result;
}

template<typename T>
inline const T& get_index(const StaticMeshGeometryDataDesc::IndexStream<T>& stream, uint32_t index)
{
    static const T zero{0};
    return stream.data
        ? *reinterpret_cast<const T*>(reinterpret_cast<const uint8_t*>(stream.data) + index * stream.stride)
        : zero;
}

} // namespace

void pack_mesh_vertices(
    std::span<const MeshVertexAttributeDesc> attributes,
    std::span<const uint32_t> remap,
    std::span<shared::PackedTriangleVertex> output
)
{
    const MeshVertexAttributeDesc* position_attribute = find_attribute(attributes, MeshAttributeSemantic::position, 0);
    SGL_CHECK(position_attribute, "Static mesh geometry requires a float3 position attribute");
    const MeshVertexAttributeDesc* normal_attribute = find_attribute(attributes, MeshAttributeSemantic::normal, 0);
    const MeshVertexAttributeDesc* tangent_attribute = find_attribute(attributes, MeshAttributeSemantic::tangent, 0);
    const MeshVertexAttributeDesc* handedness_attribute
        = find_attribute(attributes, MeshAttributeSemantic::handedness, 0);
    const MeshVertexAttributeDesc* texcoord_attribute = find_attribute(attributes, MeshAttributeSemantic::tex_coord, 0);
    if (texcoord_attribute
        && (texcoord_attribute->type != sgl::DataType::float32 || texcoord_attribute->component_count < 2))
        texcoord_attribute = nullptr;

    FALCOR_ASSERT(output.size() == remap.size());
    for (size_t i = 0; i < output.size(); ++i) {
        uint32_t orig_index = remap[i];
        shared::TriangleVertex vertex = {};

        const float4 position = read_attribute_float4(position_attribute, orig_index);
        const float4 normal = read_attribute_float4(normal_attribute, orig_index);
        const float4 tangent = read_attribute_float4(tangent_attribute, orig_index);
        const float4 texcoord = read_attribute_float4(texcoord_attribute, orig_index);
        vertex.position = float3(position.x, position.y, position.z);
        vertex.normal = float3(normal.x, normal.y, normal.z);
        vertex.tangent = float3(tangent.x, tangent.y, tangent.z);
        vertex.handedness = read_attribute_float4(handedness_attribute, orig_index).x;
        vertex.uv[0] = float2(texcoord.x, texcoord.y);
        output[i] = shared::detail::pack_triangle_vertex(vertex);
    }
}
void make_mesh_geometry_data(
    const StaticMeshGeometryDataDesc& desc,
    std::vector<MeshGeometryData>& sub_meshes,
    std::vector<std::vector<uint32_t>>* remaps
)
{
    if (remaps)
        remaps->resize(desc.sub_meshes.size());
    validate_unique_geometry_attributes(desc.vertex_attributes, "static mesh");
    for (const MeshVertexAttributeDesc& attribute : desc.vertex_attributes) {
        validate_attribute(attribute, desc.vertex_count);
        warn_if_unsupported_geometry_attribute(attribute, "Static mesh");
    }

    SGL_CHECK(
        find_attribute(desc.vertex_attributes, MeshAttributeSemantic::position, 0),
        "Static mesh geometry requires a float3 position attribute"
    );
    sub_meshes.resize(desc.sub_meshes.size());

    // Convert a single sub-mesh from the input data.
    // Extracts the required vertices, packs the data and remaps indices.
    // The output sub-mesh is self-contained.
    auto convert_sub_mesh = [&](size_t sub_mesh_index)
    {
        const StaticMeshGeometryDataDesc::SubMesh& src = desc.sub_meshes[sub_mesh_index];
        MeshGeometryData& dst = sub_meshes[sub_mesh_index];

        dst.name = src.name;
        dst.vertex_attributes.clear();

        dst.indices.resize(src.index_count);
        uint32_t vertex_count = 0;
        std::vector<uint32_t> index_map(desc.vertex_count, 0xffffffff);
        std::vector<uint32_t> reverse_index_map(desc.vertex_count);
        for (uint32_t i = 0; i < src.index_count; ++i) {
            uint32_t orig_index = get_index(src.index_stream, i);
            SGL_CHECK(orig_index < desc.vertex_count, "Static mesh index is outside the vertex range");
            if (index_map[orig_index] == 0xffffffff) {
                uint32_t new_index = vertex_count++;
                index_map[orig_index] = new_index;
                reverse_index_map[new_index] = orig_index;
            }
            dst.indices[i] = index_map[orig_index];
        }

        dst.vertex_attributes.reserve(desc.vertex_attributes.size());
        if (remaps)
            (*remaps)[sub_mesh_index].assign(reverse_index_map.begin(), reverse_index_map.begin() + vertex_count);
        const auto vertex_remap = std::span(reverse_index_map).first(vertex_count);
        for (const MeshVertexAttributeDesc& attribute : desc.vertex_attributes) {
            dst.vertex_attributes.push_back(copy_geometry_vertex_attribute(attribute, vertex_remap));
        }

        dst.vertices.resize(vertex_count);
        pack_mesh_vertices(desc.vertex_attributes, vertex_remap, dst.vertices);
    };

    for (size_t sub_mesh_index = 0; sub_mesh_index < desc.sub_meshes.size(); ++sub_mesh_index)
        convert_sub_mesh(sub_mesh_index);
}

} // namespace falcor
