// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "static_mesh_geometry.h"

#include "geometry_attribute_registry.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/material.h"

#include <sgl/core/type_utils.h>

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

StaticMeshGeometry::~StaticMeshGeometry() { }

void StaticMeshGeometry::set_mesh_data(const StaticMeshGeometryDataDesc& desc)
{
    set_name(desc.name);

    validate_unique_geometry_attributes(desc.vertex_attributes, "static mesh");
    for (const MeshVertexAttributeDesc& attribute : desc.vertex_attributes) {
        validate_attribute(attribute, desc.vertex_count);
        warn_if_unsupported_geometry_attribute(attribute, "Static mesh");
    }

    if (m_scene) {
        for (const MeshVertexAttributeDesc& attribute : desc.vertex_attributes)
            register_geometry_attribute(*m_scene, attribute);
    }

    const MeshVertexAttributeDesc* position_attribute
        = find_attribute(desc.vertex_attributes, MeshAttributeSemantic::position, 0);
    SGL_CHECK(position_attribute, "Static mesh geometry requires a float3 position attribute");
    const MeshVertexAttributeDesc* normal_attribute
        = find_attribute(desc.vertex_attributes, MeshAttributeSemantic::normal, 0);
    const MeshVertexAttributeDesc* tangent_attribute
        = find_attribute(desc.vertex_attributes, MeshAttributeSemantic::tangent, 0);
    const MeshVertexAttributeDesc* handedness_attribute
        = find_attribute(desc.vertex_attributes, MeshAttributeSemantic::handedness, 0);
    const MeshVertexAttributeDesc* texcoord_attribute
        = find_attribute(desc.vertex_attributes, MeshAttributeSemantic::tex_coord, 0);
    if (texcoord_attribute
        && (texcoord_attribute->type != sgl::DataType::float32 || texcoord_attribute->component_count < 2))
        texcoord_attribute = nullptr;

    m_sub_meshes.resize(desc.sub_meshes.size());

    // Convert a single sub-mesh from the input data.
    // Extracts the required vertices, packs the data and remaps indices.
    // The output sub-mesh is self-contained.
    auto convert_sub_mesh = [&](size_t sub_mesh_index)
    {
        const StaticMeshGeometryDataDesc::SubMesh& src = desc.sub_meshes[sub_mesh_index];
        SubMesh& dst = m_sub_meshes[sub_mesh_index];

        dst.name = src.name;
        dst.material = nullptr;
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
        const auto vertex_remap = std::span(reverse_index_map).first(vertex_count);
        for (const MeshVertexAttributeDesc& attribute : desc.vertex_attributes) {
            dst.vertex_attributes.push_back(copy_geometry_vertex_attribute(attribute, vertex_remap));
        }

        dst.vertices.resize(vertex_count);
        for (uint32_t i = 0; i < vertex_count; ++i) {
            shared::TriangleVertex vertex = {};
            uint32_t orig_index = reverse_index_map[i];
            const float4 position = read_attribute_float4(position_attribute, orig_index);
            const float4 normal = read_attribute_float4(normal_attribute, orig_index);
            const float4 tangent = read_attribute_float4(tangent_attribute, orig_index);
            const float4 texcoord = read_attribute_float4(texcoord_attribute, orig_index);
            vertex.position = float3(position.x, position.y, position.z);
            vertex.normal = float3(normal.x, normal.y, normal.z);
            vertex.tangent = float3(tangent.x, tangent.y, tangent.z);
            vertex.handedness = read_attribute_float4(handedness_attribute, orig_index).x;
            vertex.uv[0] = float2(texcoord.x, texcoord.y);
            dst.vertices[i] = shared::detail::pack_triangle_vertex(vertex);
        }
    };

    for (size_t sub_mesh_index = 0; sub_mesh_index < desc.sub_meshes.size(); ++sub_mesh_index)
        convert_sub_mesh(sub_mesh_index);

    recompute_local_aabb();

    mark_dirty(DirtyFlags::render_state);
}

MeshVertexAttribute*
StaticMeshGeometry::find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index)
{
    for (MeshVertexAttribute& attribute : m_sub_meshes.at(sub_mesh_index).vertex_attributes) {
        if (attribute.semantic == semantic && attribute.index == index)
            return &attribute;
    }
    return nullptr;
}

const MeshVertexAttribute*
StaticMeshGeometry::find_vertex_attribute(size_t sub_mesh_index, MeshAttributeSemantic semantic, uint32_t index) const
{
    return const_cast<StaticMeshGeometry*>(this)->find_vertex_attribute(sub_mesh_index, semantic, index);
}

void StaticMeshGeometry::clear_invalid_references() { }

void StaticMeshGeometry::on_add_to_scene()
{
    register_geometry_attributes();
}

void StaticMeshGeometry::register_geometry_attributes()
{
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const MeshVertexAttribute& attribute : sub_mesh.vertex_attributes)
            register_geometry_attribute(*m_scene, attribute);
    }
}

void StaticMeshGeometry::recompute_local_aabb()
{
    m_local_aabb = AABB();
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        for (const shared::PackedTriangleVertex& vertex : sub_mesh.vertices) {
            m_local_aabb.expand(float3(vertex.position[0], vertex.position[1], vertex.position[2]));
        }
    }
}

void StaticMeshGeometry::update_render_state()
{
    clear_render_state();
    RenderScene* render_scene = m_scene->_render_scene();
    std::vector<RenderHandle> geometries;
    for (const SubMesh& sub_mesh : m_sub_meshes) {
        RenderTriangleGeometryDesc desc = {};
        desc.vertex_count = sgl::narrow_cast<uint32_t>(sub_mesh.vertices.size());
        desc.index_count = sgl::narrow_cast<uint32_t>(sub_mesh.indices.size());
        RenderTriangleGeometryHandle geometry = render_scene->create_triangle_geometry(desc);
        RenderTriangleGeometryData data = {};
        data.vertices = sub_mesh.vertices;
        data.indices = sub_mesh.indices;
        PreparedGeometryAttributes additional_attributes(sub_mesh.vertex_attributes.size());
        for (const MeshVertexAttribute& attribute : sub_mesh.vertex_attributes) {
            // An ordinary float2 TEXCOORD_0 is already stored in PackedTriangleVertex.
            if (attribute.semantic == MeshAttributeSemantic::tex_coord && attribute.index == 0
                && attribute.type == sgl::DataType::float32 && attribute.component_count == 2)
                continue;
            additional_attributes.append(*m_scene, attribute, sub_mesh.vertices.size());
        }
        data.additional_attributes = additional_attributes.attributes();
        render_scene->update_triangle_geometry(geometry, data);
        m_render_geometries.push_back(geometry);
        geometries.push_back(geometry);
    }

    if (geometries.size() > 0) {
        RenderGeometryGroupDesc desc = {};
        desc.geometries = geometries;
        m_render_geometry_group = render_scene->create_geometry_group(desc);
    }
}

void StaticMeshGeometry::clear_render_state()
{
    RenderScene* render_scene = m_scene->_render_scene();

    if (m_render_geometry_group) {
        render_scene->destroy_geometry_group(m_render_geometry_group);
        m_render_geometry_group = {};
    }

    for (RenderTriangleGeometryHandle geometry : m_render_geometries) {
        render_scene->destroy_triangle_geometry(geometry);
    }

    m_render_geometries.clear();
}

FALCOR_SCENE_REGISTER_GEOMETRY(StaticMeshGeometry);

} // namespace falcor
