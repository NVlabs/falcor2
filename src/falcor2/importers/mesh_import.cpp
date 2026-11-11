// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_import.h"

namespace falcor {

void update_imported_mesh(StaticMeshGeometry* geometry, const ImporterMesh& importer_mesh, UVOrigin uv_origin)
{
    StaticMeshGeometryDataDesc desc = {};
    std::vector<MeshVertexAttributeDesc> vertex_attributes;
    std::vector<std::vector<uint8_t>> converted_texcoords;

    desc.name = importer_mesh.name;
    desc.vertex_count = sgl::narrow_cast<uint32_t>(importer_mesh.vertex_count());
    auto to_runtime_semantic = [](ImporterSemantic semantic)
    {
        switch (semantic) {
        case ImporterSemantic::position:
            return MeshAttributeSemantic::position;
        case ImporterSemantic::normal:
            return MeshAttributeSemantic::normal;
        case ImporterSemantic::tangent:
            return MeshAttributeSemantic::tangent;
        case ImporterSemantic::bitangent:
            return MeshAttributeSemantic::bitangent;
        case ImporterSemantic::handedness:
            return MeshAttributeSemantic::handedness;
        case ImporterSemantic::tex_coord:
            return MeshAttributeSemantic::tex_coord;
        case ImporterSemantic::color:
            return MeshAttributeSemantic::color;
        case ImporterSemantic::joints:
            return MeshAttributeSemantic::joints;
        case ImporterSemantic::weights:
            return MeshAttributeSemantic::weights;
        default:
            return MeshAttributeSemantic::unknown;
        }
    };
    converted_texcoords.reserve(importer_mesh.attributes().size());
    for (const ImporterMeshAttribute& attribute : importer_mesh.attributes()) {
        // Runtime geometry consumes only the reference motion sample.
        if (attribute.index != 0
            && (attribute.semantic == ImporterSemantic::position || attribute.semantic == ImporterSemantic::normal
                || attribute.semantic == ImporterSemantic::tangent))
            continue;
        const void* data = nullptr;
        size_t stride = 0;
        importer_mesh.get_stream_info(attribute, data, stride);
        const size_t element_size = attribute.num_components * sgl::data_type_size(attribute.type);
        const uint8_t* runtime_data = reinterpret_cast<const uint8_t*>(data);
        size_t runtime_stride = stride;
        size_t data_size
            = importer_mesh.vertex_count() == 0 ? 0 : (importer_mesh.vertex_count() - 1) * stride + element_size;
        if (attribute.semantic == ImporterSemantic::tex_coord && attribute.type == sgl::DataType::float32
            && attribute.num_components >= 2 && importer_mesh.uv_origin != uv_origin) {
            auto& converted = converted_texcoords.emplace_back(importer_mesh.vertex_count() * element_size);
            for (size_t i = 0; i < importer_mesh.vertex_count(); ++i) {
                std::memcpy(converted.data() + i * element_size, runtime_data + i * stride, element_size);
                float* values = reinterpret_cast<float*>(converted.data() + i * element_size);
                values[1] = 1.f - values[1];
            }
            runtime_data = converted.data();
            runtime_stride = element_size;
            data_size = converted.size();
        }
        vertex_attributes.push_back({
            .name = attribute.name,
            .semantic = to_runtime_semantic(attribute.semantic),
            .index = attribute.index,
            .type = attribute.type,
            .component_count = attribute.num_components,
            .data = runtime_data,
            .stride = runtime_stride,
            .data_size = data_size,
        });
    }
    desc.vertex_attributes = vertex_attributes;
    for (const ImporterMesh::Subgeometry& subgeo : importer_mesh.subgeometries) {
        StaticMeshGeometryDataDesc::SubMesh sub_mesh = {};
        sub_mesh.name = subgeo.name;
        sub_mesh.index_count = sgl::narrow_cast<uint32_t>(subgeo.indices.size()) * 3;
        sub_mesh.index_stream.data = reinterpret_cast<const uint32_t*>(subgeo.indices.data());
        sub_mesh.index_stream.stride = sizeof(float);
        desc.sub_meshes.push_back(sub_mesh);
    }

    geometry->set_mesh_data(desc);
    geometry->_set_source_asset_ref(importer_mesh.source_asset_ref);
}

} // namespace falcor
