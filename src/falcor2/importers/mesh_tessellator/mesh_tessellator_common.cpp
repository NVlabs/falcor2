// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator_internal.h"

#include <array>
#include <cstring>

namespace falcor {
namespace mesh_tessellator {
namespace detail {

namespace {

size_t interpolation_element_count(
    MeshInterpolation interpolation,
    size_t face_count,
    size_t vertex_count,
    size_t face_vertex_count
)
{
    switch (interpolation) {
    case MeshInterpolation::constant:
        return 1;
    case MeshInterpolation::uniform:
        return face_count;
    case MeshInterpolation::vertex:
    case MeshInterpolation::varying:
        return vertex_count;
    case MeshInterpolation::face_varying:
        return face_vertex_count;
    }
    SGL_UNREACHABLE();
}

} // namespace

void check_stream_layout(
    const MeshAttributeStream& stream,
    size_t face_count,
    size_t vertex_count,
    size_t face_vertex_count
)
{
    FALCOR_CHECK(stream.component_count() > 0, "Mesh tessellator stream has zero components");
    FALCOR_CHECK(
        stream.values.size() % stream.component_count() == 0,
        "Mesh tessellator stream has {} scalars, which is not divisible by its {} components",
        stream.values.size(),
        stream.component_count()
    );

    const size_t element_count
        = interpolation_element_count(stream.interpolation, face_count, vertex_count, face_vertex_count);
    if (stream.indices.empty()) {
        FALCOR_CHECK(
            stream.value_count() >= element_count,
            "Mesh tessellator stream has {} values but its interpolation requires {}",
            stream.value_count(),
            element_count
        );
    } else {
        FALCOR_CHECK(
            stream.indices.size() >= element_count,
            "Indexed mesh tessellator stream has {} indices but its interpolation requires {}",
            stream.indices.size(),
            element_count
        );
    }
}

void OutputStream::prepare_control_values(size_t value_count)
{
    if (input->indices.empty())
        return;

    control_values.resize(value_count * component_count());
    // Vertex and varying surfaces address values directly by control-vertex index, so gather any authored indices.
    for (size_t value_index = 0; value_index < value_count; ++value_index)
        copy_stream_element(*input, int(value_index), control_values.data() + value_index * component_count());
}

size_t OutputStreams::add(ImporterMeshDefaultAttribute attribute, const MeshAttributeStream* input)
{
    FALCOR_CHECK(m_vertex_count == 0, "Cannot add an output subdivision stream after allocating vertices");
    FALCOR_CHECK(attribute.components > 0, "Output subdivision stream has zero components");
    for (const OutputStream& stream : m_streams) {
        FALCOR_CHECK(
            stream.attribute.semantic != attribute.semantic || stream.attribute.index != attribute.index,
            "Duplicate output subdivision stream for semantic {} index {}",
            attribute.semantic,
            attribute.index
        );
    }
    m_streams.push_back({.attribute = attribute, .scalar_offset = m_scalar_stride, .input = input});
    m_scalar_stride += attribute.components;
    return m_streams.size() - 1;
}

void OutputStreams::resize_vertices(size_t count)
{
    FALCOR_CHECK(count >= m_vertex_count, "Subdivision output vertex count cannot shrink");
    m_vertex_count = count;
    m_values.resize(m_vertex_count * m_scalar_stride);
}

void OutputStreams::write(ImporterMesh& result) const
{
    std::vector<ImporterMeshDefaultAttribute> attributes;
    attributes.reserve(m_streams.size());
    for (const OutputStream& stream : m_streams)
        attributes.push_back(stream.attribute);
    result.ensure_attributes(attributes);
    result.allocate_vertices(m_vertex_count);

    uint32_t buffer_index = 0;
    for (size_t stream_index = 0; stream_index < m_streams.size(); ++stream_index) {
        const OutputStream& stream = m_streams[stream_index];
        const ImporterMeshAttribute* attribute
            = result.find_attribute(stream.attribute.semantic, stream.attribute.index);
        FALCOR_CHECK(attribute, "Failed to create output subdivision stream");
        if (stream_index == 0)
            buffer_index = attribute->buffer;
        FALCOR_CHECK(attribute->buffer == buffer_index, "Output subdivision streams use different buffers");
        FALCOR_CHECK(
            attribute->offset == stream.scalar_offset * sizeof(float),
            "Output subdivision stream layout does not match ImporterMesh"
        );
    }

    ImporterMeshBuffer& buffer = result.buffers()[buffer_index];
    FALCOR_CHECK(
        buffer.stride == m_scalar_stride * sizeof(float),
        "Output subdivision vertex stride does not match ImporterMesh"
    );
    std::memcpy(buffer.data.data(), m_values.data(), m_values.size() * sizeof(float));
}

void add_output_attribute(std::vector<ImporterMeshDefaultAttribute>& attributes, ImporterMeshDefaultAttribute attribute)
{
    FALCOR_CHECK(attribute.components > 0, "Output triangulation stream has zero components");
    for (const ImporterMeshDefaultAttribute& existing : attributes) {
        FALCOR_CHECK(
            existing.semantic != attribute.semantic || existing.index != attribute.index,
            "Duplicate output triangulation stream for semantic {} index {}",
            attribute.semantic,
            attribute.index
        );
    }
    attributes.push_back(attribute);
}

OutputBinding bind_output(ImporterMesh& result, ImporterMeshDefaultAttribute attribute)
{
    const ImporterMeshAttribute* output = result.find_attribute(attribute.semantic, attribute.index);
    FALCOR_CHECK(output, "Failed to create output triangulation stream");
    FALCOR_CHECK(output->type == sgl::DataType::float32, "Output triangulation stream is not float32");
    FALCOR_CHECK(
        output->num_components == attribute.components,
        "Output triangulation stream has {} components instead of {}",
        output->num_components,
        attribute.components
    );

    void* data = nullptr;
    size_t byte_stride = 0;
    result.get_stream_info(*output, data, byte_stride);
    return {
        .data = reinterpret_cast<std::byte*>(data),
        .byte_stride = byte_stride,
        .component_count = output->num_components,
    };
}

void initialize_result(const TessellatorInputMesh& input, ImporterMesh& result, std::vector<int>& subgeometry_mapping)
{
    result.name = input.name;
    result.subgeometries.resize(input.subgeometries.size());
    subgeometry_mapping.assign(input.face_vertex_counts.size(), -1);
    for (size_t i = 0; i < input.subgeometries.size(); ++i) {
        result.subgeometries[i].material_name = input.subgeometries[i].material_name;
        result.subgeometries[i].name = input.subgeometries[i].name;
        for (int face_index : input.subgeometries[i].face_indices) {
            FALCOR_CHECK(
                face_index >= 0 && size_t(face_index) < subgeometry_mapping.size(),
                "Mesh tessellator subgeometry '{}' references face {} but the mesh has {} faces",
                input.subgeometries[i].name,
                face_index,
                subgeometry_mapping.size()
            );
            FALCOR_CHECK(
                subgeometry_mapping[face_index] < 0,
                "Mesh tessellator face {} is assigned to multiple subgeometries",
                face_index
            );
            subgeometry_mapping[face_index] = int(i);
        }
    }
}

void exclude_holes_from_output(const TessellatorInputMesh& input, std::span<int> subgeometry_mapping)
{
    // A negative mapping suppresses output without removing the face from the subdivision topology.
    for (int face_index : input.hole_indices) {
        FALCOR_CHECK(
            face_index >= 0 && size_t(face_index) < subgeometry_mapping.size(),
            "Mesh tessellator hole references face {} but the mesh has {} faces",
            face_index,
            subgeometry_mapping.size()
        );
        subgeometry_mapping[face_index] = -1;
    }
}

const MeshAttributeStream*
find_stream(std::span<const MeshAttributeStream> streams, ImporterSemantic semantic, uint32_t index)
{
    for (const MeshAttributeStream& stream : streams) {
        if (stream.attribute.semantic == semantic && stream.attribute.index == index)
            return &stream;
    }
    return nullptr;
}

std::vector<const MeshAttributeStream*>
collect_streams(const TessellatorInputMesh& input, MeshAttributeStream& default_uv)
{
    std::vector<const MeshAttributeStream*> streams;
    streams.reserve(input.streams.size() + 1);
    for (const MeshAttributeStream& stream : input.streams)
        streams.push_back(&stream);

    if (!find_stream(input.streams, ImporterSemantic::tex_coord)) {
        static constexpr std::array<float, 2> default_uv_values{0.f, 0.f};
        default_uv.attribute = {ImporterSemantic::tex_coord, 0, 2};
        default_uv.interpolation = MeshInterpolation::constant;
        default_uv.values = default_uv_values;
        streams.push_back(&default_uv);
    }
    return streams;
}

} // namespace detail
} // namespace mesh_tessellator
} // namespace falcor
