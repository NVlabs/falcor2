// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "mesh_tessellator.h"

#include "falcor2/core/error.h"
#include "falcor2/core/types.h"

#include <cstddef>
#include <cstring>
#include <span>
#include <type_traits>
#include <vector>

namespace falcor {
namespace mesh_tessellator {
namespace detail {
struct PrimvarElementIndices {
    int uniform = 0;
    int vertex = 0;
    int face_varying = 0;
};

inline int interpolation_element_index(MeshInterpolation interpolation, const PrimvarElementIndices& indices)
{
    switch (interpolation) {
    case MeshInterpolation::constant:
        return 0;
    case MeshInterpolation::uniform:
        return indices.uniform;
    case MeshInterpolation::vertex:
    case MeshInterpolation::varying:
        return indices.vertex;
    case MeshInterpolation::face_varying:
        return indices.face_varying;
    }
    SGL_UNREACHABLE();
}

inline std::span<const float> stream_element(const MeshAttributeStream& stream, int element_index)
{
    const int value_index = stream.indices.empty() ? element_index : stream.indices[element_index];
    return stream.values.subspan(value_index * stream.component_count(), stream.component_count());
}

template<typename T>
inline T stream_element(const MeshAttributeStream& stream, int element_index)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const std::span<const float> element = stream_element(stream, element_index);
    FALCOR_ASSERT_EQ(sizeof(T), element.size_bytes());
    T value;
    std::memcpy(&value, element.data(), sizeof(T));
    return value;
}

inline void copy_stream_element(const MeshAttributeStream& stream, int element_index, float* destination)
{
    const std::span<const float> element = stream_element(stream, element_index);
    std::memcpy(destination, element.data(), element.size_bytes());
}

void check_stream_layout(
    const MeshAttributeStream& stream,
    size_t face_count,
    size_t vertex_count,
    size_t face_vertex_count
);

inline bool is_left_handed(MeshOrientation orientation)
{
    return orientation == MeshOrientation::left_handed;
}

/// Boundary-continuity guarantee for one stream. Matching regions reuse or copy an existing value instead of
/// evaluating the current face; values are not compared, so incorrectly sharing discontinuous data silently keeps
/// the older value. Examples: positions are continuous, uniform values and bilinear normals are per-face, and UVs
/// follow face-varying topology.
enum class RegionMode {
    // Share across all incident faces.
    continuous,
    // Every coarse face owns a distinct value.
    per_face,
    // Sharing follows an OpenSubdiv face-varying topology channel.
    face_varying,
};

/// Runtime state for one final ImporterMesh attribute.
struct OutputStream {
    ImporterMeshDefaultAttribute attribute;
    // Scalar offset in OutputStreams' interleaved final-vertex storage.
    size_t scalar_offset = 0;
    // Null only for generated attributes such as the limit-surface normal.
    const MeshAttributeStream* input = nullptr;
    RegionMode region_mode = RegionMode::continuous;
    int fvar_channel = -1;
    // Flat float control table prepared only for streams evaluated by OpenSubdiv.
    std::vector<float> control_values;

    size_t component_count() const { return attribute.components; }
    const float* mesh_values() const { return control_values.empty() ? input->values.data() : control_values.data(); }
    void prepare_control_values(size_t value_count);
};

/// Owns all final vertex attributes in the same interleaved order used by ImporterMesh.
/// Stream order also defines the slot order in each boundary-region signature.
class OutputStreams {
public:
    void reserve(size_t count) { m_streams.reserve(count); }
    size_t add(ImporterMeshDefaultAttribute attribute, const MeshAttributeStream* input = nullptr);

    OutputStream& operator[](size_t index) { return m_streams[index]; }
    const OutputStream& operator[](size_t index) const { return m_streams[index]; }
    std::span<OutputStream> streams() { return m_streams; }
    std::span<const OutputStream> streams() const { return m_streams; }
    size_t size() const { return m_streams.size(); }

    void reserve_vertices(size_t count) { m_values.reserve(count * m_scalar_stride); }
    void resize_vertices(size_t count);
    size_t vertex_count() const { return m_vertex_count; }

    float* value(size_t stream_index, int vertex_index)
    {
        return m_values.data() + vertex_index * m_scalar_stride + m_streams[stream_index].scalar_offset;
    }

    const float* value(size_t stream_index, int vertex_index) const
    {
        return m_values.data() + vertex_index * m_scalar_stride + m_streams[stream_index].scalar_offset;
    }

    template<typename T>
    T& value(size_t stream_index, int vertex_index)
    {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(alignof(T) <= alignof(float));
        static_assert(sizeof(T) % sizeof(float) == 0);
        FALCOR_ASSERT_EQ(sizeof(T), m_streams[stream_index].component_count() * sizeof(float));
        return *reinterpret_cast<T*>(value(stream_index, vertex_index));
    }

    void set_value(size_t stream_index, int vertex_index, std::span<const float> source)
    {
        const OutputStream& stream = m_streams[stream_index];
        FALCOR_ASSERT_EQ(source.size(), stream.component_count());
        std::memcpy(value(stream_index, vertex_index), source.data(), source.size_bytes());
    }

    void copy_value(size_t stream_index, int source_vertex, int destination_vertex)
    {
        set_value(
            stream_index,
            destination_vertex,
            std::span<const float>(value(stream_index, source_vertex), m_streams[stream_index].component_count())
        );
    }

    void write(ImporterMesh& result) const;

private:
    size_t m_vertex_count = 0;
    size_t m_scalar_stride = 0;
    std::vector<OutputStream> m_streams;
    std::vector<float> m_values;
};

/// Stable destination view acquired after ImporterMesh has been exactly allocated.
/// The mesh attribute layout and vertex count must not change while a binding is used.
struct OutputBinding {
    std::byte* data = nullptr;
    size_t byte_stride = 0;
    size_t component_count = 0;

    float* value(int vertex_index) const { return reinterpret_cast<float*>(data + size_t(vertex_index) * byte_stride); }

    template<typename T>
    T& value(int vertex_index) const
    {
        static_assert(std::is_trivially_copyable_v<T>);
        static_assert(alignof(T) <= alignof(float));
        static_assert(sizeof(T) % sizeof(float) == 0);
        FALCOR_ASSERT_EQ(sizeof(T), component_count * sizeof(float));
        return *reinterpret_cast<T*>(value(vertex_index));
    }

    void set_value(int vertex_index, std::span<const float> source) const
    {
        FALCOR_ASSERT_EQ(source.size(), component_count);
        std::memcpy(value(vertex_index), source.data(), source.size_bytes());
    }
};

void add_output_attribute(
    std::vector<ImporterMeshDefaultAttribute>& attributes,
    ImporterMeshDefaultAttribute attribute
);
OutputBinding bind_output(ImporterMesh& result, ImporterMeshDefaultAttribute attribute);
void initialize_result(const TessellatorInputMesh& input, ImporterMesh& result, std::vector<int>& subgeometry_mapping);
void exclude_holes_from_output(const TessellatorInputMesh& input, std::span<int> subgeometry_mapping);
const MeshAttributeStream*
find_stream(std::span<const MeshAttributeStream> streams, ImporterSemantic semantic, uint32_t index = 0);
std::vector<const MeshAttributeStream*>
collect_streams(const TessellatorInputMesh& input, MeshAttributeStream& default_uv);

ImporterMesh triangulate_mesh(const TessellatorInputMesh& input);
ImporterMesh tessellate_subdivision_surface(const TessellatorInputMesh& input);

} // namespace detail
} // namespace mesh_tessellator
} // namespace falcor
