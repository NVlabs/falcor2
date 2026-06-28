// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/mikkt.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/types.h"
#include "falcor2/core/blob_cache.h"

#include <sgl/core/crypto.h>
#include <sgl/core/logger.h>

#include <mikktspace.h>

#include <vector>

namespace falcor {

namespace {

struct MikkTState {
    ImporterMesh& mesh;
    std::vector<int3> flattened_faces;
    ImporterVertexStream<float3> position_stream;
    ImporterVertexStream<float3> normal_stream;
    ImporterVertexStream<float2> uv_stream;
    ImporterVertexStream<float3> tangent_stream;
    ImporterVertexStream<float> handedness_stream;

    static MikkTState* from_context(const SMikkTSpaceContext* context)
    {
        return reinterpret_cast<MikkTState*>(context->m_pUserData);
    }

    template<typename T>
    T read_vertex(ImporterVertexStream<T> stream, int32_t face, int32_t vert) const
    {
        return stream[flattened_faces[face][vert]];
    }

    template<typename T>
    void write_vertex(ImporterVertexStream<T> stream, int32_t face, int32_t vert, const T& value)
    {
        stream[flattened_faces[face][vert]] = value;
    }
};

} // namespace

void mikkt_generate_tangent_space(ImporterMesh& mesh)
{
    MikkTState state{.mesh = mesh};

    for (auto& geom : mesh.subgeometries)
        state.flattened_faces.insert(state.flattened_faces.end(), geom.indices.begin(), geom.indices.end());

    // Get the 3 streams required to exist to read from
    state.position_stream = mesh.position_stream();
    state.normal_stream = mesh.normal_stream();
    state.uv_stream = mesh.texcoord_stream();
    SGL_CHECK(state.position_stream.valid(), "Cannot generate tangents without position stream");
    SGL_CHECK(state.normal_stream.valid(), "Cannot generate tangents without normal stream");
    SGL_CHECK(state.uv_stream.valid(), "Cannot generate tangents without texcoord stream");

    // Create/get tangent and handedness attributes
    mesh.ensure_attributes({{ImporterSemantic::tangent, 0}, {ImporterSemantic::handedness, 0}});
    state.tangent_stream = mesh.tangent_stream();
    state.handedness_stream = mesh.handedness_stream();

    using ContextCPtr = const SMikkTSpaceContext*;

    SMikkTSpaceInterface mikktspace = {};
    mikktspace.m_getNumFaces = [](ContextCPtr context)
    {
        return (int)MikkTState::from_context(context)->flattened_faces.size();
    };
    mikktspace.m_getNumVerticesOfFace = [](ContextCPtr, int)
    {
        return 3;
    };

    mikktspace.m_getPosition = [](ContextCPtr context, float val[], int face, int vert)
    {
        auto* s = MikkTState::from_context(context);
        *reinterpret_cast<float3*>(val) = s->read_vertex(s->position_stream, face, vert);
    };
    mikktspace.m_getNormal = [](ContextCPtr context, float val[], int face, int vert)
    {
        auto* s = MikkTState::from_context(context);
        *reinterpret_cast<float3*>(val) = s->read_vertex(s->normal_stream, face, vert);
    };
    mikktspace.m_getTexCoord = [](ContextCPtr context, float val[], int face, int vert)
    {
        auto* s = MikkTState::from_context(context);
        *reinterpret_cast<float2*>(val) = s->read_vertex(s->uv_stream, face, vert);
    };

    mikktspace.m_setTSpaceBasic = [](ContextCPtr context, const float val[], float sign, int face, int vert)
    {
        auto* s = MikkTState::from_context(context);
        s->write_vertex(s->tangent_stream, face, vert, *reinterpret_cast<const float3*>(val));
        s->write_vertex(s->handedness_stream, face, vert, sign);
    };

    SMikkTSpaceContext context = {};
    context.m_pInterface = &mikktspace;
    context.m_pUserData = &state;

    if (!genTangSpaceDefault(&context))
        SGL_THROW("Failed to generate tangents");
}

void mikkt_generate_tangent_space(ImporterMesh& mesh, BlobCache& cache)
{
    // Compute cache key from all inputs that affect the output.
    sgl::SHA1 sha1;
    sha1.update("mikkt_v1");

    auto position_stream = mesh.position_stream();
    auto normal_stream = mesh.normal_stream();
    auto uv_stream = mesh.texcoord_stream();
    SGL_CHECK(position_stream.valid(), "Cannot generate tangents without position stream");
    SGL_CHECK(normal_stream.valid(), "Cannot generate tangents without normal stream");
    SGL_CHECK(uv_stream.valid(), "Cannot generate tangents without texcoord stream");

    size_t vertex_count = position_stream.size;

    // Hash vertex data per-vertex to handle arbitrary strides.
    for (size_t i = 0; i < vertex_count; ++i)
        sha1.update(&position_stream[i], sizeof(float3));
    for (size_t i = 0; i < vertex_count; ++i)
        sha1.update(&normal_stream[i], sizeof(float3));
    for (size_t i = 0; i < vertex_count; ++i)
        sha1.update(&uv_stream[i], sizeof(float2));

    // Hash indices.
    for (auto& geom : mesh.subgeometries)
        sha1.update(geom.indices.data(), geom.indices.size() * sizeof(int3));

    sgl::SHA1::Digest key = sha1.digest();

    // Try reading from cache.
    size_t tangent_bytes = vertex_count * sizeof(float3);
    size_t handedness_bytes = vertex_count * sizeof(float);
    size_t total_bytes = tangent_bytes + handedness_bytes;

    std::vector<uint8_t> blob(total_bytes);
    std::error_code read_ec;
    bool cache_hit = cache.try_read(key, blob, std::nullopt, read_ec);
    if (read_ec)
        sgl::log_warn("Failed to read MikkTSpace tangent cache: {}", read_ec.message());

    if (cache_hit) {
        // Cache hit: write tangents and handedness into mesh.
        mesh.ensure_attributes({{ImporterSemantic::tangent, 0}, {ImporterSemantic::handedness, 0}});
        auto tangent_stream = mesh.tangent_stream();
        auto handedness_stream = mesh.handedness_stream();

        for (size_t i = 0; i < vertex_count; ++i) {
            float3 tangent;
            float handedness;
            std::memcpy(&tangent, blob.data() + i * sizeof(float3), sizeof(float3));
            std::memcpy(&handedness, blob.data() + tangent_bytes + i * sizeof(float), sizeof(float));
            tangent_stream[i] = tangent;
            handedness_stream[i] = handedness;
        }
        return;
    }

    // Cache miss: compute tangents.
    mikkt_generate_tangent_space(mesh);

    // Serialize and write to cache.
    auto tangent_stream = mesh.tangent_stream();
    auto handedness_stream = mesh.handedness_stream();

    blob.resize(total_bytes);
    for (size_t i = 0; i < vertex_count; ++i)
        std::memcpy(blob.data() + i * sizeof(float3), &tangent_stream[i], sizeof(float3));
    for (size_t i = 0; i < vertex_count; ++i)
        std::memcpy(blob.data() + tangent_bytes + i * sizeof(float), &handedness_stream[i], sizeof(float));

    std::error_code write_ec;
    cache.try_write(key, blob, std::nullopt, write_ec);
    if (write_ec)
        sgl::log_warn("Failed to write MikkTSpace tangent cache: {}", write_ec.message());
}

} // namespace falcor
