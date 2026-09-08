// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/mikkt.h"
#if defined(FALCOR_ENABLE_MIKKT_COMPARISON)
#include "falcor2/importers/mikkt_internal.h"
#endif
#include "falcor2/importers/importer_types.h"
#include "falcor2/core/types.h"
#include "falcor2/core/blob_cache.h"

#include <sgl/core/crypto.h>
#include <sgl/core/logger.h>

#include <mikktspace.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <span>
#include <utility>
#include <vector>

namespace falcor {

namespace {

struct OfficialMikkContext {
    std::vector<uint3> flattened_faces;
    std::span<const uint3> faces;
    ImporterVertexStreamRO<float3> positions;
    ImporterVertexStreamRO<float3> normals;
    ImporterVertexStreamRO<float2> uvs;
    std::vector<float4> tangent_space;

    explicit OfficialMikkContext(const ImporterMesh& mesh)
        : positions(mesh.position_stream())
        , normals(mesh.normal_stream())
        , uvs(mesh.texcoord_stream())
    {
        SGL_CHECK(positions.valid(), "Cannot generate tangents without position stream");
        SGL_CHECK(normals.valid(), "Cannot generate tangents without normal stream");
        SGL_CHECK(uvs.valid(), "Cannot generate tangents without texcoord stream");

        if (mesh.subgeometries.size() == 1) {
            faces = mesh.subgeometries.front().indices;
        } else {
            size_t face_count = 0;
            for (const auto& subgeometry : mesh.subgeometries)
                face_count += subgeometry.indices.size();
            flattened_faces.reserve(face_count);
            for (const auto& subgeometry : mesh.subgeometries)
                flattened_faces.insert(flattened_faces.end(), subgeometry.indices.begin(), subgeometry.indices.end());
            faces = flattened_faces;
        }

        SGL_CHECK(
            faces.size() <= size_t(std::numeric_limits<int>::max()),
            "MikkTSpace face count exceeds its integer range"
        );
        tangent_space.resize(faces.size() * 3);
    }

    OfficialMikkContext(const OfficialMikkContext&) = delete;
    OfficialMikkContext& operator=(const OfficialMikkContext&) = delete;
    OfficialMikkContext(OfficialMikkContext&&) = delete;
    OfficialMikkContext& operator=(OfficialMikkContext&&) = delete;

    static OfficialMikkContext* from_context(const SMikkTSpaceContext* context)
    {
        return reinterpret_cast<OfficialMikkContext*>(context->m_pUserData);
    }

    template<typename T>
    T read_vertex(ImporterVertexStreamRO<T> stream, int32_t face, int32_t corner) const
    {
        return stream[faces[face][corner]];
    }
};

std::vector<float4> generate_official_mikktspace(const ImporterMesh& mesh)
{
    // Adapt the indexed ImporterMesh to MikkTSpace's callback interface and capture its unindexed per-corner output.
    OfficialMikkContext state(mesh);

    using ContextCPtr = const SMikkTSpaceContext*;

    SMikkTSpaceInterface mikktspace = {};
    mikktspace.m_getNumFaces = [](ContextCPtr context)
    {
        return int(OfficialMikkContext::from_context(context)->faces.size());
    };
    mikktspace.m_getNumVerticesOfFace = [](ContextCPtr, int)
    {
        return 3;
    };

    mikktspace.m_getPosition = [](ContextCPtr context, float value[], int face, int corner)
    {
        const OfficialMikkContext* state = OfficialMikkContext::from_context(context);
        const float3 position = state->read_vertex(state->positions, face, corner);
        value[0] = position.x;
        value[1] = position.y;
        value[2] = position.z;
    };
    mikktspace.m_getNormal = [](ContextCPtr context, float value[], int face, int corner)
    {
        const OfficialMikkContext* state = OfficialMikkContext::from_context(context);
        const float3 normal = state->read_vertex(state->normals, face, corner);
        value[0] = normal.x;
        value[1] = normal.y;
        value[2] = normal.z;
    };
    mikktspace.m_getTexCoord = [](ContextCPtr context, float value[], int face, int corner)
    {
        const OfficialMikkContext* state = OfficialMikkContext::from_context(context);
        const float2 uv = state->read_vertex(state->uvs, face, corner);
        value[0] = uv.x;
        value[1] = uv.y;
    };
    mikktspace.m_setTSpaceBasic = [](ContextCPtr context, const float value[], float sign, int face, int corner)
    {
        OfficialMikkContext* state = OfficialMikkContext::from_context(context);
        state->tangent_space[size_t(face) * 3 + corner] = float4(value[0], value[1], value[2], sign);
    };

    SMikkTSpaceContext context = {};
    context.m_pInterface = &mikktspace;
    context.m_pUserData = &state;

    if (!genTangSpaceDefault(&context))
        SGL_THROW("Failed to generate tangents");
    return std::move(state.tangent_space);
}

struct TangentVariant {
    uint32_t source_vertex;
    uint32_t output_vertex;
    float4 tangent_space;
    uint32_t next;
};

size_t estimate_tangent_variant_count(size_t source_vertex_count, size_t corner_count)
{
    // Most indexed meshes need few tangent splits. Modest headroom avoids a large worst-case reservation while
    // retaining normal vector growth when a mesh contains more discontinuities.
    const size_t headroom
        = std::min(source_vertex_count / 5, corner_count - std::min(corner_count, source_vertex_count));
    return std::min(corner_count, source_vertex_count + headroom);
}

void assemble_tangent_variants(ImporterMesh& mesh, std::vector<float4> corner_tangent_space)
{
    // MikkTSpace returns an unindexed tangent frame for every triangle corner. Assemble that data into the indexed
    // mesh in three phases:
    // 1. Walk the corners and collect the distinct tangent variants required by each source vertex.
    // 2. Reuse the source vertex for its first variant, append later variants, and rewrite triangle indices.
    // 3. Allocate the appended vertices once, copy all their streams, then write tangent and handedness values.
    // This function only splits vertices. Callers that want exact compaction must deduplicate the authored mesh first.
    constexpr uint32_t INVALID_INDEX = std::numeric_limits<uint32_t>::max();
    const size_t source_vertex_count = mesh.vertex_count();
    FALCOR_CHECK(source_vertex_count < INVALID_INDEX, "Mesh vertex count exceeds the 32-bit index range");
    FALCOR_CHECK(
        corner_tangent_space.size() < INVALID_INDEX
            && source_vertex_count < INVALID_INDEX - corner_tangent_space.size(),
        "MikkTSpace output exceeds the 32-bit mesh index range"
    );

    // Each source vertex owns a linked list of tangent variants. The head array gives O(1) access to that usually
    // short list without allocating a separate container per vertex.
    std::vector<uint32_t> variant_heads(source_vertex_count, INVALID_INDEX);
    std::vector<TangentVariant> variants;
    variants.reserve(estimate_tangent_variant_count(source_vertex_count, corner_tangent_space.size()));

    // Phase 1: Walk every corner and find its tangent frame in the source vertex's variant list.
    size_t extra_variant_count = 0;
    size_t face_index = 0;
    for (ImporterMesh::Subgeometry& subgeometry : mesh.subgeometries) {
        for (uint3& triangle : subgeometry.indices) {
            for (int corner = 0; corner < 3; ++corner) {
                const uint32_t source_vertex = triangle[corner];
                const float4 tangent_space = corner_tangent_space[face_index * 3 + corner];

                uint32_t variant_index = variant_heads[source_vertex];
                while (variant_index != INVALID_INDEX) {
                    if (variants[variant_index].tangent_space == tangent_space)
                        break;
                    variant_index = variants[variant_index].next;
                }

                // Phase 2: Reuse the source slot for its first variant, append later variants, and rewrite the corner.
                if (variant_index == INVALID_INDEX) {
                    const bool first_variant = variant_heads[source_vertex] == INVALID_INDEX;
                    const uint32_t output_vertex
                        = first_variant ? source_vertex : uint32_t(source_vertex_count + extra_variant_count++);
                    variant_index = uint32_t(variants.size());
                    variants.push_back(
                        TangentVariant{
                            .source_vertex = source_vertex,
                            .output_vertex = output_vertex,
                            .tangent_space = tangent_space,
                            .next = variant_heads[source_vertex],
                        }
                    );
                    variant_heads[source_vertex] = variant_index;
                }
                triangle[corner] = variants[variant_index].output_vertex;
            }
            ++face_index;
        }
    }
    FALCOR_CHECK(face_index * 3 == corner_tangent_space.size(), "MikkTSpace corner count does not match mesh topology");

    // Phase 1 copied every required tangent into variants. Release the per-corner output and lookup table before
    // growing the mesh, which can itself allocate substantial vertex buffers.
    std::vector<float4>().swap(corner_tangent_space);
    std::vector<uint32_t>().swap(variant_heads);

    // Phase 3: Grow the mesh once, copy every existing stream for appended variants, then create any missing output
    // attributes at the final vertex count and write the generated tangent data.
    if (extra_variant_count > 0) {
        // TODO: Growing these vectors invalidates non-owning Python stream views. The Python add_tangents_from_uvs()
        // binding guards this path until vertex buffers use allocation-owned copy-on-write storage.
        mesh.allocate_vertices(extra_variant_count);

        for (ImporterMeshBuffer& buffer : mesh.buffers()) {
            for (const TangentVariant& variant : variants) {
                if (variant.output_vertex < source_vertex_count)
                    continue;
                std::memcpy(
                    buffer.data.data() + size_t(variant.output_vertex) * buffer.stride,
                    buffer.data.data() + size_t(variant.source_vertex) * buffer.stride,
                    buffer.stride
                );
            }
        }
    }
    mesh.ensure_attributes({{ImporterSemantic::tangent, 0}, {ImporterSemantic::handedness, 0}});

    auto tangents = mesh.tangent_stream();
    auto handedness = mesh.handedness_stream();
    for (const TangentVariant& variant : variants) {
        tangents[variant.output_vertex] = variant.tangent_space.xyz();
        handedness[variant.output_vertex] = variant.tangent_space.w;
    }
}

std::span<uint8_t> as_writable_bytes(std::span<float4> values)
{
    return {reinterpret_cast<uint8_t*>(values.data()), values.size_bytes()};
}

std::span<const uint8_t> as_bytes(std::span<const float4> values)
{
    return {reinterpret_cast<const uint8_t*>(values.data()), values.size_bytes()};
}

template<typename T>
void hash_vertex_stream(sgl::SHA1& sha1, ImporterVertexStreamRO<T> stream)
{
    if (stream.stride == sizeof(T)) {
        sha1.update(stream.data, stream.size * sizeof(T));
        return;
    }

    // Preserve the tightly packed logical byte sequence while amortizing SHA1 calls for interleaved buffers.
    constexpr size_t VALUES_PER_BLOCK = std::max<size_t>(1, (64 * 1024) / sizeof(T));
    std::array<T, VALUES_PER_BLOCK> values;
    for (size_t first = 0; first < stream.size; first += values.size()) {
        const size_t count = std::min(values.size(), stream.size - first);
        for (size_t i = 0; i < count; ++i)
            values[i] = stream[first + i];
        sha1.update(values.data(), count * sizeof(T));
    }
}

std::vector<float4> get_cached_corner_tangent_space(const ImporterMesh& mesh, BlobCache& cache)
{
    // Compute cache key from all inputs that affect the output.
    sgl::SHA1 sha1;
    // The suffix versions the cache payload schema, independently of the in-memory assembly algorithm.
    sha1.update("mikkt_corner_tangent_space_v1");

    const auto position_stream = mesh.position_stream();
    const auto normal_stream = mesh.normal_stream();
    const auto uv_stream = mesh.texcoord_stream();
    SGL_CHECK(position_stream.valid(), "Cannot generate tangents without position stream");
    SGL_CHECK(normal_stream.valid(), "Cannot generate tangents without normal stream");
    SGL_CHECK(uv_stream.valid(), "Cannot generate tangents without texcoord stream");

    hash_vertex_stream(sha1, position_stream);
    hash_vertex_stream(sha1, normal_stream);
    hash_vertex_stream(sha1, uv_stream);

    // Hash indices.
    for (const auto& subgeometry : mesh.subgeometries)
        sha1.update(subgeometry.indices.data(), subgeometry.indices.size() * sizeof(uint3));

    const sgl::SHA1::Digest key = sha1.digest();

    size_t face_count = 0;
    for (const auto& subgeometry : mesh.subgeometries)
        face_count += subgeometry.indices.size();
    std::vector<float4> tangent_space(face_count * 3);

    std::error_code read_ec;
    const bool cache_hit = cache.try_read(key, as_writable_bytes(tangent_space), std::nullopt, read_ec);
    if (read_ec)
        sgl::log_warn("Failed to read MikkTSpace tangent cache: {}", read_ec.message());

    if (!cache_hit) {
        // try_read() requires a destination of the expected size. Do not retain that failed-read allocation while
        // the official implementation produces its own per-corner output.
        std::vector<float4>().swap(tangent_space);
        tangent_space = generate_official_mikktspace(mesh);
    }

    if (cache_hit)
        return tangent_space;

    std::error_code write_ec;
    cache.try_write(key, as_bytes(tangent_space), std::nullopt, write_ec);
    if (write_ec)
        sgl::log_warn("Failed to write MikkTSpace tangent cache: {}", write_ec.message());
    return tangent_space;
}

} // namespace

void mikkt_generate_tangent_space(ImporterMesh& mesh)
{
    std::vector<float4> tangent_space = generate_official_mikktspace(mesh);
    assemble_tangent_variants(mesh, std::move(tangent_space));
}

#if defined(FALCOR_ENABLE_MIKKT_COMPARISON)
namespace mikkt_detail {

std::vector<float4> generate_corner_tangent_space(const ImporterMesh& mesh, Generator generator)
{
    switch (generator) {
    case Generator::official_mikktspace:
        return generate_official_mikktspace(mesh);
    case Generator::parallel_mikktspace:
        return generate_parallel_mikktspace(mesh);
    }
    FALCOR_UNREACHABLE();
}

void generate_tangent_space(ImporterMesh& mesh, Generator generator)
{
    std::vector<float4> tangent_space = generate_corner_tangent_space(mesh, generator);
    assemble_tangent_variants(mesh, std::move(tangent_space));
}

} // namespace mikkt_detail
#endif

void mikkt_generate_tangent_space(ImporterMesh& mesh, BlobCache& cache)
{
    std::vector<float4> tangent_space = get_cached_corner_tangent_space(mesh, cache);
    assemble_tangent_variants(mesh, std::move(tangent_space));
}

} // namespace falcor
