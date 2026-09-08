// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/mikkt_internal.h"
#include "falcor2/importers/importer_types.h"

#include "falcor2/core/error.h"

#include <sgl/core/thread.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <limits>
#include <numeric>
#include <span>
#include <utility>
#include <vector>

namespace falcor::mikkt_detail {

namespace {

// This is a parallel reimplementation of the MikkTSpace algorithm specialized for ImporterMesh. On representative
// large meshes it takes roughly 35-40% of the official implementation's Release time end to end (about 2.5-3x the
// throughput); the exact gain depends on mesh topology and available CPU parallelism.
//
// It uses MikkTSpace revision 3e895b49d05ea07e4c2133156cfa94369e19e409 as its behavioral oracle.
// Representative selection, BuildNeighborsFast sorting, GROUP_WITH_ANY traversal, and degenerate copying intentionally
// preserve ordering that is observable in the official implementation.
constexpr size_t PACKED_CORNER_STRIDE = 4;
constexpr size_t MIN_WORK_ITEMS_PER_PARALLEL_JOB = 4096;

template<typename Func>
void for_blocks(size_t count, size_t block_size, size_t work_item_count, Func&& func)
{
    if (count == 0)
        return;
    const size_t max_job_count = work_item_count / MIN_WORK_ITEMS_PER_PARALLEL_JOB;
    if (max_job_count < 2 || count <= block_size) {
        func(sgl::thread::blocked_range<size_t>(0, count, count));
        return;
    }

    // Scene ingestion already processes meshes concurrently. Limit inner parallelism by useful work, so crossing the
    // threshold grows gradually instead of immediately submitting every possible block for a medium-sized mesh.
    const size_t work_limited_block_size = 1 + (count - 1) / max_job_count;
    const size_t parallel_block_size = std::max(block_size, work_limited_block_size);
    sgl::thread::parallel_for(
        sgl::thread::blocked_range<size_t>(0, count, parallel_block_size),
        std::forward<Func>(func)
    );
}

struct ParallelMikkContext {
    std::vector<uint3> flattened_faces;
    std::span<const uint3> faces;
    ImporterVertexStreamRO<float3> positions;
    ImporterVertexStreamRO<float3> normals;
    ImporterVertexStreamRO<float2> uvs;

    explicit ParallelMikkContext(const ImporterMesh& mesh)
        : positions(mesh.position_stream())
        , normals(mesh.normal_stream())
        , uvs(mesh.texcoord_stream())
    {
        FALCOR_CHECK(positions.valid(), "Cannot generate tangents without position stream");
        FALCOR_CHECK(normals.valid(), "Cannot generate tangents without normal stream");
        FALCOR_CHECK(uvs.valid(), "Cannot generate tangents without texcoord stream");

        if (mesh.subgeometries.size() == 1) {
            faces = mesh.subgeometries.front().indices;
        } else {
            size_t face_count = 0;
            for (const ImporterMesh::Subgeometry& subgeometry : mesh.subgeometries)
                face_count += subgeometry.indices.size();
            flattened_faces.reserve(face_count);
            for (const ImporterMesh::Subgeometry& subgeometry : mesh.subgeometries)
                flattened_faces.insert(flattened_faces.end(), subgeometry.indices.begin(), subgeometry.indices.end());
            faces = flattened_faces;
        }
        FALCOR_CHECK(
            faces.size() <= size_t(std::numeric_limits<int>::max()) / PACKED_CORNER_STRIDE,
            "Mesh has too many triangle faces"
        );
    }

    ParallelMikkContext(const ParallelMikkContext&) = delete;
    ParallelMikkContext& operator=(const ParallelMikkContext&) = delete;
    ParallelMikkContext(ParallelMikkContext&&) = delete;
    ParallelMikkContext& operator=(ParallelMikkContext&&) = delete;
};

SGL_INLINE float dot(float3 lhs, float3 rhs)
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

SGL_INLINE float length(float3 value)
{
    return std::sqrt(dot(value, value));
}

SGL_INLINE bool is_nonzero(float value)
{
    return std::abs(value) > FLT_MIN;
}

SGL_INLINE bool is_nonzero(float3 value)
{
    return is_nonzero(value.x) || is_nonzero(value.y) || is_nonzero(value.z);
}

SGL_INLINE float3 normalize_nonzero(float3 value)
{
    return is_nonzero(value) ? value * (1.f / length(value)) : value;
}

SGL_INLINE float3 project(float3 value, float3 normal)
{
    return value - normal * dot(normal, value);
}

struct FaceTangent {
    float3 tangent{};
    float3 bitangent{};
    uint8_t orientation = 0; // Bit 0 is positive, bit 1 negative, and zero is degenerate.
    bool contributes = false;
    bool position_degenerate = false;
};

struct WeldVertex {
    float3 position{};
    uint32_t corner = 0;
    uint32_t vertex = 0;
};

#if defined(_MSC_VER)
__declspec(noinline)
#else
__attribute__((noinline))
#endif
int weld_grid_cell(float minimum, float maximum, float value)
{
    constexpr int CELL_COUNT = 2048;
    // The histogram and fill passes evaluate this independently. Like upstream FindGridCell, keep one out-of-line
    // implementation so compiler context cannot make the same input select different cells in the two passes.
    // The grid is only an acceleration structure. A collapsed extent belongs to one cell and still uses the exact
    // P/N/UV comparison below; avoid the otherwise-undefined NaN-to-integer conversion.
    if (minimum == maximum)
        return 0;
    const float index = CELL_COUNT * ((value - minimum) / (maximum - minimum));
    const int cell = int(index);
    return std::clamp(cell, 0, CELL_COUNT - 1);
}

SGL_INLINE bool vertices_equal(const ParallelMikkContext& mesh, uint32_t lhs, uint32_t rhs)
{
    return mesh.positions[lhs] == mesh.positions[rhs] && mesh.normals[lhs] == mesh.normals[rhs]
        && mesh.uvs[lhs] == mesh.uvs[rhs];
}

void merge_weld_vertices(
    std::span<int> representatives,
    std::span<WeldVertex> vertices,
    const ParallelMikkContext& mesh,
    int left,
    int right
)
{
    float3 minimum = vertices[left].position;
    float3 maximum = minimum;
    for (int i = left + 1; i <= right; ++i) {
        const float3 position = vertices[i].position;
        minimum.x = std::min(minimum.x, position.x);
        minimum.y = std::min(minimum.y, position.y);
        minimum.z = std::min(minimum.z, position.z);
        maximum.x = std::max(maximum.x, position.x);
        maximum.y = std::max(maximum.y, position.y);
        maximum.z = std::max(maximum.z, position.z);
    }
    const float3 extent = maximum - minimum;
    int channel = 0;
    if (extent.y > extent.x && extent.y > extent.z)
        channel = 1;
    else if (extent.z > extent.x)
        channel = 2;
    const float separation = 0.5f * (maximum[channel] + minimum[channel]);
    FALCOR_CHECK(std::isfinite(separation), "Parallel MikkTSpace position range exceeds the supported float domain");

    if (separation >= maximum[channel] || separation <= minimum[channel]) {
        for (int i = left; i <= right; ++i) {
            const uint32_t corner = vertices[i].corner;
            for (int previous = left; previous < i; ++previous) {
                if (vertices_equal(mesh, vertices[i].vertex, vertices[previous].vertex)) {
                    representatives[corner] = representatives[vertices[previous].corner];
                    break;
                }
            }
        }
        return;
    }

    int lhs = left;
    int rhs = right;
    while (lhs < rhs) {
        while (lhs < rhs && vertices[lhs].position[channel] < separation)
            ++lhs;
        while (lhs < rhs && !(vertices[rhs].position[channel] < separation))
            --rhs;
        if (lhs < rhs) {
            std::swap(vertices[lhs], vertices[rhs]);
            ++lhs;
            --rhs;
        }
    }
    if (lhs == rhs) {
        if (vertices[rhs].position[channel] < separation)
            ++lhs;
        else
            --rhs;
    }
    if (left < rhs)
        merge_weld_vertices(representatives, vertices, mesh, left, rhs);
    if (lhs < right)
        merge_weld_vertices(representatives, vertices, mesh, lhs, right);
}

std::vector<int> generate_ordered_corner_representatives(const ParallelMikkContext& mesh)
{
    // Equal P/N/UV corners could be hashed more cheaply, but Mikk's chosen representative feeds its legacy edge
    // ordering and is observable on non-manifold assets. Reproduce that spatial partition directly over our streams.
    struct CellVertex {
        uint32_t corner = 0;
        uint32_t vertex = 0;
    };
    constexpr int CELL_COUNT = 2048;
    const uint32_t corner_count = uint32_t(mesh.faces.size() * 3);
    std::vector<int> representatives(corner_count);
    float3 minimum = mesh.positions[mesh.faces.front().x];
    float3 maximum = minimum;
    for (size_t face = 0; face < mesh.faces.size(); ++face) {
        const uint3 vertices = mesh.faces[face];
        for (int corner = 0; corner < 3; ++corner) {
            // Preserve Mikk's four-wide triangle-list encoding because its representative IDs affect legacy edge
            // ordering on non-manifold geometry.
            representatives[face * 3 + corner] = int(face * PACKED_CORNER_STRIDE + corner);
            const uint32_t vertex = corner == 0 ? vertices.x : corner == 1 ? vertices.y : vertices.z;
            const float3 position = mesh.positions[vertex];
            minimum.x = std::min(minimum.x, position.x);
            minimum.y = std::min(minimum.y, position.y);
            minimum.z = std::min(minimum.z, position.z);
            maximum.x = std::max(maximum.x, position.x);
            maximum.y = std::max(maximum.y, position.y);
            maximum.z = std::max(maximum.z, position.z);
        }
    }
    const float3 extent = maximum - minimum;
    int channel = 0;
    if (extent.y > extent.x && extent.y > extent.z)
        channel = 1;
    else if (extent.z > extent.x)
        channel = 2;
    FALCOR_CHECK(
        std::isfinite(extent[channel]),
        "Parallel MikkTSpace position extent exceeds the supported float range"
    );

    std::vector<uint32_t> cell_offsets(CELL_COUNT + 1, 0);
    for (const uint3& vertices : mesh.faces) {
        for (int corner = 0; corner < 3; ++corner) {
            const uint32_t vertex = corner == 0 ? vertices.x : corner == 1 ? vertices.y : vertices.z;
            const float value = mesh.positions[vertex][channel];
            ++cell_offsets[size_t(weld_grid_cell(minimum[channel], maximum[channel], value)) + 1];
        }
    }
    std::partial_sum(cell_offsets.begin(), cell_offsets.end(), cell_offsets.begin());
    std::vector<CellVertex> cell_vertices(corner_count);
    std::vector<uint32_t> next_offset = cell_offsets;
    for (size_t face = 0; face < mesh.faces.size(); ++face) {
        const uint3 vertices = mesh.faces[face];
        for (uint32_t corner = 0; corner < 3; ++corner) {
            const uint32_t vertex = corner == 0 ? vertices.x : corner == 1 ? vertices.y : vertices.z;
            const float value = mesh.positions[vertex][channel];
            const int cell = weld_grid_cell(minimum[channel], maximum[channel], value);
            cell_vertices[next_offset[size_t(cell)]++] = {uint32_t(face * 3 + corner), vertex};
        }
    }

    for_blocks(
        CELL_COUNT,
        16,
        corner_count,
        [&](const auto& range)
        {
            size_t scratch_size = 0;
            for (size_t cell : range)
                scratch_size = std::max(scratch_size, size_t(cell_offsets[cell + 1] - cell_offsets[cell]));
            std::vector<WeldVertex> vertices(scratch_size);
            for (size_t cell : range) {
                const uint32_t begin = cell_offsets[cell];
                const uint32_t count = cell_offsets[cell + 1] - begin;
                if (count < 2)
                    continue;
                for (uint32_t i = 0; i < count; ++i) {
                    const CellVertex cell_vertex = cell_vertices[begin + i];
                    vertices[i] = {mesh.positions[cell_vertex.vertex], cell_vertex.corner, cell_vertex.vertex};
                }
                merge_weld_vertices(representatives, vertices, mesh, 0, int(count) - 1);
            }
        }
    );
    return representatives;
}

std::vector<FaceTangent> compute_face_tangents(const ParallelMikkContext& mesh)
{
    std::vector<FaceTangent> result(mesh.faces.size());
    for_blocks(
        mesh.faces.size(),
        4096,
        mesh.faces.size(),
        [&](const auto& range)
        {
            for (size_t face_index : range) {
                const uint3 face = mesh.faces[face_index];
                const float3 p0 = mesh.positions[face.x];
                const float3 p1 = mesh.positions[face.y];
                const float3 p2 = mesh.positions[face.z];
                const float2 uv0 = mesh.uvs[face.x];
                const float2 uv1 = mesh.uvs[face.y];
                const float2 uv2 = mesh.uvs[face.z];

                FaceTangent& tangent = result[face_index];
                tangent.position_degenerate = p0 == p1 || p0 == p2 || p1 == p2;
                if (tangent.position_degenerate)
                    continue;

                const float3 edge1 = p1 - p0;
                const float3 edge2 = p2 - p0;
                const float2 uv_edge1 = uv1 - uv0;
                const float2 uv_edge2 = uv2 - uv0;
                const float signed_area = uv_edge1.x * uv_edge2.y - uv_edge1.y * uv_edge2.x;
                tangent.orientation = signed_area > 0.f ? 1 : signed_area < 0.f ? 2 : 0;

                const float3 derivative = uv_edge2.y * edge1 - uv_edge1.y * edge2;
                const float3 bitangent = -uv_edge2.x * edge1 + uv_edge1.x * edge2;
                if (is_nonzero(signed_area)) {
                    const float derivative_length = length(derivative);
                    const float bitangent_length = length(bitangent);
                    const float absolute_area = std::abs(signed_area);
                    const float orientation_sign = tangent.orientation == 1 ? 1.f : -1.f;
                    if (is_nonzero(derivative_length))
                        tangent.tangent = derivative * (orientation_sign / derivative_length);
                    if (is_nonzero(bitangent_length))
                        tangent.bitangent = bitangent * (orientation_sign / bitangent_length);

                    // Match Mikk's GROUP_WITH_ANY test: eligibility is based on the pre-normalization magnitudes, not
                    // merely on non-zero derivative lengths. This distinction is observable for extreme finite inputs.
                    const float tangent_magnitude = derivative_length / absolute_area;
                    const float bitangent_magnitude = bitangent_length / absolute_area;
                    tangent.contributes = is_nonzero(tangent_magnitude) && is_nonzero(bitangent_magnitude);
                }
                if (!tangent.contributes)
                    tangent.orientation = 0;
            }
        }
    );
    return result;
}

struct Edge {
    int vertex0 = 0;
    int vertex1 = 0;
    uint32_t face = 0;
    uint8_t local = 0;
};

struct EdgeSortPartition {
    int left_end;
    int right_begin;
    uint32_t child_seed;
};

SGL_INLINE int edge_channel(const Edge& edge, int channel)
{
    return channel == 0 ? edge.vertex0 : channel == 1 ? edge.vertex1 : int(edge.face);
}

EdgeSortPartition partition_edges(std::span<Edge> edges, int left, int right, int channel, uint32_t seed)
{
    const int count = right - left + 1;
    seed = seed + std::rotl(seed, int(seed & 31)) + 3;
    int lhs = left;
    int rhs = right;
    const int pivot = edge_channel(edges[left + int(seed % uint32_t(count))], channel);
    do {
        while (edge_channel(edges[lhs], channel) < pivot)
            ++lhs;
        while (edge_channel(edges[rhs], channel) > pivot)
            --rhs;
        if (lhs <= rhs) {
            std::swap(edges[lhs], edges[rhs]);
            ++lhs;
            --rhs;
        }
    } while (lhs <= rhs);
    return {rhs, lhs, seed};
}

void sort_edges(std::span<Edge> edges, int left, int right, int channel, uint32_t seed)
{
    const int count = right - left + 1;
    if (count < 2)
        return;
    if (count == 2) {
        if (edge_channel(edges[left], channel) > edge_channel(edges[right], channel))
            std::swap(edges[left], edges[right]);
        return;
    }

    const EdgeSortPartition partition = partition_edges(edges, left, right, channel, seed);
    if (left < partition.left_end)
        sort_edges(edges, left, partition.left_end, channel, partition.child_seed);
    if (partition.right_begin < right)
        sort_edges(edges, partition.right_begin, right, channel, partition.child_seed);
}

void parallel_sort_edges(std::span<Edge> edges, int channel, uint32_t seed)
{
    constexpr int SORT_JOB_SIZE = 64 * 1024;
    struct Job {
        int left;
        int right;
        uint32_t seed;
    };
    std::vector<Job> jobs;
    auto partition_jobs = [&](auto&& self, int left, int right, uint32_t job_seed) -> void
    {
        const int count = right - left + 1;
        if (count <= SORT_JOB_SIZE) {
            jobs.push_back({left, right, job_seed});
            return;
        }
        const EdgeSortPartition partition = partition_edges(edges, left, right, channel, job_seed);
        if (left < partition.left_end)
            self(self, left, partition.left_end, partition.child_seed);
        if (partition.right_begin < right)
            self(self, partition.right_begin, right, partition.child_seed);
    };
    if (!edges.empty())
        partition_jobs(partition_jobs, 0, int(edges.size()) - 1, seed);

    // The partition tree is identical to Mikk's quicksort. Only disjoint subtrees execute concurrently.
    for_blocks(
        jobs.size(),
        1,
        edges.size(),
        [&](const auto& range)
        {
            for (size_t job_index : range) {
                const Job& job = jobs[job_index];
                sort_edges(edges, job.left, job.right, channel, job.seed);
            }
        }
    );
}

std::vector<uint32_t>
build_edge_neighbors(std::span<const int> representatives, std::span<const FaceTangent> face_tangents)
{
    constexpr uint32_t NO_NEIGHBOR = std::numeric_limits<uint32_t>::max();
    const size_t face_count = representatives.size() / 3;
    std::vector<Edge> edges(representatives.size());
    size_t edge_count = 0;
    for (uint32_t face = 0; face < face_count; ++face) {
        if (face_tangents[face].position_degenerate)
            continue;
        for (uint8_t local = 0; local < 3; ++local) {
            const int start = representatives[size_t(face) * 3 + local];
            const int end = representatives[size_t(face) * 3 + (local + 1) % 3];
            edges[edge_count++] = {std::min(start, end), std::max(start, end), face, local};
        }
    }
    edges.resize(edge_count);
    // The randomized ordering is part of Mikk compatibility: a normal sort changes which faces are paired when an
    // edge is non-manifold. Only the tie-break is retained here; the rest of the implementation is purpose-built.
    constexpr uint32_t SORT_SEED = 39871946;
    parallel_sort_edges(edges, 0, SORT_SEED);
    size_t run_begin = 0;
    for (size_t i = 1; i < edges.size(); ++i) {
        if (edges[run_begin].vertex0 != edges[i].vertex0) {
            sort_edges(edges, int(run_begin), int(i - 1), 1, SORT_SEED);
            run_begin = i;
        }
    }
    run_begin = 0;
    for (size_t i = 1; i < edges.size(); ++i) {
        if (edges[run_begin].vertex0 != edges[i].vertex0 || edges[run_begin].vertex1 != edges[i].vertex1) {
            sort_edges(edges, int(run_begin), int(i - 1), 2, SORT_SEED);
            run_begin = i;
        }
    }
    // Keep BuildNeighborsFast compatibility: upstream does not flush either trailing sub-sort run. Although unusual,
    // changing it can alter face pairing on a non-manifold edge.

    // A non-manifold edge may have more than two incident faces. Mikk pairs each directed edge only once in face
    // order rather than merging the entire fan; this otherwise-observable rule is important on production assets.
    std::vector<uint32_t> result(representatives.size(), NO_NEIGHBOR);
    for (size_t edge_run_begin = 0; edge_run_begin < edges.size();) {
        size_t run_end = edge_run_begin + 1;
        while (run_end < edges.size() && edges[run_end].vertex0 == edges[edge_run_begin].vertex0
               && edges[run_end].vertex1 == edges[edge_run_begin].vertex1)
            ++run_end;
        for (size_t i = edge_run_begin; i < run_end; ++i) {
            const Edge& lhs = edges[i];
            const uint32_t lhs_edge = lhs.face * 3 + lhs.local;
            if (result[lhs_edge] != NO_NEIGHBOR)
                continue;
            const int lhs_start = representatives[lhs_edge];
            const int lhs_end = representatives[size_t(lhs.face) * 3 + (lhs.local + 1) % 3];
            for (size_t j = i + 1; j < run_end; ++j) {
                const Edge& rhs = edges[j];
                const uint32_t rhs_edge = rhs.face * 3 + rhs.local;
                if (result[rhs_edge] != NO_NEIGHBOR)
                    continue;
                const int rhs_start = representatives[rhs_edge];
                const int rhs_end = representatives[size_t(rhs.face) * 3 + (rhs.local + 1) % 3];
                if (lhs_start == rhs_end && lhs_end == rhs_start) {
                    result[lhs_edge] = rhs_edge;
                    result[rhs_edge] = lhs_edge;
                    break;
                }
            }
        }
        edge_run_begin = run_end;
    }
    return result;
}

void merge_corner_groups(
    std::span<uint32_t> corner_groups,
    std::span<uint8_t> face_orientations,
    std::span<const uint32_t> edge_neighbors,
    std::span<const int> representatives,
    std::span<const FaceTangent> face_tangents
)
{
    constexpr uint32_t UNASSIGNED = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t NO_NEIGHBOR = std::numeric_limits<uint32_t>::max();
    std::fill(corner_groups.begin(), corner_groups.end(), UNASSIGNED);

    // Mikk seeds groups in face/corner order and recursively visits the two edges incident to that corner. This order
    // is observable only for UV-degenerate faces: the first contributing group that reaches one chooses its
    // orientation. Use an explicit stack to preserve the traversal without risking recursion depth on large meshes.
    std::vector<uint32_t> pending_faces;
    auto append_neighbor = [&](uint32_t edge)
    {
        const uint32_t neighbor = edge_neighbors[edge];
        if (neighbor != NO_NEIGHBOR)
            pending_faces.push_back(neighbor / 3);
    };

    for (uint32_t seed_face = 0; seed_face < face_tangents.size(); ++seed_face) {
        if (!face_tangents[seed_face].contributes)
            continue;
        for (uint32_t seed_local = 0; seed_local < 3; ++seed_local) {
            const uint32_t seed_corner = seed_face * 3 + seed_local;
            if (corner_groups[seed_corner] != UNASSIGNED)
                continue;

            const uint32_t group = seed_corner;
            const int representative = representatives[seed_corner];
            const uint8_t orientation = face_orientations[seed_face];
            corner_groups[seed_corner] = group;
            pending_faces.clear();
            // Push right before left so the LIFO traversal processes Mikk's left neighbor first.
            append_neighbor(seed_face * 3 + (seed_local + 2) % 3);
            append_neighbor(seed_face * 3 + seed_local);

            while (!pending_faces.empty()) {
                const uint32_t face = pending_faces.back();
                pending_faces.pop_back();

                uint32_t local = 0;
                while (local < 3 && representatives[face * 3 + local] != representative)
                    ++local;
                FALCOR_ASSERT(local < 3);
                const uint32_t corner = face * 3 + local;
                if (corner_groups[corner] == group)
                    continue;
                if (corner_groups[corner] != UNASSIGNED)
                    continue;

                if (!face_tangents[face].contributes && corner_groups[face * 3] == UNASSIGNED
                    && corner_groups[face * 3 + 1] == UNASSIGNED && corner_groups[face * 3 + 2] == UNASSIGNED)
                    face_orientations[face] = orientation;
                if (face_orientations[face] != orientation)
                    continue;

                corner_groups[corner] = group;
                append_neighbor(face * 3 + (local + 2) % 3);
                append_neighbor(face * 3 + local);
            }
        }
    }

    // Corners not reachable from a contributing face keep Mikk's conventional default tangent independently.
    for (uint32_t corner = 0; corner < corner_groups.size(); ++corner) {
        if (corner_groups[corner] == UNASSIGNED)
            corner_groups[corner] = corner;
    }
}

void accumulate_corner_tangents(
    std::span<float4> output,
    const ParallelMikkContext& mesh,
    std::span<const FaceTangent> face_tangents,
    std::span<const uint32_t> corner_groups,
    std::span<const uint8_t> face_orientations,
    std::span<const int> representatives
)
{
    constexpr float ANGULAR_THRESHOLD_COSINE = -1.f;
    constexpr int NEXT[3] = {1, 2, 0};
    constexpr int PREVIOUS[3] = {2, 0, 1};
    struct CornerContribution {
        float3 tangent{};
        float3 bitangent{};
        float angle = 0.f;
    };
    // Release the O(corner-count) accumulation data before allocating the positional-degenerate copy map below.
    {
        std::vector<CornerContribution> contributions(output.size());

        for_blocks(
            mesh.faces.size(),
            4096,
            mesh.faces.size(),
            [&](const auto& range)
            {
                for (size_t face_index : range) {
                    if (!face_tangents[face_index].contributes || face_tangents[face_index].position_degenerate)
                        continue;
                    const uint3 face = mesh.faces[face_index];
                    for (int corner = 0; corner < 3; ++corner) {
                        const uint32_t vertex = face[corner];
                        const float3 normal = mesh.normals[vertex];
                        CornerContribution& contribution = contributions[face_index * 3 + corner];
                        contribution.tangent = normalize_nonzero(project(face_tangents[face_index].tangent, normal));
                        contribution.bitangent
                            = normalize_nonzero(project(face_tangents[face_index].bitangent, normal));

                        float3 edge1 = mesh.positions[face[PREVIOUS[corner]]] - mesh.positions[vertex];
                        float3 edge2 = mesh.positions[face[NEXT[corner]]] - mesh.positions[vertex];
                        edge1 = normalize_nonzero(project(edge1, normal));
                        edge2 = normalize_nonzero(project(edge2, normal));
                        const float cosine = std::clamp(dot(edge1, edge2), -1.f, 1.f);
                        contribution.angle = std::acos(cosine);
                    }
                }
            }
        );

        std::vector<uint32_t> group_offsets(output.size() + 1, 0);
        for (uint32_t group : corner_groups)
            ++group_offsets[size_t(group) + 1];
        std::partial_sum(group_offsets.begin(), group_offsets.end(), group_offsets.begin());
        std::vector<uint32_t> group_corners(output.size());
        {
            std::vector<uint32_t> next_offset = group_offsets;
            for (uint32_t corner = 0; corner < output.size(); ++corner)
                group_corners[next_offset[size_t(corner_groups[corner])]++] = corner;
        }

        std::fill(output.begin(), output.end(), float4(1.f, 0.f, 0.f, -1.f));
        auto write_tangent = [&](uint32_t corner, float3 accumulated)
        {
            const bool has_tangent = is_nonzero(accumulated);
            const float3 tangent = has_tangent ? normalize_nonzero(accumulated) : float3{};
            const float sign = (face_orientations[corner / 3] & 1) != 0 ? 1.f : -1.f;
            output[corner] = float4(tangent, sign);
        };

        for_blocks(
            output.size(),
            4096,
            output.size(),
            [&](const auto& range)
            {
                for (size_t group : range) {
                    const uint32_t begin = group_offsets[group];
                    const uint32_t end = group_offsets[group + 1];
                    if (begin == end)
                        continue;

                    bool has_contributing_face = false;
                    for (uint32_t i = begin; i < end; ++i)
                        has_contributing_face |= face_tangents[group_corners[i] / 3].contributes;
                    if (!has_contributing_face)
                        continue;

                    bool has_subgroups = false;
                    for (uint32_t i = begin; i < end && !has_subgroups; ++i) {
                        const uint32_t lhs = group_corners[i];
                        if (!face_tangents[lhs / 3].contributes)
                            continue;
                        for (uint32_t j = i + 1; j < end; ++j) {
                            const uint32_t rhs = group_corners[j];
                            if (!face_tangents[rhs / 3].contributes)
                                continue;
                            has_subgroups = dot(contributions[lhs].tangent, contributions[rhs].tangent)
                                    <= ANGULAR_THRESHOLD_COSINE
                                || dot(contributions[lhs].bitangent, contributions[rhs].bitangent)
                                    <= ANGULAR_THRESHOLD_COSINE;
                            if (has_subgroups)
                                break;
                        }
                    }

                    if (!has_subgroups) {
                        float3 accumulated{};
                        for (uint32_t i = begin; i < end; ++i) {
                            const CornerContribution& contribution = contributions[group_corners[i]];
                            accumulated += contribution.tangent * contribution.angle;
                        }
                        for (uint32_t i = begin; i < end; ++i)
                            write_tangent(group_corners[i], accumulated);
                        continue;
                    }

                    for (uint32_t i = begin; i < end; ++i) {
                        const uint32_t corner = group_corners[i];
                        const CornerContribution& seed = contributions[corner];
                        float3 accumulated{};
                        for (uint32_t j = begin; j < end; ++j) {
                            const uint32_t member = group_corners[j];
                            if (!face_tangents[member / 3].contributes
                                || (dot(seed.tangent, contributions[member].tangent) > ANGULAR_THRESHOLD_COSINE
                                    && dot(seed.bitangent, contributions[member].bitangent)
                                        > ANGULAR_THRESHOLD_COSINE)) {
                                accumulated += contributions[member].tangent * contributions[member].angle;
                            }
                        }
                        write_tangent(corner, accumulated);
                    }
                }
            }
        );
    }

    // Positional degenerates do not participate in grouping. Mikk copies each of their welded corners from the
    // first non-degenerate occurrence, leaving the conventional (1, 0, 0, -1) value when none exists.
    std::vector<uint32_t> first_good_corner(
        mesh.faces.size() * PACKED_CORNER_STRIDE,
        std::numeric_limits<uint32_t>::max()
    );
    for (uint32_t corner = 0; corner < output.size(); ++corner) {
        if (!face_tangents[corner / 3].position_degenerate) {
            const int representative = representatives[corner];
            if (first_good_corner[representative] == std::numeric_limits<uint32_t>::max())
                first_good_corner[representative] = corner;
        }
    }
    for_blocks(
        output.size(),
        4096,
        output.size(),
        [&](const auto& range)
        {
            for (size_t corner : range) {
                if (face_tangents[corner / 3].position_degenerate) {
                    const uint32_t source = first_good_corner[representatives[corner]];
                    if (source != std::numeric_limits<uint32_t>::max())
                        output[corner] = output[source];
                }
            }
        }
    );
}

} // namespace

std::vector<float4> generate_parallel_mikktspace(const ImporterMesh& mesh)
{
    const ParallelMikkContext view(mesh);
    FALCOR_CHECK(!view.faces.empty(), "Cannot generate tangents for an empty mesh");

    // Phase 1: weld only the attributes that define a tangent frame, then build all incident corners for each class.
    const std::vector<FaceTangent> face_tangents = compute_face_tangents(view);
    std::vector<int> representatives = generate_ordered_corner_representatives(view);
    std::vector<uint32_t> corner_groups(representatives.size());
    std::vector<uint8_t> face_orientations(view.faces.size());
    for (size_t face = 0; face < view.faces.size(); ++face)
        face_orientations[face] = face_tangents[face].orientation;

    // Phase 2: edge connectivity and UV orientation partition each welded class into Mikk-compatible tangent groups.
    {
        const std::vector<uint32_t> edge_neighbors = build_edge_neighbors(representatives, face_tangents);
        merge_corner_groups(corner_groups, face_orientations, edge_neighbors, representatives, face_tangents);
    }

    // Phase 3: project and angle-weight each face derivative directly into its group, then scatter per corner.
    std::vector<float4> output(representatives.size());
    accumulate_corner_tangents(output, view, face_tangents, corner_groups, face_orientations, representatives);
    return output;
}

} // namespace falcor::mikkt_detail
