// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/vertex_remap.h"

#include "falcor2/core/error.h"

#include <bit>
#include <cstring>
#include <limits>

namespace falcor {

namespace {

constexpr uint32_t EMPTY_SLOT = std::numeric_limits<uint32_t>::max();

uint32_t hash_vertex(std::span<const VertexRemapStream> streams, uint32_t vertex_index)
{
    // This hash only selects a table bucket; exact stream comparison below determines vertex identity. Favor a
    // cheap word-at-a-time mix here because standard byte hashes are materially slower in Debug builds.
    uint32_t hash = UINT32_C(0x811c9dc5);
    uint32_t total_size = 0;
    for (const VertexRemapStream& stream : streams) {
        const std::byte* data = stream.data + vertex_index * stream.stride;
        size_t remaining = stream.element_size;
        while (remaining >= sizeof(uint32_t)) {
            uint32_t word;
            std::memcpy(&word, data, sizeof(word));
            hash = (hash ^ word) * UINT32_C(0x9e3779b1);
            data += sizeof(word);
            remaining -= sizeof(word);
        }

        uint32_t tail = 0;
        if (remaining > 0) {
            std::memcpy(&tail, data, remaining);
            hash = (hash ^ tail) * UINT32_C(0x9e3779b1);
        }
        total_size += static_cast<uint32_t>(stream.element_size);
    }

    hash ^= total_size;
    hash ^= hash >> 16;
    hash *= UINT32_C(0x85ebca6b);
    hash ^= hash >> 13;
    hash *= UINT32_C(0xc2b2ae35);
    hash ^= hash >> 16;
    return hash;
}

bool vertices_equal(std::span<const VertexRemapStream> streams, uint32_t lhs, uint32_t rhs)
{
    for (const VertexRemapStream& stream : streams) {
        if (std::memcmp(stream.data + lhs * stream.stride, stream.data + rhs * stream.stride, stream.element_size) != 0)
            return false;
    }
    return true;
}

} // namespace

VertexRemap generate_vertex_remap(std::span<const VertexRemapStream> streams, size_t vertex_count)
{
    FALCOR_CHECK(vertex_count < EMPTY_SLOT, "Vertex count exceeds the 32-bit mesh index range");
    for (const VertexRemapStream& stream : streams) {
        FALCOR_CHECK(stream.data || vertex_count == 0, "Vertex remap stream has no data");
        FALCOR_CHECK(stream.element_size > 0, "Vertex remap stream has an empty element");
        FALCOR_CHECK(stream.stride >= stream.element_size, "Vertex remap stream stride is smaller than its element");
    }

    VertexRemap result;
    if (vertex_count == 0)
        return result;

    result.old_to_new.resize(vertex_count);
    uint32_t unique_vertex_count = 0;
    {
        // Request 25% headroom before rounding to a power of two. Triangular probing visits every slot in such a table,
        // and the mask replaces a modulus in the hot loop.
        const size_t table_size = std::bit_ceil(vertex_count + vertex_count / 4);
        std::vector<uint32_t> table(table_size, EMPTY_SLOT);

        const size_t table_mask = table_size - 1;
        for (uint32_t vertex_index = 0; vertex_index < static_cast<uint32_t>(vertex_count); ++vertex_index) {
            size_t bucket = hash_vertex(streams, vertex_index) & table_mask;
            for (size_t probe = 0; probe <= table_mask; ++probe) {
                uint32_t& representative = table[bucket];
                if (representative == EMPTY_SLOT) {
                    representative = vertex_index;
                    result.old_to_new[vertex_index] = unique_vertex_count++;
                    break;
                }
                if (vertices_equal(streams, representative, vertex_index)) {
                    result.old_to_new[vertex_index] = result.old_to_new[representative];
                    break;
                }
                bucket = (bucket + probe + 1) & table_mask;
            }
        }
    }

    if (unique_vertex_count != vertex_count) {
        result.new_to_old.assign(unique_vertex_count, EMPTY_SLOT);
        for (uint32_t old_index = 0; old_index < static_cast<uint32_t>(vertex_count); ++old_index) {
            uint32_t& representative = result.new_to_old[result.old_to_new[old_index]];
            if (representative == EMPTY_SLOT)
                representative = old_index;
        }
    }
    return result;
}

} // namespace falcor
