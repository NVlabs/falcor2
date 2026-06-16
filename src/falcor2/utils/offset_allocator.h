// SPDX-License-Identifier: Apache-2.0 AND MIT

// Based on https://github.com/sebbbi/OffsetAllocator
// See LICENSES/offsetallocator.txt for the upstream MIT notice.

// (C) Sebastian Aaltonen 2023
// MIT License

#pragma once

// #define USE_16_BIT_NODE_INDICES

#include "falcor2/core/macros.h"

#include <cstdint>

namespace falcor {

class FALCOR_API OffsetAllocator {
public:
// 16 bit offsets mode will halve the metadata storage cost
// But it only supports up to 65536 maximum allocation count
#ifdef USE_16_BIT_NODE_INDICES
    typedef uint16_t NodeIndex;
#else
    typedef uint32_t NodeIndex;
#endif

    static constexpr uint32_t NUM_TOP_BINS = 32;
    static constexpr uint32_t BINS_PER_LEAF = 8;
    static constexpr uint32_t TOP_BINS_INDEX_SHIFT = 3;
    static constexpr uint32_t LEAF_BINS_INDEX_MASK = 0x7;
    static constexpr uint32_t NUM_LEAF_BINS = NUM_TOP_BINS * BINS_PER_LEAF;

    struct Allocation {
        static constexpr uint32_t NO_SPACE = 0xffffffff;

        uint32_t offset = NO_SPACE;
        NodeIndex metadata = NO_SPACE; // internal: node index

        bool is_valid() const { return offset != NO_SPACE; }
        explicit operator bool() const { return is_valid(); }
    };

    struct StorageReport {
        uint32_t total_free_space;
        uint32_t largest_free_region;
    };

    struct StorageReportFull {
        struct Region {
            uint32_t size;
            uint32_t count;
        };
        Region free_regions[NUM_LEAF_BINS];
    };

    OffsetAllocator(uint32_t size, uint32_t max_allocs = 128 * 1024);
    OffsetAllocator(OffsetAllocator&& other);
    ~OffsetAllocator();

    void reset();

    Allocation allocate(uint32_t size);
    void free(Allocation allocation);

    uint32_t allocation_size(Allocation allocation) const;
    StorageReport storage_report() const;
    StorageReportFull storage_report_full() const;

    uint32_t get_size() const { return m_size; }
    uint32_t get_max_allocs() const { return m_max_allocs; }
    uint32_t get_free_storage() const { return m_free_storage; }
    uint32_t get_current_allocs() const { return m_current_allocs; }

private:
    uint32_t insert_node_into_bin(uint32_t size, uint32_t data_offset);
    void remove_node_from_bin(uint32_t node_index);

    struct Node {
        static constexpr NodeIndex UNUSED = 0xffffffff;

        uint32_t data_offset = 0;
        uint32_t data_size = 0;
        NodeIndex bin_list_prev = UNUSED;
        NodeIndex bin_list_next = UNUSED;
        NodeIndex neighbor_prev = UNUSED;
        NodeIndex neighbor_next = UNUSED;
        bool used = false; // TODO: Merge as bit flag
    };

    uint32_t m_size;
    uint32_t m_max_allocs;
    uint32_t m_free_storage;
    uint32_t m_current_allocs;

    uint32_t m_used_bins_top;
    uint8_t m_used_bins[NUM_TOP_BINS];
    NodeIndex m_bin_indices[NUM_LEAF_BINS];

    Node* m_nodes;
    NodeIndex* m_free_nodes;
    uint32_t m_free_offset;
};

} // namespace falcor
