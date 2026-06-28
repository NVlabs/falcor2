// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Based on https://github.com/sebbbi/OffsetAllocator
// See LICENSES/offsetallocator.txt for the upstream MIT notice.

// (C) Sebastian Aaltonen 2023
// MIT License

#include "offset_allocator.h"

#include "falcor2/core/error.h"

#ifdef DEBUG_VERBOSE
#include <cstdio>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#endif

#include <cstring>

namespace falcor {

inline uint32_t lzcnt_nonzero(uint32_t v)
{
#ifdef _MSC_VER
    unsigned long retVal;
    _BitScanReverse(&retVal, v);
    return 31 - retVal;
#else
    return __builtin_clz(v);
#endif
}

inline uint32_t tzcnt_nonzero(uint32_t v)
{
#ifdef _MSC_VER
    unsigned long retVal;
    _BitScanForward(&retVal, v);
    return retVal;
#else
    return __builtin_ctz(v);
#endif
}

namespace SmallFloat {
static constexpr uint32_t MANTISSA_BITS = 3;
static constexpr uint32_t MANTISSA_VALUE = 1 << MANTISSA_BITS;
static constexpr uint32_t MANTISSA_MASK = MANTISSA_VALUE - 1;

// Bin sizes follow floating point (exponent + mantissa) distribution (piecewise linear log approx)
// This ensures that for each size class, the average overhead percentage stays the same
FALCOR_API uint32_t uint_to_float_round_up(uint32_t size)
{
    uint32_t exp = 0;
    uint32_t mantissa = 0;

    if (size < MANTISSA_VALUE) {
        // Denorm: 0..(MANTISSA_VALUE-1)
        mantissa = size;
    } else {
        // Normalized: Hidden high bit always 1. Not stored. Just like float.
        uint32_t leading_zeros = lzcnt_nonzero(size);
        uint32_t highest_set_bit = 31 - leading_zeros;

        uint32_t mantissa_start_bit = highest_set_bit - MANTISSA_BITS;
        exp = mantissa_start_bit + 1;
        mantissa = (size >> mantissa_start_bit) & MANTISSA_MASK;

        uint32_t low_bits_mask = (1 << mantissa_start_bit) - 1;

        // Round up!
        if ((size & low_bits_mask) != 0)
            mantissa++;
    }

    return (exp << MANTISSA_BITS) + mantissa; // + allows mantissa->exp overflow for round up
}

FALCOR_API uint32_t uint_to_float_round_down(uint32_t size)
{
    uint32_t exp = 0;
    uint32_t mantissa = 0;

    if (size < MANTISSA_VALUE) {
        // Denorm: 0..(MANTISSA_VALUE-1)
        mantissa = size;
    } else {
        // Normalized: Hidden high bit always 1. Not stored. Just like float.
        uint32_t leading_zeros = lzcnt_nonzero(size);
        uint32_t highest_set_bit = 31 - leading_zeros;

        uint32_t mantissa_start_bit = highest_set_bit - MANTISSA_BITS;
        exp = mantissa_start_bit + 1;
        mantissa = (size >> mantissa_start_bit) & MANTISSA_MASK;
    }

    return (exp << MANTISSA_BITS) | mantissa;
}

FALCOR_API uint32_t float_to_uint(uint32_t float_value)
{
    uint32_t exponent = float_value >> MANTISSA_BITS;
    uint32_t mantissa = float_value & MANTISSA_MASK;
    if (exponent == 0) {
        // Denorms
        return mantissa;
    } else {
        return (mantissa | MANTISSA_VALUE) << (exponent - 1);
    }
}
} // namespace SmallFloat

// Utility functions
uint32_t find_lowest_set_bit_after(uint32_t bit_mask, uint32_t start_bit_index)
{
    uint32_t mask_before_start_index = (1 << start_bit_index) - 1;
    uint32_t mask_after_start_index = ~mask_before_start_index;
    uint32_t bits_after = bit_mask & mask_after_start_index;
    if (bits_after == 0)
        return OffsetAllocator::Allocation::NO_SPACE;
    return tzcnt_nonzero(bits_after);
}

// OffsetAllocator...
OffsetAllocator::OffsetAllocator(uint32_t size, uint32_t max_allocs)
    : m_size(size)
    , m_max_allocs(max_allocs)
    , m_current_allocs(0)
    , m_nodes(nullptr)
    , m_free_nodes(nullptr)
{
    if constexpr (sizeof(NodeIndex) == 2) {
        FALCOR_ASSERT(max_allocs <= 65536);
    }
    reset();
}

OffsetAllocator::OffsetAllocator(OffsetAllocator&& other)
    : m_size(other.m_size)
    , m_max_allocs(other.m_max_allocs)
    , m_free_storage(other.m_free_storage)
    , m_current_allocs(other.m_current_allocs)
    , m_used_bins_top(other.m_used_bins_top)
    , m_nodes(other.m_nodes)
    , m_free_nodes(other.m_free_nodes)
    , m_free_offset(other.m_free_offset)
{
    memcpy(m_used_bins, other.m_used_bins, sizeof(uint8_t) * NUM_TOP_BINS);
    memcpy(m_bin_indices, other.m_bin_indices, sizeof(NodeIndex) * NUM_LEAF_BINS);

    other.m_nodes = nullptr;
    other.m_free_nodes = nullptr;
    other.m_free_offset = 0;
    other.m_max_allocs = 0;
    other.m_used_bins_top = 0;
}

void OffsetAllocator::reset()
{
    m_free_storage = 0;
    m_used_bins_top = 0;
    m_current_allocs = 0;
    m_free_offset = m_max_allocs - 1;

    for (uint32_t i = 0; i < NUM_TOP_BINS; i++)
        m_used_bins[i] = 0;

    for (uint32_t i = 0; i < NUM_LEAF_BINS; i++)
        m_bin_indices[i] = Node::UNUSED;

    if (m_nodes)
        delete[] m_nodes;
    if (m_free_nodes)
        delete[] m_free_nodes;

    m_nodes = new Node[m_max_allocs];
    m_free_nodes = new NodeIndex[m_max_allocs];

    // Freelist is a stack. Nodes in inverse order so that [0] pops first.
    for (uint32_t i = 0; i < m_max_allocs; i++) {
        m_free_nodes[i] = m_max_allocs - i - 1;
    }

    // Start state: Whole storage as one big node
    // Algorithm will split remainders and push them back as smaller nodes
    insert_node_into_bin(m_size, 0);
}

OffsetAllocator::~OffsetAllocator()
{
    delete[] m_nodes;
    delete[] m_free_nodes;
}

OffsetAllocator::Allocation OffsetAllocator::allocate(uint32_t size)
{
    // Out of allocations?
    if (m_free_offset == 0) {
        return {};
    }

    // Round up to bin index to ensure that alloc >= bin
    // Gives us min bin index that fits the size
    uint32_t min_bin_index = SmallFloat::uint_to_float_round_up(size);

    uint32_t min_top_bin_index = min_bin_index >> TOP_BINS_INDEX_SHIFT;
    uint32_t min_leaf_bin_index = min_bin_index & LEAF_BINS_INDEX_MASK;

    uint32_t top_bin_index = min_top_bin_index;
    uint32_t leaf_bin_index = Allocation::NO_SPACE;

    // If top bin exists, scan its leaf bin. This can fail (NO_SPACE).
    if (m_used_bins_top & (1 << top_bin_index)) {
        leaf_bin_index = find_lowest_set_bit_after(m_used_bins[top_bin_index], min_leaf_bin_index);
    }

    // If we didn't find space in top bin, we search top bin from +1
    if (leaf_bin_index == Allocation::NO_SPACE) {
        top_bin_index = find_lowest_set_bit_after(m_used_bins_top, min_top_bin_index + 1);

        // Out of space?
        if (top_bin_index == Allocation::NO_SPACE) {
            return {};
        }

        // All leaf bins here fit the alloc, since the top bin was rounded up. Start leaf search from bit 0.
        // NOTE: This search can't fail since at least one leaf bit was set because the top bit was set.
        leaf_bin_index = tzcnt_nonzero(m_used_bins[top_bin_index]);
    }

    uint32_t bin_index = (top_bin_index << TOP_BINS_INDEX_SHIFT) | leaf_bin_index;

    // Pop the top node of the bin. Bin top = node.next.
    uint32_t node_index = m_bin_indices[bin_index];
    Node& node = m_nodes[node_index];
    uint32_t node_total_size = node.data_size;
    node.data_size = size;
    node.used = true;
    m_bin_indices[bin_index] = node.bin_list_next;
    if (node.bin_list_next != Node::UNUSED)
        m_nodes[node.bin_list_next].bin_list_prev = Node::UNUSED;
    m_free_storage -= node_total_size;
#ifdef DEBUG_VERBOSE
    printf("Free storage: %u (-%u) (allocate)\n", m_free_storage, node_total_size);
#endif

    // Bin empty?
    if (m_bin_indices[bin_index] == Node::UNUSED) {
        // Remove a leaf bin mask bit
        m_used_bins[top_bin_index] &= ~(1 << leaf_bin_index);

        // All leaf bins empty?
        if (m_used_bins[top_bin_index] == 0) {
            // Remove a top bin mask bit
            m_used_bins_top &= ~(1 << top_bin_index);
        }
    }

    // Push back reminder N elements to a lower bin
    uint32_t reminder_size = node_total_size - size;
    if (reminder_size > 0) {
        uint32_t new_node_index = insert_node_into_bin(reminder_size, node.data_offset + size);

        // Link nodes next to each other so that we can merge them later if both are free
        // And update the old next neighbor to point to the new node (in middle)
        if (node.neighbor_next != Node::UNUSED)
            m_nodes[node.neighbor_next].neighbor_prev = new_node_index;
        m_nodes[new_node_index].neighbor_prev = node_index;
        m_nodes[new_node_index].neighbor_next = node.neighbor_next;
        node.neighbor_next = new_node_index;
    }

    // Track total allocations
    m_current_allocs++;

    return {node.data_offset, node_index};
}

void OffsetAllocator::free(Allocation allocation)
{
    FALCOR_ASSERT(allocation.metadata != Allocation::NO_SPACE);
    if (!m_nodes)
        return;

    uint32_t node_index = allocation.metadata;
    Node& node = m_nodes[node_index];

    // Double delete check
    FALCOR_ASSERT(node.used == true);

    // Merge with neighbors...
    uint32_t offset = node.data_offset;
    uint32_t size = node.data_size;

    if ((node.neighbor_prev != Node::UNUSED) && (m_nodes[node.neighbor_prev].used == false)) {
        // Previous (contiguous) free node: Change offset to previous node offset. Sum sizes
        Node& prev_node = m_nodes[node.neighbor_prev];
        offset = prev_node.data_offset;
        size += prev_node.data_size;

        // Remove node from the bin linked list and put it in the freelist
        remove_node_from_bin(node.neighbor_prev);

        FALCOR_ASSERT(prev_node.neighbor_next == node_index);
        node.neighbor_prev = prev_node.neighbor_prev;
    }

    if ((node.neighbor_next != Node::UNUSED) && (m_nodes[node.neighbor_next].used == false)) {
        // Next (contiguous) free node: Offset remains the same. Sum sizes.
        Node& next_node = m_nodes[node.neighbor_next];
        size += next_node.data_size;

        // Remove node from the bin linked list and put it in the freelist
        remove_node_from_bin(node.neighbor_next);

        FALCOR_ASSERT(next_node.neighbor_prev == node_index);
        node.neighbor_next = next_node.neighbor_next;
    }

    uint32_t neighbor_next = node.neighbor_next;
    uint32_t neighbor_prev = node.neighbor_prev;

    // Insert the removed node to freelist
#ifdef DEBUG_VERBOSE
    printf("Putting node %u into freelist[%u] (free)\n", node_index, m_free_offset + 1);
#endif
    m_free_nodes[++m_free_offset] = node_index;

    // Insert the (combined) free node to bin
    uint32_t combined_node_index = insert_node_into_bin(size, offset);

    // Connect neighbors with the new combined node
    if (neighbor_next != Node::UNUSED) {
        m_nodes[combined_node_index].neighbor_next = neighbor_next;
        m_nodes[neighbor_next].neighbor_prev = combined_node_index;
    }
    if (neighbor_prev != Node::UNUSED) {
        m_nodes[combined_node_index].neighbor_prev = neighbor_prev;
        m_nodes[neighbor_prev].neighbor_next = combined_node_index;
    }

    // Track total allocations
    m_current_allocs--;
}

uint32_t OffsetAllocator::insert_node_into_bin(uint32_t size, uint32_t data_offset)
{
    // Round down to bin index to ensure that bin >= alloc
    uint32_t bin_index = SmallFloat::uint_to_float_round_down(size);

    uint32_t top_bin_index = bin_index >> TOP_BINS_INDEX_SHIFT;
    uint32_t leaf_bin_index = bin_index & LEAF_BINS_INDEX_MASK;

    // Bin was empty before?
    if (m_bin_indices[bin_index] == Node::UNUSED) {
        // Set bin mask bits
        m_used_bins[top_bin_index] |= 1 << leaf_bin_index;
        m_used_bins_top |= 1 << top_bin_index;
    }

    // Take a freelist node and insert on top of the bin linked list (next = old top)
    uint32_t top_node_index = m_bin_indices[bin_index];
    uint32_t node_index = m_free_nodes[m_free_offset--];
#ifdef DEBUG_VERBOSE
    printf("Getting node %u from freelist[%u]\n", node_index, m_free_offset + 1);
#endif
    Node& node = m_nodes[node_index];
    node = {};
    node.data_offset = data_offset;
    node.data_size = size;
    node.bin_list_next = top_node_index;
    if (top_node_index != Node::UNUSED)
        m_nodes[top_node_index].bin_list_prev = node_index;
    m_bin_indices[bin_index] = node_index;

    m_free_storage += size;
#ifdef DEBUG_VERBOSE
    printf("Free storage: %u (+%u) (insert_node_into_bin)\n", m_free_storage, size);
#endif

    return node_index;
}

void OffsetAllocator::remove_node_from_bin(uint32_t node_index)
{
    Node& node = m_nodes[node_index];

    if (node.bin_list_prev != Node::UNUSED) {
        // Easy case: We have previous node. Just remove this node from the middle of the list.
        m_nodes[node.bin_list_prev].bin_list_next = node.bin_list_next;
        if (node.bin_list_next != Node::UNUSED)
            m_nodes[node.bin_list_next].bin_list_prev = node.bin_list_prev;
    } else {
        // Hard case: We are the first node in a bin. Find the bin.

        // Round down to bin index to ensure that bin >= alloc
        uint32_t bin_index = SmallFloat::uint_to_float_round_down(node.data_size);

        uint32_t top_bin_index = bin_index >> TOP_BINS_INDEX_SHIFT;
        uint32_t leaf_bin_index = bin_index & LEAF_BINS_INDEX_MASK;

        m_bin_indices[bin_index] = node.bin_list_next;
        if (node.bin_list_next != Node::UNUSED)
            m_nodes[node.bin_list_next].bin_list_prev = Node::UNUSED;

        // Bin empty?
        if (m_bin_indices[bin_index] == Node::UNUSED) {
            // Remove a leaf bin mask bit
            m_used_bins[top_bin_index] &= ~(1 << leaf_bin_index);

            // All leaf bins empty?
            if (m_used_bins[top_bin_index] == 0) {
                // Remove a top bin mask bit
                m_used_bins_top &= ~(1 << top_bin_index);
            }
        }
    }

    // Insert the node to freelist
#ifdef DEBUG_VERBOSE
    printf("Putting node %u into freelist[%u] (remove_node_from_bin)\n", node_index, m_free_offset + 1);
#endif
    m_free_nodes[++m_free_offset] = node_index;

    m_free_storage -= node.data_size;
#ifdef DEBUG_VERBOSE
    printf("Free storage: %u (-%u) (remove_node_from_bin)\n", m_free_storage, node.data_size);
#endif
}

uint32_t OffsetAllocator::allocation_size(Allocation allocation) const
{
    if (allocation.metadata == Allocation::NO_SPACE)
        return 0;
    if (!m_nodes)
        return 0;

    return m_nodes[allocation.metadata].data_size;
}

OffsetAllocator::StorageReport OffsetAllocator::storage_report() const
{
    uint32_t largest_free_region = 0;
    uint32_t free_storage = 0;

    // Out of allocations? -> Zero free space
    if (m_free_offset > 0) {
        free_storage = m_free_storage;
        if (m_used_bins_top) {
            uint32_t top_bin_index = 31 - lzcnt_nonzero(m_used_bins_top);
            uint32_t leaf_bin_index = 31 - lzcnt_nonzero(m_used_bins[top_bin_index]);
            largest_free_region = SmallFloat::float_to_uint((top_bin_index << TOP_BINS_INDEX_SHIFT) | leaf_bin_index);
            FALCOR_ASSERT(free_storage >= largest_free_region);
        }
    }

    StorageReport report;
    report.total_free_space = free_storage;
    report.largest_free_region = largest_free_region;
    return report;
}

OffsetAllocator::StorageReportFull OffsetAllocator::storage_report_full() const
{
    StorageReportFull report;
    for (uint32_t i = 0; i < NUM_LEAF_BINS; i++) {
        uint32_t count = 0;
        uint32_t node_index = m_bin_indices[i];
        while (node_index != Node::UNUSED) {
            node_index = m_nodes[node_index].bin_list_next;
            count++;
        }
        report.free_regions[i] = {SmallFloat::float_to_uint(i), count};
    }
    return report;
}

} // namespace falcor
