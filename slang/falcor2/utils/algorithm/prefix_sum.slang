// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Parallel prefix sum computed in place using exclusive scan.
///
/// The implementation is based on G. Blelloch, "Vector Models for Data-Parallel Computing", MIT Press, 1990.
/// See CUDA code: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch39.html
/// See also: http://www.umiacs.umd.edu/~ramani/cmsc828e_gpusci/ScanTalk.pdf

#ifndef TYPE
#error "TYPE must be defined"
#define TYPE uint
#endif

#ifndef USE_FLOAT_BYTE_ADDRESS_BUFFER
#error "USE_FLOAT_BYTE_ADDRESS_BUFFER must be defined"
#define USE_FLOAT_BYTE_ADDRESS_BUFFER 0
#endif

/// Type used in the data buffer.
typedef TYPE T;

// D3D12 doesn't support atomics on RWStructuredBuffer<float>.
#if USE_FLOAT_BYTE_ADDRESS_BUFFER

typedef ByteAddressBuffer Buffer;
typedef RWByteAddressBuffer RWBuffer;

#define LOAD(buf, index) buf.Load<T>((index) * 4)
#define STORE(buf, index, value) buf.Store<T>((index) * 4, value)
#define INTERLOCKED_ADD(buf, index, value) buf.InterlockedAddF32((index) * 4, value)

#else // USE_FLOAT_BYTE_ADDRESS_BUFFER

typedef StructuredBuffer<T> Buffer;
typedef RWStructuredBuffer<T> RWBuffer;

#define LOAD(buf, index) buf[index]
#define STORE(buf, index, value) buf[index] = value
#define INTERLOCKED_ADD(buf, index, value) InterlockedAdd(buf[index], value)

#endif // USE_FLOAT_BYTE_ADDRESS_BUFFER

/// Thread group size, must be a power-of-two <= 1024.
/// Must be kept in sync with host side.
static const uint GROUP_SIZE = 1024;

/// Temporary working buffer in shared memory for 2N elements.
groupshared T shared_data[2 * GROUP_SIZE];

/// Parallel prefix sum in shared memory over consecutive groups of 2N elements,
/// where N is the thread group size.
/// This shader reads from data and writes one element per group to prefix_group_sums.
/// If there are more elements than 2 * GROUP_SIZE * GROUP_SIZE, the prefix sum will be split
/// into multiple iterations.
/// @param group_id Group index.
/// @param group_thread_id Local thread ID in the range 0..N-1.
/// @param group_count Group count, each group is 2N elements.
/// @param total_element_count Total element count. This does not have to be a multiple of the group size.
/// @param iteration Current iteration.
/// @param data Data buffer.
/// @param data_offset Data offset (number of elements).
/// @param prefix_group_sums One element per group, each holds the sum of all elements in the group and to the left.
/// @param total_sum One element, holds the total sum of a prefix sum iteration.
/// @param prev_total_sum One element, holds the previous total sum of the previous prefix sum iteration.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void group_scan(
    uint group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID,
    uniform uint group_count,
    uniform uint total_element_count,
    uniform uint iteration,
    RWBuffer data,
    uniform uint data_offset,
    RWBuffer prefix_group_sums,
    RWBuffer total_sum,
    Buffer prev_total_sum
)
{
    // Load data for group into shared memory. Each thread loads two elements.
    // Interleaved load at consecutive addresses can lead to 2x bank conflicts.
    // It's probably better to load one element into each half of the array as we do here.
    // We pad the data with zeros in shared memory if actual number of elements is less than working set.
    const T prev_sum = LOAD(prev_total_sum, 0);
    const uint idx = 2 * GROUP_SIZE * GROUP_SIZE * iteration + group_id * (2 * GROUP_SIZE) + group_thread_id;
    shared_data[group_thread_id] = (idx < total_element_count ? LOAD(data, data_offset + idx) : T(0));
    shared_data[group_thread_id + GROUP_SIZE]
        = ((idx + GROUP_SIZE) < total_element_count ? LOAD(data, data_offset + idx + GROUP_SIZE) : T(0));

    // Reducation phase.
    // We do log2(N)+1 iterations for d = 2^(N), 2^(N-1), .., 2, 1.
    uint offset = 1;
    for (uint d = GROUP_SIZE; d > 0; d >>= 1) {
        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < d) {
            uint ai = offset * (2 * group_thread_id + 1) - 1;
            uint bi = ai + offset;
            shared_data[bi] += shared_data[ai];
        }
        offset *= 2; // offset = 1, 2, ... N
    }

    GroupMemoryBarrierWithGroupSync();

    // Compute prefix sum over groups.
    // Since groups run out-of-order, we use atomics to add our group's sum to all relevent group sums.
    // This can get slow for large inputs, but for moderate sized inputs (tens to hundreds of groups) it's probably
    // still very fast. The alternative is to run an extra shader pass computing the prefix sum over the groups.
    if (group_thread_id >= group_id && group_thread_id < group_count) {
        T sum = shared_data[2 * GROUP_SIZE - 1];
        INTERLOCKED_ADD(prefix_group_sums, group_thread_id, sum);

        // One thread per group also adds the group sum to the total sum.
        if (group_thread_id == group_count - 1) {
            INTERLOCKED_ADD(total_sum, 0, sum);
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Zero out top element, this is required for down-sweep phase to work correctly.
    // Only one thread in each group does this.
    if (group_thread_id == 0)
        shared_data[2 * GROUP_SIZE - 1] = T(0);

    // Down-sweep phase.
    // We do log2(N)+1 iterations for d = 1, 2, 4, ..., N.
    for (uint d = 1; d <= GROUP_SIZE; d *= 2) {
        offset >>= 1; // offset = N, N/2, ..., 1

        GroupMemoryBarrierWithGroupSync();

        if (group_thread_id < d) {
            uint ai = offset * (2 * group_thread_id + 1) - 1;
            uint bi = ai + offset;
            T tmp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += tmp;
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // Write results to memory and add prev sum. Lower half first then upper half.
    if (idx < total_element_count)
        STORE(data, data_offset + idx, shared_data[group_thread_id] + prev_sum);
    if ((idx + GROUP_SIZE) < total_element_count)
        STORE(data, data_offset + idx + GROUP_SIZE, shared_data[group_thread_id + GROUP_SIZE] + prev_sum);
}

/// Pass for finalizing a prefix sum computed over multiple thread groups.
/// Each thread here operates on one element of the data buffer.
/// Note that we're skipping the first N elements as those don't need to be added
/// (their group's prefix sum is zero).
/// @param group_id Group index.
/// @param group_thread_id Local thread ID in the range 0..N-1.
/// @param total_element_count Total element count. This does not have to be a multiple of the group size.
/// @param iteration Current iteration.
/// @param data Data buffer.
/// @param data_offset Data offset (number of elements).
/// @param prefix_group_sums One element per group, each holds the sum of all elements in the group and to the left.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void finalize(
    uint group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID,
    uniform uint total_element_count,
    uniform uint iteration,
    RWBuffer data,
    uniform uint data_offset,
    Buffer prefix_group_sums,
)
{
    T sum = LOAD(prefix_group_sums, group_id >> 1);
    // Skip first 2N elements.
    uint idx = (group_id * GROUP_SIZE) + group_thread_id + 2 * GROUP_SIZE * (1 + GROUP_SIZE * iteration);
    if (idx < total_element_count)
        STORE(data, data_offset + idx, LOAD(data, data_offset + idx) + sum);
}
