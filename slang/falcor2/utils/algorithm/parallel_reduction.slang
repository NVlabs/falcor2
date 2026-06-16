// SPDX-License-Identifier: Apache-2.0

/// Parallel reduction shader supporting Sum, Min, and Max operations.
/// Configured via defines:
///   TYPE        - scalar type (int32_t, uint32_t, int64_t, uint64_t, float32_t) for buffer reduction
///   REDUCE_SUM  - enable sum reduction
///   REDUCE_MIN  - enable min reduction
///   REDUCE_MAX  - enable max reduction
///   IS_TEXTURE  - enable texture reduction path (uses FORMAT_TYPE)
///   FORMAT_TYPE - float4/int4/uint4 for texture path

#ifndef TYPE
#error "TYPE must be defined"
#define TYPE uint
#endif

typedef TYPE T;

static const uint GROUP_SIZE = 1024;
static const uint ELEMENTS_PER_THREAD = 8;
static const uint TILE_SIZE = 32;
static const uint MIN_WAVE_SIZE = 32;
static const uint MAX_WAVES = GROUP_SIZE / MIN_WAVE_SIZE;

// ----------------------------------------------------------------------------
// Buffer reduction
// ----------------------------------------------------------------------------

/// Accumulator struct for buffer reduction. Contains only the enabled operation fields.
/// Also used as the element type for intermediate structured buffers.
struct ReduceState
{
#ifdef REDUCE_SUM
    T sum;
#endif
#ifdef REDUCE_MIN
    T min_val;
#endif
#ifdef REDUCE_MAX
    T max_val;
#endif

    static ReduceState identity()
    {
        ReduceState s;
#ifdef REDUCE_SUM
        s.sum = T(0);
#endif
#ifdef REDUCE_MIN
        s.min_val = T.maxValue;
#endif
#ifdef REDUCE_MAX
        s.max_val = T.minValue;
#endif
        return s;
    }

    static ReduceState from_value(T val)
    {
        ReduceState s;
#ifdef REDUCE_SUM
        s.sum = val;
#endif
#ifdef REDUCE_MIN
        s.min_val = val;
#endif
#ifdef REDUCE_MAX
        s.max_val = val;
#endif
        return s;
    }

    [mutating]
    void accumulate(T val)
    {
#ifdef REDUCE_SUM
        sum += val;
#endif
#ifdef REDUCE_MIN
        min_val = min(min_val, val);
#endif
#ifdef REDUCE_MAX
        max_val = max(max_val, val);
#endif
    }

    ReduceState combine(ReduceState other)
    {
        ReduceState s;
#ifdef REDUCE_SUM
        s.sum = sum + other.sum;
#endif
#ifdef REDUCE_MIN
        s.min_val = min(min_val, other.min_val);
#endif
#ifdef REDUCE_MAX
        s.max_val = max(max_val, other.max_val);
#endif
        return s;
    }

    ReduceState wave_reduce()
    {
        ReduceState s;
#ifdef REDUCE_SUM
        s.sum = WaveActiveSum(sum);
#endif
#ifdef REDUCE_MIN
        s.min_val = WaveActiveMin(min_val);
#endif
#ifdef REDUCE_MAX
        s.max_val = WaveActiveMax(max_val);
#endif
        return s;
    }
};

groupshared ReduceState g_cache[MAX_WAVES];

/// Reduce a per-thread state to a single group result stored in g_cache[0].
void reduce_group(uint group_thread_id, ReduceState state)
{
    ReduceState wave_result = state.wave_reduce();
    uint wave_id = group_thread_id / WaveGetLaneCount();

    if (WaveIsFirstLane())
        g_cache[wave_id] = wave_result;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = MAX_WAVES / 2; stride > 0; stride >>= 1) {
        if (group_thread_id < stride)
            g_cache[group_thread_id] = g_cache[group_thread_id].combine(g_cache[group_thread_id + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[WaveSize(32)]
[numthreads(GROUP_SIZE, 1, 1)]
void buffer_reduce(
    uint group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID,
    uniform uint element_count,
    StructuredBuffer<T> data,
    uniform uint data_offset,
    RWStructuredBuffer<ReduceState> output
)
{
    if (group_thread_id < MAX_WAVES)
        g_cache[group_thread_id] = ReduceState::identity();
    GroupMemoryBarrierWithGroupSync();

    uint base_idx = group_id * (GROUP_SIZE * ELEMENTS_PER_THREAD) + group_thread_id;
    ReduceState state = ReduceState::identity();

    if (base_idx < element_count)
        state = ReduceState::from_value(data[data_offset + base_idx]);

    [unroll]
    for (uint i = 1; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = base_idx + i * GROUP_SIZE;
        if (idx < element_count)
            state.accumulate(data[data_offset + idx]);
    }

    reduce_group(group_thread_id, state);

    if (group_thread_id == 0)
        output[group_id] = g_cache[0];
}

[shader("compute")]
[WaveSize(32)]
[numthreads(GROUP_SIZE, 1, 1)]
void final_reduce(
    uint group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID,
    uniform uint element_count,
    StructuredBuffer<ReduceState> input,
    RWStructuredBuffer<ReduceState> output
)
{
    if (group_thread_id < MAX_WAVES)
        g_cache[group_thread_id] = ReduceState::identity();
    GroupMemoryBarrierWithGroupSync();

    uint base_idx = group_id * (GROUP_SIZE * ELEMENTS_PER_THREAD) + group_thread_id;
    ReduceState state = ReduceState::identity();

    if (base_idx < element_count)
        state = input[base_idx];

    [unroll]
    for (uint i = 1; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = base_idx + i * GROUP_SIZE;
        if (idx < element_count)
            state = state.combine(input[idx]);
    }

    reduce_group(group_thread_id, state);

    if (group_thread_id == 0)
        output[group_id] = g_cache[0];
}

// ----------------------------------------------------------------------------
// Texture reduction
// ----------------------------------------------------------------------------

#ifdef IS_TEXTURE

#ifndef FORMAT_TYPE
#error "FORMAT_TYPE must be defined for texture reduction"
#define FORMAT_TYPE float4
#endif

typedef FORMAT_TYPE TexT;

static const uint TILE_THREADS = TILE_SIZE * TILE_SIZE;
static const uint TEX_MAX_WAVES = TILE_THREADS / MIN_WAVE_SIZE;

/// Accumulator struct for texture reduction.
struct TexReduceState
{
#ifdef REDUCE_SUM
    TexT sum;
#endif
#ifdef REDUCE_MIN
    TexT min_val;
#endif
#ifdef REDUCE_MAX
    TexT max_val;
#endif

    static TexReduceState identity()
    {
        TexReduceState s;
#ifdef REDUCE_SUM
        s.sum = TexT(0);
#endif
#ifdef REDUCE_MIN
        s.min_val = TexT(T.maxValue);
#endif
#ifdef REDUCE_MAX
        s.max_val = TexT(T.minValue);
#endif
        return s;
    }

    static TexReduceState from_value(TexT val)
    {
        TexReduceState s;
#ifdef REDUCE_SUM
        s.sum = val;
#endif
#ifdef REDUCE_MIN
        s.min_val = val;
#endif
#ifdef REDUCE_MAX
        s.max_val = val;
#endif
        return s;
    }

    [mutating]
    void accumulate(TexT val)
    {
#ifdef REDUCE_SUM
        sum += val;
#endif
#ifdef REDUCE_MIN
        min_val = min(min_val, val);
#endif
#ifdef REDUCE_MAX
        max_val = max(max_val, val);
#endif
    }

    TexReduceState combine(TexReduceState other)
    {
        TexReduceState s;
#ifdef REDUCE_SUM
        s.sum = sum + other.sum;
#endif
#ifdef REDUCE_MIN
        s.min_val = min(min_val, other.min_val);
#endif
#ifdef REDUCE_MAX
        s.max_val = max(max_val, other.max_val);
#endif
        return s;
    }

    TexReduceState wave_reduce()
    {
        TexReduceState s;
#ifdef REDUCE_SUM
        s.sum = WaveActiveSum(sum);
#endif
#ifdef REDUCE_MIN
        s.min_val = WaveActiveMin(min_val);
#endif
#ifdef REDUCE_MAX
        s.max_val = WaveActiveMax(max_val);
#endif
        return s;
    }
};

groupshared TexReduceState g_tex_cache[TEX_MAX_WAVES];
groupshared uint g_tex_wave_count;

/// Reduce a per-thread state to a single group result stored in g_tex_cache[0].
void reduce_tex_group(uint group_thread_id, TexReduceState state)
{
    TexReduceState wave_result = state.wave_reduce();

    uint my_wave_id = 0;
    if (WaveIsFirstLane()) {
        InterlockedAdd(g_tex_wave_count, 1u, my_wave_id);
        g_tex_cache[my_wave_id] = wave_result;
    }
    GroupMemoryBarrierWithGroupSync();

    uint wave_count = g_tex_wave_count;
    for (uint stride = TEX_MAX_WAVES / 2; stride > 0; stride >>= 1) {
        if (group_thread_id < stride && group_thread_id + stride < wave_count)
            g_tex_cache[group_thread_id] = g_tex_cache[group_thread_id].combine(g_tex_cache[group_thread_id + stride]);
        GroupMemoryBarrierWithGroupSync();
    }
}

[shader("compute")]
[WaveSize(32)]
[numthreads(TILE_SIZE, TILE_SIZE, 1)]
void texture_reduce(
    uint3 group_id: SV_GroupID,
    uint3 group_thread_id_3d: SV_GroupThreadID,
    uint group_thread_index: SV_GroupIndex,
    uniform uint tex_width,
    uniform uint tex_height,
    Texture2D<FORMAT_TYPE> input_texture,
    RWStructuredBuffer<TexReduceState> output,
    uniform uint tiles_x
)
{
    if (group_thread_index == 0)
        g_tex_wave_count = 0;
    if (group_thread_index < TEX_MAX_WAVES)
        g_tex_cache[group_thread_index] = TexReduceState::identity();
    GroupMemoryBarrierWithGroupSync();

    uint2 pixel = group_id.xy * TILE_SIZE + group_thread_id_3d.xy;
    bool valid = (pixel.x < tex_width) && (pixel.y < tex_height);

    TexReduceState state = TexReduceState::identity();
    if (valid)
        state = TexReduceState::from_value(input_texture[pixel]);

    reduce_tex_group(group_thread_index, state);

    if (group_thread_index == 0) {
        uint tile_id = group_id.y * tiles_x + group_id.x;
        output[tile_id] = g_tex_cache[0];
    }
}

[shader("compute")]
[WaveSize(32)]
[numthreads(GROUP_SIZE, 1, 1)]
void texture_final_reduce(
    uint group_id: SV_GroupID,
    uint group_thread_id: SV_GroupThreadID,
    uniform uint element_count,
    StructuredBuffer<TexReduceState> input,
    RWStructuredBuffer<TexReduceState> output
)
{
    if (group_thread_id == 0)
        g_tex_wave_count = 0;
    if (group_thread_id < TEX_MAX_WAVES)
        g_tex_cache[group_thread_id] = TexReduceState::identity();
    GroupMemoryBarrierWithGroupSync();

    uint base_idx = group_id * (GROUP_SIZE * ELEMENTS_PER_THREAD) + group_thread_id;
    TexReduceState state = TexReduceState::identity();

    if (base_idx < element_count)
        state = input[base_idx];

    [unroll]
    for (uint i = 1; i < ELEMENTS_PER_THREAD; i++) {
        uint idx = base_idx + i * GROUP_SIZE;
        if (idx < element_count)
            state = state.combine(input[idx]);
    }

    reduce_tex_group(group_thread_id, state);

    if (group_thread_id == 0)
        output[group_id] = g_tex_cache[0];
}

#endif // IS_TEXTURE
