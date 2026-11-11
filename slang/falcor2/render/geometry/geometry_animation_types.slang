// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/shared_types_compat.slangh"
#ifndef FALCOR_SHARED_HOST
implementing render;
#endif

public enum class GeometryHistoryUpdate : uint {
    none,            ///< No work to dispatch.
    initialize,      ///< Evaluate current and copy the result to previous.
    capture_current, ///< Copy old current to previous, then evaluate new current.
    catch_up,        ///< Copy current to previous without evaluating again.
};

/// One output range. Pointers address its first vertex in each source/output buffer.
public struct GeometryAnimationWork
{
    public BufferPtr source0;
    public BufferPtr source1;
    public RWBufferPtr current;
    public RWBufferPtr previous;
    public uint vertex_count;
    public float source1_weight;
    public GeometryHistoryUpdate history_update;
    public uint reserved;
};

#include "falcor2/shared_types_compat_undef.slangh"
