// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module mdl_material_data;
import falcor2.utils;

public struct MDLMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.

    public BufferHandle data;
    /// Offset into `data` where the argument buffer starts.
    public uint arg_block_offset;
    /// Offset into `data` where the texture table starts.
    public uint texture_table_offset;

    public T read_only_data_load<T>(uint offset) { return data.load_aligned<T>(offset, 4); }

    public T arg_block_load<T>(uint offset) { return data.load_aligned<T>(arg_block_offset + offset, 4); }

    public T texture_table_load<T>(uint offset) { return data.load_aligned<T>(texture_table_offset + offset, 4); }

    /// Number of bsdf slots used by the material.
    public uint surface_scatter_bsdf_count = 0;
};
