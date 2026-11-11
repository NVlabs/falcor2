// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module mdl_material_data;
import falcor2.utils;

public struct MDLMaterialData
{
    // MaterialHeader (16B) is stored just before this struct in memory.

    public BufferHandle data;
    /// Absolute byte offsets into `data` for the three MDL data regions.
    public uint ro_data_offset;
    public uint arg_block_offset;
    public uint texture_table_offset;

    public T read_only_data_load<T>(uint offset) { return data.load_aligned<T, 4>(ro_data_offset + offset); }

    public T arg_block_load<T>(uint offset) { return data.load_aligned<T, 4>(arg_block_offset + offset); }

    public T texture_table_load<T>(uint offset) { return data.load_aligned<T, 4>(texture_table_offset + offset); }

    /// Number of bsdf slots used by the material.
    public uint surface_scatter_bsdf_count = 0;
};
