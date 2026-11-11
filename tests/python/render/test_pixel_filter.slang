// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

public float2x3 sample_filter(IReconstructionFilter pixel_filter, float2 u, float2 offset)
{
    let sample = pixel_filter.sample(u);
    let evaluated_sample = pixel_filter.eval(sample.offset);
    let evaluated_offset = pixel_filter.eval(offset);
    return float2x3(float3(sample.offset, sample.weight), float3(evaluated_sample, evaluated_offset, 0.0));
}
