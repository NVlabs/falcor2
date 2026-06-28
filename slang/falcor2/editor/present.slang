// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing editor;
import slangpy;

public void blit_tensor_to_texture(
    int2 pixel,
    int2 size,
    int2 offset,
    bool bilinear,
    Tensor<float4, 2> input,
    RWTexture2D<float4> output
)
{
    float2 uv = (float2(pixel) + 0.5) / float2(size);
    uint[2] shape = input.shape;
    uint2 sample_pos
        = uint2(min(uint(shape[1] - 1), (uint)(shape[1] * uv.x)), min(uint(shape[0] - 1), (uint)(shape[0] * uv.y)));

    float4 value = bilinear ? input.sample(float2(uv.x, uv.y)) : input.load(sample_pos.y, sample_pos.x);

    output[offset + pixel] = value;
}
