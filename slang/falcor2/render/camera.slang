// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import slangpy;

// Camera struct with dims for slangpy dispatch

public struct Camera
{
    public uint2 dims;
    public float3 position;
    public float3 image_u;
    public float3 image_v;
    public float3 image_w;

    public Ray get_ray(float2 uv)
    {
        uv = uv * 2 - 1;
        float3 dir = normalize(uv.x * image_u + uv.y * image_v + image_w);
        return Ray(position, dir);
    }

    public float2 project_world_to_pixel(float3 position_ws)
    {
        float3 offset_ws = position_ws - position;
        float depth = dot(offset_ws, image_w);
        float safe_depth = abs(depth) > 1e-6 ? depth : (depth < 0.0 ? -1e-6 : 1e-6);
        float image_u_len_sq = max(dot(image_u, image_u), 1e-8);
        float image_v_len_sq = max(dot(image_v, image_v), 1e-8);
        float ndc_x = dot(offset_ws, image_u) / (safe_depth * image_u_len_sq);
        float ndc_y = dot(offset_ws, image_v) / (safe_depth * image_v_len_sq);
        float2 uv = float2(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
        return uv * float2(dims);
    }

    public float2 get_motion_vector(Camera previous_camera, float3 position_ws)
    {
        float2 current_pixel = this.project_world_to_pixel(position_ws);
        float2 previous_pixel = previous_camera.project_world_to_pixel(position_ws);
        return previous_pixel - current_pixel;
    }
};

// SlangPy load extensions for Camera

public extension Camera
{
    public void __slangpy_load(Context0D context, out Camera value)
    {
        value = this;
    }
    public void __slangpy_store(Context0D context, in Camera value) {};

    public void __slangpy_load(Context2D context, out Ray value)
    {
        float2 uv = (float2(context.call_id_vec) + 0.5) / float2(this.dims);
        uv.y = 1 - uv.y;
        value = get_ray(uv);
    }
    public void __slangpy_store(Context2D context, in Ray value) {};

    public void __slangpy_load(Context2D context, out RaySampler value)
    {
        value.camera = this;
        value.id = context.call_id_vec;
    }
    public void __slangpy_store(Context2D context, in RaySampler value) {};
}

// RaySampler: camera + pixel id
public struct RaySampler
{
    public Camera camera;
    public uint2 id;
};
