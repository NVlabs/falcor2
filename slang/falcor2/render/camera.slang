// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

import falcor2.utils;
import slangpy;

// Camera struct with dims for slangpy dispatch

public struct Camera
{
    public uint2 dims;
    public float3 position;
    public float3 image_u;
    public float3 image_v;
    public float3 image_w;
    public float aperture_radius;
    public float focus_distance;

    private float3 get_unnormalized_ray_direction(float2 uv)
    {
        uv = uv * 2 - 1;
        return uv.x * image_u + uv.y * image_v + image_w;
    }

    public Ray get_ray(float2 uv) { return Ray(position, normalize(get_unnormalized_ray_direction(uv))); }

    /// Generate a ray through a uniformly sampled circular thin lens.
    public Ray get_ray(float2 uv, float2 lens_sample)
    {
        if (aperture_radius <= 0.0 || focus_distance <= 0.0)
            return get_ray(uv);

        float3 unnormalized_dir = get_unnormalized_ray_direction(uv);
        float3 focus_point = position + focus_distance * unnormalized_dir;
        float2 aperture_sample = square_to_uniform_disk_concentric(lens_sample);
        float3 lens_position = position
            + aperture_radius * (aperture_sample.x * normalize(image_u) + aperture_sample.y * normalize(image_v));
        return Ray(lens_position, normalize(focus_point - lens_position));
    }

    public float2 project_world_to_pixel(float3 pos_ws)
    {
        float3 offset_ws = pos_ws - position;
        float depth = dot(offset_ws, image_w);
        float safe_depth = abs(depth) > 1e-6 ? depth : (depth < 0.0 ? -1e-6 : 1e-6);
        float image_u_len_sq = max(dot(image_u, image_u), 1e-8);
        float image_v_len_sq = max(dot(image_v, image_v), 1e-8);
        float ndc_x = dot(offset_ws, image_u) / (safe_depth * image_u_len_sq);
        float ndc_y = dot(offset_ws, image_v) / (safe_depth * image_v_len_sq);
        float2 uv = float2(ndc_x * 0.5 + 0.5, 0.5 - ndc_y * 0.5);
        return uv * float2(dims);
    }

    public float2 get_motion_vector(Camera previous_camera, float3 pos_ws)
    {
        float2 current_pixel = this.project_world_to_pixel(pos_ws);
        float2 previous_pixel = previous_camera.project_world_to_pixel(pos_ws);
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
