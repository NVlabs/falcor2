// SPDX-License-Identifier: Apache-2.0

implementing utils;

public struct Ray : IDifferentiable
{
    public float3 origin = {};
    public float t_min = 0;
    public float3 dir = {};
    public float t_max = 0;

    [Differentiable]
    public __init(float3 origin, float3 dir, float t_min = 0.f, float t_max = 1000.0)
    {
        this.origin = origin;
        this.dir = dir;
        this.t_min = t_min;
        this.t_max = t_max;
    }

    public RayDesc to_ray_desc() { return { origin, t_min, dir, t_max }; }
};
