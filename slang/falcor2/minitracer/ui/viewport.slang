// SPDX-License-Identifier: Apache-2.0

implementing ui;

import falcor2.minitracer.scene;
import falcor2.utils;

bool is_edge(uint2 pixel, Texture2D<uint> id_texture, uint selected_id)
{
    uint center = id_texture[pixel];
    if (center != selected_id)
        return false;
    uint left = id_texture[pixel - uint2(1, 0)];
    uint right = id_texture[pixel + uint2(1, 0)];
    uint top = id_texture[pixel - uint2(0, 1)];
    uint bottom = id_texture[pixel + uint2(0, 1)];
    return left != center || right != center || top != center || bottom != center;
}

void render_selection(uint2 pixel, Texture2D<uint> id_texture, uint selected_id, inout float4 color)
{
    uint id = id_texture[pixel].x;
    if (id == selected_id) {
        if (is_edge(pixel, id_texture, selected_id))
            color += float4(1, 1, 0, 0) * 0.3;
        color += float4(1, 1, 0, 0) * 0.01;
    }
}

[shader("compute")]
void render_geometry_instance_id(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    uniform uint2 mouse_pos,
    uniform RWTexture2D<uint> id_texture,
    uniform RWStructuredBuffer<uint> id_mouse_pos
)
{
    uint2 dims;
    id_texture.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;
    float2 uv = (tid + 0.5) / dims;
    // TODO: this is because camera flips Y due to writing to buffers instead of textures!
    uv.y = 1 - uv.y;
    Ray ray = camera.get_ray(uv);
    RayQuerySceneIntersector intersector;
    HitInfo hit = intersector.intersect(ray);
    GeometryInstanceID id = GeometryInstanceID::invalid;
    if (hit.valid) {
        id = hit.geometry_instance_id;
    }
    id_texture[tid].x = uint(id);
    if (all(tid == mouse_pos)) {
        id_mouse_pos[0] = uint(id);
    }
}

[shader("compute")]
void render_overlay(
    uint2 tid: SV_DispatchThreadID,
    uniform Camera camera,
    uniform uint selected_id,
    uniform Texture2D<uint> id_texture,
    uniform Texture2D<float4> render_texture,
    uniform RWTexture2D<float4> output_texture,
)
{
    uint2 dims;
    output_texture.GetDimensions(dims.x, dims.y);
    if (any(tid >= dims))
        return;
    float4 color = render_texture[tid];
    if (selected_id != uint(GeometryInstanceID::invalid)) {
        render_selection(tid, id_texture, selected_id, color);
    }
    output_texture[tid] = color;
}
