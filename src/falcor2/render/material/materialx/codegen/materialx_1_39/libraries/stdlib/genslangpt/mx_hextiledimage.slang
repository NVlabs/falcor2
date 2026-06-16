// SPDX-License-Identifier: Apache-2.0

float2 hextile_hash(float2 p)
{
    float3 p3 = frac(float3(p.x, p.y, p.x) * float3(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, float3(p3.y, p3.z, p3.x) + 33.33);
    return frac((float2(p3.x, p3.x) + float2(p3.y, p3.z)) * float2(p3.z, p3.y));
}

float hextile_schlick_gain(float x, float r)
{
    float rr = clamp(r, 0.001, 0.999);
    float a = (1.0 / rr - 2.0) * (1.0 - 2.0 * x);
    return (x < 0.5) ? x / (a + 1.0) : (a - x) / (a - 1.0);
}

float3 hextile_compute_blend_weights(float3 luminance_weights, float3 tile_weights, float falloff)
{
    float3 w = luminance_weights * pow(tile_weights, float3(7.0));
    w /= max(w.x + w.y + w.z, M_FLOAT_EPS);

    if (falloff != 0.5) {
        w.x = hextile_schlick_gain(w.x, falloff);
        w.y = hextile_schlick_gain(w.y, falloff);
        w.z = hextile_schlick_gain(w.z, falloff);
        w /= max(w.x + w.y + w.z, M_FLOAT_EPS);
    }
    return w;
}

float2 hextile_skew(float2 v)
{
    return float2(v.x - 0.57735027 * v.y, 1.15470054 * v.y);
}

float2 hextile_unskew(float2 v)
{
    return float2(v.x + 0.5 * v.y, v.y / 1.15470054);
}

float2 hextile_rotate(float2 v, float sin_r, float cos_r)
{
    return float2(cos_r * v.x - sin_r * v.y, sin_r * v.x + cos_r * v.y);
}

struct HextileData
{
    float2 coord0;
    float2 coord1;
    float2 coord2;
    float3 weights;
    float3 rotations;
}

float3 normals_to_gradient(float3 n, float3 np)
{
    float d = dot(n, np);
    return (d * n - np) / max(M_FLOAT_EPS, abs(d));
}

float3
gradient_blend_3_normals(float3 n, float3 n1, float n1_weight, float3 n2, float n2_weight, float3 n3, float n3_weight)
{
    float w1 = clamp(n1_weight, 0.0, 1.0);
    float w2 = clamp(n2_weight, 0.0, 1.0);
    float w3 = clamp(n3_weight, 0.0, 1.0);

    float3 g1 = normals_to_gradient(n, n1);
    float3 g2 = normals_to_gradient(n, n2);
    float3 g3 = normals_to_gradient(n, n3);
    return normalize(n - (w1 * g1 + w2 * g2 + w3 * g3));
}

float3 rotate_around_axis(float3 v, float3 axis, float angle)
{
    float s = sin(angle);
    float c = cos(angle);
    return v * c + cross(axis, v) * s + axis * dot(axis, v) * (1.0 - c);
}

HextileData hextile_coord(
    float2 coord,
    float rotation,
    float2 rotationrange,
    float scale,
    float2 scalerange,
    float offset,
    float2 offsetrange
)
{
    float sqrt3_2 = sqrt(3.0) * 2.0;
    float2 st_skewed = hextile_skew(coord * sqrt3_2);

    float2 st_frac = frac(st_skewed);
    float3 temp = float3(st_frac.x, st_frac.y, 1.0 - st_frac.x - st_frac.y);

    float s = step(0.0, -temp.z);
    float s2 = 2.0 * s - 1.0;

    float w1 = -temp.z * s2;
    float w2 = s - temp.y * s2;
    float w3 = s - temp.x * s2;

    int2 base_id = int2(floor(st_skewed));
    int si = int(s);
    int2 id1 = base_id + int2(si, si);
    int2 id2 = base_id + int2(si, 1 - si);
    int2 id3 = base_id + int2(1 - si, si);

    float2 ctr1 = hextile_unskew(float2(id1) / sqrt3_2);
    float2 ctr2 = hextile_unskew(float2(id2) / sqrt3_2);
    float2 ctr3 = hextile_unskew(float2(id3) / sqrt3_2);

    float2 seed_offset = float2(0.12345);
    float2 rand1 = hextile_hash(float2(id1) + seed_offset);
    float2 rand2 = hextile_hash(float2(id2) + seed_offset);
    float2 rand3 = hextile_hash(float2(id3) + seed_offset);

    float2 rr = mx_radians(rotationrange);
    float3 rand_x = float3(rand1.x, rand2.x, rand3.x);
    float3 rotations = lerp(float3(rr.x), float3(rr.y), rand_x * rotation);
    float3 sin_r = sin(rotations);
    float3 cos_r = cos(rotations);

    float3 rand_y = float3(rand1.y, rand2.y, rand3.y);
    float3 scales = lerp(float3(1.0), lerp(float3(scalerange.x), float3(scalerange.y), rand_y), scale);

    float2 offset1 = lerp(float2(offsetrange.x), float2(offsetrange.y), rand1 * offset);
    float2 offset2 = lerp(float2(offsetrange.x), float2(offsetrange.y), rand2 * offset);
    float2 offset3 = lerp(float2(offsetrange.x), float2(offsetrange.y), rand3 * offset);

    HextileData tile_data;
    tile_data.weights = float3(w1, w2, w3);
    tile_data.rotations = rotations;
    tile_data.coord0 = hextile_rotate(coord - ctr1, sin_r.x, cos_r.x) / scales.x + ctr1 + offset1;
    tile_data.coord1 = hextile_rotate(coord - ctr2, sin_r.y, cos_r.y) / scales.y + ctr2 + offset2;
    tile_data.coord2 = hextile_rotate(coord - ctr3, sin_r.z, cos_r.z) / scales.z + ctr3 + offset3;
    return tile_data;
}

void mx_hextiledimage_color3(
    TextureHandle file,
    float3 default_value,
    float2 texcoord,
    float2 tiling,
    float rotation,
    float2 rotationrange,
    float scale,
    float2 scalerange,
    float offset,
    float2 offsetrange,
    float falloff,
    float falloffcontrast,
    float3 lumacoeffs,
    out float3 result
)
{
    float2 coord = texcoord * tiling;
    HextileData tile_data = hextile_coord(coord, rotation, rotationrange, scale, scalerange, offset, offsetrange);

    float4 default_sample = float4(default_value, 1.0);
    float3 c1 = sample_texture(file, tile_data.coord0, lod_sampler, default_sample).rgb;
    float3 c2 = sample_texture(file, tile_data.coord1, lod_sampler, default_sample).rgb;
    float3 c3 = sample_texture(file, tile_data.coord2, lod_sampler, default_sample).rgb;

    float3 cw = float3(dot(c1, lumacoeffs), dot(c2, lumacoeffs), dot(c3, lumacoeffs));
    cw = lerp(float3(1.0), cw, float3(falloffcontrast));

    float3 w = hextile_compute_blend_weights(cw, tile_data.weights, falloff);
    result = w.x * c1 + w.y * c2 + w.z * c3;
}

void mx_hextiledimage_color4(
    TextureHandle file,
    float4 default_value,
    float2 texcoord,
    float2 tiling,
    float rotation,
    float2 rotationrange,
    float scale,
    float2 scalerange,
    float offset,
    float2 offsetrange,
    float falloff,
    float falloffcontrast,
    float3 lumacoeffs,
    out float4 result
)
{
    float2 coord = texcoord * tiling;
    HextileData tile_data = hextile_coord(coord, rotation, rotationrange, scale, scalerange, offset, offsetrange);

    float4 c1 = sample_texture(file, tile_data.coord0, lod_sampler, default_value);
    float4 c2 = sample_texture(file, tile_data.coord1, lod_sampler, default_value);
    float4 c3 = sample_texture(file, tile_data.coord2, lod_sampler, default_value);

    float3 cw = float3(dot(c1.rgb, lumacoeffs), dot(c2.rgb, lumacoeffs), dot(c3.rgb, lumacoeffs));
    cw = lerp(float3(1.0), cw, float3(falloffcontrast));

    float3 w = hextile_compute_blend_weights(cw, tile_data.weights, falloff);
    float3 aw = hextile_compute_blend_weights(float3(1.0), tile_data.weights, falloff);
    result = float4(w.x * c1.rgb + w.y * c2.rgb + w.z * c3.rgb, aw.x * c1.a + aw.y * c2.a + aw.z * c3.a);
}

void mx_hextilednormalmap_vector3(
    TextureHandle file,
    float3 default_value,
    float2 texcoord,
    float2 tiling,
    float rotation,
    float2 rotationrange,
    float scale,
    float2 scalerange,
    float offset,
    float2 offsetrange,
    float falloff,
    float strength,
    bool flip_g,
    float3 normal,
    float3 tangent,
    float3 bitangent,
    out float3 result
)
{
    float2 coord = texcoord * tiling;
    HextileData tile_data = hextile_coord(coord, rotation, rotationrange, scale, scalerange, offset, offsetrange);

    float4 default_sample = float4(default_value, 1.0);
    float3 nm1 = sample_texture(file, tile_data.coord0, lod_sampler, default_sample).xyz;
    float3 nm2 = sample_texture(file, tile_data.coord1, lod_sampler, default_sample).xyz;
    float3 nm3 = sample_texture(file, tile_data.coord2, lod_sampler, default_sample).xyz;

    nm1.y = flip_g ? 1.0 - nm1.y : nm1.y;
    nm2.y = flip_g ? 1.0 - nm2.y : nm2.y;
    nm3.y = flip_g ? 1.0 - nm3.y : nm3.y;

    nm1 = 2.0 * nm1 - 1.0;
    nm2 = 2.0 * nm2 - 1.0;
    nm3 = 2.0 * nm3 - 1.0;

    float3 n = normalize(normal);
    float3 t = normalize(tangent);
    float3 b = normalize(bitangent);
    // MaterialX 1.39 GLSL calls mx_axis_rotation_matrix(N, -rotation), but its matrix
    // constructor convention makes that equivalent to a standard column-vector
    // positive-axis rotation here.
    float3 t1 = rotate_around_axis(t, n, tile_data.rotations.x) * strength;
    float3 t2 = rotate_around_axis(t, n, tile_data.rotations.y) * strength;
    float3 t3 = rotate_around_axis(t, n, tile_data.rotations.z) * strength;
    float3 b1 = rotate_around_axis(b, n, tile_data.rotations.x) * strength;
    float3 b2 = rotate_around_axis(b, n, tile_data.rotations.y) * strength;
    float3 b3 = rotate_around_axis(b, n, tile_data.rotations.z) * strength;

    float3 n1 = normalize(t1 * nm1.x + b1 * nm1.y + n * nm1.z);
    float3 n2 = normalize(t2 * nm2.x + b2 * nm2.y + n * nm2.z);
    float3 n3 = normalize(t3 * nm3.x + b3 * nm3.y + n * nm3.z);
    float3 w = hextile_compute_blend_weights(float3(1.0), tile_data.weights, falloff);
    result = gradient_blend_3_normals(n, n1, w.x, n2, w.y, n3, w.z);
}
