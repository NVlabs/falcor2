// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Luminance of an RGB in Rec.709.
[Differentiable]
public float luminance(float3 rgb)
{
    return dot(rgb, float3(0.2126, 0.7152, 0.0722));
}

/// Transform RGB in Rec.709 to CIE XYZ.
[Differentiable]
public float3 rgb_to_xyz_rec709(float3 c)
{
    static const float3x3 M = {
        // clang-format off
        0.4123907992659595, 0.3575843393838780, 0.1804807884018343, // row 0
        0.2126390058715104, 0.7151686787677559, 0.0721923153607337, // row 1
        0.0193308187155918, 0.1191947797946259, 0.9505321522496608, // row 2
        // clang-format off
    };
    return mul(M, c);
}

/// Transform CIE XYZ to RGB in Rec.709.
[Differentiable]
public float3 xyz_to_rgb_rec709(float3 c)
{
    static const float3x3 M = {
        // clang-format off
        3.240969941904522,   -1.537383177570094,  -0.4986107602930032, // row 0
        -0.9692436362808803, 1.875967501507721,   0.04155505740717569, // row 1
        0.05563007969699373, -0.2039769588889765, 1.056971514242878,   // row 2
        // clang-format on
    };
    return mul(M, c);
}

/// Transform RGB to YCgCo space.
[Differentiable]
public float3 rgb_to_ycgco(float3 rgb)
{
    float Y = dot(rgb, float3(0.25, 0.50, 0.25));
    float Cg = dot(rgb, float3(-0.25, 0.50, -0.25));
    float Co = dot(rgb, float3(0.50, 0.00, -0.50));
    return float3(Y, Cg, Co);
}

/// Transform YCgCo to RGB space.
public float3 ycgco_to_rgb(float3 ycgco)
{
    float tmp = ycgco.x - ycgco.y;
    float r = tmp + ycgco.z;
    float g = ycgco.x + ycgco.y;
    float b = tmp - ycgco.z;
    return float3(r, g, b);
}

/// Transform RGB to YUV (Rec.709).
[Differentiable]
public float3 rgb_to_yuv(float3 rgb)
{
    float3 ret;
    ret.x = dot(rgb, float3(0.2126, 0.7152, 0.0722));
    ret.y = dot(rgb, float3(-0.09991, -0.33609, 0.436));
    ret.z = dot(rgb, float3(0.615, -0.55861, -0.05639));
    return ret;
}

/// Transform YUV to RGB (Rec.709).
[Differentiable]
public float3 yuv_to_rgb(float3 yuv)
{
    float3 ret;
    ret.x = dot(yuv, float3(1.0, 0.0, 1.28033));
    ret.y = dot(yuv, float3(1.0, -0.21482, -0.38059));
    ret.z = dot(yuv, float3(1.0, 2.12798, 0.0));
    return ret;
}

/// Transform sRGB to linear RGB.
[Differentiable]
public float srgb_to_linear(float srgb)
{
    if (srgb <= 0.04045)
        return srgb * (1.0 / 12.92);
    else
        return pow((srgb + 0.055) * (1.0 / 1.055), 2.4);
}

/// Transform sRGB to linear RGB.
[Differentiable]
public float3 srgb_to_linear(float3 srgb)
{
    return float3(srgb_to_linear(srgb.x), srgb_to_linear(srgb.y), srgb_to_linear(srgb.z));
}

/// Transform linear RGB to sRGB.
[Differentiable]
public float linear_to_srgb(float linear)
{
    if (linear <= 0.0031308)
        return linear * 12.92;
    else
        return pow(linear, (1.0 / 2.4)) * (1.055) - 0.055;
}

/// Transform linear RGB to sRGB.
[Differentiable]
public float3 linear_to_srgb(float3 linear)
{
    return float3(linear_to_srgb(linear.x), linear_to_srgb(linear.y), linear_to_srgb(linear.z));
}


public static const float3 D65_REFERENCE_ILLUMINANT = float3(0.950428545, 1.000000000, 1.088900371);
public static const float3 D65_REFERENCE_ILLUMINANT_INV = float3(1.052156925, 1.000000000, 0.918357670);


/// Transform linear RGB to XYZ space (D65 illuminant).
[Differentiable]
public float3 linear_rgb_to_xyz(float3 rgb)
{
    // Source:
    // https://www.image-engineering.de/library/technotes/958-how-to-convert-between-srgb-and-ciexyz
    // Assumes D65 standard illuminant.
    const float a11 = 10135552.0 / 24577794.0;
    const float a12 = 8788810.0 / 24577794.0;
    const float a13 = 4435075.0 / 24577794.0;
    const float a21 = 2613072.0 / 12288897.0;
    const float a22 = 8788810.0 / 12288897.0;
    const float a23 = 887015.0 / 12288897.0;
    const float a31 = 1425312.0 / 73733382.0;
    const float a32 = 8788810.0 / 73733382.0;
    const float a33 = 70074185.0 / 73733382.0;

    float3 xyz;
    xyz.x = a11 * rgb.r + a12 * rgb.g + a13 * rgb.b;
    xyz.y = a21 * rgb.r + a22 * rgb.g + a23 * rgb.b;
    xyz.z = a31 * rgb.r + a32 * rgb.g + a33 * rgb.b;

    return xyz;
}

/// Transform XYZ to linear RGB space (D65 illuminant).
[Differentiable]
public float3 xyz_to_linear_rgb(float3 xyz)
{
    // Return values in linear RGB, assuming D65 standard illuminant.
    const float a11 = 3.241003275;
    const float a12 = -1.537398934;
    const float a13 = -0.498615861;
    const float a21 = -0.969224334;
    const float a22 = 1.875930071;
    const float a23 = 0.041554224;
    const float a31 = 0.055639423;
    const float a32 = -0.204011202;
    const float a33 = 1.057148933;

    float3 rgb;
    rgb.r = a11 * xyz.x + a12 * xyz.y + a13 * xyz.z;
    rgb.g = a21 * xyz.x + a22 * xyz.y + a23 * xyz.z;
    rgb.b = a31 * xyz.x + a32 * xyz.y + a33 * xyz.z;

    return rgb;
}

[Differentiable]
float actual_pow(float x, float y)
{
    if (x >= 0.0)
        return pow(x, y);
    else
        return -pow(-x, y);
}

[Differentiable]
float3 actual_pow(float3 in, float y)
{
    return float3(actual_pow(in.x, y), actual_pow(in.y, y), actual_pow(in.z, y));
}

[Differentiable]
public float3 xyz_to_oklab(float3 xyz)
{
    static const float3x3 M1 = {
        // clang-format off
        0.8189330101, 0.3618667424, -0.1288597137, // row 0
        0.0329845436, 0.9293118715, 0.0361456387,  // row 1
        0.0482003018, 0.2643662691, 0.6338517070   // row 2
        // clang-format on
    };
    float3 lms = mul(M1, xyz);
    lms = actual_pow(lms, 1.0 / 3.0);
    static const float3x3 M2 = {
        // clang-format off
        0.2104542553, 0.7936177850, -0.0040720468, // row 0
        1.9779984951, -2.4285922050, 0.4505937099, // row 1
        0.0259040371, 0.7827717662, -0.8086757660  // row 2
        // clang-format on
    };
    return mul(M2, lms);
}

[Differentiable]
public float3 oklab_to_xyz(float3 oklab)
{
    static const float3x3 M2_INV = float3x3(
        // clang-format off
        1.00000000, 0.39633779,  0.21580376,  // row 0
        1.00000001, -0.10556134, -0.06385417, // row 1
        1.00000005, -0.08948418, -1.29148554  // row 2
        // clang-format off
    );
    float3 lms = mul(M2_INV, oklab);
    lms = lms * lms * lms;
    static const float3x3 M1_INV = float3x3(
        // clang-format off
        1.22701385,  -0.55779998, 0.28125615,  // row 0
        -0.04058018, 1.11225687,  -0.07167668, // row 1
        -0.07638128, -0.42148198, 1.58616322   // row 2
        // clang-format on
    );
    return mul(M1_INV, lms);
}

[Differentiable]
public float3 oklab_to_hcl(float3 oklab)
{
    float L = oklab.x;
    float a = oklab.y;
    float b = oklab.z;

    float H = atan2(b, a);
    float C = sqrt(a * a + b * b);
    return float3(H, C, L);
}

[Differentiable]
public float3 hcl_to_oklab(float3 hcl)
{
    float H = hcl.x;
    float C = hcl.y;
    float L = hcl.z;

    float a = C * cos(H);
    float b = C * sin(H);
    return float3(L, a, b);
}

[Differentiable]
public float3 rgb_to_oklab(float3 rgb)
{
    return xyz_to_oklab(linear_rgb_to_xyz(rgb));
}

[Differentiable]
public float3 oklab_to_rgb(float3 oklab)
{
    return xyz_to_linear_rgb(oklab_to_xyz(oklab));
}

[Differentiable]
public float3 xyz_to_cie_lab(float3 xyz, const float3 inv_reference_illuminant = D65_REFERENCE_ILLUMINANT_INV)
{
    // The default illuminant is D65.
    float3 tmp = xyz * inv_reference_illuminant;

    float delta = 6.0 / 29.0;
    float delta2 = delta * delta;
    float delta3 = delta * delta2;
    float factor = 1.0 / (3.0 * delta2);
    float term = 4.0 / 29.0;

    tmp.r = (tmp.r > delta3 ? pow(tmp.r, 1.0 / 3.0) : factor * tmp.r + term);
    tmp.g = (tmp.g > delta3 ? pow(tmp.g, 1.0 / 3.0) : factor * tmp.g + term);
    tmp.b = (tmp.b > delta3 ? pow(tmp.b, 1.0 / 3.0) : factor * tmp.b + term);

    float3 cie_lab;
    cie_lab.r = 116.0f * tmp.g - 16.0f;
    cie_lab.g = 500.0f * (tmp.r - tmp.g);
    cie_lab.b = 200.0f * (tmp.g - tmp.b);

    return cie_lab;
}

[Differentiable]
public float3 cie_lab_to_xyz(float3 cie_lab, const float3 reference_illuminant = D65_REFERENCE_ILLUMINANT)
{
    // The default illuminant is D65.
    float Y = (cie_lab.r + 16.0) / 116.0;
    float X = cie_lab.g / 500.0 + Y;
    float Z = Y - cie_lab.b / 200.0;

    float delta = 6.0 / 29.0;
    float factor = 3.0 * delta * delta;
    float term = 4.0 / 29.0;
    X = ((X > delta) ? X * X * X : (X - term) * factor);
    Y = ((Y > delta) ? Y * Y * Y : (Y - term) * factor);
    Z = ((Z > delta) ? Z * Z * Z : (Z - term) * factor);

    return float3(X, Y, Z) * reference_illuminant;
}

[Differentiable]
public float3 xyz_to_ycxcz(float3 xyz, const float3 inv_reference_illuminant = D65_REFERENCE_ILLUMINANT_INV)
{
    // The default illuminant is D65.
    float3 tmp = xyz * inv_reference_illuminant;

    float3 ycxcz;
    ycxcz.x = 116.0 * tmp.g - 16.0;
    ycxcz.y = 500.0 * (tmp.r - tmp.g);
    ycxcz.z = 200.0 * (tmp.g - tmp.b);

    return ycxcz;
}

[Differentiable]
public float3 ycxcz_to_xyz(float3 ycxcz, const float3 reference_illuminant = D65_REFERENCE_ILLUMINANT)
{
    // The default illuminant is D65.
    float Y = (ycxcz.r + 16.0) / 116.0;
    float X = ycxcz.g / 500.0 + Y;
    float Z = Y - ycxcz.b / 200.0;

    return float3(X, Y, Z) * reference_illuminant;
}

[Differentiable]
public float3 linear_rgb_to_cie_lab(float3 rgb)
{
    return xyz_to_cie_lab(linear_rgb_to_xyz(rgb));
}

[Differentiable]
public float3 ycxcz_to_linear_rgb(float3 ycxcz)

{
    return xyz_to_linear_rgb(ycxcz_to_xyz(ycxcz));
}

[Differentiable]
public float3 linear_rgb_to_ycxcz(float3 rgb)
{
    return xyz_to_ycxcz(linear_rgb_to_xyz(rgb));
}


/// Evaluates the 1931 CIE XYZ color matching curves.
/// This uses a multi-lobe piecewise Gaussian fit described in:
/// Wyman et al., "Simple Analytic Approximations to the CIE XYZ Color
/// Matching Functions", JCGT, vol 2(2), 2013.
/// @param lambda Wavelength in nm.
/// @return XYZ tristimulus values.
public float3 wavelength_to_xyz(float lambda)
{
    float x = 0.362 * G(lambda, 442.0, 0.0624, 0.0374) + 1.056 * G(lambda, 599.8, 0.0264, 0.0323)
        - 0.065 * G(lambda, 501.1, 0.0490, 0.0382);
    float y = 0.821 * G(lambda, 568.8, 0.0213, 0.0247) + 0.286 * G(lambda, 530.9, 0.0613, 0.0322);
    float z = 1.217 * G(lambda, 437.0, 0.0845, 0.0278) + 0.681 * G(lambda, 459.0, 0.0385, 0.0725);
    return float3(x, y, z);
}

float G(float x, float lambda, float inv_sigma_1, float inv_sigma_2)
{
    float t = (x - lambda) * (x < lambda ? inv_sigma_1 : inv_sigma_2);
    return exp(-0.5 * t * t);
}
