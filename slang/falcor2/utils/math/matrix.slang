// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public extension<T : __BuiltinFloatingPointType, let N : int> matrix<T, N, N>
{
    // Create an identity matrix.
    public static matrix<T, N, N> identity()
    {
        matrix<T, N, N> m = matrix<T, N, N>((T)0);
        [ForceUnroll]
        for (int i = 0; i < N; i++)
            m[i][i] = (T)1;
        return m;
    }
}

/// Invert a 2x2 matrix.
public matrix<T, 2, 2> inverse<T>(matrix<T, 2, 2> m)
    where T : IFloat, __BuiltinFloatingPointType
{
    matrix<T, 2, 2> inv;

    T inv_det = T(1) / determinant(m);
    inv[0][0] = m[1][1] * inv_det;
    inv[1][1] = m[0][0] * inv_det;
    inv[0][1] = -m[0][1] * inv_det;
    inv[1][0] = -m[1][0] * inv_det;
    return inv;
}

/// Invert a 2x3 matrix.
public matrix<T, 2, 3> inverse<T>(matrix<T, 2, 3> m)
    where T : IFloat, __BuiltinFloatingPointType
{
    matrix<T, 2, 2> n = matrix<T, 2, 2>(m._m00, m._m01, m._m10, m._m11);
    matrix<T, 2, 2> inv_n = inverse(n);
    vector<T, 2> t = -mul(inv_n, vector<T, 2>(m._m02, m._m12));
    return matrix<T, 2, 3>(inv_n._m00, inv_n._m01, t.x, inv_n._m10, inv_n._m11, t.y);
}

/// Invert a 3x3 matrix.
public matrix<T, 3, 3> inverse<T>(matrix<T, 3, 3> m)
    where T : IFloat, __BuiltinFloatingPointType
{
    matrix<T, 3, 3> inv;
    T inv_det = T(1) / determinant(m);
    inv[0][0] = (m[1][1] * m[2][2] - m[2][1] * m[1][2]) * inv_det;
    inv[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det;
    inv[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det;
    inv[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det;
    inv[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det;
    inv[1][2] = (m[1][0] * m[0][2] - m[0][0] * m[1][2]) * inv_det;
    inv[2][0] = (m[1][0] * m[2][1] - m[2][0] * m[1][1]) * inv_det;
    inv[2][1] = (m[2][0] * m[0][1] - m[0][0] * m[2][1]) * inv_det;
    inv[2][2] = (m[0][0] * m[1][1] - m[1][0] * m[0][1]) * inv_det;
    return inv;
}

/// Invert a 4x4 matrix.
public matrix<T, 4, 4> inverse<T>(matrix<T, 4, 4> m)
    where T : IFloat, __BuiltinFloatingPointType
{
    T n11 = m[0][0], n12 = m[1][0], n13 = m[2][0], n14 = m[3][0];
    T n21 = m[0][1], n22 = m[1][1], n23 = m[2][1], n24 = m[3][1];
    T n31 = m[0][2], n32 = m[1][2], n33 = m[2][2], n34 = m[3][2];
    T n41 = m[0][3], n42 = m[1][3], n43 = m[2][3], n44 = m[3][3];

    T t11 = n23 * n34 * n42 - n24 * n33 * n42 + n24 * n32 * n43 - n22 * n34 * n43 - n23 * n32 * n44 + n22 * n33 * n44;
    T t12 = n14 * n33 * n42 - n13 * n34 * n42 - n14 * n32 * n43 + n12 * n34 * n43 + n13 * n32 * n44 - n12 * n33 * n44;
    T t13 = n13 * n24 * n42 - n14 * n23 * n42 + n14 * n22 * n43 - n12 * n24 * n43 - n13 * n22 * n44 + n12 * n23 * n44;
    T t14 = n14 * n23 * n32 - n13 * n24 * n32 - n14 * n22 * n33 + n12 * n24 * n33 + n13 * n22 * n34 - n12 * n23 * n34;

    T det = n11 * t11 + n21 * t12 + n31 * t13 + n41 * t14;
    T inv_det = T(1) / det;

    matrix<T, 4, 4> inv;

    // clang-format off
    inv[0][0] = t11 * inv_det;
    inv[0][1] = (n24 * n33 * n41 - n23 * n34 * n41 - n24 * n31 * n43 + n21 * n34 * n43 + n23 * n31 * n44 - n21 * n33 * n44) * inv_det;
    inv[0][2] = (n22 * n34 * n41 - n24 * n32 * n41 + n24 * n31 * n42 - n21 * n34 * n42 - n22 * n31 * n44 + n21 * n32 * n44) * inv_det;
    inv[0][3] = (n23 * n32 * n41 - n22 * n33 * n41 - n23 * n31 * n42 + n21 * n33 * n42 + n22 * n31 * n43 - n21 * n32 * n43) * inv_det;

    inv[1][0] = t12 * inv_det;
    inv[1][1] = (n13 * n34 * n41 - n14 * n33 * n41 + n14 * n31 * n43 - n11 * n34 * n43 - n13 * n31 * n44 + n11 * n33 * n44) * inv_det;
    inv[1][2] = (n14 * n32 * n41 - n12 * n34 * n41 - n14 * n31 * n42 + n11 * n34 * n42 + n12 * n31 * n44 - n11 * n32 * n44) * inv_det;
    inv[1][3] = (n12 * n33 * n41 - n13 * n32 * n41 + n13 * n31 * n42 - n11 * n33 * n42 - n12 * n31 * n43 + n11 * n32 * n43) * inv_det;

    inv[2][0] = t13 * inv_det;
    inv[2][1] = (n14 * n23 * n41 - n13 * n24 * n41 - n14 * n21 * n43 + n11 * n24 * n43 + n13 * n21 * n44 - n11 * n23 * n44) * inv_det;
    inv[2][2] = (n12 * n24 * n41 - n14 * n22 * n41 + n14 * n21 * n42 - n11 * n24 * n42 - n12 * n21 * n44 + n11 * n22 * n44) * inv_det;
    inv[2][3] = (n13 * n22 * n41 - n12 * n23 * n41 - n13 * n21 * n42 + n11 * n23 * n42 + n12 * n21 * n43 - n11 * n22 * n43) * inv_det;

    inv[3][0] = t14 * inv_det;
    inv[3][1] = (n13 * n24 * n31 - n14 * n23 * n31 + n14 * n21 * n33 - n11 * n24 * n33 - n13 * n21 * n34 + n11 * n23 * n34) * inv_det;
    inv[3][2] = (n14 * n22 * n31 - n12 * n24 * n31 - n14 * n21 * n32 + n11 * n24 * n32 + n12 * n21 * n34 - n11 * n22 * n34) * inv_det;
    inv[3][3] = (n12 * n23 * n31 - n13 * n22 * n31 + n13 * n21 * n32 - n11 * n23 * n32 - n12 * n21 * n33 + n11 * n22 * n33) * inv_det;
    // clang-format on

    return inv;
}
