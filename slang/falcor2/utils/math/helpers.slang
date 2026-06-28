// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Returns the square of the input value.
[Differentiable]
public T sqr<T : __BuiltinFloatingPointType>(T x)
{
    return x * x;
}

/// Returns the norm of the input vector.
[Differentiable]
public T norm<T : __BuiltinFloatingPointType, int N>(vector<T, N> x)
{
    return length(x);
}

/// Returns the squared norm of the input vector.
[Differentiable]
public T squared_norm<T : __BuiltinFloatingPointType, int N>(vector<T, N> x)
{
    return dot(x, x);
}

/// Returns the square root of the input value, or 0 if the input is negative.
[Differentiable]
public T safe_sqrt<T : __BuiltinFloatingPointType>(T x)
{
    return sqrt(max(x, T(0)));
}

/// Multiplies 2 values together
[Differentiable]
public T multiply<T : IArithmetic>(T x, T y)
{
    return x * y;
}

/// Adds 2 values together
[Differentiable]
public T add<T : IArithmetic>(T x, T y)
{
    return x + y;
}

/// Subtracts 2 values
[Differentiable]
public T subtract<T : IArithmetic>(T x, T y)
{
    return x - y;
}

/// Divides 2 values
[Differentiable]
public T divide<T : IArithmetic>(T x, T y)
{
    return x / y;
}

/// Error function.
[Differentiable]
public T erf<T : __BuiltinFloatingPointType>(T x)
{
    // From "Numerical Recipes in C, The Art of Scientific Computing"
    // (Second Edition) by Press et al. 1992. Page 221.
    // Maxiumum error: 1.2 x 10^-7.
    T t = T(1.0) / (T(1.0) + T(0.5) * abs(x));
    T p = T(0.17087277);
    p = T(-0.82215223) + p * t;
    p = T(1.48851587) + p * t;
    p = T(-1.13520398) + p * t;
    p = T(0.27886807) + p * t;
    p = T(-0.18628806) + p * t;
    p = T(0.09678418) + p * t;
    p = T(0.37409196) + p * t;
    p = T(1.00002368) + p * t;
    p = T(-1.26551223) + p * t;
    T tau = t * exp(-x * x + p);
    return x >= T(0.0) ? T(1.0) - tau : tau - T(1.0);
}

/// Inverse error function.
[Differentiable]
public T erfinv<T : __BuiltinFloatingPointType>(T x)
{
    // From "Approximating the erfinv function" by Mike Giles 2012.
    // Maximum error: 7 x 10^-7.
    if (x <= T(-1.0)) {
        return T(-1.0) / T(0.0);
    } else if (x >= T(1.0)) {
        return T(1.0) / T(0.0);
    }

    T w = -log((T(1.0) - x) * (T(1.0) + x));
    T p;
    if (w < T(5.0)) {
        w = w - T(2.5);
        p = T(2.81022636e-08);
        p = T(3.43273939e-07) + p * w;
        p = T(-3.5233877e-06) + p * w;
        p = T(-4.39150654e-06) + p * w;
        p = T(0.00021858087) + p * w;
        p = T(-0.00125372503) + p * w;
        p = T(-0.00417768164) + p * w;
        p = T(0.246640727) + p * w;
        p = T(1.50140941) + p * w;
    } else {
        w = sqrt(w) - T(3.0);
        p = T(-0.000200214257);
        p = T(0.000100950558) + p * w;
        p = T(0.00134934322) + p * w;
        p = T(-0.00367342844) + p * w;
        p = T(0.00573950773) + p * w;
        p = T(-0.0076224613) + p * w;
        p = T(0.00943887047) + p * w;
        p = T(1.00167406) + p * w;
        p = T(2.83297682) + p * w;
    }
    return p * x;
}

/// Logarithm of the Gamma function.
[Differentiable]
public T lgamma<T : __BuiltinFloatingPointType>(T x_)
{
    // Lanczos approximation from
    // "Numerical Recipes in C, The Art of Scientific Computing"
    // (Second Edition) by Press et al. 1992. Page 214.
    // Maxiumum error: 2 x 10^-10.
    const T g = T(5.0);
    const T coeffs[7] = {
        T(1.000000000190015),
        T(76.18009172947146),
        T(-86.50532032941677),
        T(24.01409824083091),
        T(-1.231739572450155),
        T(0.1208650973866179e-2),
        T(-0.5395239384953e-5),
    };

    bool reflect = x_ < T(0.5);
    T x = reflect ? -x_ : x_ - T(1.0);
    T base = x + g + T(0.5);

    T s = T(0.0);
    for (int i = 6; i >= 1; --i) {
        s += coeffs[i] / (x + T(i));
    }
    s += coeffs[0];

    T res = ((log(sqrt(T(2.0) * T.getPi())) + log(s)) - base) + log(base) * (x + T(0.5));
    if (reflect) {
        // Apply the Gamma reflection formula (in log-space).
        res = log(abs(T.getPi() / sin(T.getPi() * x_))) - res;
    }
    return res;
}

/// Gamma function.
[Differentiable]
public T gamma<T : __BuiltinFloatingPointType>(T x)
{
    return exp(lgamma(x));
}

/// Beta function.
[Differentiable]
public T beta<T : __BuiltinFloatingPointType>(T x, T y)
{
    return gamma(x) * gamma(y) / gamma(x + y);
}
