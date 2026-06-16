// SPDX-License-Identifier: Apache-2.0

implementing utils;

public interface IMathConstants
{
    public static const This E;
    public static const This LOG2_E;
    public static const This LOG10_E;
    public static const This LN_TWO;
    public static const This LN_TEN;
    public static const This PI;
    public static const This TWO_PI;
    public static const This FOUR_PI;
    public static const This INV_PI;
    public static const This INV_TWO_PI;
    public static const This INV_FOUR_PI;
    public static const This PI_OVER_TWO;
    public static const This PI_OVER_FOUR;
    public static const This TWO_OVER_PI;
    public static const This FOUR_OVER_PI;
    public static const This SQRT_TWO;
    public static const This INV_SQRT_TWO;
    public static const This SQRT_THREE;
    public static const This SQRT_PI;
    public static const This INV_SQRT_PI;
    public static const This TWO_OVER_SQRT_PI;

    // TODO these are not math constants, so maybe they should be moved
    public static const This EPSILON;  // smallest such that 1.0+EPSILON != 1.0
    public static const This MAX;      // max value
    public static const This MIN;      // min positive value (normalized)
    public static const This TRUE_MIN; // min positive value
    public static const This NAN;      // not a number
};

#define concat(a, b) a##b

// clang-format off
#define DEFINE_MATH_CONSTANTS(type, suffix)                                                                     \
    public static const type E                  = concat(2.71828182845904523536, suffix);   /* e            */  \
    public static const type LOG2_E             = concat(1.44269504088896340736, suffix);   /* log2(e)      */  \
    public static const type LOG10_E            = concat(0.434294481903251827651, suffix);  /* log10(e)     */  \
    public static const type LN_TWO             = concat(0.693147180559945309417, suffix);  /* ln(2)        */  \
    public static const type LN_TEN             = concat(2.30258509299404568402, suffix);   /* ln(10)       */  \
    public static const type PI                 = concat(3.14159265358979323846, suffix);   /* pi           */  \
    public static const type TWO_PI             = concat(6.28318530717958647693, suffix);   /* 2pi          */  \
    public static const type FOUR_PI            = concat(12.5663706143591729539, suffix);   /* 4pi          */  \
    public static const type INV_PI             = concat(0.318309886183790671538, suffix);  /* 1/pi         */  \
    public static const type INV_TWO_PI         = concat(0.159154943091895335769, suffix);  /* 1/2pi        */  \
    public static const type INV_FOUR_PI        = concat(0.079577471545947667884, suffix);  /* 1/4pi        */  \
    public static const type PI_OVER_TWO        = concat(1.57079632679489661923, suffix);   /* pi/2         */  \
    public static const type PI_OVER_FOUR       = concat(0.785398163397448309616, suffix);  /* pi/4         */  \
    public static const type TWO_OVER_PI        = concat(0.636619772367581343076, suffix);  /* 2/pi         */  \
    public static const type FOUR_OVER_PI       = concat(1.27323954473516268615, suffix);   /* 4/pi         */  \
    public static const type SQRT_TWO           = concat(1.41421356237309504880, suffix);   /* sqrt(2)      */  \
    public static const type INV_SQRT_TWO       = concat(0.707106781186547524401, suffix);  /* 1/sqrt(2)    */  \
    public static const type SQRT_THREE         = concat(1.732050807568877293527, suffix);  /* sqrt(3)      */  \
    public static const type SQRT_PI            = concat(1.77245385090551602730, suffix);   /* sqrt(pi)     */  \
    public static const type INV_SQRT_PI        = concat(0.564189583547756279280, suffix);  /* 1/sqrt(pi)   */  \
    public static const type TWO_OVER_SQRT_PI   = concat(1.12837916709551257390, suffix);   /* 2/sqrt(pi)   */

// clang-format on

public extension half : IMathConstants
{
    DEFINE_MATH_CONSTANTS(half, h);

    public static const half EPSILON = 0.00097656h;
    public static const half MAX = 65504.0h;
    public static const half MIN = 6.10351562e-05h;
    public static const half TRUE_MIN = 5.96046448e-08h;
    public static const half NAN = 0.h / 0.h;
};

public extension float : IMathConstants
{
    DEFINE_MATH_CONSTANTS(float, f);

    public static const float EPSILON = 1.192092896e-07f;
    public static const float MAX = 3.402823466e+38f;
    public static const float MIN = 1.175494351e-38f;
    public static const float TRUE_MIN = 1.401298464e-45f;
    public static const float NAN = 0.f / 0.f;
};

public extension double : IMathConstants
{
    DEFINE_MATH_CONSTANTS(double, l);

    public static const double EPSILON = 2.2204460492503131e-016l;
    public static const double MAX = 1.7976931348623158e+308l;
    public static const double MIN = 2.2250738585072014e-308l;
    public static const double TRUE_MIN = 4.9406564584124654e-324l;
    public static const double NAN = 0.l / 0.l;
};
