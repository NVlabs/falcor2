// SPDX-License-Identifier: Apache-2.0

module utils;
import slangpy;

__include "utils/accumulator.slang";
__include "utils/buffer_handle.slang";
__include "utils/chi2.slang";
__include "utils/enum.slang";
__include "utils/fill.slang";
__include "utils/loss.slang";
__include "utils/tonemap.slang";

__include "utils/color/colormap.slang";
__include "utils/color/helpers.slang";

__include "utils/geometry/frame.slang";
__include "utils/geometry/intersection.slang";
__include "utils/geometry/raytracing.slang";

__include "utils/materials/fresnel.slang";
__include "utils/materials/isotropic_ggx.slang";

__include "utils/math/bitfield.slang";
__include "utils/math/bitmanip.slang";
__include "utils/math/constants.slang";
__include "utils/math/hash.slang";
__include "utils/math/helpers.slang";
__include "utils/math/matrix.slang";
__include "utils/math/mis.slang";
__include "utils/math/packing.slang";
__include "utils/math/ray.slang";
__include "utils/math/transform.slang";
__include "utils/math/warp.slang";

__include "utils/optimizer/adam.slang";
__include "utils/optimizer/gradient_descent.slang";
__include "utils/optimizer/optimizer.slang";

__include "utils/sampling/distribution_1d.slang";
__include "utils/sampling/rng.slang";
__include "utils/sampling/sample_generator.slang";
__include "utils/sampling/rng/lcg.slang";
__include "utils/sampling/rng/splitmix64.slang";
__include "utils/sampling/rng/xorshift32.slang";
__include "utils/sampling/rng/xoshiro128starstar.slang";
__include "utils/sampling/sample_generator/tiny_uniform_sample_generator.slang";
__include "utils/sampling/sample_generator/uniform_sample_generator.slang";
