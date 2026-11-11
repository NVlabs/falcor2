# Sample generation

The reference path tracer defaults to Uniform. Select Sobol with:

```python
from falcor2.rendernodes import SampleGeneratorType

pipeline.path_tracer.sample_generator = SampleGeneratorType.sobol_pbrt
```

Changing the generator invalidates shader specialization and resets accumulation
on the next frame. Set `SampleGeneratorType.uniform` to restore the default.

## Initialization and consumption

```csharp
PBRTSobolSampleGenerator sg = PBRTSobolSampleGenerator(sample_index, scrambling_seed);
float2 u = next_float2(sg);
```

`sample_index` selects a point; successive draws select its coordinates. Keep
`scrambling_seed` fixed while advancing the index. Its default of zero selects a
fixed scramble. `next_floatX` consumes X consecutive coordinates in [0,1).
Copying a generator copies its current position.

The pixel constructor `SG(uint2 pixel, uint sample_index)` derives a stable QMC
seed using `get_pixel_scrambling_seed(pixel)`. The index is the sample ordinal
within that pixel. Generic integration uses the two-integer constructor directly.
Uniform and TinyUniform retain their original pixel and single-integer streams;
with a nonzero scrambling seed they combine both integers to initialize the PRNG.

`ISampleGenerator` requires these constructors and `next_uint()`. Shader
specialization retains `extern struct SampleGenerator : ISampleGenerator =
UniformSampleGenerator`. Standalone SlangPy callers bind the Sobol buffer with:

```python
from falcor2.utils.sample_generators import SampleGeneratorType, bind_sample_generator

sampler = SampleGeneratorType.sobol_pbrt
fn = module[f"integrate<{sampler.slang_type_name}>"].write(
    lambda cursor: bind_sample_generator(cursor, device, sampler)
)
```

## Sobol implementation and limits

Coordinates 0-1023 use distinct PBRT Sobol direction matrices. Kollig-Keller
random-digit scrambling XORs each integer coordinate with
`jenkins_hash(scrambling_seed + dimension)`, fixed across sample indices.
This preserves digital-net stratification, but retains more structure than Owen
scrambling. See [Kollig and Keller (2002), section 3.2.3](https://doi.org/10.1111/1467-8659.00706).

Subsequent draws use Xorshift32, initialized once from the original index and seed.
Its state reuses the sample-index word; the dimension counter saturates so Sobol
coordinates never repeat through counter wrap. Total state is 12 bytes. Materials
consume the generator directly, without fixed budgets or synchronization.

The sample index supports 2^32 points per seed/pixel before wrapping. Coordinates
are computed as 32-bit integers; floats use the upper 24 bits. QMC does not improve
convergence for every integrand.

## Data and reproduction

The implementation adapts `SobolSample` and `BinaryPermuteScrambler` from public
[PBRT v4 revision b4ce9687](https://github.com/mmp/pbrt-v4/tree/b4ce9687e6c695f5582997c61b0c66cf064bdb4a)
under Apache-2.0, adding Slang buffers, 32-bit indexing, integer outputs and PRNG
continuation. No internal Falcor code or data is used.

`falcor2/utils/qmc_tables.json` packages 1,024 matrices with 32 columns, source
SHA-256 hashes, the PBRT license, and the complete direction-table notice including
Leonhard Gruenschloss's permissive license and the Joe-Kuo construction reference.
PBRT's remaining 20 columns only serve indices above our uint range; discarding
them leaves every supported sample unchanged.

`falcor2/utils/sample_generators.py` uploads the matrices into one read-only
128 KiB structured buffer per device, cached until device close. Rendering requires
no downloads. The shader XORs matrix columns selected by the sample index's bits.

To reproduce the data, obtain these files at the pinned revision:

- PBRT `src/pbrt/util/sobolmatrices.cpp`, saved as `<source_dir>/pbrt/sobolmatrices.cpp`.
- PBRT `LICENSE.txt`, saved as `<source_dir>/pbrt/LICENSE.txt`.

Run `python tools/generate_qmc_tables.py <source_dir>`. It verifies both input
hashes before writing the packaged JSON. Preserve the included licenses and notices
when redistributing.
