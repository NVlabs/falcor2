---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "94b4f58e85ced7cc7d558b7e22eb5f189c2f27e3"
---

(falcor2-design-sampling-inputs)=
# Sampling inputs

Falcor2 separates sample generation from the rendering operations that consume samples. A caller owns the
generator and constructs explicit inputs for a light, material, BSDF, phase function or volume-distance
sample. Each input factory consumes a fixed number of scalar coordinates; implementations can reuse and
remap those coordinates without advancing the caller's generator. Material algorithms that need additional
randomness derive a local stream from an auxiliary seed. This boundary makes consumption and replay
inspectable, while the caller's branches still determine the overall coordinate schedule. Fixed factory
budgets alone do not synchronize dimensions across paths or guarantee better convergence.

## Purpose and scope

A contributor familiar with rendering interfaces can use this page to predict which operation advances the
sample generator, how nested choices reuse inputs, and what changes can shift later samples.
[BSDF contract](bsdf-contract.md) owns estimator weights, component densities and direction conventions.
[Lighting and sampling](lighting-and-sampling.md) owns emitter distributions and visibility.
[Sample generation](../sample-generation.md) explains generator selection and reproducibility.

## Architecture

### The caller owns the coordinate sequence

`ISampleGenerator` exposes sequential `next_uint()` calls. Its float helpers convert the upper 24 bits to
values in `[0, 1)`; vector helpers consume consecutive coordinates in component order. Copying a generator
copies its current position. A copy is therefore a replay point, not an independently assigned stream.

Rendering samplers receive input values rather than mutable generators. Callers can construct inputs with
the standard factories or supply coordinates directly. Every float input must be in `[0, 1)`; every bit
pattern is valid for `aux_seed`. The factory budgets below count calls to `next_uint()`, including the raw
seed word. Each factory's field order is part of the contract.

| Input type | Fields in factory consumption order | Scalar draws |
| --- | --- | --- |
| `LightSampleInput` | `u_select`, `u_light.x`, `u_light.y` | 3 |
| `BSDFSampleInput` | `u_lobe`, `u_dir.x`, `u_dir.y`, `aux_seed` | 4 |
| `MaterialSampleInput` | Alias of `BSDFSampleInput` | 4 |
| `PhaseFunctionSampleInput` | `u_lobe`, `u_dir.x`, `u_dir.y` | 3 |
| `VolumeDistanceSampleInput` | `u_channel`, `u_distance` | 2 |

Factories such as `make_material_sample_input()` advance the generator by their full budget even when the
consumer ignores fields or rejects the sample. A caller that skips the factory spends nothing. Material
inputs can be forwarded to a BSDF without a coordinate-frame conversion: `u_dir` contains uniform sample
coordinates, not a physical direction.

### Nested choices reuse conditional residuals

A selector chooses an interval of a discrete distribution. A child that needs another choice receives the
selector's conditional residual: `u_child = (u - cdf_begin) / p`, where `p` is the selected interval's
positive probability. `remap_sample_selector()` clamps the result to `[0, 1)` to contain rounding error.
Callers must provide a positive probability; this helper does not make a zero denominator valid.

`choose_sample_branch()` handles zero and unit probabilities without changing the selector. For intermediate
probabilities it selects the first branch when `u < p` and remaps into the chosen interval. With `p = 0.25`,
`u = 0.625` selects the second branch and becomes `0.5`; exactly `u = 0.25` selects that branch at zero.

`BSDFSampleInput.with_remapped_lobe()` changes only `u_lobe`, preserving direction coordinates and seed.
Composite BSDFs and generated MaterialX closure selection forward residuals where the child needs its own
selector. For example, Standard material's combined reflection/transmission branch needs a subsequent
Fresnel choice. Children that do not need another choice can receive the original input; a single MaterialX
closure also preserves its selector. Remapping follows proposal probabilities, which can differ from physical
contribution weights.

Scene light sampling separates emitter choice from sampling that emitter. Power and hierarchical samplers
condition `u_select` through their domain and distribution choices, then pass `u_light` unchanged to the
chosen light. Individual `ILight.sample()` implementations receive only two coordinates. An environment
light can repeatedly warp those same coordinates through its importance-map hierarchy, retaining residuals
for the final position, without drawing again.

### Auxiliary randomness stays within a material sample

`BSDFSampleInput.make_aux_rng()` hashes `aux_seed` with a fixed salt and constructs a concrete
`UniformSampleGenerator`. Cloth, scratch and dielectric-plate algorithms can draw as many local values as
their algorithms require without advancing the caller's sequence. The factory preserves all 32 seed bits.
Reconstructing the auxiliary generator from the same seed restarts the same local stream, even if the other
coordinates differ. This mechanism does not establish formal independence between streams.

Material adapters interpret the shared input according to their target interface. OpenPBR receives
`float3(u_lobe, u_dir)`. Common MaterialX leaf adapters, including dielectric and generalized-Schlick,
use `float3(u_dir, u_lobe)`. Generated MDL wrappers use
`float4(u_dir, u_lobe, aux_seed_as_float())`, where the last value is the seed's upper 24 bits scaled into
`[0, 1)`. MDL therefore uses that value as a fourth SDK sampling coordinate, rather than constructing the
local auxiliary generator. All these adapters retain the four-word caller budget.

### Phase and distance inputs have separate roles

Phase sampling separates lobe choice from direction sampling. Isotropic and single Henyey-Greenstein phase
functions ignore `u_lobe`; dual Henyey-Greenstein uses it to choose a lobe and passes `u_dir` unchanged
to direction sampling. Their common factory still spends three coordinates.

Homogeneous-volume distance sampling uses `u_channel` to select an RGB proposal channel and
`u_distance` to sample its scattering distance. Zero-probability channels are skipped. Selecting a channel
with zero scattering rate returns a successful escape at infinity; an entirely non-scattering medium
returns false with infinite distance. Neither result changes the two-word budget of an already constructed
input. The reference tracer checks `has_scattering()` before constructing distance inputs.

### Evaluation does not own randomness

BSDF, material and phase evaluation and PDF queries do not take a generator or auxiliary seed. Volume
transmittance and weight evaluation are also deterministic functions of their inputs. `BSDFContext` carries
indices of refraction and hints, not an evaluation random stream. Light endpoint reconstruction uses a retained sample
reference and current context. These boundaries permit evaluation without consuming or reconstructing the
sampling sequence.

## Runtime flow

The reference path tracer illustrates why a fixed input budget is local to an operation:

1. Each camera sample initializes the generator from the pixel and sample ordinal. Pixel filtering consumes
   two coordinates even for a zero-width box filter. Lens sampling consumes two more only when depth of field
   is enabled with positive aperture and focus distance.
2. Preparing a new ray segment inside a scattering homogeneous medium constructs distance inputs. Segment
   continuation returns before this work. Each new segment then consumes one word for its opacity evaluator.
3. An eligible surface next-event estimate constructs three light coordinates. A valid light and direction
   then require a separate shadow-opacity seed. Material evaluation adds no generator consumption.
4. Material continuation constructs a four-word input. The CUDA path's local generator copy and writeback
   preserve that same parent advancement. A volume event instead constructs three phase coordinates after
   its throughput and depth gates.
5. Russian roulette consumes a coordinate only when its depth and successful-scattering conditions apply.
   Subsequent operations resume the generator at the resulting position.

Opacity randomness is separate from `MaterialSampleInput.aux_seed`. Candidate-hit decisions hash the
per-ray seed with geometry identifiers, so additional candidate hits do not consume additional path
coordinates. New segments still draw their seed when the selected evaluator accepts every hit.

The default generator is Uniform; the optional Sobol generator uses its first 1,024 dimensions and then a
deterministic PRNG continuation. Both implement the same sequential interface. Branches, skipped factories,
depth of field, media and roulette can shift later semantic uses to different dimensions. There is no
global dimension allocator or per-bounce synchronization supplied by these input types.

## Invariants and rationale

The caller owns parent advancement; consumers own how supplied coordinates map to their distributions.
Preserving that boundary lets a material algorithm change its local auxiliary work without changing later
parent draws. Preserving conditional residuals lets nested selection reuse a bounded input while maintaining
the child's conditional sampling distribution.

Fixed factory order, full seed bits and half-open float domains are shared contracts across generators.
Rejection does not refund coordinates. Evaluation cannot silently draw new values. These invariants support
controlled replay and consumption tests; they do not promise identical rendered images after an algorithm
change or establish a convergence advantage for Sobol.

## Source landmarks

- `slang/falcor2/utils/sampling/sample_generator.slang`,
  `slang/falcor2/utils/sampling/sample_generator/uniform_sample_generator.slang` and
  `slang/falcor2/utils/sampling/sample_generator/pbrt_sobol_sample_generator.slang` - sequential generation
  and concrete streams.
- `falcor2/utils/sample_generators.py` - exposed generator choices and device resources.
- `slang/falcor2/render/bsdf.slang` and `slang/falcor2/render/material.slang` - inputs, factories,
  selector helpers and auxiliary streams.
- `slang/falcor2/render/light_sampler.slang`, `slang/falcor2/render/light.slang`,
  `slang/falcor2/render/emissive_triangle_tree.slang` and
  `slang/falcor2/render/lights/env_map_light.slang` - emitter selection and coordinate reuse.
- `slang/falcor2/render/phase_function.slang`, `slang/falcor2/render/phase_functions/` and
  `slang/falcor2/render/homogeneous_volume_sampler.slang` - phase and distance contracts.
- `src/falcor2/render/material/mdl/mdl_material.cpp` and
  `src/falcor2/render/material/mtlx/codegen/structured/layering/bsdf_mix/flat_root.cpp` - generated adapter
  and closure-input forwarding.
- `slang/falcor2/render/materials/openpbr_material.slang` and
  `slang/falcor2/render/materials/mtlx/bsdf_wrapper.slang` - target-specific input interpretation.
- `slang/falcor2/rendernodes/reference_pathtracer.slang` and
  `slang/falcor2/render/intersector.slang` - caller branches and separate opacity seeds.
- `tests/python/render/test_sample_inputs.slang` and
  `tests/python/render/test_homogeneous_volume_sampler.slang` - field order, budgets, nested forwarding,
  auxiliary seeds and distance boundaries.
- `tests/python/utils/test_sample_generator.py` and `tests/python/pathtracer/test_sample_generator.py` -
  generator reference/overflow behavior, copies, render repeatability and generator-switch resets.

## Related design documents

- [BSDF contract](bsdf-contract.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [MDL integration](mdl-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Render pipeline](render-pipeline.md)
- [Falcor2 design documentation](index.md)
