---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "a42654be7704e9a4b36b4b2134c1131db2e2b60e"
---

(falcor2-design-bsdf-contract)=
# BSDF contract

Falcor2's bidirectional scattering distribution function (BSDF) contract connects per-hit materials to renderer
transport. An `IMaterial` creates a world-space `IMaterialInstance`, which resolves hit state and either dispatches
local `IBSDF` implementations or adapts an upstream scattering model. A BSDF can evaluate projected scattering,
sample an outgoing direction, or return the value and directional probability density function (PDF) together.
Continuous events use solid-angle densities; ideal reflection and refraction are delta events identified by flags
and report zero in the scalar PDF field. Composition keeps physical lobe weights separate from proposal-selection
probabilities so renderers can use the returned densities for multiple importance sampling.

## Purpose and scope

A contributor familiar with Monte Carlo rendering can use this page to interpret material evaluation and sample
results, reconstruct a contribution when the selected policy permits it, and preserve that contract across
composition. Continuous directional PDFs have units per steradian. Multiple importance sampling (MIS) combines
proposals using densities for the same direction and measure.

Material ownership and device records belong in [Material system](material-system.md). Detailed MaterialX graph
lowering belongs in [MaterialX integration](materialx-integration.md), and renderer and light-sampler construction
belong in [Render pipeline](render-pipeline.md) and [Lighting and sampling](lighting-and-sampling.md). This page does
not catalogue every BSDF formula or assert that all models are reciprocal. The layered evidence used to validate
these semantics is described in [BSDF correctness and testing](bsdf-correctness-and-testing.md).

## Architecture

### Directions and scattering measure

`IBSDF` is the low-level Slang interface. Its operations use a local orthonormal frame whose normal, tangent, and
bitangent are positive Z, positive X, and positive Y. Both the incident direction `wi` and outgoing direction `wo`
are normalized and point away from the shading point. Surface BSDFs receive `wi` in the positive hemisphere;
reflection normally returns to that hemisphere and transmission crosses it. Curve models, such as hair or cloth
fibers, may operate over the full sphere in their curve frame.

Surface `eval()` values already include outgoing foreshortening: they return `f(wi, wo) * abs(wo.z)`. A renderer
must not multiply by another surface cosine. Curve BSDFs return their model response with any curve-specific measure
terms and Jacobians included by that model, not an extra surface cosine. `BSDF_MIN_COS_THETA` provides one shared
epsilon for implementations that are undefined exactly at grazing incidence.

`BSDFContext` supplies the incidence and transmission indices of refraction. Its `sample_lobe_types_hint` is a soft
efficiency hint: a caller will discard events outside the mask, but ignoring the hint is still correct. An
implementation that honors it must report the density of the restricted proposal it actually sampled, while
leaving the physical value unchanged.

### Evaluation and sampling operations

`IBSDF.eval_with_pdf(wi, wo, bc, component)` is the canonical evaluation operation and returns `BSDFEval`, containing
the selected projected value and solid-angle density. `eval()` and `eval_pdf()` select the corresponding field.
Each operation has one signature with a defaulted `BSDFComponent::full_response` selector. Implementations,
including custom BSDFs and materials, must accept that selector explicitly. Evaluation takes neither a generator
nor a seed. Low-level `IBSDF` evaluation has no field-selection hint. Concrete aliases can preserve compiler
requirements without duplicating formulas: `DiffuseSpecularBRDF` retains differentiable projections of the fused
result because differentiation through the interface default is unsupported.

`sample()` chooses `wo` and returns a `BSDFSample` only on success. The record carries a PDF, an RGB throughput
weight, a contribution selector, and `BSDFFlags` identifying diffuse, glossy, or delta reflection or transmission,
plus glossy curve events. Continuous events expose their marginal directional density. An ideal mirror or refraction
is a discrete delta event rather than a finite solid-angle density, so its scalar PDF is zero and its event
probability is represented through the sample weight and flags. This zero-PDF sentinel belongs to the sampled
delta event. Full-response evaluation at the same direction may still report a nonzero continuous density from
overlapping diffuse or glossy lobes.

At the world-space boundary, `IMaterialInstance` returns `BSDFEval` from evaluation and `MaterialSample` from
sampling. Its `eval_with_pdf<Hints>()` defaults to `BSDFEvalHints::both`. The `eval()` and `eval_pdf()` aliases
request `eval` and `pdf`, respectively, so aggregate implementations can omit unneeded work. Implementations may
compute extra fields, but callers cannot rely on unrequested results. Selecting `.pdf` from a default fused
call still requests both fields; use the PDF-only alias or hint when only the density is needed.

`BSDFSampleInput` supplies caller-owned coordinates: `u_lobe` selects a proposal, `u_dir` samples within it, and
`aux_seed` supplies auxiliary random bits. `MaterialSampleInput` aliases that type. Frame conversions leave
these inputs unchanged. [Sampling inputs](sampling-inputs.md) defines the four-word factory budget, conditional
selector reuse and local auxiliary streams.

### Material sample weight policy

`MATERIAL_SAMPLE_WEIGHT_POLICY` is a renderer-owned link-time constant in `render/config.slang`, with values from
`MaterialSampleWeightPolicy` in `material.slang`. The default `native` policy preserves valid continuation
estimators, marginal PDFs, and event flags. It does not guarantee that a material sample's weight can be reconstructed
through component evaluation, even when the component marker is `full_response`.

Renderers that reconstruct sampled contributions for reuse must select `eval_consistent`.
Successful continuous material samples then satisfy the component value/joint-density identity below. This does not
force selected-component implementations to collapse their response. Delta events retain integrated weights.
The policy changes the estimator guarantee, without changing evaluation or the physical scattering function.

Each linked renderer chooses its policy independently of scene configuration or material properties.
The reference path tracer uses the default `native` policy. The same generated material module can therefore serve
ordinary continuation and evaluation-consistent reuse. MDL's native path returns the SDK's `bsdf_over_pdf`
directly; its evaluation-consistent path additionally evaluates the full response and divides by the SDK's
marginal sampled PDF. This changes the continuous weight while preserving the sampled direction, flags and PDF.

Renderers specialize the enum with a typed exported constant, for example in a shader prelude:

```cpp
export static const MaterialSampleWeightPolicy MATERIAL_SAMPLE_WEIGHT_POLICY
    = MaterialSampleWeightPolicy::eval_consistent;
```

### Contribution identity and sample weights

Before reconstructing a material sample's weight, identify its event flags and the linked renderer's weight
policy. Delta events use integrated weights and do not use the continuous reconstruction below. For continuous
events, `eval_consistent` guarantees reconstruction; `native` does not. Then use the returned component to choose
the value and joint density, while retaining the marginal `sample.pdf` for MIS.

`BSDFComponent` identifies the physical contribution represented by a sample's weight. A component may group
several lobes or internal proposals. Its index describes the returned contribution, independently of which
proposal generated the direction. Indices are local to a component layout and stay fixed when contributions become
inactive. Component-aware implementations return zero for unknown indices. BSDFs and material adapters that expose
only full response ignore the selector and always evaluate their entire response. Callers use a material's layout
ID to determine whether it exposes a component partition.

For a finite component partition, let `F_c(wi, wo)` be component `c`'s projected physical contribution and
`q_c(wi, wo)` its joint component/direction density, including component-selection probability. At a continuous
direction, full response is the sum of `F_c`, and the marginal directional density is the sum of `q_c`. The sample
weight identities hold for material samples under `eval_consistent` and for component-consistent low-level BSDFs.

| Operation or sample field | Full response | Selected component `c` |
| --- | --- | --- |
| `eval_with_pdf(..., component).value` | Sum of physical contributions | `F_c(wi, wo)` |
| `eval_with_pdf(..., component).pdf` | Marginal directional density | `q_c(wi, wo)` |
| `sample.weight` | Full projected value / marginal density | `F_c(wi, wo) / q_c(wi, wo)` |
| `sample.pdf` | Marginal directional density | Marginal directional density |

Physical material weights belong in `F_c`; proposal probabilities belong in `q`. For example, selecting one of two
overlapping contributions with probabilities `p_0` and `p_1` and conditional densities `r_c(wo)` gives joint density
`q_c(wi, wo) = p_c * r_c(wo)`. Dividing the
selected contribution by that joint density compensates for selection. Its weight generally differs from full
`eval / sample.pdf`, even though both estimators integrate the same response.

`BSDFComponent::full_response` is the sentinel `0xffffffff`. Under `eval_consistent`, it means that the weight
represents the whole response divided by the marginal density; it is not an extra term in the finite component
partition. Native material estimators may retain hidden branch weights under this marker. A wrapper that collapses
child contributions must return this marker. Delta weights are integrated event weights with selection compensation;
continuous division and reconstruction formulas do not apply to them.

### Composition and proposal selection

`StandardBSDF` combines diffuse reflection, diffuse transmission, microfacet reflection, and microfacet
reflection/transmission. It biases lobe selection using Fresnel terms and luminance. MaterialX weight wrappers scale
physical values; frame wrappers rotate directions without changing solid-angle densities under orthonormal
transforms. MaterialX layer, mix, and add wrappers choose proposals from the luminance of weighted directional-albedo
estimates. Layer opacity attenuates the base contribution; mix weights scale both children's albedo estimates.
Delta branches return before continuous densities are folded in because those measures differ.

`StandardBSDF` and `GLTFSpecGlossBSDF` retain selected contributions.


MaterialX weight and frame wrappers preserve
child component identity, whereas layer, mix, and add wrappers collapse continuous samples to full response.
MaterialX `bsdf_mix` and `closure_tree` expose full response at
the material boundary. Cloth, scratch, OpenPBR, and native metallic-roughness glTF also expose full continuous
response. Cloth and scratch reconstruct that weight locally from evaluation and the marginal density because
their adapters do not expose a stable finite partition of latent choices. MDL reconstructs full-response weights
only when the linked material sample weight policy requires evaluation consistency.

### Component compatibility and roughness

`IMaterialInstance.get_component_layout_id()` identifies compatible component-index meanings. Equal nonzero IDs
allow callers to interpret a saved component at another compatible instance; they do not imply equal parameters,
values, or probabilities. Layout zero is reserved for full-response sampling. The scene-local ID lifetime and
native slot layouts are described in [Material system](material-system.md#component-layout-identity).

`is_component_rough(si, component, marginal_pdf, roughness_threshold)` is a reconnection heuristic. Callers first
exclude delta events using their flags, then supply the marginal density at the current interaction and outgoing
direction. The default classifies a response as rough when the threshold is nonpositive or the density is at most
`1 / threshold^2`. Standard and SpecGloss classify their diffuse slots as rough and compare specular perceptual
roughness with the threshold; full-response requests use the default density heuristic. This is a classification
policy, not a universal physical roughness measure.

### Material-instance dispatch

A geometry hit supplies a synchronized `MaterialID`. `MaterialSystem` recovers the dynamic `IMaterial`, and
`setup_material_instance()` resolves textures, attributes, media, and the shading frame for that accepted hit.
Standard materials transform world-space directions into one local `StandardBSDF`, reject disagreement between
shading-frame and geometric hemispheres, and transform sampled directions back. Advanced MaterialX evaluates graph lobes in their generated local frames and
aggregates them behind the same interface.

Adapters preserve the public contract even when an upstream model has a different API. OpenPBR prepares its BSDF
once per evaluation query and conditionally calls separate upstream value and PDF functions according to the
requested hints. Generated MDL uses its evaluate operation when a value is needed and retains a specialized
upstream PDF operation for a PDF-only request. The renderer therefore dispatches `IMaterialInstance`, not a concrete
BSDF family.

## Runtime flow

```text
 GeometryInstance MaterialID
            |
            v
       dynamic IMaterial
            |
  setup_material_instance()
            |
            v
      IMaterialInstance
       /             \
 direct-light     path-continuation
 candidate        sample()
       |             |
 eval_with_pdf()  weight + PDF + flags + component
       |             |
     MIS weight   throughput, emitter MIS,
                  medium-boundary update
```

1. After traversal accepts a hit, the concrete material creates an instance conditioned on the surface interaction
   and current texture LOD policy. The instance owns the world-to-local conversion or upstream adapter needed by its
   scattering implementation.
2. For next-event estimation, the path tracer uses the material's lobe flags to decide which hemispheres can
   contribute. When MIS is enabled and the light sample is not marked `nee_only`, it evaluates the direction
   once through `eval_with_pdf()` and uses both value and PDF. Disabled MIS and `nee_only` light samples request
   only the value because no competing BSDF density is needed.
3. For path continuation, `sample()` supplies a direction and weight. The path tracer multiplies throughput by the
   weight and retains the marginal PDF for MIS if the next segment reaches an emitter. Delta flags disable an
   incompatible continuous-density comparison; transmission flags drive interior-boundary tracking.
4. Implementations are expected to fail sampling rather than expose non-finite directions, negative or non-finite
   weights, or invalid densities. The generic tests require successful non-delta samples to have a positive density;
   successful delta samples instead require zero in the scalar PDF field and a delta flag. These checks do not
   establish that the flag identifies the physically correct event.

## Invariants and rationale

Evaluation and PDF are one semantic calculation. Keeping `eval_with_pdf()` authoritative prevents separate formulas
from drifting and lets shared half-vector, distribution, Fresnel, or graph work be reused. Single-result aliases
remain useful at call sites, while compile-time hints prevent large material aggregates from doing unnecessary work.
Evaluation must reproduce the selected contribution and its density at the supplied interaction and directions.

Marginalizing over hidden proposal choices lets a generic renderer store one density for later MIS. Returning only
the selected conditional density would make that comparison depend on hidden sampler state.

Under `eval_consistent`, reconstruct a successful continuous material sample's weight with its component evaluation
and joint density, while checking `sample.pdf` separately against full-response evaluation. Using the marginal
density to reconstruct a selected-component weight loses its selection compensation. When reevaluating at another interaction,
use that interaction's component value and density; a saved proposal probability is not a substitute. Rejected
attempts contribute zero and must not be silently renormalized out of the sampling distribution.

Flags are behavioral data, not descriptive decoration. They distinguish reflection from transmission, delta from
continuous events, and curves from surfaces; renderers use them for hemisphere selection, MIS, path-state updates,
and medium transitions. A sampling-lobe hint may change the proposal and its PDF, but not the physical scattering
function. Reciprocity and energy behavior remain properties of individual models and transport modes rather than
blanket interface guarantees.

## Source landmarks

- `slang/falcor2/render/bsdf.slang` and `slang/falcor2/render/material.slang` - core records, directions, measures,
  fused operations, hints, aliases, flags, and material-instance interfaces.
- `slang/falcor2/render/config.slang` - renderer-owned sample-weight policy and default.
- `slang/falcor2/render/bsdfs/diffuse_specular_brdf.slang` - differentiable fused result and concrete aliases.
- `slang/falcor2/render/bsdfs/standard_bsdf.slang`, `slang/falcor2/render/bsdfs/lambert_diffuse_brdf.slang`,
  `slang/falcor2/render/bsdfs/specular_microfacet.slang`, `slang/falcor2/render/bsdfs/proceduralcloth_bsdf.slang`,
  `slang/falcor2/render/bsdfs/gltf_specgloss_bsdf.slang`, and
  `slang/falcor2/render/bsdfs/scratch_conductor_brdf.slang` - continuous, delta, component, and full-response behavior.
- `slang/falcor2/render/materials/standard_material_instance.slang` and
  `slang/falcor2/render/materials/openpbr_material.slang` - local-frame adaptation and an upstream split evaluator.
- `slang/falcor2/render/materials/mtlx/interface.slang` and
  `slang/falcor2/render/materials/mtlx/bsdf_wrapper.slang` - layerable BSDFs and graph composition.
- `src/falcor2/render/material/mdl/mdl_material.cpp` - generated MDL material-instance evaluation and sampling.
- `slang/falcor2/rendernodes/reference_pathtracer.slang` and
  `slang/falcor2/tools/materials/scene_sampling_tools.slang` - production transport consumption and diagnostic entry
  points.
- `tests/python/render/scattering_tests.slang`, `tests/python/render/scattering_tests.py`,
  `tests/python/render/test_bsdfs.py`, and `tests/python/render/test_material_sampling.py` - shared component contracts,
  independent estimator means, component partitions, layout compatibility, roughness, and delta-measure checks.

## Related design documents

- [Sampling inputs](sampling-inputs.md) - coordinate ownership, fixed factory budgets and nested selection.
- [Falcor2 system overview](overview.md)
- [BSDF correctness and testing](bsdf-correctness-and-testing.md)
- [Render pipeline](render-pipeline.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [Falcor2 design documentation](index.md)
- [MaterialX integration](materialx-integration.md)
- [MDL integration](mdl-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
