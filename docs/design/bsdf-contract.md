---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-08
verified_against: "01b22e34e4715605fdd894a7dcbe736be8030736"
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

This page defines the direction, measure, evaluation, sampling, composition, and dispatch conventions shared by
Falcor2 scattering code. A PDF is a probability density function; for continuous outgoing directions its measure is
solid angle, with units per steradian. Multiple importance sampling (MIS) combines samples from different proposals
using their densities for the same direction.

Material ownership and device records belong in [Material system](material-system.md). Detailed MaterialX graph
lowering belongs in [MaterialX integration](materialx-integration.md), and renderer and light-sampler construction
belong in [Render pipeline](render-pipeline.md) and [Lighting and sampling](lighting-and-sampling.md). This page does
not catalogue every BSDF formula or assert that all models are reciprocal. The layered evidence used to validate
these semantics is described in [BSDF correctness and testing](bsdf-correctness-and-testing.md).

## Architecture

### Directions and scattering measure

`IBSDF` is the low-level Slang interface. Its operations use a local orthonormal frame whose normal, tangent, and
bitangent are positive Z, positive X, and positive Y. Both the incident direction `wi` and outgoing direction `wo`
point away from the shading point. Surface BSDFs receive `wi` in the positive hemisphere; reflection normally
returns to that hemisphere and transmission crosses it. Curve models, such as hair or cloth fibers, may operate over
the full sphere in their curve frame.

Surface `eval()` values already include outgoing foreshortening: they return `f(wi, wo) * abs(wo.z)`. A renderer
must not multiply by another surface cosine. Curve BSDFs return their model response with any curve-specific measure
terms and Jacobians included by that model, not an extra surface cosine. `BSDF_MIN_COS_THETA` provides one shared
epsilon for implementations that are undefined exactly at grazing incidence.

`BSDFContext` supplies the incidence and transmission indices of refraction. Its `sample_lobe_types_hint` is a soft
efficiency hint: a caller will discard events outside the mask, but ignoring the hint is still correct. An
implementation that honors it must report the density of the restricted proposal it actually sampled.

### Evaluation and sampling operations

`IBSDF.eval_with_pdf()` is the canonical evaluation operation and returns `BSDFEval`, containing the projected value
and the outgoing solid-angle PDF. `eval()` is a field-selecting convenience alias. The generator-free `eval_pdf()`
alias calls the same fused operation with a deterministic midpoint sample generator. Randomness may affect an
implementation's value calculation, but the PDF must be deterministic and independent of generator state.

`sample()` chooses `wo` and returns a `BSDFSample` only on success. The record carries a PDF, an RGB throughput
weight, and `BSDFFlags` identifying diffuse, glossy, or delta reflection or transmission, plus glossy curve events.
Continuous events expose their directional density. An ideal mirror or refraction is a discrete delta event rather
than a finite solid-angle density, so its scalar PDF is zero and its event probability is represented through the
sample weight and flags.

At the world-space boundary, `IMaterialInstance` exposes the same three operations through `MaterialSample`.
Its `eval_with_pdf<Hints>()` takes a compile-time `BSDFEvalHints` mask so aggregate implementations can omit value or
PDF work when only one field is requested. Implementations may compute extra fields, but callers cannot rely on
unrequested results.

### Composition and proposal selection

Composite BSDFs separate the physical coefficient of a lobe from the probability used to select it for sampling.
For continuous lobes selected with probabilities `p_j` and conditional densities `q_j`, the density visible to MIS
is the marginal sum `p_j * q_j` over every lobe that could produce the direction. Returning only the selected
conditional density would make a later MIS decision depend on hidden sampler state.

`StandardBSDF` combines diffuse reflection, diffuse transmission, microfacet reflection, and microfacet
reflection/transmission. It biases lobe selection using Fresnel terms and luminance, while fused evaluation sums the
physical values and marginal proposal density. MaterialX weight and frame wrappers scale values or rotate directions
without changing a solid-angle PDF under an orthonormal transform. Its layer, mix, and add wrappers choose proposals
using opacity, mix luminance, or albedo estimates, then combine physical values and marginal PDFs. Delta branches
return before continuous densities are folded in because discrete probability mass and solid-angle density are
different measures.

### Material-instance dispatch

A geometry hit supplies a synchronized `MaterialID`. `MaterialSystem` recovers the dynamic `IMaterial`, and
`setup_material_instance()` resolves textures, attributes, media, and the shading frame for that accepted hit.
Standard materials transform world-space directions into one local `StandardBSDF`, validate its hemispheres, and
transform sampled directions back. Advanced MaterialX evaluates graph lobes in their generated local frames and
aggregates them behind the same interface.

Adapters preserve the public contract even when an upstream model has a different API. OpenPBR prepares its BSDF
once and conditionally calls separate upstream value and PDF functions according to the requested hints. Generated
MDL uses its evaluate operation when a value is needed and retains a specialized upstream PDF operation for a
PDF-only request. The renderer therefore dispatches `IMaterialInstance`, not a concrete BSDF family.

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
 eval_with_pdf()  weight + PDF + flags
       |             |
     MIS weight   throughput, emitter MIS,
                  medium-boundary update
```

1. After traversal accepts a hit, the concrete material creates an instance conditioned on the surface interaction
   and current texture LOD policy. The instance owns the world-to-local conversion or upstream adapter needed by its
   scattering implementation.
2. For next-event estimation, the path tracer uses the material's lobe flags to decide which hemispheres can
   contribute. With MIS enabled, it evaluates a sampled light direction once through `eval_with_pdf()` and uses both
   the scattering value and BSDF PDF. Without MIS it requests only the value.
3. For path continuation, `sample()` supplies a direction and weight. The path tracer multiplies throughput by the
   weight and retains the PDF for MIS if the next segment reaches an emitter. Delta flags disable an incompatible
   continuous-density comparison; transmission flags drive interior-boundary tracking.
4. Implementations are expected to fail sampling rather than expose non-finite directions, negative or non-finite
   weights, or invalid densities. The generic tests require successful non-delta samples to have a positive density;
   successful delta samples instead require zero in the scalar PDF field and a matching delta flag.

## Invariants and rationale

Evaluation and PDF are one semantic calculation. Keeping `eval_with_pdf()` authoritative prevents separate formulas
from drifting and lets shared half-vector, distribution, Fresnel, or graph work be reused. Single-result aliases
remain useful at call sites, while compile-time hints prevent large material aggregates from doing unnecessary work.
The deterministic fallback is safe only because PDF results cannot depend on random generator state.

Probability measure is part of the interface. Continuous PDFs are non-negative solid-angle densities, while delta
events report zero and carry their discrete selection compensation in the sample weight. Composite non-delta PDFs
must be marginal over hidden branch choices. These rules let a generic renderer store one PDF for later MIS without
knowing which concrete BSDF or internal lobe produced the sample.

The declared `BSDFSample` and `MaterialSample` contract says that `weight` is projected `eval / pdf`. Most simple
continuous BSDFs and the non-delta MaterialX composition wrappers collapse to that value. Current `StandardBSDF`,
the advanced MaterialX root, `ProceduralClothBSDF`, and the MaterialX scratch model instead retain a
selection-compensated branch estimator while reporting the marginal PDF used by MIS. That estimator can remain
unbiased, but its weight is not generally equal to aggregate `eval / pdf` when lobes overlap. This is an explicit
current contract ambiguity: generic callers must not diagnose those implementations solely by that equality, and
the test suite deliberately exempts their weights while separately checking relevant PDFs. Choosing one universal
weight form remains unresolved.

Flags are behavioral data, not descriptive decoration. They distinguish reflection from transmission, delta from
continuous events, and curves from surfaces; renderers use them for hemisphere selection, MIS, path-state updates,
and medium transitions. A sampling-lobe hint may change the proposal and its PDF, but not the physical scattering
function. Reciprocity and energy behavior remain properties of individual models and transport modes rather than
blanket interface guarantees.

## Source landmarks

- `slang/falcor2/render/bsdf.slang` and `slang/falcor2/render/material.slang` - core records, directions, measures,
  fused operations, hints, aliases, flags, and material-instance interfaces.
- `slang/falcor2/render/bsdfs/standard_bsdf.slang`, `slang/falcor2/render/bsdfs/lambert_diffuse_brdf.slang`,
  `slang/falcor2/render/bsdfs/specular_microfacet.slang`, `slang/falcor2/render/bsdfs/proceduralcloth_bsdf.slang`,
  and `slang/falcor2/render/bsdfs/scratch_conductor_brdf.slang` - continuous, delta, composite, and
  selection-compensated native behavior.
- `slang/falcor2/render/materials/standard_material_instance.slang` and
  `slang/falcor2/render/materials/openpbr_material.slang` - local-frame adaptation and an upstream split evaluator.
- `slang/falcor2/render/materials/mtlx/interface.slang`,
  `slang/falcor2/render/materials/mtlx/bsdf_wrapper.slang`, and
  `src/falcor2/render/material/mtlx/codegen/snippets/methods/basic/basic_material_instance.slang` - layerable BSDFs,
  graph composition, and generated material-instance dispatch.
- `src/falcor2/render/material/mdl/mdl_material.cpp` - generated MDL material-instance evaluation and sampling.
- `slang/falcor2/rendernodes/reference_pathtracer.slang` and
  `slang/falcor2/tools/materials/scene_sampling_tools.slang` - production transport consumption and diagnostic entry
  points.
- `tests/python/render/bsdf_tests.slang`, `tests/python/render/bsdf_tests.py`,
  `tests/python/render/test_bsdfs.py`, and `tests/python/render/test_materials.py` - distribution, sample consistency,
  reciprocity, fused result, flags, lobe-hint, and delta-measure checks.

## Related design documents

- [Falcor2 system overview](overview.md)
- [BSDF correctness and testing](bsdf-correctness-and-testing.md)
- [Render pipeline](render-pipeline.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [Falcor2 design documentation](index.md)
- [MaterialX integration](materialx-integration.md)
- [Lighting and sampling](lighting-and-sampling.md)
