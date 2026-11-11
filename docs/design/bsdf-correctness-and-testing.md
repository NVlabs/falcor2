---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-06
verified_against: "2ac1506f0384e5691c17a0c98cef21f1a9663f66"
---

(falcor2-design-bsdf-correctness-and-testing)=
# BSDF correctness and testing

Falcor2 treats bidirectional scattering distribution function (BSDF) correctness as a layered evidence problem.
Reference formulas and interface rules establish intended behavior; deterministic tests isolate boundary cases;
statistical GPU tests exercise sampling distributions and shared properties; material-system tests verify generated
and dynamically dispatched implementations; and rendered tests expose transport-level failures. No layer proves a
model correct by itself. In particular, agreement between a sampler and its own PDF can preserve the same error in
both. The test architecture therefore combines independent checks while recording model-specific exclusions and
numerical defects explicitly.

## Purpose and scope

This page explains how Falcor2 establishes and maintains confidence in BSDF implementations. It covers the evidence
hierarchy, the reusable leaf-BSDF harness, material-instance checks, generated-code checks, and rendered integration
tests. It also defines what a passing test does and does not establish.

The runtime meanings of evaluation, sampling, PDFs, flags, composition, and dispatch are defined by the
[BSDF contract](bsdf-contract.md). Material ownership belongs in [Material system](material-system.md), and the
path tracer's use of scattering values belongs in [Render pipeline](render-pipeline.md). This page is not a catalog
of every test case, a replacement for model specifications, or a claim that every Falcor2 model is reciprocal,
energy conserving, or independently verified against an external authority.

## Architecture

### Correctness authorities

The current implementation is authoritative for what Falcor2 does, but not automatically for what a physical model
should do. Shared interface semantics come from the BSDF contract. Model-specific equations, parameter domains, and
intent should be checked against a named paper, standard, upstream implementation, or other stable authority when
one exists. Tests then encode observable consequences of both levels.

A review can consequently reach different conclusions: behavior may be verified against an authority, confirmed
to violate it, intentionally different and documented, or still unverified because the necessary reference or
coverage is missing. Internal consistency alone is weaker evidence. For example, a sampler and evaluator copied
from the same mistaken formula may agree perfectly while implementing the wrong distribution.

### Shared scattering checks

`tests/python/render/test_bsdfs.py` is the model matrix for reusable leaf checks. Each configuration identifies a
Slang BSDF type, bindings, incident directions, context, and explicit exclusions. Direct BSDF and scene-backed
material adapters implement `IScatteringTestTarget` from `scattering_tests.slang`, exposing sampling and
component-aware evaluation in one orthonormal frame. The shared Python runners in `scattering_tests.py` drive
SlangPy kernels on the configured GPU backends rather than substituting host-language replicas.

The harness separates four properties:

- The chi-square test compares sampled spherical bins with probability mass obtained by integrating the reported
  PDF. It pools bins with low expected counts, rejects impossible or out-of-domain samples, and applies a statistical
  acceptance threshold.
- The sample contract rejects non-finite or negative payloads and nonunit directions. For continuous events, it
  compares `sample.pdf` with the full-response marginal density, then reconstructs weight using the returned
  component's value and joint density. A full-response marker uses the full evaluation for reconstruction.
  Delta samples require a delta flag and zero scalar solid-angle PDF; their integrated weights need separate checks.
- The independent estimator-mean test compares mean continuous sample weights with a uniform-sphere integral of
  full-response evaluation, using separate random streams. Rejections and delta events contribute zero to the
  continuous estimate, and every attempt remains in the denominator. The acceptance bound is six estimated standard
  errors plus a small absolute tolerance. This detects some errors that a self-consistent value/PDF pair cannot.
- Reciprocity compares opposite-direction projected evaluations with the required cosine conversion for models and
  modes where reciprocity applies.

Property tests are complemented by deterministic kernels. Explicit `BSDFSampleInput` coordinates control proposal
choices and random variates so tests can reach exact branches, near-smooth limits, lobe-hint restrictions, delta
events, and formula-specific cases that random sampling would hit unreliably. These targeted checks include albedo,
branch probabilities, dielectric and microfacet behavior, and selected MaterialX lobes.

An exclusion in the model matrix is an applicability statement, not proof and not a blanket waiver. Pure delta
models do not participate in continuous chi-square tests. Non-reciprocal models do not participate in reciprocity
checks. Selected-component and full-response estimators use the same contract checker. Known numerical defects are
excluded only from the particular assertion they invalidate, so the remaining properties continue to run. The
plate density mismatch currently excludes that model's contract check; the generalized-Schlick thin-film mismatch
excludes weight reconstruction while retaining payload and marginal-PDF checks.

Independent means use broad lobes because uniform-sphere integration cannot reliably resolve arbitrarily narrow
peaks. Cloth and scratch participate at normal and oblique incidence. Smooth dielectric tests separately check
analytic Fresnel event frequencies and integrated weights, including the index-of-refraction scaling for
transmission. Mixed diffuse/delta tests check that discrete mass is excluded from continuous PDFs.

Synthetic targets in `scattering_test_fixtures.slang` deliberately return wrong component identities, weights,
joint densities, marginal densities, or sampling frequencies. `test_scattering_contract.py` verifies that the
corresponding checker fails. These negative witnesses test the harness itself; consistency, distribution, and mean
checks remain separate evidence.

### Material and generated-code layers

Leaf tests bind a BSDF type directly. `tests/python/render/test_materials.py` adds the material boundary: it builds
surface interactions, asks material instances to evaluate or sample, and compares fused values, PDFs, flags, and
selected properties with stored expectations. The helpers in
`slang/falcor2/tools/materials/scene_sampling_tools.slang` exercise both direct cursor-writer bindings and
`MaterialSystem` dynamic dispatch after scene synchronization. This catches setup, frame, resource, specialization,
and dispatch faults that a leaf test cannot see.

`test_material_sampling.py` uses the shared contract and mean runners through production `MaterialSystem` dispatch.
Its fixtures cover native, custom Python, generated MaterialX, and MDL materials. Primary MaterialX cases use the
build's default layering mode; compatibility cases explicitly select the other modes. Material reconstruction
checks explicitly link `MATERIAL_SAMPLE_WEIGHT_POLICY` as `MaterialSampleWeightPolicy::eval_consistent`. Native
policy checks retain payload/PDF validation and independent estimator means without requiring reconstruction.
Policy tests link the same scene with the default, native, and evaluation-consistent policies, checking unchanged
sampled directions and marginal PDFs, native default behavior, and the layered MDL weight difference.
Component-partition checks
sum fixed slots at arbitrary directions and compare both value and density with full response. Additional checks
exercise unknown and inactive indices for component-aware implementations, weighting/frame forwarding, layout IDs,
and component roughness.

Native material tests cover host/device records, metadata, and generated material classes. Focused Python and Slang
tests cover generated MaterialX graph behavior such as repaired lobe normals, while MDL pipeline tests verify that
generated classes compile and receive the required state. These tests establish integration behavior; they do not
by themselves validate every equation emitted by an external code generator.

### Rendered integration

Path-tracer tests are the final consumer-level layer. They exercise combinations such as direct lighting,
transmission, emissive and environment MIS, media transitions, delta reflection, Russian roulette, and generated
materials in complete paths. A rendered result can reveal a missing cosine, wrong measure, bad flag, or broken
boundary update that isolated calls miss. Conversely, image agreement is a coarse oracle and can conceal
compensating errors, so rendered tests supplement rather than replace focused mathematical checks.

## Runtime and test flow

```text
 Interface contract + model authority
                 |
                 v
       deterministic leaf cases
                 |
                 v
       generic GPU properties
                 |
                 v
     material-instance dispatch
                 |
                 v
       rendered path transport
```

1. A leaf configuration selects a BSDF, bindings, incident directions, and the properties that apply to that model.
2. SlangPy compiles the shared kernels for a configured device backend. Deterministic tests inject selected random
   values; broad tests generate many samples on the GPU.
3. GPU reductions report invalid payloads and worst consistency or reciprocity errors. For distribution tests,
   Python compares the sampled histogram with integrated PDF mass and produces diagnostic data on failure.
4. Material tests construct synthetic surface interactions and repeat the public operations through direct and
   dynamically selected material instances, including generated implementations.
5. Path-tracer tests consume the same values, PDFs, weights, and flags in full transport to check their interaction
   with lights, media, accumulation, and presentation-independent output.

## Invariants and rationale

Every non-delta successful sample must expose finite direction and weight data and a valid continuous density. Delta
events occupy a discrete measure: they must be identified by flags and report zero in the scalar solid-angle PDF.
Mixing those measures in one statistical comparison produces meaningless results, so continuous-distribution tests
exclude delta mass while generic payload checks still cover delta events.

A chi-square pass establishes agreement between sampling frequency and the reported PDF within the chosen
statistical test; it does not establish that either matches the intended scattering model. Likewise, a stored
expected value can preserve an old mistake. Formula corrections therefore need an independent authority or a test
whose expected behavior follows from a separate invariant. Useful regression cases distinguish a plausible wrong
implementation from the accepted one instead of merely asserting a visible constant, clamp, or branch.

Fused evaluation is the canonical semantic operation. Tests should inspect its externally observable value and PDF
against meaningful expectations. Comparing `eval()` or `eval_pdf()` aliases with the fused operation after those
aliases delegate to it is tautological and does not provide independent evidence, so temporary parity scaffolding is
not part of the permanent suite.

Under `eval_consistent`, the material sample's component determines weight reconstruction. A selected-component
weight can differ from aggregate `eval / sample.pdf` without violating the contract: its denominator is the joint
component/direction density.
Checking the marginal density separately preserves the MIS contract. Neither reconstruction nor partition sums
prove that the sampler actually follows its reported density, which is why distribution and independent mean
checks are also needed. A mean-test pass is statistical evidence for the tested configuration, not proof of
unbiasedness over every parameter or incident direction.

Reciprocity, energy behavior, and sampling domains are model properties rather than universal interface promises.
Each generic assertion must state where it applies, and each skip must remain as narrow and explanatory as possible.
Tolerance or statistical changes should reflect numerical analysis, sample count, and backend behavior, not hide a
repeatable model mismatch. Running shared shader tests across supported device backends matters because compiler,
code-generation, and floating-point behavior are part of the exercised implementation.

## Source landmarks

- `slang/falcor2/render/bsdf.slang` - shared BSDF records, measures, operations, flags, and hints.
- `tests/python/render/bsdf_tests.py`, `tests/python/render/bsdf_tests.slang`, and
  `tests/python/render/test_bsdfs.py` - direct-BSDF adapters, model matrix, property checks, and targeted cases.
- `tests/python/render/scattering_tests.py`, `tests/python/render/scattering_tests.slang`,
  `tests/python/render/scattering_test_fixtures.slang`, and `tests/python/render/test_scattering_contract.py` - shared
  contract/mean runners and negative witnesses.
- `tests/python/render/material_sampling_tests.slang` and `tests/python/render/test_material_sampling.py` - production
  material sampling, component partitions, layout IDs, roughness, and independent means.
- `falcor2/utils/chi2.py` and `slang/falcor2/utils/chi2.slang` - statistical decision logic, diagnostics, histogram
  generation, and PDF integration.
- `slang/falcor2/tools/materials/scene_sampling_tools.slang` and `tests/python/render/test_materials.py` - direct and
  dynamically dispatched material-instance checks.
- `tests/native/render/test_materials.cpp` - native material records, metadata, and generated-class coverage.
- `tests/python/render/test_materialx_lobe_normal_repair.py` and
  `tests/python/render/test_materialx_lobe_normal_repair.slang` - targeted generated MaterialX graph behavior.
- `tests/python/render/test_mdl_generated_material_pipeline.py` - generated MDL compilation and state integration.
- `tests/python/pathtracer/test_pathtracer.py` - rendered transport interactions involving BSDF values, PDFs, flags,
  media, and MIS.

## Related design documents

- [BSDF contract](bsdf-contract.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Render pipeline](render-pipeline.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Falcor2 design documentation](index.md)
