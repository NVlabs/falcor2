---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "c4799c96cf61c2c0053285e0e8b1468d9aa42db3"
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

A contributor familiar with Monte Carlo rendering can use this page to choose evidence for a scattering change,
locate the checks that exercise its implementation boundary, and interpret their results. It connects model
authorities, leaf properties, material-instance dispatch and rendered transport, including what a passing check
does and does not establish.

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

Independence has several meanings. Separate random streams prevent a mean comparison from reusing the sampler's
directions, but the reference integral still evaluates Falcor2's implementation. An analytic expectation, such as
the smooth dielectric's normal-incidence Fresnel probability and transmission scaling, supplies a different kind
of check. Stored material expectations detect regressions without independently establishing the model's formula.

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
- The sample contract rejects non-finite directions, negative or non-finite weights and densities, and nonunit
  directions. For continuous events, it compares `sample.pdf` with the full-response marginal density, then
  reconstructs weight using the returned
  component's value and joint density. A full-response marker uses the full evaluation for reconstruction.
  Delta samples require a delta flag and zero scalar solid-angle PDF; this does not establish that the flag names
  the physically correct event. Their integrated weights need separate checks.
- The independent estimator-mean test compares mean continuous sample weights with a uniform-sphere integral of
  full-response evaluation, using separate random streams. Rejections and delta events contribute zero to the
  continuous estimate, and every attempt remains in the denominator. The acceptance bound is six estimated standard
  errors plus a small absolute tolerance. This detects some errors that a self-consistent value/PDF pair cannot.
- Reciprocity compares evaluations with incident and outgoing directions exchanged, using the required cosine
  conversion for supported reflection cases. The runner reports valid and skipped counts; zero valid cases produces a warning, not a failure.
  A passing status without valid cases supplies no reciprocity evidence.

`SampleContractTest` counts rejected attempts separately from successful continuous, delta and invalid samples.
It checks marginal-PDF agreement independently of component-weight reconstruction. Its default also requires at
least one continuous event; a dedicated delta test can use `require_continuous=False`. It does not validate the
contents of a rejected sample record, which the caller must discard.

### Deterministic cases and coverage exceptions

Property tests are complemented by deterministic kernels. Explicit `BSDFSampleInput` coordinates control proposal
choices and random variates so tests can reach exact branches, near-smooth limits, lobe-hint restrictions, delta
events, and formula-specific cases that random sampling would hit unreliably. These targeted checks include albedo,
branch probabilities, dielectric and microfacet behavior, and selected MaterialX lobes.

The model matrix separates property applicability from known defects. Pure delta models do not participate in
continuous chi-square tests, and reciprocity is excluded where it is inapplicable. Other exclusions limit present
coverage: the cloth configuration omits chi-square as well as reciprocity, while retaining contract and mean checks.
Selected-component and full-response estimators use the same contract checker.

Two numerical exceptions need particular care when interpreting coverage:

| Model | Disabled checks | Checks retained |
| --- | --- | --- |
| `DielectricPlateBSDF` | Chi-square and the entire sample-contract check, for a documented sampling/density mismatch | Other applicable model-matrix checks; no contract pass is implied |
| `mtlx::MxGeneralizedSchlickAiryBSDF` | Component-weight reconstruction because of a thin-film Fresnel mismatch | Payload validation, marginal-PDF agreement and applicable distribution checks |

`_CONTRACT_SKIP_REASONS` and `_CONTRACT_WEIGHT_EXCEPTIONS` in `test_bsdfs.py` make the distinction explicit. The
Airy exception uses `check_weight=False`, which still checks finite evaluation and positive continuous density.
An exclusion documents missing evidence; it does not certify the excluded behavior.

Independent means use broad lobes because uniform-sphere integration cannot reliably resolve arbitrarily narrow
peaks. Cloth and scratch participate at normal and oblique incidence. Smooth dielectric tests separately check
analytic Fresnel event frequencies and integrated weights, including the index-of-refraction scaling for
transmission. Mixed diffuse/delta tests check that discrete mass is excluded from continuous PDFs.

Synthetic targets in `scattering_test_fixtures.slang` deliberately return wrong component identities, weights,
joint densities or marginal densities, including mismatches between sampling frequency and reported density.
`test_scattering_contract.py` verifies that the
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
policy checks do not require reconstruction. The policy-comparison fixtures link layered MDL and mixed Standard
materials under default, native and evaluation-consistent policies. They check payload/PDF validity, unchanged
sampled events and marginal PDFs, native default behavior, and the layered MDL weight difference. Independent mean
fixtures exercise both policies for selected custom, glTF, Standard and MDL materials. These are bounded cases,
not coverage of every model and parameter combination.

Component-partition checks sum fixed slots at arbitrary directions and compare both value and density with full
response. Additional checks exercise unknown and inactive indices for component-aware implementations,
weighting/frame forwarding, layout IDs,
and component roughness.

The shared direct-BSDF and material test adapters assemble `BSDFEval` from the value and PDF aliases. Those aliases
derive from canonical fused evaluation; agreement among them is not an independent numerical oracle. In contrast,
the all-material result test calls the fused scene tool directly and checks both fields against stored expectations.

Native material tests cover records, reflected metadata and update behavior. Focused Python and Slang tests cover
generated MaterialX graph behavior such as repaired lobe normals, while MDL pipeline tests verify compilation and
scene-state delivery. These establish integration behavior, not every equation emitted by an external generator.

### Rendered integration

Path-tracer tests are the final consumer-level layer. They exercise direct lighting, transmission, MIS, media,
delta events and generated materials in complete paths. Their assertions differ in strength: the environment-hit
MIS case checks finiteness and nonzero output, while the homogeneous-extinction case compares rendered attenuation
with the Beer-Lambert expectation. Neither assertion substitutes for the other. Rendered tests can reveal a missing
cosine, wrong measure, bad flag or broken boundary update; image agreement can also conceal compensating errors.

### Execution selection and evidence

Regular and nightly tiers control which parameter/backend combinations run, independently of which mathematical
property a test checks. Representative cases remain regular while expensive or repeated matrices are deferred.
The BSDF matrix uses backend-conditional and model-specific nightly marks; material compatibility and mean checks
also include nightly cases. A successful selected run therefore does not imply the complete matrix ran. Capability
skips and deselected cases supply no passing evidence. [Test tiers](../../tests/README.md) owns selection details;
reports need the actual cases and backends exercised.

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

Fused evaluation is the canonical semantic operation. Its value and PDF need meaningful expectations. Comparing
`eval()` or `eval_pdf()` aliases with their fused source only verifies field selection. Numerical agreement also
does not establish GPU efficiency; performance needs separate measurements of the current generated code.

Under `eval_consistent`, the material sample's component determines weight reconstruction. A selected-component
weight can differ from aggregate `eval / sample.pdf` without violating the contract: its denominator is the joint
component/direction density. Checking the marginal density separately preserves the MIS contract. Neither
reconstruction nor partition sums
prove that the sampler actually follows its reported density, which is why distribution and independent mean
checks are also needed. A mean-test pass is statistical evidence for the tested configuration, not proof of
unbiasedness over every parameter or incident direction.

Reciprocity, energy behavior, and sampling domains are model properties rather than universal interface promises.
Each generic assertion must state where it applies, and each skip must remain as narrow and explanatory as possible.
Tolerance or statistical changes should reflect numerical analysis, sample count, and backend behavior, not hide a
repeatable model mismatch. Running shared shader tests across supported device backends matters because compiler,
code-generation, and floating-point behavior are part of the exercised implementation.

## Source landmarks

- `slang/falcor2/render/bsdf.slang`, `slang/falcor2/render/material.slang` and
  `slang/falcor2/render/config.slang` - scattering contracts, components and material weight policy.
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
- `tests/native/render/test_materials.cpp` - native material records, reflected metadata and update behavior.
- `tests/python/render/test_materialx_lobe_normal_repair.py` and
  `tests/python/render/test_materialx_lobe_normal_repair.slang` - targeted generated MaterialX graph behavior.
- `tests/python/render/test_mdl_generated_material_pipeline.py` - generated MDL compilation and state integration.
- `tests/python/pathtracer/test_pathtracer.py` - rendered transport interactions involving BSDF values, PDFs, flags,
  media, and MIS.
- `tests/README.md` and `tests/python/conftest.py` - execution tiers and collection controls, separate from the
  numerical assertions above.

## Related design documents

- [BSDF contract](bsdf-contract.md)
- [Material system](material-system.md)
- [MaterialX integration](materialx-integration.md)
- [Sampling inputs](sampling-inputs.md)
- [Render pipeline](render-pipeline.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Falcor2 design documentation](index.md)
