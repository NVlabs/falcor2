---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-09
verified_against: "1f258f94a34e029c86b0901513d741b3809ecc48"
---

(falcor2-design-bsdf-correctness-and-testing)=
# BSDF correctness and testing

Falcor2 treats bidirectional scattering distribution function (BSDF) correctness as a layered evidence problem.
Reference formulas and interface rules establish intended behavior; deterministic tests isolate boundary cases;
statistical GPU tests exercise sampling distributions and shared properties; material-system tests verify generated
and dynamically dispatched implementations; and rendered tests expose transport-level failures. No layer proves a
model correct by itself. In particular, agreement between a sampler and its own PDF can preserve the same error in
both. The test architecture therefore combines independent checks while recording model-specific exclusions and
unresolved contract questions explicitly.

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

### Leaf-BSDF harness

`tests/python/render/test_bsdfs.py` is the model matrix for reusable leaf checks. Each configuration identifies a
Slang BSDF type, bindings, incident directions, context, and explicit exclusions. Python drives SlangPy kernels from
`tests/python/render/bsdf_tests.slang`, so the same harness exercises shader implementations on the configured GPU
backends rather than substituting host-language replicas.

The harness provides three broad properties:

- The chi-square test compares sampled spherical bins with probability mass obtained by integrating the reported
  PDF. It pools bins with low expected counts, rejects impossible or out-of-domain samples, and applies a statistical
  acceptance threshold.
- Evaluation/sample consistency rejects non-finite or negative payloads, checks continuous sample PDFs against
  evaluation PDFs, and checks `eval / pdf` against sample weight where that equality is part of the model's current
  contract. Delta samples instead require a delta flag and zero scalar solid-angle PDF.
- Reciprocity compares opposite-direction projected evaluations with the required cosine conversion for models and
  modes where reciprocity applies.

Property tests are complemented by deterministic kernels. `FixedSampleGenerator` controls proposal choices and
random variates so tests can reach exact branches, near-smooth limits, lobe-hint restrictions, delta events, and
formula-specific cases that random sampling would hit unreliably. These targeted checks include albedo, branch
probabilities, dielectric and microfacet behavior, and selected MaterialX lobes.

An exclusion in the model matrix is an applicability statement, not proof and not a blanket waiver. Pure delta
models do not participate in continuous chi-square tests. Non-reciprocal models do not participate in reciprocity
checks. Known contract differences or defects are excluded only from the particular assertion they invalidate, so
the remaining properties continue to run.

### Material and generated-code layers

Leaf tests bind a BSDF type directly. `tests/python/render/test_materials.py` adds the material boundary: it builds
surface interactions, asks material instances to evaluate or sample, and compares fused values, PDFs, flags, and
selected properties with stored expectations. The helpers in
`slang/falcor2/tools/materials/scene_sampling_tools.slang` exercise both direct cursor-writer bindings and
`MaterialSystem` dynamic dispatch after scene synchronization. This catches setup, frame, resource, specialization,
and dispatch faults that a leaf test cannot see.

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

The sample-weight rule remains deliberately narrow where the runtime contract is unresolved. `StandardBSDF`, the
advanced MaterialX root, `ProceduralClothBSDF`, and the MaterialX scratch model currently use
selection-compensated branch estimators that need not equal aggregate `eval / pdf` when lobes overlap. Their
relevant generic equality checks are excluded, but marginal PDF behavior required for MIS remains independently
testable. The repository backlog retains the owner decision about whether all models should eventually use one
weight form; this page does not resolve it or present proposed behavior as current.

Reciprocity, energy behavior, and sampling domains are model properties rather than universal interface promises.
Each generic assertion must state where it applies, and each skip must remain as narrow and explanatory as possible.
Tolerance or statistical changes should reflect numerical analysis, sample count, and backend behavior, not hide a
repeatable model mismatch. Running shared shader tests across supported device backends matters because compiler,
code-generation, and floating-point behavior are part of the exercised implementation.

## Source landmarks

- `slang/falcor2/render/bsdf.slang` - shared BSDF records, measures, operations, flags, and hints.
- `tests/python/render/bsdf_tests.py`, `tests/python/render/bsdf_tests.slang`, and
  `tests/python/render/test_bsdfs.py` - reusable GPU harness, model matrix, property checks, and targeted cases.
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
