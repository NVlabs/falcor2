# MX139 Static Input Query

This directory owns reusable static input analysis for MX139 MaterialX graphs.
It answers positive exact-value questions for graph inputs and outputs without
mutating the graph or depending on pruning counters.

Current files:

- `static_values.h`: static value states for closure values and scalar/color
  values.
- `static_value_analysis.h`: per-output static-value analysis result and the
  static-only analysis entry point.
- `static_input_query.*`: convenience query object for asking whether a node
  input is exactly a requested value after interface and connection propagation.
- `input_static_values.h`: helpers for lifting MaterialX input values into
  static values.
- `call_site_index.*`: deterministic reachable-graph and shared-subgraph
  call-site indexing.
- `operator_match.h`: narrow MaterialX implementation-family and closure-domain
  predicates shared by analysis consumers.

## Exactness Policy

In this directory, "value" means the conservative static value known to the
analysis, not necessarily the runtime shader value.

Closure values:

- `KnownEmpty`: statically zero closure contribution.
- `KnownNonEmpty`: statically known to carry contribution.
- `Unknown`: unknown or mixed; pruning must not assume empty.

Scalar/color values:

- `KnownZero`: exactly zero in every component.
- `KnownOne`: exactly one in every component.
- `KnownConstant`: exact nonzero/nonone scalar, vector, or color value.
- `Unknown`: unknown, unsupported, or not proven exact.

Do not treat near-zero as zero. Do not use NaN, infinity, or an unsupported
value as proof; unsupported values should become `Unknown`. Consumers should
act only on a positive exact result and skip otherwise.

## Shared Graph Rule

MaterialX 1.39 nodegraphs may be shared by multiple reachable call sites. A
static value inside a shared graph is only exact if every currently reachable
call site agrees after values are merged into the callee input environment.

Do not flatten, clone, or specialize call sites to prove a value in this
implementation.

## Adding A Static Value Rule

Add static value rules here, not in pruning consumers.

Checklist:

1. Keep the rule mutation-free.
2. Return `Unknown` for unsupported, runtime, non-finite, or ambiguous values.
3. Merge values across reachable call sites.
4. Add focused native tests for direct values, connected values, shared
   nodegraph call sites, and unknown fallback.
5. Verify source identity if the slice can affect code generation.

Current non-pruning consumer: MX139 Fresnel policy selection in
`policy.cpp`, which uses `StaticInputQuery` to select the cheapest
generalized Schlick Fresnel implementation proven valid by exact static values.
