# MX139 Graph Prepare

This directory owns MaterialX graph preparation passes used before MX139 code
generation. Single-pair preparation actors live here directly. Multi-file
capabilities live in subdirectories named after the capability, so ownership is
visible from the file tree.

Current modules:

- `static_input_query/`: reusable static input analysis shared by graph-prepare
  consumers.
- `closure_pruning/`: public pruning driver, pruning analysis, graph
  simplification, counters/report data, and pruning-specific test hooks.

Current single-pair actors:

- `default_geomprop_materialization.*`: materializes reachable default
  geomprop inputs.

Shared helpers:

- `core_reachability.*`: upstream reachability over mutable MaterialX Core
  document nodes. This is separate from ShaderGraph call-site reachability in
  `static_input_query/`.

## Closure Pruning Flow

Production pruning mode:

```text
repeat:
  -> rebuild the reachable call-site index for the current graph
  -> analyze the current graph with analyze_closure_pruning
  -> apply one pruning pass from that classification snapshot
  -> if the graph changed, remove unused nodes and repair topology
until no mutation occurred and classification plus static values are stable
```

`run_closure_pruning` owns mode dispatch, pruning-counter reset, iteration
accounting, repeat-until-stable termination, and final top-shader collapse
checks. `make_editable` disables the pass effectively even when a pruning mode
is requested.

The analysis must never mutate `ShaderGraph`, must not require
`CodegenUserData`, and should stay usable by non-pruning consumers after the
static-value refactor. Non-pruning consumers should include
`static_input_query/static_value_analysis.h` and use `analyze_static_values`;
pruning consumers use `closure_pruning/classify.h` and read the
`ClosurePruningAnalysisResult::pruning` side data.

## Pruning Consumer Boundaries

`ClosurePruningAnalysisResult` separates reusable static values from pruning data:

- `static_values.output_static_values`: conservative per-output closure/scalar
  values that non-pruning consumers may use.
- `pruning.classification` and `pruning.asymmetric_mix_fold_plans`: pruning-only
  decisions consumed by `closure_pruning/closure_simplification.*`.

The pruning consumer may rewire graph connections, materialize helper nodes,
call `removeUnusedNodes()`, repair topology after mutation, and increment
pruning counters.

The analysis layer must not create nodes, delete nodes, rewire outputs, call
`removeUnusedNodes()`, write pruning counters, or depend on `CodegenUserData`.

## Closure Algebra Policy

MX139 pruning follows OSL-style closure algebra where the meaning is generic
closure composition. It uses local MX139 policy where MaterialX layering or
current codegen support is more specific than core OSL closure algebra.

Accepted identities include exact zero/one closure multiply, additive identity
for empty closures, exact mix endpoints, asymmetric mix rewrites into multiply
helpers, layer identities with empty BSDF/EDF closures, and the current narrow
`layer_vdf(top: BSDF, base: VDF)` boundary preservation rule.

These identities are about closure contribution only. They do not remove or
rewrite opacity, presence, cutout, displacement, volume, physical layer
thickness, or other aggregate shader state.

## VDF Policy

MX139 VDF pruning support is intentionally narrow. VDF constructor nodes such
as `absorption_vdf` and `anisotropic_vdf` may feed codegen, but dead-closure
pruning only has one VDF-specific rule: preserve the MaterialX boundary
`layer_vdf(top: BSDF, base: VDF) -> BSDF`.

- An `IM_layer_vdf*` node with BSDF output and a connected VDF `base` input is
  `Keep`, even when its BSDF `top` is statically `KnownEmpty`.
- A missing or explicitly unconnected VDF `base` is not a live boundary; the
  node falls back to the ordinary layer rule.
- Direct VDF closure algebra such as `add_vdf`, `mix_vdf`, or `multiply_vdf` is
  not part of MX139 pruning today.

Top-shader all-empty diagnostics still ignore VDF-typed inputs and
`volumeshader` producers because those are not surface BSDF/EDF closure
channels.

## Readability Rules

This code is for graphics and AI researchers, not compiler specialists. Future
refactors should keep the MaterialX graph story visible.

- Use domain names for phases: call-site index, static values, classification,
  rewrite plan, pruning mutation, and graph repair.
- Preserve naming boundaries: `classify_*` is mutation-free, `scan_*` is
  read-only, `rewrite_*` and `apply_*` may mutate, and `*_static_value*` must
  not depend on pruning counters or `CodegenUserData`.
- Keep fixed-point loop state explicit near the loop.
- Prefer one small function per closure identity family over a registry or
  generic rule engine.
- Comments should explain MaterialX/rendering invariants and mutation risks,
  not compiler-framework vocabulary.
