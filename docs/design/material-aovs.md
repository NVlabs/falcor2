---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-18
verified_against: "2a5bad67ea7ca53a3d4a44bdd475848dd9d51e8e"
---

(falcor2-design-material-aovs)=
# Named material outputs

Falcor2 material arbitrary output variables (AOVs) carry authored numeric values, such as masks or mixing weights,
from material evaluation into renderer-selected images or tensors. A scene-owned catalog gives each declared name
a permanent numeric ID and type. MaterialX and MDL generate supported outputs independently of consumers; the
reference path tracer resolves requested names after ordinary scene preparation and supplies storage through its
guide-output machinery. Consumer changes affect capture targets without regenerating materials. The shader writer
belongs to `IMaterial` and evaluates from a surface interaction without constructing a scattering instance.
MaterialX writes local graph results and MDL evaluates expressions against locally initialized state;
`IMaterialInstance` carries no AOV writer or retained MaterialX AOV fields.

## Purpose and scope

This page explains how a named material value reaches a guide buffer: declaration, code generation, lookup, storage,
and primary-hit delivery. General material dispatch, BSDF semantics, and pipeline composition belong to the related
pages.

Material AOVs are separate from fixed `MaterialProperties` guides. Runtime transport supports float32 values with
one through four components, including colors. A declaration allocates no output image. Values are per-dispatch
primary-surface results, rather than per-bounce accumulation or light-path expressions.

## Architecture

### Permanent typed declarations

`Scene` owns a `MaterialAOVRegistry`. Each entry is an `AOVDesc` containing `id`, an owned `name`, scalar `type`,
and `component_count`. Only producer registration creates entries. An exact, case-sensitive name receives a new
uint32 ID on first registration; compatible subsequent declarations return that ID. Empty names, unsupported
types or widths, and incompatible redeclarations fail.

The catalog is append-only. Removing a material, changing its output selection, or failing later compilation does
not erase successful declarations. A known name remains attachable even when no live material writes it. IDs are
never recycled within a catalog and have no meaning across scenes or serialized sessions. Permanent type agreement
protects compiled shaders and cached consumer metadata from an ID silently acquiring a different interpretation.

The registry tracks declarations, not producer or consumer lifetimes. Code generators register outputs directly;
host `Material` retains no AOV handles or output inventory. The registry holds no Scene, Material, renderer, or
GPU-resource references. A retained registry reference can outlive its Scene; copied descriptors can outlive the registry.

`find_output(name)` returns a copied declaration, or an empty optional exposed as Python `None`, without allocating
an ID. `outputs` provides copied declarations in ID order for discovery; `has_output` checks whether a name has ever
been registered. Enumeration does not establish live producer availability. Registration and lookup are serialized.

### Consumer binding and caching

Assigning `guide_output_specs` records names and container specifications without accessing the catalog. `None`
disables a target. Binding happens during rendering, after scene preparation. An unknown enabled name fails before
dispatch without changing the registry; a later invocation can resolve it once a producer has registered it.
Normal rendering needs no notification subscription, automatic retry loop, or additional preparation phase.

`ReferencePathTracerNode` caches resolved IDs and component counts. Switching registry identity discards those
bindings, even if both catalogs have the same size. The catalog revision, `output_generation`, is its entry count
and changes only when a new ID is added. It invalidates cached fixed-guide collision checks; permanent custom AOV
bindings remain valid. An enabled fixed guide whose name also appears in the catalog is rejected as ambiguous,
including after the declaring material is removed.

Stable targets reuse `IndexedOutput` tuples. `OutputPrelude` caches sorted descriptors and signatures for immutable
tuples of texture targets. Mutable sequences and tensors are revalidated because their layout can change without
identity changing. Binding and clearing still happen on each dispatch.

### MaterialX and MDL generation

MaterialX selects its primary material independently of additional numeric outputs. Candidates come from the
selected node's NodeDef outputs and top-level document outputs, rather than arbitrary intermediate nodes. Numeric
outputs do not make a single material ambiguous or select another surface.

`CodeGenDesc.additional_output_names` defaults to `{"*"}`. Exact names and case-sensitive `*` patterns select
candidates; `*` matches zero or more characters, without regular-expression or full filesystem-glob syntax. An
empty list opts out. Unmatched patterns are allowed, while missing exact names and explicit closure selections fail.
Distinct selected candidates with the same name are rejected.

The scene integration supplies its registry, and codegen registers every matched, supported float1-4 output
directly. MaterialX carries registered outputs as `AOVDesc` values; every entry has an ID and a generated writer.
Low-level codegen without a registry can generate ordinary numeric Graph outputs, including matrices, without
runtime IDs or an AOV writer. This does not extend the renderer's transport types.

The combined MaterialX graph shares authored nodes between the surface and numeric outputs and keeps the primary
surface in output slot zero. Instance setup consumes the surface result without retaining numeric outputs on the
instance. The material's AOV writer evaluates the same Graph separately, using the same surface preparation and
inline or buffered parameters as setup, and writes directly from its output members.
One writer invocation constructs one Graph for all enabled outputs, not one Graph per output.

MDL discovers supported top-level numeric fields in a user-defined `material_category` struct, excluding built-in
render-contract fields and ignoring unsupported types with diagnostics. The effective material must expose a BSDF
at `surface.scattering` or top-level `scattering`. Scene materials supply their registry for the supported fields.
AOV expressions join beauty expressions and `init` in the same SDK link-unit submission. The material's writer
calls `init_shading_state` once to bind material data and initialize local state, then evaluates enabled expressions
against that state. It does not use state from a scattering instance or bind material data again for each output.
Standalone MDL compilation without a registry skips AOV discovery and generation.

Both scene material backends generate supported selected AOVs with zero consumers. There is no global AOV mode or
on/off switch. MaterialX's per-material output selection remains independent of consumer configuration. Pending
generation precedes texture enumeration so AOV-only resources are ready during normal material preparation.

### Material-level evaluation

`IMaterial.write_material_aovs<Writer>(coord, si, lod_sampler, hints)` takes the surface interaction, LOD sampler,
and shading hints directly. It can be called without `setup_material_instance`; its default implementation is a
no-op. Generated materials override it only when they offer registered AOVs. Fixed material-property guides still
use `IMaterialInstance.collect_properties` and do not share the arbitrary-output interface.

Each generated writer first checks whether any of its registered IDs is enabled. If none is enabled, it returns
before loading buffered parameters, preparing the MaterialX interaction, constructing Graph, or initializing MDL
state. Otherwise it evaluates locally and checks each output's enablement before storing a zero-padded float4.
No values or evaluation flags need to be retained between scattering setup and the AOV call.

## Runtime flow

```text
material outputs -> typed IDs + generated material code
                                  |
                         normal scene preparation
                                  |
guide specifications -> lookup -> containers + WriteGuide
                                  |
                  accepted primary material + surface -> writes
```

1. Material creation or regeneration declares supported outputs and generates their code. Ordinary scene update
   realizes material resources and shader requirements.
2. Standalone path-tracer callers update the scene after edits. `PathTracerPipeline` updates it once per frame
   before resolving the active camera and rendering samples. Guide changes trigger no implicit scene update.
3. Binding resolves enabled custom names through the prepared scene's catalog and checks fixed-guide collisions.
   `ContainerSpec.auto()` derives dimensions from the render target and format from declared width: R32F, RG32F,
   RGBA32F, and RGBA32F for one through four components. Explicit targets require a supported floating-point
   format with enough channels.
4. `OutputPrelude` specializes `WriteGuide` for fixed targets and ID-bearing indexed descriptors. Numeric IDs
   form indexed resource identifiers and switch cases. Textures bind through the parameter block; writable tensors
   bind directly. Selected indexed targets are cleared to zero before every dispatch.
5. The path tracer acquires the material once for instance setup and reuses it at the accepted primary surface,
   after nested-interface rejection and before unlit termination. It invokes `IMaterial.write_material_aovs`
   with the same surface interaction, LOD sampler, and shading hints, independently of fixed-property collection.
   `MaterialAOVWriteGuide` adapts the reflected writer to `IMaterialAOVWriter`, whose contract is
   `generic_data_enabled(id)` and `generic_data(coord, id, float4)`. Generated writers zero-pad values and guard
   each write by enablement. An ID absent from the writer is disabled and performs no store.

The guide dictionary returns arbitrary outputs under their requested names. Unwritten pixels remain zero,
including when no surviving producer writes a registered output. These values are not accumulated with beauty.
The DLSS branch preserves custom guides at raw path-tracing resolution without routing them into DLSS guide slots
or upscaling them as arbitrary AOVs.

## Invariants and rationale

Consumer choices never enter material codegen hashes or invalidate material modules. Target changes can still
specialize the path-tracer dispatch; stable material code does not imply that all renderer shader compilation
disappears. The registry has no consumer request set, use counts, or AOV notification signal.

The unused paths keep orchestration small. With no eligible producer outputs, materials perform no registration
and retain no host AOV bookkeeping. Scene retains one empty catalog object, but scene update performs no AOV
reconciliation or material scan. With all guides disabled, binding makes no registry calls. Fixed guides alone
require one catalog access and revision read per render invocation; name checks are cached, and an empty catalog
requires no name lookup. No custom targets means no indexed resources. Removing custom guides releases their
targets and cached bindings through the existing guide flow.

Separating entry points does not separate their computation dependencies. MaterialX instance setup still evaluates
the combined Graph, which includes selected numeric outputs even without consumers. An enabled AOV writer evaluates
that Graph again, including its surface computation. Per-material selection can exclude an independent AOV-only
branch, while a selected compound may still compute multiple outputs. MDL's shared `init` work occurs during
instance setup and again during an enabled AOV call. Its beauty state and SDK expression grouping remain shared.

The disabled-writer guard prevents the additional writer evaluation; it does not remove AOV-related work from
scattering setup. No eligible outputs means the material inherits the no-op writer. There is no AOV-specific
instance hint or retained output storage. This boundary supports independent callers without assuming compiler
dead-code elimination, fewer texture reads, or smaller MDL beauty state.

## Source landmarks

- Catalog and host registration: `src/falcor2/render/material_aov_registry.h`,
  `src/falcor2/render/material_aov_registry.cpp`, and
  `src/falcor2/render/scene.h`. Python bindings are in `src/falcor2_ext/render/scene.cpp`.
- MaterialX selection and generation: `src/falcor2/render/material/mtlx/codegen/structured/document/loader.cpp`,
  `src/falcor2/render/material/mtlx/codegen/structured/generator.cpp`,
  `src/falcor2/render/material/mtlx/codegen/structured/material/material_aovs.cpp`, and
  `src/falcor2/render/material/mtlx/mtlx_material.cpp`.
- MDL compilation and material emission: `src/falcor2/render/material/mdl/mdl_context.cpp` and
  `src/falcor2/render/material/mdl/mdl_material.cpp`.
- Delivery: `slang/falcor2/render/material.slang`, `slang/falcor2/rendernodes/reference_pathtracer.slang`,
  `falcor2/rendergraph/output_prelude.py`, `falcor2/rendernodes/reference_pathtracer_node.py`, and
  `falcor2/rendernodes/pathtracer_pipeline_node.py`.
- Catalog, generation, and integration tests: `tests/native/render/test_material_aov_registry.cpp`,
  `tests/native/render/test_materials.cpp`, `tests/python/render/test_scene.py`,
  `tests/python/rendergraph/test_output_prelude.py`, and `tests/python/pathtracer/test_material_aovs.py`.
  The direct-call fixture `tests/python/pathtracer/test_material_aovs.slang` invokes the material writer without
  instance setup; its Python test covers inline data, buffered live edits, MDL state, and disabled writes.
  The paired marble fixtures are `data/assets/test_material_aovs/marble_blue_veins.mtlx` and
  `data/assets/test_material_aovs/marble_blue_veins.mdl`.

## Related design documents

- [Material system](material-system.md): scene ownership, resource managers, and runtime dispatch.
- [MaterialX integration](materialx-integration.md): graph preparation and generated material structure.
- [Scene model](scene-model.md): lifecycle and scene-update ordering.
- [Render pipeline](render-pipeline.md): containers, render nodes, and image processing.
