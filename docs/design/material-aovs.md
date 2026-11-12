---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "3ea6306b922446e17dc3ba8d07c0fe2228d31a66"
---

(falcor2-design-material-aovs)=
# Named material outputs

Falcor2 material arbitrary output variables (AOVs) carry authored numeric values, such as masks and mixing weights,
into renderer-selected images or tensors. A scene-owned catalog gives each declared name a permanent ID and type.
MaterialX and MDL generate eligible outputs independently of consumers. Consumers resolve names after scene
synchronization, select shader capabilities and provide storage through guide outputs. `IMaterial` evaluates
enabled outputs from a surface interaction without constructing a scattering instance. Capture changes can
specialize the renderer while leaving generated materials intact. Declarations, compiled execution and output
storage therefore have separate owners and lifetimes.

## Purpose and scope

A contributor familiar with scene preparation and material evaluation can use this page to trace a named value
into a guide buffer, explain missing or unwritten outputs, and predict which changes affect producer generation.
General material dispatch, BSDF semantics, and pipeline composition belong to the related pages.

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
been registered. Enumeration does not establish live producer availability. Registration and lookup have a
single-threaded contract; callers must serialize access.

### Consumer binding and caching

Assigning `guide_output_specs` records names and container specifications without accessing the catalog. `None`
disables a target. Binding happens during rendering, after scene preparation. An unknown enabled name fails before
dispatch without changing the registry; a later invocation can resolve it after producer registration and scene update.
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

Low-level `mtlx::CodeGenDesc.additional_output_names` defaults to `{"*"}`. Exact names and case-sensitive `*` patterns select
candidates; `*` matches zero or more characters, without regular-expression or full filesystem-glob syntax. An
empty list opts out. Unmatched patterns are allowed, while missing exact names and explicit closure selections fail.
Distinct selected candidates with the same name are rejected.

Scene `MaterialXMaterial` uses the default selection and exposes no separate output-pattern property. Low-level
codegen callers can change the descriptor's selection independently of capture configuration.

The scene integration supplies its registry, and codegen registers every matched, supported float1-4 output
directly. MaterialX carries registered outputs as `AOVDesc` values; every entry has an ID and a generated writer.
Low-level codegen without a registry can generate ordinary numeric Graph outputs, including matrices, without
runtime IDs or an AOV writer. This does not extend the renderer's transport types.

The combined MaterialX graph shares authored nodes between the surface and numeric outputs and keeps the primary
surface in output slot zero. Instance setup consumes the surface result without retaining numeric outputs on the
instance. The material's AOV writer evaluates the same Graph separately, using the same
surface preparation and inline or buffered parameters as setup, and writes directly from its output members.
One writer invocation constructs one Graph for all enabled outputs, not one Graph per output.


MDL discovers supported top-level numeric fields in a user-defined `material_category` struct, excluding built-in
render-contract fields and ignoring unsupported types with diagnostics. The effective material must expose a BSDF
at `surface.scattering` or top-level `scattering`. Scene materials supply their registry for the supported fields.
AOV expressions join beauty expressions and `init` in the same SDK link-unit submission. The material's writer
calls `init_shading_state` once to bind material data and initialize local state, then evaluates enabled expressions
against that state. It does not use state from a scattering instance or bind material data again for each output.
Standalone MDL compilation without a registry skips AOV discovery and generation.

Both scene material backends generate eligible AOVs with zero consumers. There is no consumer-dependent producer
generation mode. Pending generation precedes texture enumeration so AOV-only resources are ready during normal
material preparation.

### Consumer requests select compiled execution

`SceneRenderRequest.material_aovs` names consumer demand without describing target formats or storage.
`Scene.resolve_requirements()` looks up each name and requires its ID to belong to the catalog generation
synchronized by `Scene.update()`. Registering a new name alone is insufficient: resolution fails until that
update. Unknown or unsynchronized names fail without allocating IDs.

Requested AOVs currently require both `material_aovs` and `geometry_properties` capabilities, conservatively
including names with no surviving producer. The `required` policy combines consumer and material demand;
`specified` rejects an insufficient mask; `all` enables every supported capability. Requested names still
undergo validation under each policy. [Scene model](scene-model.md#consumer-requests-and-immutable-requirements)
explains the snapshot and policy contracts.

`ReferencePathTracerNode` supplies enabled custom names to `SceneShaderHelper.prepare()` before linking.
The selected requirements set `ENABLE_MATERIAL_AOVS`, which gates execution of the public material writer.
A direct caller must also select AOV support; passing names to the helper validates synchronization and supplies
the conservative capabilities. An accepting output writer cannot enable support omitted from the linked shader.
Consumer changes can select different linked renderer modules without regenerating material producers.

### Material-level evaluation

Consumers call `IMaterial.write_material_aovs<Writer>(coord, si, lod_sampler, hints)` with the interaction, LOD
sampler and shading hints directly, without `setup_material_instance`. This public wrapper tests
`ENABLE_MATERIAL_AOVS` before calling `write_material_aovs_impl()`, the default no-op producer hook. Generated
materials override the hook when they offer registered AOVs; consumers use the wrapper to respect capability
selection. Fixed material-property guides still use `IMaterialInstance.collect_properties`.

Each generated producer hook checks whether any of its registered IDs is enabled. If none is enabled, it returns
before loading buffered parameters, preparing the MaterialX interaction, constructing Graph, or initializing MDL
state. Otherwise it evaluates locally and checks each output's enablement before storing a zero-padded float4.
No values or evaluation flags need to be retained between scattering setup and the AOV call.

## Runtime flow

```text
material outputs -> typed IDs + generated producer hooks
                                  |
                         normal scene preparation
                                  |
guide names -> lookup -> containers + selected requirements
                                  |
                         linked WriteGuide
                                  |
                  accepted primary material + surface -> writes
```

1. Material creation or regeneration declares supported outputs and generates their code. Ordinary scene update
   realizes resources and snapshots catalog generation for requirement resolution.
2. Standalone path-tracer callers update the scene after edits. `PathTracerPipeline` updates it once per frame
   before resolving the active camera and rendering samples. Guide changes trigger no implicit scene update.
3. Binding resolves enabled custom names through the prepared scene's catalog and checks fixed-guide collisions.
   `ContainerSpec.auto()` derives dimensions from the render target and format from declared width: R32F, RG32F,
   RGBA32F, and RGBA32F for one through four components. Explicit targets require a supported floating-point
   format with enough channels.
4. Enabled custom names select capabilities through `SceneShaderHelper`. `OutputPrelude` separately specializes
   `WriteGuide` for fixed targets and ID-bearing indexed descriptors. Numeric IDs
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

Consumer choices never enter material codegen hashes or invalidate material modules. Changed capabilities and
targets can specialize the linked renderer and dispatch; stable material code does not imply that all renderer
compilation disappears. The registry has no consumer request set, use counts, or AOV notification signal.

### Disabled outputs and retained computation

Disabling all custom AOV guides avoids additional AOV writer evaluation and custom target bookkeeping.
With no eligible producer outputs, materials perform no registration and retain no host AOV bookkeeping.
Scene retains one catalog object
and reads its generation during update. New declarations can advance requirement generation.
No producer/consumer reconciliation scan is needed.
With all guides disabled, guide binding makes no registry calls. Fixed guides alone
require one catalog access and revision read per render invocation; name checks are cached, and an empty catalog
requires no name lookup. No custom targets means no indexed resources. Removing custom guides releases their
targets and cached bindings through the existing guide flow.

Separating entry points does not separate their computation dependencies. MaterialX instance setup still evaluates
the combined Graph, which includes selected numeric outputs even without consumers. An enabled AOV writer evaluates
that Graph again, including its surface computation. Low-level codegen selection can exclude an independent
AOV-only branch, while a selected compound may still compute multiple outputs. MDL's shared `init` work occurs during
instance setup and again during an enabled AOV call. Its beauty state and SDK expression grouping remain shared.

The capability gate and disabled-ID guard prevent additional writer evaluation; they do not establish removal of
AOV-related work from scattering setup. No eligible outputs means the material inherits the no-op producer hook.
There is no AOV-specific
instance hint or retained output storage. This boundary supports independent callers without assuming compiler
dead-code elimination, fewer texture reads, or smaller MDL beauty state.

## Source landmarks

- Catalog and host registration: `src/falcor2/render/material_aov_registry.h`,
  `src/falcor2/render/material_aov_registry.cpp`, and
  `src/falcor2/render/scene.h`. Python bindings are in `src/falcor2_ext/render/scene.cpp`.
- Synchronization and capability selection: `src/falcor2/render/scene.cpp`,
  `src/falcor2/render/scene_render_request.h` and `falcor2/editor/scene_shader.py`.
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
  instance setup; its Python test covers inline data, buffered live edits, MDL state, disabled IDs and disabled
  compiled AOV support. `tests/python/render/test_scene.py` also checks synchronized name resolution and policy masks.
  The paired marble fixtures are `data/assets/test_material_aovs/marble_blue_veins.mtlx` and
  `data/assets/test_material_aovs/marble_blue_veins.mdl`.

## Related design documents

- [Material system](material-system.md): scene ownership, resource managers, and runtime dispatch.
- [MaterialX integration](materialx-integration.md): graph preparation and generated material structure.
- [MDL integration](mdl-integration.md): SDK expression grouping, state and resources.
- [Scene model](scene-model.md): lifecycle and scene-update ordering.
- [Render pipeline](render-pipeline.md): containers, render nodes, and image processing.
