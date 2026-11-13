---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "5aff5ad087edafa287f398b325654fbe7fbfcd16"
---

(falcor2-design-material-system)=
# Material system

Falcor2's material system synchronizes editable scene objects with a dynamically dispatched GPU material table.
Each material supplies its concrete Slang type, resources, shader requirements and shading metadata.
`MaterialSystem` assigns IDs, packs fixed-size records and manages shared external storage; `TextureManager`
owns registered textures and samplers behind compact handles. Scene updates finalize resources before publishing
material data and refreshing emissive geometry. Shaders recover an `IMaterial` from a geometry instance's
`MaterialID`, then create an `IMaterialInstance` for the current surface. Native, generated and Python-defined
materials share this boundary while retaining different parameter layouts and compilation policies.

## Purpose and scope

A contributor familiar with Scene objects and Slang can use this page to identify who owns material data,
follow an edit into rendering, and distinguish data updates from shader-requirement changes. Detailed graph
compilation belongs to [MDL integration](mdl-integration.md) and [MaterialX integration](materialx-integration.md);
scattering semantics belong to [BSDF contract](bsdf-contract.md).

## Architecture

### Scene ownership and host materials

`Scene` owns its material collection, `MaterialSystem` and `TextureManager`. Each `Material` supplies a concrete
Slang type name, an optional generated module, material flags, an `OpacityDesc` and required scene capabilities.
Native reflection uses `class_descriptor` plus optional `dynamic_properties` for editable state.

Resource edits trigger `on_load_resources()`; `update()` prepares family state; `write_to_cursor()` serializes
the concrete shader representation. Edits that can change emission must also set emission dirtiness, even when
the material is currently dark. Custom materials with unknown emission dependencies conservatively mark all
authored edits this way. Otherwise light sampling can retain stale classification or power.

Material flags describe two-sided shading, thin-walled behavior and an unlit terminal. Nested dielectric priority
shares the header's packed metadata. `enable_backface_culling` is a separate traversal property; two-sided shading
does not by itself choose hardware culling. [Geometry and ray tracing](geometry-and-ray-tracing.md) describes
how renderer policy and instance state apply it.

### Device table and dynamic dispatch

The device table is a `StructuredBuffer<MaterialData>`. Each 128-byte record has a 16-byte `MaterialHeader`
and 112-byte payload. The header stores a dispatch type ID, flags/priority and compact opacity state.
Material ID zero always selects `InvalidMaterial`; released nonzero IDs can be reused.

On a dirty material update, `MaterialSystem` sorts the unique concrete type names and assigns their
`IMaterial` conformances, reserving dispatch type zero for `InvalidMaterial`. Required modules are deduplicated
by their underlying Slang component identity, then sorted by name. The system reflects
`StructuredBuffer<ConcreteType>` in the shared render module or the material's generated module and gives the
material a typed cursor over its record.

`MaterialSystem.get_material()` reconstructs a dynamic `IMaterial` from the header and bytes. That interface
provides opacity, emission, [named material outputs](material-aovs.md) and surface-instance creation.
The resulting `IMaterialInstance` provides scattering operations at the current interaction. Renderers consume
these interfaces without a closed material-family enum.

### Shared external material storage

`MaterialSystem` owns a `BufferSuballocator` backed by CPU storage and one GPU buffer. A participating
`Material` owns only an allocation token. MDL data, external MaterialX data and native glTF extra records use
this pool; OpenPBR retains a separate attribute buffer. Each family controls the bytes within its allocation.

`ensure_data_size()` preserves existing bytes and zeroes added capacity when growth is needed. Smaller nonzero
requests retain capacity; zero releases the allocation. Ranges are 16-byte aligned. `mutable_data()` marks
storage and material properties dirty and returns a borrowed span including allocation padding. Pool growth
can invalidate every borrowed CPU span, so writes must finish before another allocation grows the pool.

During material update, the pool uploads before records are written. Records store the current `BufferPtr`
returned by `data_ref()`, with family-specific relative offsets where needed. GPU growth can replace the
underlying buffer, so every valid record is rewritten, including otherwise unchanged materials.
Callers must reacquire pointers after updates. Removal frees the allocation during synchronization, even if
Python or C++ still retains the material object; material destructors do not free through a dead Scene.
See [Buffer pointers](buffer-pointers.md) for pointer transport and lifetime.

### Component layout identity

`MaterialSample.component` identifies a selected contribution; its meaning is separate from both the material
ID and the marginal PDF used for multiple importance sampling (MIS). Equal nonzero
`get_component_layout_id()` values guarantee matching slot meanings, including inactive slots, without
guaranteeing equal values or probabilities.

| Material instance | Layout ID | Component slots |
| --- | --- | --- |
| Standard and StandardSpecGloss | `0x80000001` | Diffuse reflection, diffuse transmission, specular reflection, specular reflection/transmission |
| Native glTF SpecGloss | `0x80000002` | Diffuse reflection, specular reflection |
| Full-response adapters | `0` | `BSDFComponent::full_response` |


Native layout tags use the high bit to avoid generated IDs. Generated identities are Scene-local and can change
when the type table changes; reuse must invalidate its history accordingly. Compatible slots still require
reevaluation at the new interaction. Evaluation-consistent sample-weight reconstruction additionally requires
the renderer's `MATERIAL_SAMPLE_WEIGHT_POLICY = MaterialSampleWeightPolicy::eval_consistent`; the default
`native` policy permits native continuation estimators. See the [BSDF contract](bsdf-contract.md#contribution-identity-and-sample-weights).

### Texture ownership and reuse

A `TextureHandle` retains its `TextureManager` and reference-counted resource entry. The packed 32-bit value
written into material records carries no ownership. The manager retains textures and samplers and reclaims
released entries during a later update. Deferred handles are valid before loading completes; materials must
wait for `TextureManager.update()` before consuming finalized packed descriptors. The manager is not thread-safe.

Resource reuse has several distinct keys:

| Resource route | Reuse identity |
| --- | --- |
| File load | Resolved path, load flags, usage and effective mip limit |
| Encoded bytes | `Blob` object identity and load options |
| Existing GPU texture | Texture object identity |
| Sampler | Object identity, or configuration value for configuration-based registration |
| Regular final handle | Texture/sampler pair |

`TextureLoadFlags` express interpretation as well as processing. `color` is the default; `data`,
`data_rgb`, `normal`, `raw` and `environment` supply other presets. Color requests explicit sRGB
interpretation, while data requests linear interpretation. These choices select distinct cached resources
even for the same source. Existing GPU textures are registered as supplied.

Native slots choose color or data interpretation from their role. MDL follows SDK resource metadata; MaterialX
loads data because its generated shader performs colorspace conversion. Interpretation therefore belongs to
the consumer contract, rather than being inferred solely from a filename or extension.

The pinned SlangPy loader applies explicit interpretation to eligible bitmap formats and supported DDS linear/sRGB
format pairs. DDS changes select the corresponding format without re-encoding texels; automatic mode preserves
the authored format. Floating-point/HDR formats and linear alpha are preserved. Falcor's load flags select
explicit linear or sRGB mode. A `normal` load currently aliases
`data_rgb` and uses ordinary mip generation.

`SceneConfig.max_mip_count` supplies the manager default; zero means unrestricted. A per-load override can
replace it. The limit constrains the mip count implied by image dimensions: bitmap loading reduces the top
level, while DDS loading selects an authored mip suffix. If no DDS level fits, the loader warns and uses the
smallest level. Direct GPU registration bypasses loading and this limit.

Packed descriptors use combined texture/sampler views on CUDA and independent descriptors on other APIs.
UDIM sets add regular tile handles and an integer indirection texture selected by integral UV coordinates.

### Shared lookup-table lifetime

Family-wide lookup tables use named `SceneGlobals`. OpenPBR and MaterialX retain strong references to their
tables; Scene owns another reference. End-of-update collection removes a named global only when Scene is its
sole owner. A retained removed material can therefore keep tables alive. Destructors release references without
calling back into Scene. Manually added globals use explicit removal.

### Material implementation families

| Family | Host state and device representation |
| --- | --- |
| Standard, StandardSpecGloss, Unlit, EmissionOnly | Fixed reflected properties and compact native shader records |
| Native glTF metallic roughness and SpecGloss | Fixed properties, specialized shader types and shared-pool extra records |
| OpenPBR | Dynamic properties, a dedicated SGL attribute buffer and shared lookup tables |
| MDL | Generated module, dynamic arguments and shared-pool argument/resource regions |
| MaterialX | Generated module and dynamic properties; data stays inline when it fits, otherwise uses the shared pool |
| Python-defined Material | Explicit lifecycle/update, requirement and cursor hooks using the same record ABI |

Python subclass forwarding covers these supported hooks, but does not forward `flags()` or `opacity_desc()`.
The [Python/native integration page](python-native-integration.md) explains ownership and reflection boundaries.

#### Native glTF specialization

`GLTFContext` chooses compile masks separately for metallic-roughness and SpecGloss materials before type
conformances are collected. Runtime masks describe each material's effective features.

| `GLTFSpecializationMode` | Compile-mask policy |
| --- | --- |
| `required` | Each material's effective masks |
| `single` | Union of live materials within each workflow |
| `balanced` (default) | Unions grouped by transmission features within each workflow |
| `all` | All allowed features and their texture capabilities |

`SceneConfig.gltf_allowed_features` caps native glTF feature use without discarding authored properties or
their normal resource-loading behavior. `gltf_core_features` supplies a minimum compile mask and must be a
subset of that cap. `gltf_core_texture_transforms` and `gltf_core_alternate_uvs` add corresponding support; both
default false, and core features default to none. The core controls change available shader support, not the
features an instance actually uses. Converted Standard, MDL, MaterialX and OpenPBR materials are outside this native cap.
The immutable Scene configuration survives `append()`; import options select the material backend.

#### MDL geometry inputs

MDL and MaterialX use Scene-owned named attributes and production per-hit accessors. MDL reserves exact names,
can receive explicit importer bindings and includes Scene mapping in its generated-wrapper identity.
Geometry availability is selected through `SceneCapabilityFlags.geometry_properties`, which materials declare
when needed. The normal `required` policy includes that demand. The
[MDL geometry contract](mdl-integration.md#per-hit-geometry-and-sdk-state) details the 16-name budget, defaults,
UVs, transforms and retained hit state.

### Opacity and scene requirements

Cutout opacity controls traversal separately from BSDF transmission. `OpacityDesc` can disable cutout, provide
a compact factor and texture channel, or select authoritative `IMaterial.eval_opacity()`. The latter handles
dynamic graphs and native alternate-UV or vertex-alpha inputs.
Mask mode thresholds the value; blend mode uses stochastic acceptance. Graph and direct texture evaluation
use explicit LOD zero. MaterialX root cutout is applied once, without a duplicate transparent scattering branch.

`MaterialSystem` aggregates requirements from all valid materials, including unassigned ones, after their
updates. Materials declare geometry access, opacity and solid-transmission demand; unknown implementations
conservatively require solid transmission. A material's actual shading remains authoritative.

Any required opacity evaluation enables material-aware traversal. Current acceleration-structure policy then
marks all triangle instances potentially non-opaque, avoiding per-assignment rebuilding while allowing extra
any-hit work. The current opacity evaluator accepts non-triangle hits without a cutout test.
Renderer shader selection may conservatively enable additional capabilities.
[Scene model](scene-model.md) explains demand, consumer requests and immutable requirement snapshots.

## Runtime flow

1. An edit marks material properties, resources, emission or culling dirty. Scene resource hooks register
   required textures; `TextureManager.update()` completes loading and descriptor preparation.
2. A dirty `MaterialSystem` updates every valid material and glTF specialization. It assigns/releases IDs,
   frees removed allocations and uploads shared external storage.
3. The system aggregates capability demand, rebuilds type/module requirements, writes slot zero and every
   live typed record, then uploads the material table.
4. Scene refreshes its requirements, culling and render state before updating emissive geometry. Renderer helpers
   compare resolved shader inputs; parameter-only edits can reuse linked code while using current data.
5. `Scene.bind()` exposes the material table; packed texture handles address device descriptors directly.
   Each geometry hit supplies a synchronized `MaterialID`; shaders reconstruct its per-surface material instance.

## Invariants and rationale

Material IDs, dispatch IDs, pointers and linked requirements belong to one synchronized Scene state. They are not
persistent asset identifiers. Fixed records give every material the same lookup path; typed cursor writes align
host bytes with reflected shader layouts. Shared allocation reduces independent buffers while preserving
family-owned packing. Pooling reduces separate buffers and descriptors; it does not guarantee smaller uploads,
because updates serialize all valid materials and track a single dirty byte interval.
Texture cache identity includes interpretation because the same file can serve both color
and numerical data. A changed concrete type or module affects shader requirements; edits within an established
layout usually affect only data.

## Source landmarks

- `src/falcor2/render/material.h`, `src/falcor2/render/material.cpp`,
  `src/falcor2/render/material_system.cpp` and `src/falcor2/utils/buffer_suballocator.cpp` - hooks,
  allocation tokens, upload order, record serialization and requirement aggregation.
- `src/falcor2/render/scene.cpp`, `slang/falcor2/render/scene_types.slang`,
  `slang/falcor2/render/material_system.slang` and `slang/falcor2/render/material.slang` - synchronization,
  record ABI and dynamic interfaces.
- `src/falcor2/render/texture_manager.h`, `src/falcor2/render/texture_manager.cpp`,
  `src/falcor2/render/texture_load_flags.h` and
  `slang/falcor2/render/texture_manager.slang` - source identities, interpretation flags and device handles.
- `external/slangpy/src/sgl/utils/texture_loader.cpp` - bitmap/DDS interpretation and mip selection.
- `src/falcor2/render/material/gltf_context.cpp`, `src/falcor2/render/material/gltf_material.cpp`,
  `src/falcor2/render/material/openpbr_material.cpp`,
  `src/falcor2/render/material/mdl/mdl_material.cpp` and
  `src/falcor2/render/material/mtlx/mtlx_material.cpp` - specialization and family storage choices.
- `src/falcor2_ext/render/scene.cpp`, `src/falcor2_ext/render/texture_manager.cpp` and
  `tests/python/render/python_material_sample.py` - Python hooks, resource bindings and custom-material example.
- `tests/native/render/test_materials.cpp`, `tests/native/render/test_scene.cpp`,
  `tests/python/render/test_materials.py` and `tests/python/render/test_scene_capabilities.py` - record
  packing, pool growth/lifetime, globals, shader dispatch and capabilities.
- `tests/python/render/test_texture_manager.py` - resource identity, interpretation, deferred loading and limits.
- `slang/falcor2/render/intersector.slang` - traversal opacity.

## Related design documents

- [MDL integration](mdl-integration.md)
- [MaterialX integration](materialx-integration.md)
- [BSDF contract](bsdf-contract.md)
- [Named material outputs](material-aovs.md)
- [Scene model](scene-model.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Lighting and sampling](lighting-and-sampling.md)
- [Buffer pointers](buffer-pointers.md)
