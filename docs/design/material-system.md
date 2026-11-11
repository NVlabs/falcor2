---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-06
verified_against: "2ac1506f0384e5691c17a0c98cef21f1a9663f66"
---

(falcor2-design-material-system)=
# Material system

Falcor2's material system turns scene-owned material objects into a compact, dynamically dispatched device table.
Each `Material` describes its concrete Slang type, parameters, resources, flags, and opacity policy. During
`Scene.update()`, `TextureManager` finalizes texture handles before `MaterialSystem` writes material headers and
typed payloads, gathers generated shader modules, and publishes `IMaterial` conformances. Geometry stores only a
`MaterialID`; shaders use that ID to recover the concrete material and create an `IMaterialInstance` for shading.
This common path supports shipped native materials, generated MDL and MaterialX implementations, and custom Python
materials without making renderers depend on a closed material enum.

## Purpose and scope

This page explains material ownership, synchronization, device records, shader requirements, texture handles,
material families, and cutout-opacity dispatch. It defines the boundary between host material objects and shader
implementations rather than documenting every property or material API.

Detailed BSDF evaluation and sampling belong in [BSDF contract](bsdf-contract.md). MaterialX graph preparation and
code generation belong in [MaterialX integration](materialx-integration.md). Geometry assignment and traversal
structures belong in [Geometry and ray tracing](geometry-and-ray-tracing.md), while scene-wide lifecycle and
requirements belong in [Scene model](scene-model.md).

## Architecture

### Scene ownership and host materials

`Scene` owns a collection of `Material` scene objects, one `TextureManager`, and one `MaterialSystem`. A material
holds editable host properties and reports four pieces of renderer-facing information: a concrete Slang type name,
an optional required Slang module, static material flags, and an `OpacityDesc`. Resource changes call
`on_load_resources()` before update; `update()` prepares any family-specific state; and `write_to_cursor()` serializes
the concrete shader representation. Material dirtiness is aggregated by the scene like other scene-object changes.

The flags describe properties that affect generic shading, currently two-sidedness, thin-walled behavior, and an
unlit terminal. The nested-priority value shares compact header storage with those flags. Materials do not choose
their own IDs: `MaterialSystem` assigns an ID when the object enters the synchronized collection and releases it on
removal. Geometry instances carry these IDs in their device descriptors.

### Device table and dynamic dispatch

The device ABI is a `StructuredBuffer<MaterialData>`. Every 128-byte record contains a 16-byte `MaterialHeader` and
a 112-byte payload. The header carries a Slang dynamic-dispatch type ID, packed material flags and nested priority,
and packed opacity state. Slot zero is always an `InvalidMaterial` record and material ID zero is the fallback for
missing assignments.

For each update, `MaterialSystem` collects the concrete Slang type names currently used by valid materials. It sorts
and deduplicates them, builds one conformance to `IMaterial` per type, and gives `InvalidMaterial` dispatch type zero.
It similarly deduplicates required generated modules and sorts them by module name. For each material, the system
reflects `StructuredBuffer<ConcreteType>` from either the shared render module or that material's required module,
then gives the material a typed cursor over its record. Concrete structs overlay the common payload; larger or
variable family state is reached through buffer and texture handles stored in that payload.

On device, `MaterialSystem.get_material()` combines the header type ID and record bytes into a dynamically
dispatched `IMaterial`. The interface provides opacity, emission, [named material output evaluation](material-aovs.md),
and material-instance creation. The
returned `IMaterialInstance` provides the per-surface scattering operations used by a renderer. This split lets
generic scene and traversal code dispatch materials without knowing whether their implementation was shipped or
generated.

### Component layout identity

An `IMaterialInstance` may return a selected scattering contribution in `MaterialSample.component` while retaining
the marginal PDF for MIS. The [BSDF contract](bsdf-contract.md#contribution-identity-and-sample-weights) defines the
value and joint-density semantics. Sample-weight reconstruction is guaranteed only under the renderer's
`MATERIAL_SAMPLE_WEIGHT_POLICY = MaterialSampleWeightPolicy::eval_consistent` specialization; the default `native`
policy permits native continuation estimators. This is independent of `SceneConfig` and individual material
properties, so one scene can serve renderers with different weight requirements.

`get_component_layout_id()` provides a separate compatibility identity for
interpreting those component indices across instances. Zero denotes full-response sampling. Equal nonzero layout
IDs guarantee the same slot meanings, including inactive slots, without implying equal material values or sampling
probabilities.

| Material instance | Layout ID | Component slots |
| --- | --- | --- |
| Standard, including StandardSpecGloss | `0x80000001` | Diffuse reflection, diffuse transmission, specular reflection, specular reflection/transmission |
| Native glTF SpecGloss | `0x80000002` | Diffuse reflection, specular reflection |
| Full-response material adapters | `0` | `BSDFComponent::full_response` |

Native layouts use the high bit to keep their tags disjoint from generated IDs. MaterialX `bsdf_mix` and
`closure_tree` use full response.
Different layout IDs do not establish component compatibility, even if individual slot numbers happen to match.

Layout IDs are scene-local and tied to the synchronized type-conformance table. Adding or removing a material type
can renumber generated IDs. Reuse history must therefore be invalidated when that table changes; comparing IDs
across scenes or table rebuilds is insufficient. Even with a compatible layout, callers reevaluate the component's
value, joint density, and roughness classification at the new interaction. Layout compatibility alone does not
establish evaluation-consistent sample weights or make path reconnection valid.

### Textures and shared resources

`TextureManager` converts file textures, existing SGL textures, and samplers into reference-counted
`TextureHandle` values. It deduplicates file loads by path and options, registered resources by identity, and final
handles by texture-sampler pair. A deferred load returns a valid but unfinished handle;
`TextureManager.update()` loads pending images and finalizes device descriptors. Released handles are reclaimed on a
later update rather than immediately.

The shader-visible handle is one packed 32-bit value. Its descriptor encoding differs between CUDA, which requires
a combined texture-sampler view, and APIs with independent texture and sampler descriptors. The same shader API also
supports UDIM sets: the manager resolves numbered tiles, creates regular tile handles, and builds an integer
indirection texture that selects a tile from the integral UV coordinates. Materials therefore store small handles,
not direct ownership of device textures. The manager is not thread-safe and its update boundary is part of scene
synchronization.

Lookup tables and similar family-wide resources use `SceneGlobals` instead of one copy per material. OpenPBR and
MaterialX lazily acquire their respective tables through `Scene::get_or_create_scene_globals()` during update and
retain strong references. The scene owns another reference and indexes the globals by a unique name.

At the end of `Scene.update()`, garbage collection removes named globals only when the scene is their sole owner.
External references, including a retained removed material, can keep them alive. Material destructors release their
references without calling back into `Scene`, avoiding access to scene state during teardown. This ownership rule
also keeps one material's removal from invalidating another material's tables. Manually added globals use explicit
add/remove calls and are outside this garbage-collection mechanism.

### Material implementation families

The native-backed materials `StandardMaterial`, `StandardSpecGlossMaterial`, `UnlitMaterial`,
`EmissionOnlyMaterial`, and `OpenPBRMaterial` pair C++ scene objects with Slang implementations in the shared render
module. The first four expose fixed reflected properties and compact data. OpenPBR exposes a dynamic property set,
stores its larger attribute state in a managed buffer, and references shared lookup tables. Their host methods map
authored textures, factors, and flags into the common contract.

`MDLMaterial` and `MaterialXMaterial` generate a Slang module and concrete type for the selected definition or graph.
They expose reflected editable properties, register generated texture resources through the scene's
`TextureManager`, and return their generated module to `MaterialSystem`. Inline record fields point to family-owned
argument, attribute, texture-table, or generated-data buffers when the state does not fit in the common payload.
Python subclasses of the bound `Material` base supply the same hooks and cursor writer, using the same ABI rather
than a renderer-side special case.

#### Native glTF specialization

`GLTFMaterial` (metallic roughness) and `GLTFSpecGlossMaterial` are separate native workflows. Their GPU types have
compile-time masks for features, textures, texture transforms, and nonzero texture-coordinate sets. Runtime masks
still select what an individual material uses. Before collecting material type conformances, the scene's
`GLTFContext` chooses the compile masks independently for each workflow:

- `GLTFSpecializationMode.required` uses each material's effective masks. It favors low-feature
  shading, but distinct material masks can require more compiled types.
- `GLTFSpecializationMode.single` uses the componentwise union of live materials in each workflow. Editing or
  removing materials can change that union; changes within the existing union keep the shader type stable.
- `GLTFSpecializationMode.balanced` (the default) builds a small number of material specializations to cover all
  requirements.
- `GLTFSpecializationMode.all` uses every allowed feature and its texture, transform, and texcoord capabilities.
  Adding another material of an existing workflow does not change its type.

`SceneConfig.gltf_core_features` supplies a minimum compile mask after the mode selects a material's mask.
The matching texture slots are included for each workflow; texture transforms and nonzero texcoord selection
remain driven by the materials unless `gltf_core_texture_transforms` or `gltf_core_alternate_uvs` is enabled.
These options add transform and nonzero texcoord support to each core texture slot. Their defaults are `false`;
the core feature default is `none`, so existing specialization behavior is unchanged.
For example, `normal | emission | multiscatter` produces a metallic-roughness core mask of `<11,15,0,0>`
and a specular-glossiness core mask of `<3,15,0,0>` (which does not support multiscatter).
Adding `specular` and enabling both texture options broadens those masks to `<27,63,63,63>` and `<3,15,15,15>`.
The core must be a subset of `gltf_allowed_features`. It changes shader support, not the features a material
actually uses or the scene capabilities those features require.

`SceneConfig.gltf_allowed_features` is a hard cap on native glTF feature use. It intersects each material's
`feature_mask_filter` before runtime masks and extra records are derived. Disabled features retain their authored
properties and normal resource-loading behavior, but contribute no runtime feature, texture capability, or extra
record. The cap does not affect glTF materials converted to Standard, MDL, MaterialX, or OpenPBR. Unknown feature
bits are rejected. For example:

```python
import falcor2 as f2

config = f2.SceneConfig(
    gltf_specialization_mode=f2.GLTFSpecializationMode.balanced,
    gltf_allowed_features=(
        f2.GLTFFeatureFlags.all
        & ~f2.GLTFFeatureFlags.anisotropy
    ),
    gltf_core_features=(
        f2.GLTFFeatureFlags.normal
        | f2.GLTFFeatureFlags.emission
        | f2.GLTFFeatureFlags.multiscatter
        | f2.GLTFFeatureFlags.specular
    ),
    gltf_core_texture_transforms=True,
    gltf_core_alternate_uvs=True,
)
```

The mode and cap belong to the immutable scene configuration. Scene creation routes accept that configuration;
`append()` keeps the destination scene's settings. ImportOptions still chooses the glTF material backend.

#### MDL geometry inputs

MDL uses the same Scene-owned named IDs and per-hit accessor as MaterialX. The SDK's `referenced_scene_data`
inventory reserves exact, case-sensitive names within the Scene-wide budget of 16 custom attributes; exceeding
that budget fails explicitly. This inventory is conservative: helper calls can contribute unused candidates even
with constant arguments. Accepting it preserves existing materials without attempting stricter static-name
rejection. Runtime selection among mapped names works; editable string properties and aliases such as `Pref`,
`st1`, or `UV1` are not inferred.

`MDLContext` produces Scene-independent SDK artifacts. `MDLMaterial` joins their owned runtime string table to Scene
IDs, including captured defaults absent from the SDK target string table. Only candidate names reserve IDs, not
every runtime string. Geometry and materials may arrive in either order and reuse the same reservation. The final
generated-source hash includes this mapping, preventing module reuse with incompatible IDs across Scenes.

`scene::data_lookup_float*`, `scene::data_lookup_int*` and color reads use the accessor's existing conversions and
whole-value caller defaults on missing or incompatible data. Floating sources interpolate; integers use the dominant
vertex/control point. `data_isvalid` tests descriptor presence independently of read type: Boolean sources can be
present while numeric reads return defaults. Numeric uniform and matrix lookups return caller defaults because no
uniform or named-matrix storage is supplied.

`state::texture_coordinate` has four configured slots, UV0 through UV3. UV0 comes from the surface interaction;
absent secondary streams use zero in Scene coordinates, followed by the existing conversion to MDL lower-left
coordinates. Tangent frames remain derived from UV0. Object/world transforms use the geometry accessor, keeping
MDL's world-space internal convention. Normal transforms do not force unit length; callers normalize when needed.

`SceneConfig.enable_custom_attributes` remains an explicit Boolean. Disabled access yields caller defaults (or
identity transforms) and excludes optional packed hit state. With it enabled, the hit geometry determines runtime
availability. Nonempty SDK candidate inventories enable retained packed hit context for scene-data callbacks;
UV/transform-only materials resolve inputs during setup without retaining it. SDK state-usage flags omit unused
setup paths but do not prove per-index UV demand. Material previews with synthetic hits have ordinary UV0 and
missing extra-data defaults; there is no mock provider. The accessor factory's early-inlining workaround for
[Slang #13046](https://github.com/shader-slang/slang/issues/13046) is removable when the pinned compiler passes
the combined named/state-read regression without it.

SDK-generated calls can take shading state as `inout`. BSDF and volume entrypoints therefore copy retained state
locally before such calls, preserving immutable Falcor material-instance methods. Geomprop reads do not themselves
mutate geometry, and no full `SurfaceInteraction` or accessor object is persistently cached.

## Runtime flow

```text
 authored properties and resources
              |
              v
        Material dirty state
              |
        Scene.update()
              |
    on_load_resources()
              |
    TextureManager.update()
              |
  MaterialSystem.update()
       |             |
       v             v
 MaterialData   modules + conformances
       |             |
       +------ renderer link
              |
 GeometryInstance MaterialID
              |
        dynamic IMaterial
              |
      IMaterialInstance shading
```

1. Authoring creates or edits a material and marks properties or resources dirty. New resources are registered or
   queued through the scene's texture manager.
2. `Scene.update()` invokes material resource-loading hooks, then updates `TextureManager`. Deferred textures and
   UDIM indirection are therefore finalized before any material record reads their packed handles.
3. `MaterialSystem.update()` lets each current material prepare family state, assigns or releases material IDs,
   rebuilds type-conformance and module sets, writes the invalid record, serializes every valid typed
   record, and uploads the device table.
4. The scene combines material modules and conformances with its other requirements. Renderer helpers compare
   resolved shader inputs before relinking their Slang module; parameter-only updates can reuse the linked code
   and replace data.
5. `Scene.bind()` places the material table and texture descriptors below `g_scene`. A geometry hit supplies its
   synchronized `MaterialID`; generic shader code reconstructs `IMaterial`, performs setup, and creates the concrete
   material instance used for emission and scattering.

## Invariants and rationale

Material and dispatch IDs describe the current synchronized snapshot, not persistent identities or a serialized
ABI. Material ID zero is permanently invalid, but released nonzero IDs may be reused. Type IDs are deterministic for
the current sorted set of concrete Slang types and may change when that set changes. Callers must synchronize the
scene and use IDs with the matching bound material table and linked requirements.

The fixed record size gives every material a uniform lookup. Dynamic dispatch
keeps generated materials open-ended, while typed cursor writes make host and Slang layouts agree through
reflection. A change to a generated type or required module is consequently a shader-requirements change; changing
only values inside an established layout is data dirtiness.

Cutout opacity is a traversal policy, not BSDF transmission. `OpacityDesc` can disable cutout, provide a compact
factor and optional texture channel, or request authoritative `IMaterial.eval_opacity()` evaluation for a dynamic
MDL or MaterialX graph. Proven graph constants use the compact path. Mask mode applies a threshold; blend mode uses
stochastic acceptance. The selected opacity is applied once, so MaterialX root cutout opacity is not also represented
as a transparent scattering branch.

If any current material enables opacity, scene requirements select `MaterialOpacityEvaluator` instead of the
force-opaque evaluator. [Geometry and ray tracing](geometry-and-ray-tracing.md) describes the corresponding
conservative traversal setup: all triangle instances become potentially non-opaque today. This scene-wide choice
avoids rebuilding acceleration structures for each material assignment, at the cost of unnecessary any-hit work for
opaque triangles. Graph evaluation uses explicit LOD zero, matching direct opacity-texture evaluation.

Host flags conservatively cover behavior needed before constructing a full material instance. Generated graph
materials report two-sidedness conservatively, and transmissive standard materials are treated as two-sided. The
concrete material remains authoritative for its actual scattering and emission.

## Source landmarks

- `src/falcor2/render/material.h` and `src/falcor2/render/material.cpp` - host material lifecycle, cursor contract,
  dirty flags, static flags, and
  opacity descriptions.
- `src/falcor2/render/material_system.h`, `src/falcor2/render/material_system.cpp`, and
  `src/falcor2/render/scene.cpp` - IDs, record serialization, modules and conformances, update order, requirements,
  scene globals, and binding.
- `src/falcor2/render/texture_manager.h`, `src/falcor2/render/texture_manager.cpp`, and
  `src/falcor2/render/texture_manager_types.h` - texture, sampler, handle, deferred-load, UDIM, lifetime, and backend
  descriptor management.
- `src/falcor2/render/material/standard_material.cpp`,
  `src/falcor2/render/material/standard_specgloss_material.cpp`,
  `src/falcor2/render/material/unlit_material.cpp`, `src/falcor2/render/material/emission_only_material.cpp`, and
  `src/falcor2/render/material/openpbr_material.cpp` - native-backed family implementations.
- `src/falcor2/render/material/mdl/mdl_material.cpp` and
  `src/falcor2/render/material/mtlx/mtlx_material.cpp` - generated modules, dynamic properties, resource tables,
  external data, and opacity classification.
- `src/falcor2/render/material/mdl/mdl_context.cpp`,
  `slang/falcor2/render/materials/mdl/mdl_runtime.slangh`, and `slang/falcor2/render/geomprop_accessor.slang` - SDK
  name inventory and argument blocks, scene-data callbacks, defaults, and production per-hit access.
- `slang/falcor2/render/scene_types.slang`, `slang/falcor2/render/material_system.slang`,
  `slang/falcor2/render/material.slang`, and `slang/falcor2/render/intersector.slang` - shared record ABI, dynamic
  interfaces, lookup, opacity evaluation, and hit acceptance.
- `slang/falcor2/render/texture_manager.slang` and `slang/falcor2/render/materials/` - device texture access and
  shipped material implementations.
- `src/falcor2_ext/render/scene.cpp`, `src/falcor2_ext/render/texture_manager.cpp`, and
  `tests/python/render/python_material_sample.py` - Python material creation, subclassing, properties, cursor writes,
  and texture-manager access.
- `tests/native/render/test_materials.cpp`, `tests/python/render/test_materials.py`,
  `tests/python/render/test_texture_manager.py`, `tests/python/render/test_scene.py`, and
  `tests/python/pathtracer/test_pathtracer.py` - record packing, generated modules, texture lifetimes, requirements,
  opacity, dispatch, and rendered traversal behavior.
- `tests/native/render/test_scene.cpp` - named globals sharing, retention by external references, garbage
  collection, and recreation after the final reference is released.
- `tests/python/render/test_mdl_generated_material_pipeline.py`,
  `tests/python/render/test_materialx_geometry_properties.py`, and
  `tests/python/render/data/mdl_generated_material_pipeline/geometry_properties.mdl` - MDL callbacks and shared
  MaterialX/MDL geometry-input proofs using real Scene geometry.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Render pipeline](render-pipeline.md)
- [Falcor2 design documentation](index.md)
- [BSDF contract](bsdf-contract.md)
- [MaterialX integration](materialx-integration.md)
- [Named material outputs](material-aovs.md)
- [Lighting and sampling](lighting-and-sampling.md)
