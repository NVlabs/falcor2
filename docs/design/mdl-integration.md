---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "c5135aa947898c8b1e1fe2cb12e04a85c087f713"
---

(falcor2-design-mdl-integration)=
# MDL integration

Falcor2 compiles Material Definition Language (MDL) materials through the MDL SDK, then wraps the generated HLSL
in Slang implementations of its material interfaces. A process-wide SDK context can reuse translated targets,
while each live material owns its arguments, resource bindings and generated wrapper. Scene updates synchronize
those bindings before emission classification and rendering. At a hit, the wrapper supplies world-space shading
state, texture coordinates and geometry properties through Falcor2's production geometry accessor. This page
explains which work is shared, which state remains per material, and how edits reach shading without confusing
SDK target reuse with live material identity.

## Purpose and scope

A contributor familiar with Falcor2 materials can use this page to trace an MDL definition into a live Scene
material and predict the effect of changing parameters, textures or geometry bindings. Shared allocation and
texture caching belong to [Material system](material-system.md); sampling semantics belong to
[BSDF contract](bsdf-contract.md). This page covers the MDL adapter and its current boundaries.

## Architecture

### SDK context and module lifetime

`MDLContext::get()` creates one process-wide context lazily. The context owns the SDK runtime, database and
successful-target cache. A recursive mutex serializes compilation and protects `MDLSearchPathScope` throughout
its lifetime. That scope temporarily prepends module/resource roots and restores the preceding paths, including
nested scopes. Already loaded modules remain cached by name: changing search roots does not replace their
definitions. Generated MDL source therefore needs a new module name when its definition changes.

Each compile opens an SDK database transaction. Success commits it; a failed result or exception aborts it.
This transaction boundary does not promise that a failed live-material replacement preserves its previous
Slang module. Context shutdown releases cached SDK target handles before shutting down and unloading the SDK.

### Compiled targets and generated wrappers have different identities

Public `MDLMaterial.mdl_class_compilation` defaults to true. Class compilation keeps supported arguments
editable instead of folding every instance value into code. The lower-level
`MDLContext::CompileRequest.use_class_compilation` default is false; `MDLMaterial` explicitly supplies its
own setting.

Compilation still loads or accesses the definition, instantiates it, compiles the optimized graph and selects
functions before consulting the target cache. The cache key contains the final SDK graph hash, ordered backend
option assignments, and ordered groups of expression/callable requests. Group boundaries preserve separate
SDK link-unit additions. A hit avoids link-unit construction and HLSL translation and restores the associated
function and argument-block indices. Entries stay in memory until context shutdown.

The SDK class graph hash excludes argument values. Every compilation result therefore reconstructs its
argument block, resource table and metadata for the current instance, even after a target hit. The cache holds
SDK targets and indices; it does not retain a Scene registry or a complete live-material result. Equal SDK
subexpression hashes also allow duplicate backface requests to be omitted; distinct expressions remain separate.

`MDLMaterial` combines the HLSL with wrapper definitions, geometry-name mappings, emission predicates and named
output writers. It hashes that complete source before substituting its generated type name, producing
`MDLMaterial_<SHA1>`. Two materials with shared SDK code can require different wrappers because their Scene
bindings differ. `MaterialSystem` composes the resulting concrete material types and deduplicates module
requirements.

### Material-owned arguments and resources

Each `MDLMaterial` owns its compile result, dynamic properties, texture-resource records and Slang module.
Its material allocation contains readonly bytes, argument-block bytes and a packed texture-handle table.
`MDLMaterialData` addresses these regions through a `BufferPtr` and byte offsets; Scene-owned
`TextureManager` registrations supply the GPU textures behind the handles.

SDK texture IDs are one-based, with zero meaning invalid. Resource extraction preserves every ID through the
highest body-resource ID, even when a cached target contains unused instance defaults. Editable 2D texture
parameters receive independent slots, so editing one parameter cannot overwrite another parameter or a
constant body resource.

For an editable texture, an explicit GPU texture takes precedence over encoded bytes, which take precedence
over a filename. The parameter also carries sampler and color/data interpretation properties. File and byte
sources are loaded through `TextureManager`; an existing GPU texture is registered as supplied. Clearing all
sources writes resource ID zero. Embedded SDK BSDF lookup data is retained as owned bytes and realized as a
3D lookup texture. General 3D, cube and PTEX image resources, light profiles and measured BSDF resources are
unsupported.

The exposed dynamic arguments include booleans, integers (including enum values), scalar floats, float vectors
and 2D textures. Matrices are recognized in the SDK argument layout but create no editable property. Strings
and arrays are classified as unsupported argument types and likewise create no editable property.
Supported argument edits update material data and emission state without regenerating SDK code. Definition,
class-compilation and other code-generation settings invalidate compilation. A geometry-name binding edit
rebuilds the wrapper while retaining its SDK compilation result.

### Per-hit geometry and SDK state

An MDL material requests `SceneCapabilityFlags.geometry_properties` when the SDK reports referenced scene-data
names, texture-coordinate use or transform use. The normal `required` capability policy includes that demand.
Scene-data access is governed by the selected shader capability, not a separate MDL custom-attributes Boolean.

SDK string IDs and Scene attribute IDs are different namespaces. Compilation retains the SDK's conservative
referenced-name inventory and an owned string table, including captured string defaults. Each dynamic property
`mdl_scene_data:<name>` initially reserves the exact name in the Scene's 16-slot named-attribute range.
Importers can override a binding explicitly: the glTF adapter maps `COLOR_0` to its fixed semantic attribute.
There are no implicit spelling aliases. Generated source translates runtime string IDs to those bindings;
including that mapping in the wrapper hash prevents accidental sharing across different Scene name layouts.

The wrapper retains the current packed hit when scene-data callbacks need it. Each callback acquires a local
production geometry accessor, so triangle and linear swept-sphere hits read their own geometry and instance.
Float callbacks interpolate matching floating-point data; integer callbacks select the dominant vertex or
endpoint. Width differences truncate or zero-pad. A disabled capability, uniform lookup, invalid binding,
missing value or incompatible numeric family returns the caller's complete default. Matrix lookups always
default. `scene_data_isvalid` checks presence independently of the requested numeric type.

MDL shading positions and normals use world space. The default four texture-coordinate slots use the hit's
primary UV and accessor-provided secondary UVs; missing secondary coordinates start at zero. The wrapper
converts the Scene UV origin to MDL's lower-left convention. All slots use the existing tangent frame. Transform matrices
start as identity and are populated when the SDK requests transforms. Animation time is zero and
meters-per-scene-unit is one.

The material instance retains initialized SDK state; evaluation and sampling copy it before calling generated
functions that can mutate state. Named material outputs (AOVs) use the same initializer and geometry context.
Additional supported float, float-vector and color fields in a custom target material type are registered with
the Scene's AOV registry; the wrapper evaluates only enabled outputs. See [Named material outputs](material-aovs.md) for declaration
and consumer binding.

### Emission and opacity remain conservative

MDL emission classification first rejects structurally empty emission or provably zero constant intensity.
For editable materials, generated GPU predicates can test exact-zero intensity or safe scalar/color factors,
including guarded conditional branches. They use the material's current arguments and resource IDs.
A probe is admitted only when its SDK state usage and graph inspection establish independence from shading
state and scene data, and its argument layout is compatible with the material's block.

Without one of these zero proofs, an unknown, spatially dependent or unsupported expression remains potentially
emissive. The classifier does not sample a synthetic hit to prove that the entire material is dark. Exact-zero
comparisons preserve tiny nonzero emitters. Only constant radiant-exitance emission mode is supported; total-power and dynamic modes
are rejected. Light-selection weights use exitance; directional radiance multiplies intensity by the emission
distribution function (EDF), without adding a second cosine factor.

Known cutout-opacity constants determine the material's coarse opacity classification; dynamic values retain
shader opacity evaluation at the hit. The generated wrapper supplies opacity one when the expression is absent.
MDL materials are two-sided; the material header adds thin-walled status only when the SDK exposes a constant
true value. Dynamic thin-walled expressions cannot be represented by that header flag.

## Runtime flow

1. Creating an `MDLMaterial` or changing its code-generation settings prepares its definition, scoped search
   paths and compile request.
   The SDK produces or reuses a translated target, then reconstructs instance arguments, resource IDs,
   scene-data metadata and output declarations.
2. The material builds dynamic properties and its data regions. During `Scene.update()`, resource loading
   registers textures; `TextureManager` completes its update before `MaterialSystem` writes the packed
   handles and current argument values.
3. Material update creates the wrapper module when needed and publishes its data and shader requirements.
   Scene requirement refresh precedes the emissive-geometry update, so light classification sees current data.
4. Rendering dispatches through Falcor2's material interfaces. The MDL wrapper initializes per-hit SDK state,
   resolves geometry inputs and calls generated evaluation, sampling, opacity or AOV functions.
5. A subsequent numeric or texture edit marks properties, resources and emission dirty. The same update order
   refreshes lighting before rendering; it normally preserves generated code. A code-generation setting
   change repeats compilation, while a `mdl_scene_data:<name>` edit changes wrapper identity.

## Invariants and rationale

- SDK target reuse and live material ownership are separate. Independent argument blocks and texture slots
  preserve each instance's defaults and subsequent edits even when generated HLSL matches.
- Resource ID zero remains invalid, and constant body IDs remain stable. Runtime parameter slots extend
  the resource table without renumbering compiled references.
- Wrapper identity includes Scene-dependent mappings. Reusing only the SDK graph hash would hide differences
  in geometry bindings and named output functions.
- Per-hit geometry context stays local to the invocation. Missing or incompatible data uses documented
  defaults instead of borrowing another material's or hit's state.
- Emission pruning requires proof of zero. Conservative candidates cost work but preserve emitters whose
  intensity cannot be decided without a real shading point.

## Source landmarks

- `src/falcor2/render/material/mdl/mdl_context.h` and
  `src/falcor2/render/material/mdl/mdl_context.cpp` - SDK lifetime, search scopes, transactions, target keys,
  instance extraction, emission proofs and AOV discovery.
- `src/falcor2/render/material/mdl/mdl_material.h` and
  `src/falcor2/render/material/mdl/mdl_material.cpp` - public defaults, dynamic properties, resource ownership,
  source identity and generated material wrapper.
- `slang/falcor2/render/materials/mdl/mdl_material_data.slang`,
  `slang/falcor2/render/materials/mdl/mdl_runtime.slangh`,
  `slang/falcor2/render/materials/mdl/mdl_target_code_types.slangh` and
  `slang/falcor2/render/materials/mdl/mdl_emission.slangh` - device layout, SDK callbacks and shading state.
- `src/falcor2/render/scene.cpp`, `src/falcor2/render/material_system.cpp` and
  `slang/falcor2/render/geomprop_accessor.slang` - update order, module composition and per-hit geometry reads.
- `src/falcor2/importers/gltf_importer/gltf_to_mdl.cpp` and `src/falcor2_ext/render/scene.cpp` - importer bindings and Python
  material creation/property access.
- `tests/native/render/test_materials.cpp` - cache independence, function selection, emission and opacity.
- `tests/python/render/test_mdl_texture_parameters.py` and `tests/python/render/test_mdl_emission.py` -
  resource edits, independent slots, live emission and conservative fallback.
- `tests/python/render/test_mdl_generated_material_pipeline.py`,
  `tests/python/render/test_mdl_udim_uv_origin.py` and
  `tests/python/pathtracer/test_material_aovs.py` - geometry callbacks, UV conventions and named outputs.
- `tests/python/render/data/mdl_generated_material_pipeline/scene_data.mdl` - authored scene-data examples
  used by generated-material tests.

## Related design documents

- [Material system](material-system.md)
- [Scene model](scene-model.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [BSDF contract](bsdf-contract.md)
- [Named material outputs](material-aovs.md)
- [Lighting and sampling](lighting-and-sampling.md)
