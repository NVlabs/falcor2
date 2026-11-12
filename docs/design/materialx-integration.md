---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-11
verified_against: "eb7928cb77ab29034ef442e257340d891aa4067e"
---

(falcor2-design-materialx-integration)=
# MaterialX integration

Falcor2 turns a MaterialX document into a live scene material by selecting a primary root, preserving authored
inputs through graph preparation, and generating Slang implementations of its material interfaces. Generated
code can be reused while each material retains its own parameter values and texture resources.
Runtime reflection, shared material allocation, geometry requirements and
live emission predicates connect the generated implementation to Scene updates. This page explains these
ownership and identity boundaries so contributors can predict what a graph or parameter edit changes.


## Purpose and scope

A reader familiar with MaterialX graphs and Falcor2 materials can use this page to follow authored input into
generated code and per-hit shading. [Material system](material-system.md) owns common allocation, texture and
dispatch contracts; [BSDF contract](bsdf-contract.md) owns scattering semantics; [Named material outputs](material-aovs.md)
owns output registration and consumer binding.

## Architecture

### Document discovery and root selection

`MaterialXMaterial` accepts XML in `mtlx_buffer` or a file in `mtlx_path`. `mtlx_basepath`,
semicolon-separated `mtlx_search_paths` and a resolved file's parent directory construct its asset resolver.
The loader combines authored graphs with shipped standard and Falcor libraries, repairs compatible port
connections and validates the result.

Eligible simple documents attach prepared, process-cached immutable libraries. Documents needing includes,
definitions, overrides or incompatible contextual state use private imported copies. Generation retains the
documents needed by its graph; generated-code caches do not retain those graphs. Library provenance is tracked
by the identity of graphs supplied by library loading or attachment. An authored name or source URI that resembles
a library does not confer that identity.

`mtlx_node_name` can select the root explicitly; otherwise renderable discovery must identify one valid
primary root. Accepted surface, unconsumed BSDF and previewable value forms are normalized for generation.
The selected material instance is retained so its authored values and incoming connections survive implementation
resolution. Numeric outputs are selected separately: `CodeGenDesc.additional_output_names` defaults to
`{"*"}`, and selected supported outputs are generated independently of renderer consumers.

### Authored inputs and generated identity

Graph preparation materializes reachable default geometry properties and prunes closures proven inactive.


Editable inputs are unknown to static pruning. The generator uses
linear Rec. 709 working color and meter distance units.

`mtlx_canonicalize_names` defaults false and controls generated spelling. When enabled, structural traversal
canonicalizes user graphs, nested functions and bindings; trusted library graphs keep readable stable names.
`mtlx_editable_params` independently selects supported unbound values: empty selects none, `*` selects all,
and a comma-separated list selects authored names without the `inputs:` prefix. Ambiguous short names require
full authored paths. The removed `mtlx_class_compilation` property is rejected.

Request-local metadata preserves authored names and paths through color/unit conversion and BSDF-weight rewrites.
One numeric input can feed multiple generated uniforms while exposing one public `inputs:<name>` property.
Generated binding names can change without renaming that property. Public dynamic values are integers, booleans,
scalar floats and float2-4; matrix and texture metadata do not imply ordinary editable dynamic properties.
Filename resources remain runtime resources independently of optional numeric editability.

Equivalent canonical graphs can share generated code while retaining different live values. Topology, options,
editable selection and actual lobe classification can still change the class; some editable defaults affect
classification. Module/type names derive from source after syntax and policy rewrites, before fixed hardware/token
substitutions and insertion of the hash itself.

### Generated-code reuse

Each Scene owns a `CodeGenCache`; direct generation can omit it. Loading, graph construction, analysis,
classification and resource-binding preparation still run before lookup. The key includes graph topology,
implementation identity, baked values, generation options, types, classified lobe facts, resolved geometry
expressions and AOV IDs.

A hit restores generated source, names, size bound and geometry-use metadata. Current parameters, textures,
opacity, output declarations and analysis metadata remain request-local. Geometry registration therefore still
occurs on hits. The synchronized cache shares completed generated artifacts, separately from prepared libraries,
library source text and device shader compilation.


### Runtime material and resources

The `bsdf_mix` and `closure_tree` material boundaries expose full response. Layout identity and sample-weight
guarantees follow [Material system](material-system.md#component-layout-identity), including the renderer-selected
`eval_consistent` policy.

Code-generation property changes rebuild the result and dynamic properties, retaining values whose authored
property names still match. A successful generation resets the loaded module. Numeric edits update all bindings
for their public property and mark material properties and emission dirty without requesting generation.

Resource loading finishes pending generation, then registers nonempty filenames through the material resolver
and generated sampler settings. Deferred loads use `TextureLoadFlags::data` because generated shaders perform
source-to-working-space conversion. An empty filename preserves image-node fallback behavior.

The generator's portable size bound selects inline versus external parameter data. Loaded-module reflection
supplies the exact stride and checks the bound, with a padding exception for empty data. External data must fit
the shared allocator's alignment and uses `Material` allocation helpers and `BufferPtr`; small data stays in
the fixed material payload. Named MaterialX lookup-table globals are shared by strong reference. The common
[material update flow](material-system.md#runtime-flow) finalizes textures, uploads pooled data and refreshes
records before emissive geometry consumes them.

### Live emission and opacity

Generated emission queries follow graph connections and each compound invocation's input environment.
They evaluate trusted, state-independent numeric operations using current material data. Constants, exact-zero
factors and safe mix gates can prove a material dark before emissive-triangle allocation. Without such a proof, unknown,
texture-dependent or geometry-dependent emission remains eligible.

`IMaterial.is_emissive()` reads current inline or pooled data, so a numeric zero/nonzero edit can change
eligibility without changing the module. Light-integration queries include graph opacity; the surface
instance's emission avoids applying traversal cutout twice.

Root opacity is classified as opaque, constant or dynamic. Dynamic and texture-driven opacity evaluate the
generated graph at the hit. Proven constants use the compact common material path. Opacity changes participate
in scene requirements independently of scattering transmission.

### Geometry properties: fixed routes and exact names

World-space position, normal, tangent and bitangent values come from `SurfaceInteraction`; vector2 UV0 uses
`si.uv`. Extra UVs, geometry color, blind named values and non-world transforms use the production
`IGeomPropAccessor`. Emitted access records `uses_geometry_properties`, and `MaterialXMaterial` declares
that capability alongside opacity and conservative solid-transmission demand. The normal `required` Scene
policy includes this demand.

A `geompropdef` can resolve a name to a fixed semantic: a vector2 `lightmap` definition with
`geomprop="texcoord"` and `index="1"` reads UV1. Without such a definition, names are exact contracts:
`Pref` and `pref` are distinct, and neither has an intrinsic transform. A live blind read reserves a
Scene-owned ID; material-first reservation does not create data. Geometry later supplies a compatible type
and width. Unused or pruned reads do not reserve names.

Each read uses the current hit and its own fallback. Floating data interpolates; integer/Boolean data follows
the accessor's discrete rule. Missing or incompatible values return the caller default. Repeated reads may
construct separate accessors. Geometry selectors are immutable code-generation inputs; numeric fallbacks can
remain editable. Uniform geomprop reads, incompatible definitions and unsupported semantics are diagnosed;
matrix attributes are not generic per-vertex inputs.

Object/model space means local geometry space. Points include translation, vectors do not, and normals use
normalized inverse transpose; `transformnormal` normalizes even identity paths. View direction derives from
the incident ray, using normalized vector transformation outside world space. USD UV naming is configured by
`ImportOptions.usd_texcoord_primvar_names`; imported per-face values expanded into vertices do not implement
MaterialX uniform metadata. [Geometry and ray tracing](geometry-and-ray-tracing.md) explains offered streams.
AOVs share this generated graph and geometry-demand inventory.

## Runtime flow

1. Discovery loads and validates a document with an explicit resolver and enumerates renderables. Material creation
   selects the primary instance and numeric outputs; library provenance remains distinct from authored identity.
2. Preparation preserves authored inputs, classifies static/live values and geometry reads, and analyzes layering.
3. A cache miss builds the representation and emits source; a hit restores generated artifacts. Both retain fresh
   instance metadata. MaterialX builds dynamic properties and resolves resources before loading the Slang module
   and reflecting its data layout.
4. Scene update synchronizes parameter values, texture handles, lookup tables and inline/pooled records, then
   refreshes requirement snapshots and emissive geometry.
5. Renderer dispatch creates the generated per-hit material instance. A later numeric edit updates data;
   graph, selector or generation-policy changes repeat generation and may change shader requirements.

## Invariants and rationale

### Shader source belongs to the library

Authored documents may instantiate library nodes and define graph-backed implementations. They may not supply
shader source. Discovery and generation reject authored `implementation` declarations containing `sourcecode`
or `file`, including expanded includes, unused declarations and other targets, before loading Falcor libraries.
Importers keep library definitions attached for lookup rather than embedding source in serialized material XML.

Library definitions and source text are fixed during a process run; library-source edits require a restart.
Library-source caching uses source URI and filename, while canonical graph provenance uses graph identity.
These are separate contracts. New shader implementations belong in loaded libraries; user graph composition
remains supported.

### Default backend and validation policy

The primary consumer of every MaterialX feature is the build's default layering backend.
`mtlx::default_layering_mode()` is the implementation authority; ordinary
material construction and `CodeGenDesc` use it without an override. These are MaterialX layering modes, not GPU
APIs such as D3D12 and Vulkan.

Layering modes choose the generated scattering representation; GPU APIs choose its execution backend.

## Source landmarks

- `src/falcor2/render/material/mtlx/mtlx_material.h` and
  `src/falcor2/render/material/mtlx/mtlx_material.cpp` - public defaults, dynamic properties, resources, data
  layout and capability declaration.
- `src/falcor2/render/material/mtlx/codegen/structured/document/loader.cpp` and
  `src/falcor2/render/material/mtlx/codegen/structured/generator.cpp` - root/library ownership, input preservation,
  classification, cache keys, generation and source identity.
- `src/falcor2/render/material/mtlx/codegen/codegen_cache.cpp`,
  `src/falcor2/render/material/mtlx/codegen/structured/target/shader_generator.cpp` and
  `src/falcor2/render/material/mtlx/codegen/structured/target/nodes/source_code_node.cpp` - generated-code reuse,
  canonical provenance and library source caching.
- `src/falcor2/render/material/mtlx/codegen/structured/material/emission_query.cpp`,
  `src/falcor2/render/material/mtlx/codegen/structured/target/nodes/geometry_nodes.cpp` and
  `src/falcor2/render/material/mtlx/codegen/mx_parameters.cpp` - live queries, geometry access and parameter binding.
- `src/falcor2/render/material/mtlx/codegen/snippets/methods/basic/basic_material_wrapper.slang` - runtime
  emission/opacity queries and generated material entry points.
- `src/falcor2/render/material/mtlx/codegen/codegen_types.h` and
  `src/falcor2_ext/render/mtlx_codegen.cpp` - default layering mode, generation contract and Python discovery.
- `tests/native/render/test_materials.cpp` - canonical identity, provenance, cache isolation, authored inputs
  and closure representation.
- `tests/python/render/test_materials.py`, `tests/python/render/test_material_sampling.py`,
  `tests/python/render/test_materialx_data_layout.py`, `tests/python/render/test_materialx_emission.py` and
  `tests/python/render/test_materialx_geometry_properties.py` - live edits, target layout, emission and geometry.
- `tests/python/pathtracer/test_material_aovs.py` - rendered named-output integration.

## Related design documents

- [Material system](material-system.md)
- [MDL integration](mdl-integration.md)
- [BSDF contract](bsdf-contract.md)
- [Named material outputs](material-aovs.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Python and native integration](python-native-integration.md)
