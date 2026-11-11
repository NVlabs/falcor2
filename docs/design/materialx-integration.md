---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-11-06
verified_against: "2ac1506f0384e5691c17a0c98cef21f1a9663f66"
---

(falcor2-design-materialx-integration)=
# MaterialX integration

Falcor2 turns a MaterialX document into a scene material by selecting one primary root and optional numeric outputs,
normalizing their graph for the Falcor shading contract, and generating a content-addressed Slang module. It contains concrete
material, material-instance, graph, and parameter-data types that plug into `MaterialSystem` through the ordinary
`IMaterial` interface. Authored values can remain compile-time constants or become reflected, editable runtime
properties and texture handles. Optional name canonicalization gives equivalent graphs stable generated names to
share generated code while retaining per-instance values. Resource resolution, reflected buffer layout, opacity,
scene-owned geometry-attribute IDs, and shared lookup tables connect code generation back to the live scene.

## Purpose and scope

This page explains the boundary from a MaterialX document to Falcor2's generated material implementation. It covers
renderable discovery, document and root preparation, code generation, name canonicalization, editable parameters,
runtime resources, and scene specialization. General material ownership and device dispatch belong in
[Material system](material-system.md); scattering semantics belong in [BSDF contract](bsdf-contract.md); imported
material assignment belongs in [Scene loading and authoring](scene-loading-and-authoring.md).

MaterialX is an interchange and graph-description format. Falcor2 uses its libraries and graph machinery, but emits
Falcor-specific Slang for the renderer. This page describes that current path rather than every supported MaterialX
node or every code-generation option.

## Comparison tooling

The standalone `MaterialVisualizer` Python module and analytic sphere tracer are maintained in
They are no longer shipped by Falcor2. That project provides comparison reports using its copied
visualizer, Falcor2's reference pathtracer through the public Python API, and official MaterialX
OSL and GLSL reference renderers. Follow that repository's setup and run instructions for parity
investigations; Falcor2's core MaterialX runtime and regression tests remain here.

## Architecture

### Document discovery and root selection

`MaterialXMaterial` accepts either XML text in `mtlx_buffer` or a file named by `mtlx_path`. `mtlx_basepath` and the
semicolon-separated `mtlx_search_paths` seed an `AssetResolver`; a resolved document's parent directory is added so
relative includes and images follow the document. The loader imports the shipped MaterialX standard libraries and
Falcor's target-specific libraries, repairs compatible port connections, and validates the combined document.
Runtime copies live below `mtlx/standard`, `mtlx/libraries`, and `mtlx/snippets`; source-tree fallbacks support
development checkouts.

The Python discovery function uses the same loading rules and returns each renderable element's name and MaterialX
type. Material creation can identify a root with `mtlx_node_name`. Without an explicit name, root preparation uses
MaterialX renderable discovery and accepted surface-shader or unconsumed BSDF forms. A previewable value output can
be wrapped as a simple material. Selection must resolve to one valid primary root before generation begins. Numeric
AOV outputs do not make a document with one material ambiguous or select another surface. Additional output discovery
is separate: `additional_output_names` defaults to `{"*"}`, and all selected supported AOVs are generated
independently of consumers. [Named material outputs](material-aovs.md) describes selection, typed registration,
lookup-only binding, material-level evaluation, and delivery.

### Graph preparation and generation

Preparation first resolves a material node to its implementation graph. It materializes reachable default geometry
properties, can apply the BSDL layer compatibility rewrite, and later prunes closures that static analysis proves
inactive. Editable inputs are treated as unknown during that analysis, preventing a runtime property from being
compiled away. Generator policy also fixes Falcor's working color space to linear Rec. 709 and distance units to
meters.

The `genslangpt` target builds one shader graph for the primary root and selected numeric outputs, sharing reachable
nodes and keeping the surface in output slot zero. It inventories public inputs, filename textures, emitted
geometry reads, layering and emission behavior, and root opacity. Emitters then produce Falcor graph,
parameter-data, material-instance, and material types around the generated node functions. The result records the
Slang source and module name, concrete type names, parameter metadata, a conservative data-size bound, opacity
classification, and texture resources. Live blind geometry reads reserve Scene-owned IDs during emission rather
than producing a parallel material-to-host request inventory. The final postprocessed source is hashed into generated names, so
source identity rather than the authored material label defines shader identity.

### Canonical names and editable state

Normal compilation may bake unselected values and authored graph names into one specialized module. With
`mtlx_canonicalize_names` enabled, graph nodes, ports, compound functions, and generated bindings are named from a
canonical structural traversal. Graphs with the same topology and code-generation options can therefore produce
the same source and concrete Slang types when authored names differ, or when differing values are editable rather
than baked constants. A changed topology,
editable selection, or generation option remains a different class.

`mtlx_editable_params` controls which supported, unbound inputs become runtime properties. An empty value means no
editable parameters; `*` selects all supported inputs,
while a comma-separated list selects an explicit subset. Public authored property names and defaults are preserved
separately from canonical Slang binding names. Supported scalar, vector, matrix, and filename inputs carry type and
UI metadata through `MxParamInfo`; filename inputs become managed texture resources. Canonicalization does not
promote inputs or alter liveness. The old `mtlx_class_compilation` property is rejected rather than silently mapped
to a different editability policy.

### Runtime material and resources

When code-generation-affecting properties change, `MaterialXMaterial` regenerates the module description and
rebuilds its dynamic property set while preserving values for still-matching properties. Resource loading registers
generated texture entries with the scene `TextureManager`, including resolver context and sampler state. MaterialX
generated code owns color conversion, so these textures are not independently marked as sRGB by the manager.

During update, shared MaterialX lookup-table scene globals are installed, generation runs as needed, and the device
loads the resulting Slang source. Falcor reflects the exact target layout of the generated parameter-data type and
checks it against the target-independent upper bound. Small data stays in the common material payload; larger data
lives in a `ManagedBuffer` referenced by the material record. Editable values are written through reflected cursors.
The generated module, type conformance, texture handles, and opacity classification then participate in the same
scene update and requirements path as every other material. Geometry supplies attribute data; material codegen
reserves its named IDs without inventing geometry streams.

### Geometry properties: fixed routes and exact names

MaterialX uses the production `SurfaceInteraction` and `IGeomPropAccessor`, also used by built-in materials. Direct
position, normal, tangent and bitangent nodes read the interaction's world-space fields, converting only when
another space is requested. A vector2 `texcoord` at index zero reads `si.uv`; additional UV sets and geometry color
use the fixed attribute classification shared with geometry. This is not a separate MaterialX semantic registry.

A `geompropdef` can map a named MaterialX property onto one of these routes. For example, a vector2 definition
named `lightmap`, with `geomprop="texcoord"` and `index="1"`, makes `geompropvalue("lightmap")` read the same UV set
as `texcoord(index=1)`. The alias name need not match the source primvar. Without a semantic definition, a name is a
literal contract with the geometry author: `Pref` and `pref` are different properties. Falcor assigns no intrinsic
meaning or coordinate transform to a blind property named `Pref`.

Direct `viewdirection` nodes and semantic aliases share the same world, object, and model routes. Omitted space
defaults to world: the existing `-si.wi_ws` helper needs no geometry accessor. Object/model conversion transforms that
direction as a vector, without translation, then normalizes to account for nonuniform instance scaling. It does not
use the normal's inverse-transpose rule. The source remains the current incident ray, not a separate direction toward
the primary camera at secondary hits.

`MaterialXMaterial` supplies codegen with a resolver backed by the Scene registry. Either a geometry source or a
material request can reserve a name first; both receive the same stable ID. Material-only reservation does not mark
data present or establish its storage type. A source later fills the signature, and conflicting source signatures
are rejected. Each emitted live node calls the accessor with this ID and its own fallback. No mock provider,
parallel request wrapper, or graph-wide availability/value staging is involved. Unused or pruned reads do not
reserve IDs or emit lookups.

The accessor evaluates the current geometry at the hit. Numeric float data interpolates; integer and Boolean
values use the geometry accessor's discrete selection rule. Missing or incompatible data, or an invalid
interaction, returns the caller's default. A Scene-wide name reservation does not imply that every geometry carries
that attribute. The existing triangle accessor prepares indices and interpolation weights eagerly, but a missing
presence bit returns before descriptor payload and raw attribute loads. Multiple authored reads may independently
construct accessors; the implementation does not promise cached reads.

USD UV1-through-UV7 mapping is an importer convention, not a USD semantic guarantee. `ImportOptions` exposes
`usd_texcoord_primvar_names`, defaulting to `st1` through `st7`. Names are case-sensitive, an optional leading
`primvars:` is accepted, and an empty entry disables that set. Other supported float32 scalar/vector primvars retain
their exact names through tessellation and Scene geometry. Indexed and face-varying data, including seams, is
resolved into the renderer's vertex representation. USD `uniform` face values can be expanded into that
representation; this does not implement MaterialX uniform metadata. glTF's numbered texture-coordinate sets already
have fixed indices, so MaterialX reads the imported index rather than matching a USD name.

Live `geompropvalueuniform` nodes, uniform `geompropdef` reads, unit metadata, mismatched definition types, and
unsupported semantic tokens are diagnosed. Matrix primvars are not generic per-vertex reads. Per-instance
transforms instead come from the accessor's typed `GeomPropSpace` route: model/object map to local geometry space,
and world to scene space. Points include translation, vectors do not, and normals use inverse transpose and
normalization. MaterialX `transformnormal` normalizes even for same-space or unavailable-transform identity paths.
Geometry selectors (name, index, space) are immutable code-generation inputs, not editable data; numeric node
defaults remain eligible for editability.

`SceneConfig.enable_custom_attributes` remains a Boolean. With it disabled, the interaction's conditional packed
hit is absent and extra geometry data does not enable shader access. With it enabled, data availability is decided
at runtime for each geometry. Materials that make no read emit no accessor call, but enabled packed-hit/backend
costs still exist; this is not an automatic feature-selection mode.

## Runtime flow

```text
MaterialX XML or file
        |
        v
load libraries + select primary root and numeric outputs
        |
        v
prepare graph + classify inputs, opacity, and streams
        |
        v
emit and hash generated Slang module
        |
        +------ editable metadata ------> dynamic properties
        +------ filename inputs --------> TextureManager
        +------ live blind reads -------> Scene-owned attribute IDs
        |
        v
load module + reflect exact data layout
        |
        v
MaterialSystem record -> dynamic IMaterial dispatch
```

1. Discovery or material creation loads and validates the document using an explicit asset-resolution context.
2. Root and graph preparation combine the primary renderable and selected numeric outputs into a form understood by
   Falcor's generator and static analyses.
3. Generation emits one specialized or canonical class module and records all host-side data needed to instantiate
   it.
4. Resource loading and scene update realize textures, shared tables, generated parameter storage, and required
   shader modules.
5. `MaterialSystem` serializes the concrete material record. A renderer links the scene requirements and reaches
   the generated material instance through ordinary `IMaterial` dispatch.

## Invariants and rationale

### Shader source belongs to the library

Shader source is supported only in the MaterialX libraries loaded by Falcor, including `custom_nodes.mtlx`.
User material documents may instantiate library nodes and define custom nodegraphs, including graph-backed
`implementation` declarations. They must not supply shader source. The loader throws if an authored
`implementation` has a `sourcecode` or `file` attribute, including declarations from user XML includes,
unused declarations, and declarations targeting another backend. This check runs before Falcor imports its
libraries and applies to both material discovery and generation.

Importers that construct MaterialX documents must keep library definitions attached for lookup rather than copying
them into the serialized material. The generated XML must not embed library source or include library files.

Library source can change between process runs. Library definitions and source text are fixed during a run;
hot reloading them is unsupported. A new process reads the updated library, and changed emitted shader source
receives a new source-derived module identity. Material-supplied shader source is deliberately outside the supported
contract: do not add source fingerprinting, hot-reload invalidation, or defensive custom-source machinery for it.
New shader implementations belong in the library; user graph composition remains supported.

### Default backend and validation policy

The primary consumer of every MaterialX feature is the build's default layering backend.
`mtlx::default_layering_mode()` is the implementation authority; ordinary
material construction and `CodeGenDesc` use it without an override. These are MaterialX layering modes, not GPU
APIs such as D3D12 and Vulkan.

Substantial MaterialX work must first function on the default backend, then be verified on the other supported
backends available in that build. Passing only a non-default mode is not acceptance of a MaterialX change.
Changing the mode to make a failing feature pass hides a failure for the primary
consumer rather than fixing it.

Primary functional and regression tests leave `mtlx_layering_mode` unset (or use the existing default selector in
C++). Explicit overrides belong in clearly identified compatibility or backend-specific tests. Keep representative
functional coverage of the other modes, but do not multiply every importer, graph, editability, and device case
across all modes when the behavior is shared. Test shader compilation and runtime behavior on the supported GPU APIs
where needed; mode coverage does not substitute for GPU-API coverage. Pure code-generation assertions should call
the generator and inspect its output without constructing a device or Scene, loading a Slang module, or dispatching
GPU work. A GPU compilation check is a separate integration test, not an incidental cost of a codegen test.

The same default-first rule applies to examples, disposable proofs, and performance investigations unless the
question explicitly targets another backend. Reports must name the layering mode and GPU API independently.

### Generated code and runtime contracts

Scattering follows the [component contract](bsdf-contract.md#contribution-identity-and-sample-weights).

The `bsdf_mix` and `closure_tree` material boundaries expose full continuous response
with layout zero. Internally, weighting and orthonormal frame wrappers forward child component selectors, while
layer, mix, and add composition collapses continuous samples to full response. These strategies satisfy the same
public contract without requiring every backend to use the same estimator.

Generated source is the unit of shader identity. The source hash is applied after postprocessing, and all options
that affect emitted code participate in that source. Name canonicalization improves reuse only for structurally
equivalent graphs under equivalent options; it is not a promise that arbitrary documents share a module.

Runtime editability is an ABI choice as well as a user-interface choice. Making an input editable moves it out of
compile-time specialization and into generated data, so editable selection contributes to class identity. Static
analysis must remain conservative around such inputs. Canonical binding names may change independently of public
property names, which is why the implementation retains both identities.

The loaded target module is authoritative for memory layout. The generator's size calculation is deliberately only
a portable upper bound; reflection determines the exact stride used by D3D12, Vulkan, or CUDA. This prevents C++
packing assumptions from becoming an undocumented cross-backend ABI.

Asset resolution is explicit and ordered. A document's path, include roots, libraries, and image filenames must be
resolved through the same constructed context; process working directory is not a substitute. Root opacity is a
generation output that can alter traversal even when the public material interface remains unchanged. Built-in
geometry properties use shared fixed routes; custom properties reserve Scene-owned IDs by exact name. Geometry
still owns the offered values. The old `mtlx_geomprop_names`/`mtlx_geomprop_ids` mockup mapping is not the runtime
contract.

## Source landmarks

- `src/falcor2/render/material/mtlx/README.md`, `src/falcor2/render/material/mtlx/mtlx_material.h`, and
  `src/falcor2/render/material/mtlx/mtlx_material.cpp` - document properties, asset resolution, regeneration,
  runtime resources, reflected data storage, lookup tables, and `MaterialSystem` integration.
- `src/falcor2/render/material/mtlx/codegen/codegen_types.h`,
  `src/falcor2/render/material/mtlx/codegen/codegen.cpp`, and
  `src/falcor2/render/material/mtlx/codegen/structured/generator.cpp` - generation inputs and results, document loading, graph
  creation, code emission, hashing, opacity, editable data, and geometry ID resolution.
- `src/falcor2/render/material/mtlx/codegen/structured/document/` and
  `src/falcor2/render/material/mtlx/codegen/structured/graph/` - root selection, default geometry properties,
  compatibility rewrites, reachability, static input analysis, and closure pruning.
- `src/falcor2/render/material/mtlx/codegen/mx_parameters.cpp` and
  `src/falcor2/render/material/mtlx/codegen/structured/layering/` - editable parameter metadata and generated Falcor wrappers.
- `src/falcor2/render/material/mtlx/codegen/structured/target/nodes/geometry_nodes.cpp`,
  `src/falcor2/render/geometry/geometry_attribute_registry.cpp`, and
  `slang/falcor2/render/geomprop_accessor.slang` - intrinsic/alias/blind dispatch, Scene data registration, typed
  caller defaults, and hit-backed geometry access.
- `src/falcor2/importers/usd_importer/usd_importer_mesh_adapter.cpp` and
  `tests/python/render/test_materialx_geometry_properties.py` - named primvar/UV import and default-first functional
  integration, with representative secondary layering-mode coverage.
- `src/falcor2_ext/render/mtlx_codegen.cpp` - Python discovery and lookup-table binding helpers.
- `slang/falcor2/render/materials/mtlx/` - Falcor's runtime MaterialX interfaces, BSDF support, composition, and
  shared lookup-table declarations.
- `tests/native/render/test_materials.cpp`, `tests/python/render/test_materials.py`, and
  `tests/python/render/test_materialx_data_layout.py` - graph identity, class reuse, editable subsets, resources,
  opacity, geometry properties, reflected strides, and cross-backend data layout.

## Related design documents

- [Falcor2 system overview](overview.md)
- [Scene model](scene-model.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Geometry and ray tracing](geometry-and-ray-tracing.md)
- [Material system](material-system.md)
- [Named material outputs](material-aovs.md)
- [BSDF contract](bsdf-contract.md)
- [Python and native integration](python-native-integration.md)
- [Falcor2 design documentation](index.md)
