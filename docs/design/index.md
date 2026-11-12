(falcor2-design-index)=
# Falcor2 design documentation

These pages explain Falcor2's current architecture and the enduring reasons for its shape. They are intended for
human contributors and coding agents that need a concise map before reading implementation details. Plans describe
execution history or future intent; they are not a substitute for the verified current-state design recorded here.

```{toctree}
:hidden:
:maxdepth: 1

overview.md
scene-model.md
render-pipeline.md
scene-loading-and-authoring.md
curve-import-and-tessellation.md
usd-curve-attributes.md
geometry-and-ray-tracing.md
geometry-deformation.md
material-system.md
mdl-integration.md
bsdf-contract.md
materialx-integration.md
material-aovs.md
lighting-and-sampling.md
sampling-inputs.md
python-native-integration.md
buffer-pointers.md
bsdf-correctness-and-testing.md
build-and-runtime-packaging.md
documentation-generation.md
```


## Document states

- **`planned`**: The architectural question belongs in the catalog, but no page exists yet.
- **`draft`**: The page exists and has been checked against the repository, but still awaits human approval.
- **`current`**: A human has approved the page as the durable account of the design at its recorded verification
  commit.

## Document catalog

1. [System overview](overview.md) - **`current`**. How the Python package, native library, extension, SlangPy/SGL,
   Slang shaders, data, tests, and examples form one system.
2. [Scene model](scene-model.md) - **`current`**. How object and animation ownership, deferred lifecycle, camera
   selection, dirty updates, consumer requests and immutable capability snapshots connect to renderer binding.
3. [Render pipeline](render-pipeline.md) - **`draft`**. How renderer requirements, module and dispatch reuse,
   pixel filters, sample batches, image-history resets, denoising, and presentation compose into a frame.
4. [Scene loading and authoring](scene-loading-and-authoring.md) - **`current`**. How recipes, owned importer data,
   borrowed geometry views, pose baking, USD/glTF and MaterialX translation, asset resolution, live realization,
   append and callbacks interact.
5. [Curve import and tessellation](curve-import-and-tessellation.md) - **`draft`**. How shared USD attribute reading
   and curve evaluation produce round segments or ribbon meshes, including refinement, frame repair, UVs and
   fixed motion topology.
6. [USD curve attribute counts and interpolation](usd-curve-attributes.md) - **`draft`**. How authored varying
   counts, pinned topology, flattened strand ranges and interpolation feed shared round/ribbon evaluation,
   including validation exceptions and fixed motion layouts.
7. [Geometry and ray tracing](geometry-and-ray-tracing.md) - **`current`**. How static and dynamic geometry,
   typed attributes, pooled storage, deformation, instances, selective BLAS updates, TLAS policy and hit groups
   connect, including renderer-controlled culling and current-hit endpoint reuse.
8. [Geometry deformation](geometry-deformation.md) - **`draft`**. How typed deformers, shared updaters, resident
   sample storage, native timing and evaluated-output history produce current and previous GPU vertices.
9. [Material system](material-system.md) - **`current`**. How material records, shared and family-owned storage,
   texture interpretation and cache identity, shader requirements, opacity and material-family interfaces
   synchronize authored edits with rendering.
10. [MDL integration](mdl-integration.md) - **`draft`**. How SDK target reuse, independent arguments and resources,
    generated wrapper identity, per-hit geometry inputs and conservative emission classification reach rendering.
11. [BSDF contract](bsdf-contract.md) - **`draft`**. How fused evaluation and field hints, joint versus marginal
    densities, delta events, renderer-owned weight policy and component-preserving composition connect to transport.
12. [MaterialX integration](materialx-integration.md) - **`draft`**. How authored inputs, library provenance,
    generated-code reuse, shared closure storage, live material data, emission and geometry requirements connect.
13. [Named material outputs](material-aovs.md) - **`current`**. How permanent typed declarations, consumer-independent
    MaterialX/MDL generation, synchronized capability selection, material-level evaluation and guide delivery fit together.
14. [Lighting and sampling](lighting-and-sampling.md) - **`current`**. How authored and emissive lights, analytic
    surfaces, synchronized distributions, renderer-owned samplers, retained endpoints and emitter-hit MIS interact,
    including visibility and hidden-surface continuation.
15. [Sampling inputs](sampling-inputs.md) - **`current`**. How callers own fixed input budgets, nested selectors
    reuse conditional coordinates and materials isolate auxiliary randomness within a sequential sample stream.
16. [Python and native integration](python-native-integration.md) - **`current`**. How package initialization,
    shared object ownership, native and Python reflection paths, interned-string identity, shader marshalling,
    and helper preparation cross the language boundary.
17. [Buffer pointers and indirect GPU access](buffer-pointers.md) - **`draft`**. BufferPtr ownership, host/shader
    transport, backend representation, layout, alignment and conversion contracts, including current geometry,
    deformation and material consumers and the limits of backend support.
18. [BSDF correctness and testing](bsdf-correctness-and-testing.md) - **`draft`**. How model authorities,
    independent draws, property checks, material dispatch and rendered assertions combine, including numerical
    exceptions and the distinction between execution tiers and coverage.

20. [Build and runtime packaging](build-and-runtime-packaging.md) - **`current`**. How Python ABI selection,
    private USD/TBB identities, configuration-specific deployment, resource discovery and editor metadata compose
    a coherent source or installed package.

23. [Documentation generation and publication](documentation-generation.md) - **`current`**. How native help,
    selected runtime APIs, Sphinx assembly, wheel labels and cumulative version history produce the documentation site.

## Contributing

Read [Writing and maintaining design documents](../contributing/design-documents.md) before creating, auditing, or
retiring a design page.
