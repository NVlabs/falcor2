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
usd-curve-attributes.md
geometry-and-ray-tracing.md
material-system.md
bsdf-contract.md
materialx-integration.md
material-aovs.md
lighting-and-sampling.md
python-native-integration.md
bsdf-correctness-and-testing.md
```

## Document states

- **`planned`**: The architectural question belongs in the initial corpus, but no page exists yet.
- **`draft`**: The page exists and has been checked against the repository, but still awaits human approval.
- **`current`**: A human has approved the page as the durable account of the design at its recorded verification
  commit.

## Initial document catalog

1. [System overview](overview.md) - **`current`**. How the Python package, native library, extension, SlangPy/SGL,
   Slang shaders, data, tests, and examples form one system.
2. [Scene model](scene-model.md) - **`current`**. How scenes, scene objects, entities and components, scene systems,
   dirty state, requirements, update, and shader binding work.
3. [Render pipeline](render-pipeline.md) - **`current`**. How render nodes, pixel filters and camera jitter,
   accumulation, tone mapping, denoising, containers, and presentation compose.
4. [Scene loading and authoring](scene-loading-and-authoring.md) - **`current`**. How importers, deferred edits,
   `ImporterScene`, USD, glTF, USDZ, asset resolution, PyScene, append, and callbacks interact.
5. [Geometry and ray tracing](geometry-and-ray-tracing.md) - **`current`**. How geometry, typed vertex attributes,
   instances, `RenderScene`, device buffers, per-hit property access, acceleration structures, hit groups, and
   ray-tracing setup relate.
6. [Material system](material-system.md) - **`draft`**. How materials, `MaterialSystem`, `TextureManager`, device
   records, shader requirements, opacity, and native, MDL, and MaterialX implementations fit together, including
   MDL's production geometry inputs.
7. [BSDF contract](bsdf-contract.md) - **`current`**. How BSDF evaluation, sampling, probability measures, fused
   evaluation, composition, and material-instance dispatch behave.
8. [MaterialX integration](materialx-integration.md) - **`current`**. How MaterialX discovery, graph preparation,
   code generation, class compilation, runtime resources, editable parameters, and shader specialization work.
9. [Lighting and sampling](lighting-and-sampling.md) - **`current`**. How light authoring and radiometry, analytic,
   environment, and emissive lights, shaping, `LightSystem`, renderer-owned samplers, and `EmissiveTriangleTree`
   work.
10. [Python and native integration](python-native-integration.md) - **`current`**. How package initialization,
    nanobind injection, native ownership, SlangPy/SGL objects, Python orchestration, and extension modules cross the
    language boundary.
11. [BSDF correctness and testing](bsdf-correctness-and-testing.md) - **`draft`**. How reference authorities,
    deterministic cases, statistical GPU checks, material dispatch, generated code, and rendered integration
    combine to establish confidence in BSDF behavior.
12. [Named material outputs](material-aovs.md) - **`draft`**. How permanent typed declarations, consumer-independent
    MaterialX/MDL generation, material-level evaluation, lookup-only binding, and guide-buffer delivery fit together.

13. [USD curve attribute counts and interpolation](usd-curve-attributes.md) - **`draft`**. How authored varying
    counts, pinned topology, strand offsets, and round-curve sampling stay consistent despite upstream differences.

Build and packaging, CI and testing architecture, editor UI, MCP, denoisers, and user guides are deferred until this
runtime-focused set has been reviewed.

## Contributing

Read [Writing and maintaining design documents](../contributing/design-documents.md) before creating, auditing, or
retiring a design page.
