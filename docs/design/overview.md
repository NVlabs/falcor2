---
status: current
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-08
verified_against: "edb87e89d207b940c7d67b3d0c1590532cd84bf6"
---

(falcor2-design-overview)=
# Falcor2 system overview

Falcor2 is a Python-first rendering toolkit whose runtime joins Python orchestration, a native C++ rendering library,
a nanobind extension, SlangPy/SGL GPU services, and Slang shader modules. Python code assembles scenes, render nodes,
and presentation workflows; native code owns performance-sensitive scene, material, geometry, importer, and
ray-tracing systems; Slang programs perform GPU work through the same SGL device. Runtime resources make those
layers discoverable, while examples and tests exercise the assembled system. This page defines those top-level
boundaries and the vocabulary used by the more focused design documents.

## Purpose and scope

This page answers how the repository's principal runtime layers form one usable system. It covers package startup,
the boundary between Python and native code, the role of SlangPy/SGL and Slang, and the place of resources, examples,
and tests. It intentionally does not describe the internal scene model, render-pipeline stages, import process,
geometry acceleration structures, material and lighting contracts, or detailed binding ownership. Dedicated pages
in the design catalog own those questions.

## Architecture

The public `falcor2` package is the composition layer. `falcor2/__init__.py` first imports `slangpy`, prepares
the native library search path on Windows, loads `falcor2.falcor2_ext`, and imports `falcor2.extensions`. It then
exports selected Python render-graph and editor conveniences. As a result, a user sees one package containing both
native types such as `Scene` and Python types such as `RenderNode` and `PathTracerPipeline`.

The native implementation is split into two build targets. `src/falcor2/` builds the `falcor2` shared library,
which contains core utilities and rendering systems and links publicly to SGL. `src/falcor2_ext/` builds the
`falcor2_ext` nanobind module and links it to that library. The extension does not expose a parallel user-facing
namespace: its module initializer imports the existing `falcor2` package and registers native classes, functions,
constants, and submodule members into it. The extension also brackets process use with `falcor::static_init()` and
`falcor::static_shutdown()`.

SlangPy is the Python-facing GPU programming layer, and SGL supplies devices, resources, command encoding, shader
loading, and the graphics or CUDA backend beneath it. Falcor2 Python modules use SlangPy objects directly, while the
native library uses the same SGL types. `falcor2.extensions` registers marshalling rules that teach SlangPy how to
bind native Falcor2 values, including cameras, materials, and cursor-writable handles, to matching Slang types.

The `slang/` tree contains Falcor2's GPU modules and shared rendering type contracts. Device helpers add both the
Falcor2 and SlangPy shader roots to Slang's include paths. Render nodes load modules by logical names such as
`falcor2/rendernodes/reference_pathtracer.slang` and link them with scene-generated modules and any additional
modules required by scene contents. CMake installs the Falcor2 Slang tree so logical shader lookup survives
installation, and separately installs MaterialX and USD resources required by the packaged runtime.

The repository's examples are executable compositions of these layers rather than another runtime layer.
`examples/pathtracer/` demonstrates editor-driven and headless pipelines, and `examples/minitracer/` provides a
smaller rendering stack built on the same device and shader foundations. `data/scenes/` and `data/assets/` supply
representative scene descriptions and assets for examples and tests. Native and Python test suites verify their
respective APIs, while end-to-end Python tests exercise native scenes through SlangPy render nodes.

The main relationships are:

```text
 application, example, or editor loop
                  |
                  v
    falcor2 Python orchestration and render nodes
         |                         |
         v                         v
  falcor2_ext bindings       SlangPy modules and dispatch
         |                         |
         v                         v
    falcor2 C++ library <-------> SGL device and resources
                  \                 /
                   v               v
                linked Slang GPU programs
```

## Runtime flow

1. Importing `falcor2` initializes SlangPy before loading the Falcor2 extension. The extension populates the
   package with native bindings, after which Falcor-specific SlangPy marshallers are registered.
2. Application code creates one `slangpy.Device` through a Falcor2 helper or directly with the required include
   paths. The device is passed to native scenes and Python render nodes, giving both layers access to the same GPU
   session and resources.
3. A scene is created or loaded through the native API. Importers and native scene systems build host-side objects,
   device records, and the scene-specific shader requirements used by renderers.
4. Python render nodes select output containers, load Falcor2 Slang modules, and configure specialization. For the
   reference path tracer, `SceneShaderHelper` links the renderer module with common utilities, the scene render
   module, and requirement modules, then supplies a callback that binds the scene to the shader cursor.
5. Calling a pipeline updates the scene, resolves a camera, and dispatches GPU work through SlangPy and SGL. Results
   can remain in SGL textures or SlangPy tensors, or use a supported Torch container at the Python boundary.
6. An interactive workflow gives the final image to `Editor.present()` inside an update loop. A headless workflow
   accumulates frames and writes or returns the resulting container without creating an editor window.

## Invariants and rationale

The native and Python rendering layers share a device rather than hiding separate GPU contexts. Native scene
resources, SlangPy dispatch, and output containers must therefore belong to compatible device state. Passing the
device explicitly through scene and render-node construction makes that ownership visible.

Package import order is part of the integration contract. SlangPy loads before `falcor2_ext`, and the extension
binds into the already-created `falcor2` package. Falcor-specific marshalling registration follows native type
creation. This produces a unified Python surface while preserving separate native and Python implementation areas.

Slang source is a runtime dependency, not merely build input. Device include paths and installed resource layout must
preserve logical module names. Scene requirements can change the set of linked modules and type conformances, so
render code links against the scene's current shader representation instead of assuming one fixed global program.

Python owns high-level composition where rapid iteration and container interoperability are valuable. Native C++
owns long-lived scene data, importers, ray-tracing setup, and other systems that need explicit resource management or
close SGL integration. Slang owns parallel GPU behavior. This is an ownership guideline rather than a rule that every
feature must exist in only one language; the binding and marshalling layers are intentional bridges.

Examples demonstrate supported compositions, and tests are behavioral evidence, but neither defines architecture by
itself. The implementation at the recorded commit is authoritative when examples, tests, plans, or prose disagree.

## Source landmarks

- `falcor2/__init__.py` - package initialization, native extension loading, and top-level Python exports.
- `falcor2/_package_paths.py` and `falcor2/editor/utils.py` - source/install resource lookup and device include
  paths.
- `falcor2/extensions/` - Falcor-specific SlangPy marshalling registration.
- `falcor2/rendergraph/render_node.py` - the callable Python render-node protocol.
- `falcor2/editor/scene_shader.py` - runtime linking of renderer, scene, utility, and requirement modules.
- `falcor2/rendernodes/pathtracer_pipeline_node.py` - representative high-level Python pipeline composition.
- `src/falcor2/CMakeLists.txt` and `src/falcor2/falcor2.cpp` - native library contents, dependencies, resources,
  and process-wide initialization.
- `src/falcor2/render/scene.h` - the central native scene interface consumed by renderers.
- `src/falcor2_ext/CMakeLists.txt` and `src/falcor2_ext/falcor2_ext.cpp` - extension build, generated stubs, and
  binding injection into the package.
- `slang/falcor2/render.slang` and `slang/falcor2/rendernodes/reference_pathtracer.slang` - shared rendering
  declarations and a representative renderer module.
- `examples/pathtracer/simple.py` and `examples/pathtracer/headless.py` - interactive and headless composition.
- `tests/python/pathtracer/test_pipeline.py` and `tests/native/render/test_scene.cpp` - cross-layer pipeline and
  native-scene behavioral evidence.

## Related design documents

- [Falcor2 design documentation](index.md)
- [Scene model](scene-model.md)
- [Render pipeline](render-pipeline.md)
- [Scene loading and authoring](scene-loading-and-authoring.md)
- [Python and native integration](python-native-integration.md)
