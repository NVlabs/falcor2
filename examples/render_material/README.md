# Render Material

This example renders MaterialX and MDL material libraries through one shared
suite loop. The MaterialX path intentionally mirrors the upstream MaterialX
render-test suite shape: an `_options.mtlx` file chooses the inputs and render
settings, the suite writes backend-suffixed images, and the finished output
directories can be compared with the report tools.

MDL uses `_options.jsonc` with the same broad shape so both providers can be run
the same way. Options are stored in the recipe files so a user can run the suite
directly instead of reconstructing commands from separate scripts.

## Backends

`renderBackend` selects the renderer:

- `pathtracer` renders OBJ preview geometry with `PathTracerPipeline` and writes
  `_falcor2.png` images.
- `material_visualizer` renders Falcor2's `MaterialVisualizer` preview sphere
  and writes `_visualizer.png` images.

The backend can be set in `_options.mtlx` / `_options.jsonc` or overridden with
`--render-backend`. Material renders require `radianceIBLPath`; primary rays use
the flat MaterialX test-suite background color while the IBL contributes to
lighting and reflections. Preview-property renders such as albedo do not require
environment lighting.

`radianceIBLPath` defaults to the MaterialX test-suite IBL for MaterialX and to
the MDL SDK example IBL for MDL. The byte background color is intentionally
selected to match MaterialX RenderGLSL output.

## Basic Usage

Run commands from the repository root. If native code or shaders changed, build
the Release preset first so the local Python package loads current binaries.

Render the default MaterialX suite with the pathtracer:

```powershell
.\.conda\python.exe examples\render_material\materialx_suite.py `
  --options examples\render_material\materialx\_options.mtlx
```

Render the same recipe through MaterialVisualizer:

```powershell
.\.conda\python.exe examples\render_material\materialx_suite.py `
  --options examples\render_material\materialx\_options.mtlx `
  --render-backend material_visualizer
```

Render MDL the same way:

```powershell
.\.conda\python.exe examples\render_material\mdl_suite.py `
  --options examples\render_material\mdl\_options.jsonc `
  --render-backend pathtracer

.\.conda\python.exe examples\render_material\mdl_suite.py `
  --options examples\render_material\mdl\_options.jsonc `
  --render-backend material_visualizer
```

Use `--samples-per-pixel N` to override the recipe sample count for diagnostic
or documentation runs.

## Representative MaterialX Report

`examples/render_material/materialx/_options_representative.mtlx` is a smaller
curated MaterialX recipe for quick reports. To compare PathTracer against
MaterialVisualizer, run the recipe twice into separate output directories by
copying the options file and changing only `renderBackend` and
`outputDirectory`, or by keeping separate local option files under `.tmp`.

Then generate the report with MaterialVisualizer as the reference:

```powershell
.\.conda\python.exe tools\material_image_compare.py `
  --reference-root .tmp\render_material_mtlx_representative_visualizer `
  --tested-root .tmp\render_material_mtlx_representative_pathtracer `
  --output-dir .tmp\render_material_mtlx_representative_comparison
```

The report tool writes `comparison.html` and `html_assets/`. Rows are sorted
worst-first by raw RGB error, with structural issues first. The diff thumbnail
is amplified, so it is meant for diagnosis rather than as a pass/fail signal.

## Expected Differences

The two Falcor2 backends are not identical renderers. PathTracer renders imported
OBJ preview geometry, while MaterialVisualizer uses its own preview sphere path.
Sampling noise, geometry differences, and unsupported effects such as some
transmission cases can produce visible differences. For documentation runs, use
larger sample counts and keep MaterialVisualizer as the reference image set.
