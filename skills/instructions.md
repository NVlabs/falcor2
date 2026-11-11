# Working with Falcor2

This file describes the installed Falcor2 version's general usage. The consuming
project's instructions govern its layout, environment, and development workflow.

## Environment and API

Falcor2 is a Python and native rendering toolkit using SlangPy for GPU execution
and Slang shaders. Use the project's selected Python environment and dependency
manager. Check the installed version and import location before diagnosing API
mismatches; a wheel, editable checkout, and source submodule can expose different
versions. Do not use an agent's bundled Python as the application's environment.

Use the installed version's APIs and type stubs. Import Falcor2
as `import falcor2 as f2`. `falcor2.editor.create_device` configures shader include
paths; `load_scene` loads scenes; `falcor2.rendernodes.PathTracerPipeline` provides
path tracing and accumulation; `save_image` saves tensor output. A loaded scene
needs an active camera. Resolve the application's own assets relative to its
files or explicit user paths, not an assumed Falcor2 repository data checkout.

Do not enable optional PyTorch/CUDA interop, DLSS, or OptiX integrations unless the
task requires them. Package import success is distinct from successful GPU device
creation and from a verified render. Report what was actually checked, and inspect
rendered output when image correctness matters.

## Available workflows

Use the installed skill relevant to the requested task:

- `falcor2-health-check`: verify package installation, native imports, and GPU
  device creation; report driver diagnostics without changing the environment.
- `falcor2-setup-blender-hydra`: inspect a selected Blender build, acquire its
  matching official SDK, build and test HydraIPC, and install or verify the
  Blender extension on Linux or Windows using standalone repository tools.

Skills point to their reusable helpers or repository tools. Skill paths are
relative to the consuming project root. Setup installs guidance; it does not
itself authorize environment changes.

## Updating guidance

Run `f2 setup` (or `python -m runcli setup`) after package upgrades to refresh
the installed guidance using saved choices. Use `--interactive` to revisit choices
and `--dry-run` to preview changes. Keep project-specific guidance outside the
marked Falcor2 sections of AGENTS.md and CLAUDE.md. Setup preserves edits to managed
guidance by reporting conflicts; resolve those edits deliberately before retrying.
