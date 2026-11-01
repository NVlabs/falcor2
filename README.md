# Falcor2

## Getting started (NVIDIA internal)

The two easiest ways to get started are the AI setup plugin and the published Python packages below.
Both use NVIDIA's internal services. See the [Falcor2 documentation](https://nvresearch-gfx.gitlab-master-pages.nvidia.com/Tools/falcor2/latest/)
for guides and the Python API reference.

### AI-assisted setup

for Codex and Claude Code is the easiest option if you want help getting a working environment.
Follow its README to install the plugin, open your intended project folder in your assistant, and ask:

```text
Use setup-falcor2 to set up Falcor2 in this folder.
```

The assistant checks your machine and GPU, creates or reuses an isolated Python environment, installs
Falcor2, and verifies native imports and GPU device creation. It can also add Falcor2 to an existing app,
set up PyTorch integration, or guide a source installation if you need to modify Falcor2 itself.
You do not need to clone Falcor2 first.

### Install with pip

Activate a project-local venv or Conda environment first; Python 3.12 is a good starting point.
Install Falcor2 and its matching SlangPy dependency from the internal package index:

```sh
python -m pip install --index-url https://artifactory.pdx.nvidia.com/artifactory/api/pypi/sw-rtr-pypi-virtual/simple --only-binary=falcor2,slangpy falcor2
```

This uses prebuilt packages, so the core install does not require a Falcor2 checkout or a native build.
On Linux with Conda, currently add the active environment's libraries to the search path before using
Falcor2 (a temporary workaround for locating `libpython`):

```sh
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

For PyTorch integration, install a suitable PyTorch build into the same environment first, then install
the `slangpy-torch` native extension:

```sh
python -m pip install "setuptools>=64" wheel ninja
python -m pip install --index-url https://artifactory.pdx.nvidia.com/artifactory/api/pypi/sw-rtr-pypi-virtual/simple --no-build-isolation slangpy-torch
```

Unlike the core wheels, this extension compiles against your installed PyTorch and needs a C++ compiler.
On Windows, run its install from a Visual Studio 2022 x64 Native Tools environment with your Python
environment activated. Use releases with the same `+rtr.<pipeline>` suffix when pinning Falcor2 and
`slangpy-torch`, and rebuild the extension after changing PyTorch.

## Documentation

Start with the [published Falcor2 documentation](https://nvresearch-gfx.gitlab-master-pages.nvidia.com/Tools/falcor2/latest/)
for guides, architectural context, and the API reference. The [documentation sources](docs/index.md)
are also available in this repository.
See [Building documentation](docs/contributing/building-documentation.md) to generate and preview the HTML site.
See [Streaming](docs/streaming.md) to enable streaming for local builds and pip installations.
See [Blender Hydra integration](docs/blender-hydra.md) to run `f2 hydra install blender`
from an existing checkout, using a prebuilt native library or a source build.

## MaterialX comparison tools

The standalone `MaterialVisualizer` and its sphere tracer have moved to
The former `falcor2.tools.materials.material_tools` import is no longer provided by Falcor2.
Use the comparison repository's setup instructions and `tools/compare.py` to compare the copied
visualizer, Falcor2's reference pathtracer, and the official MaterialX OSL and GLSL implementations.
The comparison project owns the visualizer and its tests; Falcor2 retains its MaterialX runtime,
reference pathtracer, and core material tests.

## Building from source

Use a source build when developing Falcor2 or when you need changes that are not yet in a published package.
The following instructions build the native libraries and install editable Python packages from this checkout.

### Requirements

- CMake installed (and on path)
- Visual Studio 2022 (for windows)

### Installing from source

As Falcor2 is a Python package, it is recommended to create a virtual environment to work in before starting, as the setup process will install editable packages to the active Python environment.

To get started, run `setup.bat` on Windows or `setup.sh` on Linux. This will:
- Sync all submodules
- Initialize the VSCode workspace
- Install Python packages used for development
- Install editable python packages for Falcor2 and SlangPy
- Build native extensions.
- Install AI skills for Codex and Claude without prompts or instruction-file changes.

Repeat package installation/builds with `tools/pipinstall.bat`. Refresh AI skills
with `f2 setup` or `python -m runcli setup`. Source setup accepts `--no-skills`
and installer options such as `--agent both --instructions`. Choices are saved;
use `--interactive` to revisit them or `--non-interactive` for unattended setup.
See [AI skills](docs/ai-skills.md)
for pip installs, submodules, profile selection, and conflict recovery.

The initial install will configure and build release versions of the native libraries and bindings.

Built bindings are installed to their corresponding Python package folders (`./falcor2` and `./external/slangpy/slangpy`). These will always point to your most recent cmake build.

Once installed, the `slangpy` and `falcor2` will be available to import, and can be used both from within the Falcor2 codebase, or another Python package using the same environment.

### Developing from source

Falcor2 is designed to be easy to work with from source. Python packages are installed as 'editable', and automatically updated after any succesful CMake build. As such, you can safely modify source code and expect to see your changes take effect immediately (or after a build in the case of native code).

Folder structure:
- `falcor2`: Falcor2 Python code (will also contain latest built bindings)
- `tools`: Useful scripts
- `src/falcor2/falcor2`: Falcor2 native code
- `src/falcor2/falcor2_ext`: Falcor2 native Python extension
- `external/slangpy/slangpy`: SlangPy Python code (plus built bindings)
- `external/slangpy/src/sgl`: SGL (Slang graphics library) native graphics library used by SlangPy
- `external/slangpy/src/slangpy_ext`: SlangPy native Python extension
- `external/slangpy/external/slang-rhi`: Low level RHI used by SGL
