# Falcor2

Falcor2 is a rendering toolkit for path tracing and rendering research. It includes a Python package,
a native C++ library, and a native Python extension. See [Building from source](#building-from-source)
to set up a checkout.


## Documentation

Start with the [documentation sources](docs/index.md) for guides, architectural context, and the API reference.
See [Building documentation](docs/contributing/building-documentation.md) to generate and preview the HTML site.
See [Streaming](docs/streaming.md) to enable streaming for local builds and pip installations.


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
- `src/falcor2`: Falcor2 native code
- `src/falcor2_ext`: Falcor2 native Python extension
- `external/slangpy/slangpy`: SlangPy Python code (plus built bindings)
- `external/slangpy/src/sgl`: SGL (Slang graphics library) native graphics library used by SlangPy
- `external/slangpy/src/slangpy_ext`: SlangPy native Python extension
- `external/slangpy/external/slang-rhi`: Low level RHI used by SGL
