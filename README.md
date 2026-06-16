# Falcor2

## Requirements

- CMake installed (and on path)
- Visual Studio 2022 (for windows)

## Installing

As Falcor2 is a Python package, it is recommended to create a virtual environment to work in before starting, as the setup process will install editable packages to the active Python environment.

To get started, run `setup.bat` on Windows or `setup.sh` on Linux. This will:
- Sync all submodules
- Initialize the VSCode workspace
- Install Python packages used for development
- Install editable python packages for Falcor2 and SlangPy
- Build native extensions.

Note: The final 3 steps can be repeated by running `tools/pipinstall.bat`.

The initial install will configure and build release versions of the native libraries and bindings.

Built bindings are installed to their corresponding Python package folders (`./falcor2` and `./external/slangpy/slangpy`). These will always point to your most recent cmake build.

Once installed, the `slangpy` and `falcor2` will be available to import, and can be used both from within the Falcor2 codebase, or another Python package using the same environment.

## Developing

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
