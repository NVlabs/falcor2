(falcor2-api-index)=
# API reference

The reference documents Falcor2's public native bindings and the Python APIs used to build editors, render
pipelines, and scenes. Signatures and descriptions come from the current development build and Python docstrings.

- [Native API](native.rst): scene objects, materials, geometry, importers, reflection, UI, and denoiser bindings.
- [Python API](python.rst): editor helpers, render graph containers and nodes, and PyScene authoring.

Internal tools, testing helpers, experimental modules, and bundled dependency APIs are outside this reference.
An entry without descriptive text is an exported API whose source has no docstring yet.

```{toctree}
:hidden:
:maxdepth: 2

native
python
```
