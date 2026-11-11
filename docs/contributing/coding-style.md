(falcor2-coding-style)=
# Coding style

This guide describes Falcor2's coding conventions for C++, Python, Slang, and the native Python extension.
It is based on the [SlangPy coding-style guide](https://github.com/shader-slang/slangpy/blob/main/docs/src/developer_guide/coding_style.rst),
with consistent naming across languages, automatic formatting, and C++ accessors exposed as Python properties.

Rules have stable identifiers for use in reviews and change descriptions. An identifier stays with its rule when
sections move or wording changes; new rules receive new identifiers rather than renumbering existing ones.

## Scope

<a id="scope-1" name="scope-1"></a>
### SCOPE.1 - Apply conventions within the scope of the change

Follow these conventions in new code. When making small changes to existing code, preserve its established style
where this keeps the change focused and locally consistent. Keep broader renaming, restructuring, and style cleanup
separate from functional changes. Apply the current conventions when introducing a new component or undertaking
an intentional cleanup.

This flexibility applies to style, not correctness, ownership, or API requirements. The configured formatters
still apply. Account for callers and bindings when renaming a public API.

[design documentation](../design/index.md).

## Formatting

<a id="fmt-1" name="fmt-1"></a>
### FMT.1 - Use the configured formatters and source checks

Use the repository's configured formatters: clang-format for C++, Black for Python, and the Slang formatting hook
for Slang. Run pre-commit on changed files before submitting a change, using the active development interpreter:

```text
<python> -m pre_commit run --files path/to/changed_file.cpp path/to/changed_file.py
```

Prefer `.\.conda\python.exe` on Windows or `./.conda/bin/python` on Linux when available. The
[pre-commit configuration](../../.pre-commit-config.yaml) defines hook versions, file selection, and exclusions;
[.clang-format](../../.clang-format) defines C++ and Slang formatting settings. Files covered by the
`check-ascii-source` hook must contain ASCII characters only. Preserve the repository's license headers and
let the existing hooks check their format.

## Comments and API documentation

<a id="doc-1" name="doc-1"></a>
### DOC.1 - Write comments that add information

Write comments that add information the reader cannot readily get from the code. Explain intent, rationale, and
non-obvious behavior; avoid restating names, signatures, or routine operations. Document shared constraints once,
where they apply, rather than repeating them on each member.

<a id="doc-2" name="doc-2"></a>
### DOC.2 - Document useful API contracts

API documentation should explain how to use an API, including relevant constraints, units, ownership, lifetimes,
side effects, and failure conditions. Implementation comments should explain how or why the code works.
Self-explanatory declarations do not need comments. For example, a constructor's initially invalid state may need
explanation, while a comment saying that it constructs the object adds nothing.

For further discussion of comment purpose, see the
[C++ Core Guidelines on comments](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rl-what).

## Source files

<a id="file-1" name="file-1"></a>
### FILE.1 - Choose readable lowercase filenames

Use lowercase C++, Python, and Slang source filenames, separating words with underscores where this improves
readability. Preserve established spellings of compound technical terms. For example, `USDImporter` belongs in
`usd_importer.h` / `usd_importer.cpp`.

<a id="file-2" name="file-2"></a>
### FILE.2 - Guard headers and include the corresponding header first

Use `#pragma once` in ordinary C++ headers. In a C++ implementation file, include its corresponding header first
when one exists. If platform or toolchain headers require a different order, explain the dependency in a nearby
comment. Files intended for repeated inclusion are exempt from the `#pragma once` rule.

## Naming conventions

<a id="name-1" name="name-1"></a>
### NAME.1 - Choose clear names and preserve useful notation

Choose names that are clear in context. Established notation is appropriate when it helps readers follow an
algorithm or formula, even where it differs from the usual naming conventions. Prefer descriptive names for
public APIs and wherever the meaning would otherwise be unclear.

<a id="name-2" name="name-2"></a>
### NAME.2 - Follow identifier naming conventions

| Identifier | Convention | Examples |
| --- | --- | --- |
| Classes, structs, enums, interfaces | UpperCamelCase (PascalCase), with uppercase acronyms and letter-based short forms | `AOVHandle`, `MaterialAOVRegistry`, `MXFlatBSDFDesc`, `MXLUTGlobals`, `USDImporter`, `TypedID` |
| Functions and methods | Lowercase snake_case | `register_material_aov`, `create_dlss_rr_feature` |
| Parameters, local variables, public members | Lowercase snake_case | `aov_handle`, `bsdf_desc`, `material_id` |
| Constants and preprocessor defines | UPPER_SNAKE_CASE | `MAX_SAMPLE_COUNT` |
| Enum values | Lowercase snake_case | `requested_only`, `single_compensated` |
| C++ private and protected data members | `m_` prefix | `m_aov_outputs` |
| Python private members and module-local helpers | `_` prefix | `_aov_requests`, `_prepare_material_aov_requests` |

Slang interface names start with `I`, followed by the type name: `IBSDF`, `IRNG`, `ISampleGenerator`.

### Acronyms and abbreviations

<a id="name-3" name="name-3"></a>
#### NAME.3 - Capitalize acronyms and letter-based short forms

Keep acronyms, initialisms, and letter-based short forms uppercase within type names, regardless of length or
position. For example, use `USDImporter`, `MXFlatBSDFDesc`, and `FooID`: `MX` abbreviates MaterialX and `ID`
abbreviates identifier. Ordinary words and shortened words retain PascalCase, as in `MaterialX` and `Desc`.

In functions and variables, treat each such abbreviation as one lowercase word: `aov_handle`, not `a_o_v_handle` or
`AOV_handle`. The same spelling follows private-member prefixes, as in `m_aov_handle` and `_aov_handle`.
Enum values use lowercase snake_case; constants use UPPER_SNAKE_CASE.

<a id="name-4" name="name-4"></a>
#### NAME.4 - Preserve external API names

Preserve external API names in references and overrides. Falcor2's own wrappers follow this guide, for example
`USDImporter` around a dependency's `UsdStage`. Use the same type-name conventions for Python exports.

## C++ conventions

<a id="cpp-1" name="cpp-1"></a>
### CPP.1 - Use scoped enums

Use scoped enums (`enum class`) for Falcor2's C++ enums.

### Ownership and borrowed values

<a id="cpp-2" name="cpp-2"></a>
#### CPP.2 - Match ownership to the object model

Use `ref<T>` and `make_ref<T>` for shared ownership of Falcor/SGL `Object` instances. They manage object lifetimes
across the native/Python boundary; see [Python and native integration](../design/python-native-integration.md#ownership-across-nanobind-and-sgl).
Do not give these objects a separate `std::shared_ptr` owner. Standard smart pointers are appropriate for other
objects, such as `std::unique_ptr` for an implementation object or `std::shared_ptr` for shared non-`Object` state.
Borrowed pointers and references must not outlive their owners.

<a id="cpp-3" name="cpp-3"></a>
#### CPP.3 - Prefer views for borrowed text and ranges

Prefer `std::string_view` for borrowed text and `std::span<T>` / `std::span<const T>` for borrowed contiguous
ranges when ownership is unnecessary. Use const element types for read-only ranges. Keep owning types where
storage must be retained, and ensure any stored or returned view remains valid for its use.

### Diagnostics

<a id="cpp-4" name="cpp-4"></a>
#### CPP.4 - Use the standard diagnostics and preserve error contracts

Use `FALCOR_CHECK` and its specialized helpers for runtime validation, and `FALCOR_THROW` for ordinary native
failures. Use `FALCOR_ASSERT` and its helpers for internal invariants, not for required runtime validation.
Preserve exception types that are part of an API's contract, such as `std::system_error` for errors carrying an
error code.

## Properties

<a id="cpp-5" name="cpp-5"></a>
### CPP.5 - Use property accessors with appropriate value and lifetime semantics

Use a C++ getter named `property()` and a setter named `set_property(...)`. A getter is const when it does not
modify the object. Choose a return type appropriate to the value and its lifetime: small values can be returned
by value; a reference must refer to storage that remains valid.

For example, C++ code uses:

```cpp
auto translation = transform.translation();
transform.set_translation(translation);
```

The Python API exposes those accessors as a property:

```python
translation = transform.translation
transform.translation = translation
```

<a id="bind-1" name="bind-1"></a>
### BIND.1 - Expose properties through existing reflection and binding helpers

For reflected types, define shared properties through the existing reflection mechanism. Preserve their metadata
and change notifications when exposing them to Python or the editor. See
[Reflection and Python-defined scene objects](../design/python-native-integration.md#reflection-and-python-defined-scene-objects).

For explicit bindings, use `DEF_RO`, `DEF_RW`, and `DEF_PROP_*` helpers where applicable; they include the documentation macro.
Preserve the required return-value and ownership policies. Use `nb::sgl_enum` and
`nb::sgl_enum_flags` for enum bindings as appropriate.

## Slang conventions

<a id="slang-1" name="slang-1"></a>
### SLANG.1 - Use unsuffixed floating-point literals

In Slang, write floating-point literals as `0.0`, `1.0`, and `0.5`, not `0.f`, `1.f`, or `0.5f`.
Include digits on both sides of the decimal point and omit the `f` suffix, including in scientific notation
(for example, `1.0e-7`). Keep integer literals for integer values such as indices and loop bounds.

## Python conventions

<a id="py-1" name="py-1"></a>
### PY.1 - Annotate new and changed parameters

Annotate parameters in new functions and when adding or changing parameters. Changes confined to a function body
do not require annotating the existing signature.

<a id="py-2" name="py-2"></a>
### PY.2 - Mark private implementation details and preserve export conventions

Use `_` for private members and module-local helpers, rather than the C++ `m_` prefix. Do not introduce new
`__all__` declarations; preserve existing declarations unless explicitly refactoring package exports.

<a id="py-3" name="py-3"></a>
### PY.3 - Use the standard SlangPy import alias

Use `import slangpy as spy` when importing the top-level SlangPy module. Specialized imports, including submodule
imports with their own aliases, are appropriate where needed.

## Binding signatures

<a id="bind-2" name="bind-2"></a>
### BIND.2 - Define Python parameter names and defaults explicitly

Name Python parameters with nanobind's `"name"_a` syntax and give their defaults explicitly, for example
`"offset"_a = 0`. C++ default arguments do not define Python defaults. Positional-only parameters and special
methods may omit names where appropriate to their Python calling convention.

## Binding documentation

<a id="bind-3" name="bind-3"></a>
### BIND.3 - Reuse the appropriate documentation source

Do not put documentation directly in a binding when `D(...)` or existing reflection metadata is the appropriate
source. Reuse the description from the native declaration or reflection metadata. Python and Slang APIs without
an existing extraction path use ordinary comments and docstrings, with literal binding help where needed.

| Documentation source | Binding form |
| --- | --- |
| C++ declaration supported by `py_doc` | `D(Class, member)` or the existing `DEF_*` helpers |
| Reflection metadata | Existing forwarding to Python and the editor |
| Python, Slang, or another category without extraction support | Literal help where useful |
| An extractable declaration whose generation tooling is unavailable locally | Temporary `D_NA(...)`, as described below |

<a id="bind-4" name="bind-4"></a>
### BIND.4 - Keep adapter-specific help with the adapter

A forwarding lambda uses the documentation of the API it wraps. Document additional Python-specific behavior
with the adapter when it is not apparent from the signature or existing documentation. Keep shared native API
descriptions on the native declarations and Python-only calling details at the binding layer.

<a id="bind-5" name="bind-5"></a>
### BIND.5 - Preserve useful help without requiring documentation coverage

Using `D(...)` does not require a nonempty description, and exposing an API to Python does not by itself require
new native comments. Preserve useful existing help when replacing literals, moving shared native descriptions to
their declarations where necessary. Apply the [comment guidelines](#comments-and-api-documentation) to new text.
Use the existing extraction tooling; adding support for another category is a separate change.

### Generation and validation

<a id="bind-6" name="bind-6"></a>
#### BIND.6 - Regenerate and validate affected binding documentation

Refresh generated `py_doc.h` using the `falcor2_pydoc` target when comments or declarations
change for APIs using that extraction path. Add supported declarations to the target's inputs as needed; do not
edit generated entries by hand. Build the Release bindings and check the affected Python help and generated stubs.
Reflection descriptions and literal help need the same help checks, without an additional generation step.
In an already configured build, run:

```text
# Windows
cmake --build --preset windows-msvc-release --target falcor2_pydoc
cmake --build --preset windows-msvc-release

# Linux
cmake --build --preset linux-gcc-release --target falcor2_pydoc
cmake --build --preset linux-gcc-release
```

build, and help checks performed in the change description.

### When generation tools are unavailable

<a id="bind-7" name="bind-7"></a>
#### BIND.7 - Reserve temporary D_NA for unavailable generation tools

Use temporary `D_NA(...)` when `D(...)` is appropriate but required LLVM/libclang or other generation tools are
unavailable on the development machine and would otherwise block development. Check installed tools and their
configuration first, including `LIBCLANG_PATH`. Fix input and configuration errors, regenerate stale output, and
resolve parse errors using the existing tooling. Reserve `D_NA(...)` for unavailable tools; categories without
extraction support use direct text.

Put the intended future `D(...)` arguments in `D_NA(...)`, and preserve the source documentation that generation
will use. When replacing useful literal help, move any shared native description to the corresponding declaration
even if generation cannot run yet. The same comment guidelines apply to both `D(...)` and `D_NA(...)`.
Add a nearby TODO identifying the missing tooling and the need to regenerate documentation and restore `D(...)`.
Build the fallback bindings and note the affected APIs and unavailable validation in the change description.

## Relationship to SlangPy

| SlangPy | Falcor2 |
| --- | --- |
| PascalCase types, without an explicit acronym rule | PascalCase with uppercase acronyms and letter-based short forms |
| snake_case enum values | The same; enum values are distinct from UPPER_SNAKE_CASE constants |
| Private members described with `m_` | C++ private and protected members use `m_`; Python uses `_`. |
| clang-format/Black and SlangPy formatting scripts | Falcor2 pre-commit hooks, including Slang formatting, ASCII, and license checks |
| Getter example returning `const T&` | The same accessor naming and Python properties; return types depend on value and lifetime |
| No binding-documentation policy | Existing extraction or reflection where available; see [Binding documentation](#binding-documentation). |
| No argument-annotation or export-list policy | Argument annotations and restrictions on new `__all__` declarations; see [Python conventions](#python-conventions). |

Falcor2 also defines conventions for source files, scoped enums, Slang interfaces, SlangPy imports, ownership,
diagnostics, borrowed values, and binding signatures in the sections above.
