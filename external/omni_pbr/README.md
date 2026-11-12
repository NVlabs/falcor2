# OmniPBR and OmniSurfaceLite MDL library

The NVIDIA OmniPBR material family, OmniSurfaceLite, and their nonstandard MDL module dependencies.
This is original MDL source, not distilled or baked materials.

## License

All `.mdl` files in this directory and its subdirectories are licensed under
**BSD-3-Clause**. Their original copyright and SPDX headers are preserved.
The complete license text is in [LICENSE](LICENSE).

The license permits use, modification, and redistribution, subject to retaining
the copyright notices, license conditions, and disclaimer, and the restriction
on using the copyright holders' or contributors' names for endorsement.

This bundle contains only BSD-3-Clause files. It does not contain the upstream
MaterialX components covered by Apache-2.0 or any scene textures.

## Provenance

- Upstream: NVIDIA Omniverse Core Materials
- Package version: **2.1.4**
- Revision: **08dd9e79b494ae8511a02910333d6352e1b68be2** (2026-08-25)
- Redistributed license: [LICENSE](LICENSE)

| Local file | Upstream file at the pinned revision |
| --- | --- |
| `OmniPBR.mdl` | `source/Base/OmniPBR.mdl` |
| `OmniPBRBase.mdl` | `source/Base/OmniPBRBase.mdl` |
| `OmniPBR_ClearCoat.mdl` | `source/Base/OmniPBR_ClearCoat.mdl` |
| `OmniPBR_ClearCoat_Opacity.mdl` | `source/Base/OmniPBR_ClearCoat_Opacity.mdl` |
| `OmniPBR_Opacity.mdl` | `source/Base/OmniPBR_Opacity.mdl` |
| `OmniSurfaceLite.mdl` | `source/Base/OmniSurfaceLite.mdl` |
| `OmniSurface/OmniSurfaceLiteBase.mdl` | `source/mdl/OmniSurface/OmniSurfaceLiteBase.mdl` |
| `OmniSurface/OmniShared.mdl` | `source/mdl/OmniSurface/OmniShared.mdl` |
| `OmniSurface/OmniImage.mdl` | `source/mdl/OmniSurface/OmniImage.mdl` |
| `nvidia/aux_definitions.mdl` | `source/mdl/nvidia/aux_definitions.mdl` |
| `nvidia/core_definitions.mdl` | `source/mdl/nvidia/core_definitions.mdl` |
| `nvidia/support_definitions.mdl` | `source/mdl/nvidia/support_definitions.mdl` |

The MDL files are copied without source changes. Only their directory layout is
combined into a single MDL search root. `LICENSE` reproduces the BSD-3-Clause
text from the upstream package license. The MDL SDK supplies standard modules
such as `base`, `df`, `state`, `math`, `tex`, and `anno`.

## Use in Falcor2

Falcor deploys this directory beside its native runtime and inside its Python
package as `mdl_libraries/omni_pbr`. An empty `mdl_library_path` uses configured
defaults, including this fallback. Set `mdl_material_name` to
`::OmniPBR::OmniPBR` or `::OmniSurfaceLite::OmniSurfaceLite`, or select an OmniPBR
clear-coat or opacity variant. Explicit search roots (separated by semicolons)
precede defaults. The SDK's helper modules retain
precedence over the copies here.

Assets can depend on a particular library version; do not assume different versions of the
same MDL module can be switched by changing search paths within one process.
The MDL database retains loaded modules by their qualified names.

