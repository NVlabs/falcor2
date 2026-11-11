# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""54 named glTF configurations; converter cases are a subset, not a cross product."""

from copy import deepcopy
from dataclasses import dataclass
from math import pi
from typing import Any


@dataclass
class SphereCase:
    name: str
    material: dict[str, Any]
    mode: str = "gltf"
    radius: float = 1.0
    mirrored: bool = False
    vertex_color: bool = False
    bc5: bool = False


def sphere_cases() -> dict[str, list[SphereCase]]:
    cases: dict[str, SphereCase] = {}

    def add(
        name: str, *, r: float = 0.4, m: float = 0, base: tuple = (0.8, 0.3, 0.1)
    ) -> SphereCase:
        case = SphereCase(
            name,
            {
                "name": name,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [*base, 1],
                    "metallicFactor": m,
                    "roughnessFactor": r,
                },
            },
        )
        cases[name] = case
        return case

    def ext(case: SphereCase, extension: str, **values: Any) -> None:
        case.material.setdefault("extensions", {}).setdefault(
            "KHR_materials_" + extension, {}
        ).update(values)

    def coat(case: SphereCase, factor: float = 1, roughness: float = 0) -> None:
        ext(case, "clearcoat", clearcoatFactor=factor, clearcoatRoughnessFactor=roughness)

    def film(case: SphereCase, strength: float = 1, thickness: float = 250) -> None:
        ext(
            case,
            "iridescence",
            iridescenceFactor=strength,
            iridescenceIor=1.3,
            iridescenceThicknessMinimum=thickness,
            iridescenceThicknessMaximum=thickness,
        )

    def sheen(case: SphereCase, roughness: float, color: tuple = (0.8, 0.1, 0.2)) -> None:
        ext(case, "sheen", sheenColorFactor=list(color), sheenRoughnessFactor=roughness)

    def sg(case: SphereCase, specular: tuple = (0.04, 0.04, 0.04), gloss: float = 0.6) -> None:
        case.material.pop("pbrMetallicRoughness")
        ext(
            case,
            "pbrSpecularGlossiness",
            diffuseFactor=[0.8, 0.3, 0.1, 1],
            specularFactor=list(specular),
            glossinessFactor=gloss,
        )

    for i, (r, m) in enumerate([(0, 0), (0.4, 0), (1, 0), (0, 1), (0.4, 1), (1, 1), (0.4, 0.5)], 1):
        add(f"C{i:02}", r=r, m=m, base=(0.95,) * 3 if i == 6 else (0.8, 0.3, 0.1))
    sg(add("C08"))
    sg(add("C09"), (0.4, 0.1, 0.02), 0.9)
    case = add("C10")
    ext(case, "ior", ior=2)
    ext(case, "specular", specularFactor=0.3, specularColorFactor=[0.9, 0.2, 0.05])

    coat(add("L01"))
    coat(add("L02", m=1), 0.5, 0.4)
    case = add("L03")
    case.material["normalTexture"] = {"index": 2}
    coat(case, 1, 0.15)
    ext(case, "clearcoat", clearcoatNormalTexture={"index": 3, "texCoord": 1, "scale": 0.4})
    sheen(add("L04"), 0.05)
    sheen(add("L05"), 0.8)
    film(add("L06"), thickness=100)
    film(add("L07", m=1), thickness=400)
    film(add("L08", m=0.5), strength=0.5)
    film(add("L09"), thickness=0)
    case = add("L10")
    coat(case, 0.5, 0.2)
    sheen(case, 0.5, (0.2, 0.4, 0.7))
    film(case, strength=0.5)

    for i in range(1, 9):
        case = add(f"A{i:02}", r=0.03 if i == 1 else 0 if i == 2 else 0.4, m=0 if i == 8 else 1)
        ext(
            case,
            "anisotropy",
            anisotropyStrength=0 if i == 1 else 1 if i == 2 else 0.8,
            anisotropyRotation=pi / 2 if i in (4, 7) else pi / 4 if i == 8 else 0,
        )
    ext(cases["A05"], "anisotropy", anisotropyTexture={"index": 4})
    cases["A06"].material["normalTexture"] = {"index": 3, "scale": 0.7}
    cases["A07"].mirrored = True
    ext(cases["A07"], "anisotropy", anisotropyTexture={"index": 4, "texCoord": 1})
    coat(cases["A08"], 0.6, 0.2)
    ext(cases["A08"], "transmission", transmissionFactor=0.3)

    for i in range(1, 13):
        add(
            f"T{i:02}",
            r=0 if i in (1, 9, 11, 12) else 0.4,
            m=0.2 if i == 8 else 0,
            base=(
                (1, 1, 1)
                if i in (1, 4, 9, 10, 11, 12)
                else (0.6, 0.9, 0.3) if i == 2 else (0.8, 0.3, 0.1)
            ),
        )
    for i in (1, 2, 3, 4, 5, 8, 9, 10, 11, 12):
        ext(
            cases[f"T{i:02}"],
            "transmission",
            transmissionFactor=0.5 if i == 3 else 0.4 if i == 8 else 1,
        )
    ext(cases["T04"], "ior", ior=1)
    ext(cases["T05"], "ior", ior=2)
    coat(cases["T05"], 0.5, 0.15)
    for i, factor in ((6, 1), (7, 0.5), (8, 0.6)):
        ext(
            cases[f"T{i:02}"],
            "diffuse_transmission",
            diffuseTransmissionFactor=factor,
            diffuseTransmissionColorFactor=[0.2, 0.8, 0.4],
        )
    for i in (9, 10, 11, 12):
        ext(cases[f"T{i:02}"], "volume", thicknessFactor=1)
        if i >= 11:
            ext(
                cases[f"T{i:02}"],
                "volume",
                attenuationColor=[0.3, 0.8, 0.5],
                attenuationDistance=2 if i == 12 else 1,
            )
    cases["T12"].radius = 2

    case = add("X01", m=0.7, base=(0.8, 0.6, 0.4))
    case.vertex_color = True
    case.material["pbrMetallicRoughness"].update(
        baseColorTexture={"index": 0}, metallicRoughnessTexture={"index": 1}
    )
    case = add("X02")
    sg(case, gloss=0.8)
    ext(
        case,
        "pbrSpecularGlossiness",
        diffuseTexture={"index": 0},
        specularGlossinessTexture={"index": 5},
    )
    add("X03").material["normalTexture"] = {"index": 6, "scale": 0.7}
    add("X04").bc5 = True
    cases["X04"].material["normalTexture"] = {"index": 2, "scale": 0.7}
    transform = {
        "KHR_texture_transform": {"offset": [0.2, 0.3], "rotation": pi / 4, "scale": [1.5, 0.7]}
    }
    case = add("X05")
    case.material["pbrMetallicRoughness"].update(
        baseColorTexture={"index": 0, "texCoord": 1, "extensions": deepcopy(transform)},
        metallicRoughnessTexture={"index": 1, "texCoord": 1, "extensions": deepcopy(transform)},
    )
    case.material["normalTexture"] = {"index": 2, "texCoord": 1, "extensions": deepcopy(transform)}
    case = add("X06")
    sg(case)
    ext(
        case,
        "pbrSpecularGlossiness",
        diffuseFactor=[0.8, 0.3, 0.1, 0.8],
        diffuseTexture={"index": 0, "extensions": deepcopy(transform)},
    )
    case.material.update(
        alphaMode="MASK",
        alphaCutoff=0.5,
        emissiveFactor=[1, 1, 1],
        emissiveTexture={"index": 0, "extensions": deepcopy(transform)},
    )
    ext(case, "emissive_strength", emissiveStrength=2)
    for i, strength in ((7, 0), (8, 1)):
        add(f"X{i:02}").material["occlusionTexture"] = {"index": 7, "strength": strength}

    for i, (source, mode) in enumerate(
        [
            ("C07", "standard_material"),
            ("T03", "standard_material"),
            ("A03", "openpbr"),
            ("X05", "materialx"),
            ("X02", "mdl"),
            ("A05", "mdl"),
        ],
        1,
    ):
        case = deepcopy(cases[source])
        case.name = f"I{i:02}"
        case.material["name"] = case.name
        case.mode = mode
        cases[case.name] = case
    return {
        group: [case for name, case in cases.items() if name.startswith(prefix)]
        for group, prefix in (
            ("core", "C"),
            ("layers", "L"),
            ("anisotropy", "A"),
            ("transmission", "T"),
            ("textures", "X"),
            ("converters", "I"),
        )
    }
