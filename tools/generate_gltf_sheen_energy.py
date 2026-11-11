# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tabulate the directional albedo of the documented Charlie sheen BRDF.

Model: Enterprise PBR 2021x, "Sheen" and "Energy Compensation / Sheen BRDF":
https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html#components/sheen
https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html#energycompensation/sheenbrdf

The glTF sheen extension refers to this model and its directional-albedo table:
https://github.com/KhronosGroup/glTF/blob/5decc120c95764c319c4f92e7f7ead026d926ef3/extensions/2.0/Khronos/KHR_materials_sheen/README.md#implementation-notes

We use Enterprise's main Charlie model with its fitted masking-shadowing term.
Its distribution roughness r_b is called alpha here. glTF artist roughness r
maps to alpha = r*r; Enterprise clamps alpha to [0.07, 1].

For unit sheen color, f(v,l) = D(h) * V(v,l), with h = normalize(v+l).
The table contains E(mu_v, alpha) = integral f(v,l) * mu_l d(omega_l).
There is no normalization of f, clipping of E, or Sample Renderer V <= 1 clamp.

Rows sample alpha uniformly; columns sample mu_v on a squared grid to resolve
grazing angles. These grid and quadrature choices are local numerical details.
The matching coordinates are in slang/falcor2/render/bsdfs/gltf_sheen.slang.

Run from the repository root: python tools/generate_gltf_sheen_energy.py
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np

TABLE_SIZE = 64
MIN_ALPHA = 0.07
MIN_COSINE = 1e-6


def cosine_grid(size: int = TABLE_SIZE) -> np.ndarray:
    """Table columns, ordered from grazing incidence to normal incidence."""
    return MIN_COSINE + (1 - MIN_COSINE) * np.linspace(0, 1, size) ** 2


def _distribution_roughness(roughness: float) -> float:
    """Map glTF artist roughness to Enterprise's clamped distribution roughness."""
    return float(np.clip(roughness**2, MIN_ALPHA, 1))


def _charlie_distribution(n_dot_h: np.ndarray, alpha: float) -> np.ndarray:
    """Enterprise D_b: (2 + 1/alpha) * sin(theta_h)**(1/alpha) / (2*pi)."""
    sin_theta_h_squared = np.maximum(1 - n_dot_h**2, 0)
    return (2 + 1 / alpha) * sin_theta_h_squared ** (0.5 / alpha) / (2 * np.pi)


def _lambda_sheen(cosine: np.ndarray, alpha: float) -> np.ndarray:
    """Enterprise's piecewise Lambda fit, with coefficients from Conty/Kulla."""
    t = (1 - alpha) ** 2
    a = 21.5473 + (25.3245 - 21.5473) * t
    b = 3.82987 + (3.32435 - 3.82987) * t
    c = 0.19823 + (0.16801 - 0.19823) * t
    d = -1.97760 + (-1.27393 + 1.97760) * t
    e = -4.32054 + (-4.85967 + 4.32054) * t

    def l(x: np.ndarray) -> np.ndarray:
        return a / (1 + b * x**c) + d * x + e

    cosine = np.clip(cosine, 0, 1)
    return np.exp(np.where(cosine < 0.5, l(cosine), 2 * l(np.array(0.5)) - l(1 - cosine)))


def _sheen_visibility(mu_v: float, mu_l: np.ndarray, alpha: float) -> np.ndarray:
    """V = G/(4*mu_v*mu_l), where G = 1/(1 + Lambda(v) + Lambda(l))."""
    g = 1 / (1 + _lambda_sheen(np.array(mu_v), alpha) + _lambda_sheen(mu_l, alpha))
    return g / (4 * mu_v * mu_l)


def directional_energy(cosines: np.ndarray, roughness: float, resolution: int = 512) -> np.ndarray:
    """Integrate unit-color sheen over the outgoing hemisphere for each mu_v.

    roughness is glTF artist roughness. Because d(omega_l) = d(mu_l)*d(phi),
    integrate mu_l in [0,1] with Gauss-Legendre quadrature and phi in [0,2*pi]
    with equally spaced midpoint samples. No stochastic sampling is involved.
    """
    alpha = _distribution_roughness(roughness)
    nodes, weights = np.polynomial.legendre.leggauss(resolution)
    mu_l = ((nodes + 1) / 2)[:, None]
    mu_weights = (weights / 2)[:, None]
    azimuth_count = 2 * resolution
    d_phi = 2 * np.pi / azimuth_count
    phi = (np.arange(azimuth_count) + 0.5) * d_phi
    sin_theta_l = np.sqrt(1 - mu_l**2)

    energies = []
    for mu_v in np.clip(cosines, MIN_COSINE, 1):
        # Set v's azimuth to zero by isotropy. For unit v and l,
        # |v+l| = sqrt(2 + 2*dot(v,l)), so dot(n,h) = (mu_v+mu_l)/|v+l|.
        v_dot_l = np.sqrt(1 - mu_v**2) * sin_theta_l * np.cos(phi) + mu_v * mu_l
        n_dot_h = (mu_v + mu_l) / np.sqrt(2 + 2 * v_dot_l)
        distribution = _charlie_distribution(n_dot_h, alpha)
        visibility = _sheen_visibility(float(mu_v), mu_l, alpha)
        brdf = distribution * visibility
        energies.append(np.sum(brdf * mu_l * mu_weights) * d_phi)
    return np.array(energies)


def _generate_table(resolution: int = 512) -> np.ndarray:
    """One linear-energy scalar per (distribution roughness, view cosine)."""
    cosines = cosine_grid()
    alphas = np.linspace(MIN_ALPHA, 1, TABLE_SIZE)
    return np.stack(
        [directional_energy(cosines, np.sqrt(alpha), resolution) for alpha in alphas]
    ).astype(np.float32)


def _header_comments(table: np.ndarray, resolution: int | None) -> list[str]:
    """Describe the stored values; None preserves unknown historical quadrature settings."""
    rows, columns = table.shape
    lines = [
        "// Generated by tools/generate_gltf_sheen_energy.py; do not edit.",
        f"// Table: {rows} rows x {columns} columns x 1 channel; row-major float32.",
        f"// Axes: rows uniformly sample alpha in [{MIN_ALPHA}, 1], alpha = clamp(roughness^2, {MIN_ALPHA}, 1).",
        f"// Columns: mu = {MIN_COSINE:g} + (1 - {MIN_COSINE:g}) * (column / {columns - 1})^2; endpoint nodes.",
        "// Channel: unit-color Charlie directional albedo; linear, without energy normalization or clipping.",
        "// Model: Enterprise PBR 2021x Charlie distribution and fitted masking-shadowing term.",
        "// Quadrature: Gauss-Legendre polar nodes and midpoint azimuth samples; float64.",
    ]
    if resolution is None:
        lines += [
            "// Quadrature point counts: unknown; not recorded with the original committed table.",
            "// Default regeneration command (not verified to reproduce this table):",
            "// python tools/generate_gltf_sheen_energy.py --resolution 512",
        ]
    else:
        lines += [
            f"// Quadrature point counts: {resolution} polar, {2 * resolution} azimuth.",
            f"// Command: python tools/generate_gltf_sheen_energy.py --resolution {resolution}",
        ]
    table_sha256 = hashlib.sha256(table.astype("<f4").tobytes()).hexdigest()
    lines += [
        f"// Table SHA256 (row-major little-endian float32): {table_sha256}",
        "// References:",
        "// https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html#components/sheen",
        "// https://dassaultsystemes-technology.github.io/EnterprisePBRShadingModel/spec-2021x.md.html#energycompensation/sheenbrdf",
        "// https://github.com/KhronosGroup/glTF/blob/5decc120c95764c319c4f92e7f7ead026d926ef3/extensions/2.0/Khronos/KHR_materials_sheen/README.md#implementation-notes",
    ]
    return lines


def _write_header(path: Path, table: np.ndarray, resolution: int) -> None:
    lines = [
        "// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.",
        "// SPDX-License-Identifier: Apache-2.0",
        "",
        *_header_comments(table, resolution),
        "#pragma once",
        "namespace falcor::detail {",
        "inline constexpr float GLTF_SHEEN_ENERGY[] = {",
    ]
    values = table.ravel()
    lines.extend(
        "    " + ", ".join(f"{v:.9e}f" for v in values[i : i + 6]) + ","
        for i in range(0, len(values), 6)
    )
    lines.extend(["};", "} // namespace falcor::detail", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolution", type=int, default=512, help="Polar quadrature point count.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "src/falcor2/render/material/gltf_sheen_energy.h",
    )
    args = parser.parse_args()
    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    table = _generate_table(args.resolution)
    _write_header(args.output, table, args.resolution)
    print(
        f"Wrote {args.output}; unclipped directional albedo range [{table.min():.6f}, {table.max():.6f}]"
    )
