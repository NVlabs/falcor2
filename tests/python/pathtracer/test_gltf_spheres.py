# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Imported glTF sphere grids with one HDR comparison per family and finite checks."""

from itertools import groupby
from pathlib import Path

import numpy as np
import pytest
import slangpy as spy

import falcor2.testing.helpers as helpers
from falcor2.testing.image_test_plugin import ImageTest
from .gltf_sphere_cases import sphere_cases
from .gltf_sphere_scene import SPP, TILE_SIZE, render_spheres, save_contact_sheet


@pytest.mark.nightly_if_not_vk
@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
@pytest.mark.parametrize("family", list(sphere_cases()))
def test_gltf_sphere_grid(
    device: spy.Device, tmp_path: Path, family: str, image_test: ImageTest
) -> None:
    family_cases = sphere_cases()[family]
    columns = min(4, len(family_cases))
    rows = (len(family_cases) + columns - 1) // columns
    grid = np.zeros((rows * TILE_SIZE, columns * TILE_SIZE, 3), dtype=np.float32)
    tile_index = 0
    for mode, grouped in groupby(family_cases, key=lambda case: case.mode):
        cases = list(grouped)
        # Rough refractive paths have much higher variance than the opaque grids.
        samples = 16384 if family == "transmission" else SPP
        image, _, timings = render_spheres(device, tmp_path / mode, cases, spp=samples)
        print(f"{family}/{mode}: {timings}")
        batch_columns = min(4, len(cases))
        # Importer modes need separate scenes; assemble their tiles into one family grid.
        for i, case in enumerate(cases):
            crop = image[
                i // batch_columns * TILE_SIZE : (i // batch_columns + 1) * TILE_SIZE,
                i % batch_columns * TILE_SIZE : (i % batch_columns + 1) * TILE_SIZE,
                :3,
            ]
            assert np.isfinite(crop).all(), case.name
            assert crop.mean() > 1e-4, f"Unexpected black render for {case.name}"
            row, column = divmod(tile_index, columns)
            grid[
                row * TILE_SIZE : (row + 1) * TILE_SIZE,
                column * TILE_SIZE : (column + 1) * TILE_SIZE,
            ] = crop
            tile_index += 1

    debug_dir = image_test.image_test_dir / "debug" / image_test.test_file
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_contact_sheet(grid, family_cases, debug_dir / f"{image_test.test_name}.png")
    tolerance = 5e-4 if family == "transmission" else 2e-4
    image_test(grid, tolerance=tolerance)
