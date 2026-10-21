# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import numpy as np
import slangpy as spy

import falcor2 as f2
from falcor2.editor import SceneShaderHelper


TRIANGLE_POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
TRIANGLE_INDICES = np.array([[0, 1, 2]], dtype=np.uint32)


def read_material_probe(
    device: spy.Device,
    scene: f2.Scene,
    geometry_instance: f2.GeometryInstance | list[f2.GeometryInstance],
    function_name: str = "multi_uv_material_probe",
    *,
    helper: SceneShaderHelper | None = None,
) -> Any:
    helper = helper if helper is not None else SceneShaderHelper(device)
    module, requirements = helper.prepare(scene, "render/test_geometry_properties.slang")
    probe = (
        module[function_name]
        .as_func()
        .type_conformances(requirements.type_conformances)
        .write(helper.bind_scene)
        .return_type(spy.Tensor)
    )
    instances = geometry_instance if isinstance(geometry_instance, list) else [geometry_instance]
    result = probe(
        spy.Tensor.from_numpy(
            device,
            np.array(
                [int(instance.geometry_instance_id) for instance in instances], dtype=np.uint32
            ),
        ),
        spy.Tensor.from_numpy(device, np.array([0], dtype=np.uint32)),
        spy.Tensor.from_numpy(device, np.array([[0.2, 0.3]], dtype=np.float32)),
    )
    cursor = result.cursor()
    cursor.load()
    if isinstance(geometry_instance, list):
        return [cursor[index].read() for index in range(len(instances))]
    return cursor[0].read()


def transform_terms() -> list[tuple[str, str, str, int]]:
    """Both languages use the same weighted operations, including same-space normal normalization."""
    terms = []
    for kind in ("point", "vector", "normal"):
        spaces = [("object", "world"), ("world", "object")]
        if kind == "normal":
            spaces.append(("world", "world"))
        for from_space, to_space in spaces:
            terms.append((kind, from_space, to_space, 1 << len(terms)))
    return terms


def expected_transforms(enabled: bool, linear: np.ndarray) -> np.ndarray:
    v = np.array([0.1, 0.2, 0.3])
    inverse = np.linalg.inv(linear)
    translation = np.array([0.3, -0.2, 0.1])
    if enabled:
        result = (
            linear @ v
            + translation
            + 2 * (inverse @ (v - translation))
            + 4 * (linear @ v)
            + 8 * (inverse @ v)
            + 16 * (inverse.T @ v) / np.linalg.norm(inverse.T @ v)
            + 32 * (linear.T @ v) / np.linalg.norm(linear.T @ v)
        )
    else:
        result = 15 * v + 48 * v / np.linalg.norm(v)
    return result + 64 * v / np.linalg.norm(v)
