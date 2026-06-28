# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import Any

from falcor2.reflection import Property

import falcor2 as f2
import slangpy as spy
from falcor2.editor import create_device, load_scene, save_image
from falcor2.rendernodes import PathTracerPipeline


SLANG_SOURCE_PATH = Path(__file__).with_suffix(".slang")


class CheckerMaterial(f2.Material):
    color_a = Property(
        spy.float3(0.92, 0.62, 0.24),
        doc="First checker color",
        on_change=lambda self: self.mark_dirty(f2.Material.DirtyFlags.properties),
    )
    color_b = Property(
        spy.float3(0.10, 0.18, 0.35),
        doc="Second checker color",
        on_change=lambda self: self.mark_dirty(f2.Material.DirtyFlags.properties),
    )
    scale = Property(
        12.0,
        doc="Checker frequency in UV space",
        value_range=(1.0, 128.0),
        on_change=lambda self: self.mark_dirty(f2.Material.DirtyFlags.properties),
    )

    def __init__(self) -> None:
        super().__init__()
        self.slang_type_name = "CheckerMaterial"
        self._module: spy.SlangModule | None = None

    def update(self, ctx: f2.SceneUpdateContext) -> None:
        if self._module is None:
            source = SLANG_SOURCE_PATH.read_text(encoding="ascii")
            self._module = self.scene.device.load_module_from_source("checker_material", source)

    def write_to_cursor(self, cursor: spy.BufferElementCursor | spy.ShaderCursor) -> None:
        cursor["color_a"] = self.color_a
        cursor["color_b"] = self.color_b
        cursor["scale"] = self.scale

    def required_module(self) -> spy.SlangModule | None:
        return self._module

    def get_this(self) -> dict[str, Any]:
        return {
            "color_a": self.color_a,
            "color_b": self.color_b,
            "scale": self.scale,
            "_type": self.slang_type_name,
        }


def main() -> None:
    device = create_device()
    scene = load_scene(
        device,
        "data/assets/kronos/DamagedHelmet/glTF/DamagedHelmet.gltf",
    )

    props = f2.Properties()
    props["color_a"] = spy.float3(0.92, 0.62, 0.24)
    props["color_b"] = spy.float3(0.10, 0.18, 0.35)
    props["scale"] = 12.0
    checker = scene.create_material(CheckerMaterial, props)
    checker.name = "CheckerMaterial"

    for component in scene.components:
        if isinstance(component, f2.GeometryInstance):
            component.materials = [checker] * len(list(component.materials))

    env_entity = scene.create_entity()
    env_map = env_entity.create_component(f2.EnvMapLight)
    env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"

    position = spy.float3(0, 0, 4)
    camera_entity = scene.create_entity()
    camera = camera_entity.create_component(f2.Camera)
    camera.width = 1024
    camera.height = 1024
    camera.fov_y = 45
    camera_transform = f2.Transform()
    camera_transform.translation = position
    camera_transform.rotation = spy.math.quat_from_look_at(
        -spy.math.normalize(position),
        spy.float3(0, 1, 0),
    )
    camera_entity.transform = camera_transform
    camera.recompute()

    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True

    image = None
    for _ in range(32):
        image = pipeline(scene, camera)

    assert image is not None
    output_path = "output/pathtrace_checker_material.png"
    save_image(image, output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
