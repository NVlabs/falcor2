# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import falcor2 as f2
from falcor2 import pyscene


def _on_scene_loaded(scene: f2.Scene) -> None:
    from pyscene_native_loading_callback_support import callback_material_name

    material = scene.create_material("StandardMaterial")
    material.name = callback_material_name(
        has_existing_material=scene.materials.find("Existing Material") is not None
    )


def configure() -> None:
    pyscene.load_asset("test_pyscene_native_loading.usda")
    pyscene.nodes.create_camera_fov(name="PyScene Camera", fov_degrees=45.0)
    pyscene.on_scene_loaded(_on_scene_loaded)
