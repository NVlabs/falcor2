# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

from falcor2.reflection import Property

import falcor2 as f2
import slangpy as spy


DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


class PythonMaterialSample(f2.Material):
    texture_index = Property(
        3,
        doc="Index of the texture to use",
        value_range=(0, 3),
        on_change=lambda self: self.mark_dirty(f2.Material.DirtyFlags.properties),
    )
    texture0 = Property(
        "assets/textures/test_texture_manager/udim_1001.png",
        doc="First texture",
        on_change=lambda self: self.mark_dirty(
            f2.Material.DirtyFlags.properties | f2.Material.DirtyFlags.resources
        ),
    )
    texture1 = Property(
        "assets/textures/test_texture_manager/regular_blue.png",
        doc="Second texture",
        on_change=lambda self: self.mark_dirty(
            f2.Material.DirtyFlags.properties | f2.Material.DirtyFlags.resources
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self.slang_type_name = "PythonMaterialSample"

        self._asset_resolver = f2.AssetResolver().push(DATA_DIR)

        self._module: spy.SlangModule | None = None
        self._texture_handles = [f2.TextureHandle(), f2.TextureHandle()]

    # Marks textures that need to be loaded, only called when the material is dirty with respect to resources.
    def on_load_resources(self) -> None:
        self._texture_handles[0] = self.scene.texture_manager.load_texture(
            self.texture0,
            resolver=self._asset_resolver,
            srgb=True,
            load_deferred=True,
        )
        self._texture_handles[1] = self.scene.texture_manager.load_texture(
            self.texture1,
            resolver=self._asset_resolver,
            srgb=True,
            load_deferred=True,
        )

    # This should be the heaviest call. Use the ctx to get efficiency.
    def update(self, ctx: f2.SceneUpdateContext) -> None:
        if self._module is None:
            self._module = self.scene.device.load_module("render/python_material_sample.slang")

    def write_to_cursor(self, cursor: spy.BufferElementCursor | spy.ShaderCursor) -> None:
        cursor["textures"] = [
            self._texture_handles[0].get_this(),
            self._texture_handles[1].get_this(),
        ]
        cursor["texture_index"] = self.texture_index

    def required_module(self) -> spy.SlangModule | None:
        return self._module

    # Used temporarily until MaterialMarshall is in place.
    def get_this(self) -> dict[str, Any]:
        uniforms = {}
        uniforms["textures"] = [
            self._texture_handles[0].get_this(),
            self._texture_handles[1].get_this(),
        ]
        uniforms["texture_index"] = self.texture_index
        uniforms["_type"] = self.slang_type_name
        return uniforms
