# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MaterialVisualizer backend for render-material suites."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .material_properties import make_material_properties
from .pathtracer_backend import MATERIALX_DEFAULT_SCREEN_COLOR
from .render_material_manifest import RenderEntry, RenderSettings


@dataclass(frozen=True)
class MaterialVisualizerSettings:
    device_type: str = "automatic"
    radiance_ibl_path: Path | None = None
    background_color: tuple[float, float, float] = MATERIALX_DEFAULT_SCREEN_COLOR
    dump_generated_code: bool = False


class MaterialVisualizerRenderer:
    """Process-local renderer using Falcor2's MaterialVisualizer."""

    def __init__(self, settings: MaterialVisualizerSettings) -> None:
        self.settings = settings
        self._f2: Any | None = None
        self._save_image: Any | None = None
        self._spy: Any | None = None
        self.device: Any | None = None
        self.visualizer: Any | None = None

    def _ensure_renderer(self) -> None:
        if self.device is not None:
            return

        import falcor2 as f2
        from falcor2.editor.utils import create_device, save_image
        from falcor2.tools.materials.material_tools import MaterialVisualizer
        import slangpy as spy

        self._f2 = f2
        self._save_image = save_image
        self._spy = spy
        self.device = create_device(device_type=getattr(spy.DeviceType, self.settings.device_type))
        self.visualizer = MaterialVisualizer(self.device)

    def render_entry(self, entry: RenderEntry, settings: RenderSettings, image_path: Path) -> None:
        """Render one entry through Falcor2's MaterialVisualizer backend."""

        if entry.output.kind == "material" and self.settings.radiance_ibl_path is None:
            raise ValueError("Material renders require radianceIBLPath")

        self._ensure_renderer()
        assert self._f2 is not None
        assert self._save_image is not None
        assert self._spy is not None
        assert self.device is not None
        assert self.visualizer is not None

        scene = self._f2.Scene.create(self.device)
        if self.settings.dump_generated_code:
            entry = replace(
                entry,
                properties={
                    **entry.properties,
                    "debug_write_shader_path": str(image_path.with_suffix(".slang")),
                },
            )
        props = make_material_properties(entry)
        if (
            entry.material_class == "MaterialXMaterial"
            and "mtlx_use_slang_derivatives" not in entry.properties
        ):
            props["mtlx_use_slang_derivatives"] = True
        material = scene.create_material(entry.material_class, props)
        result_texture = self.device.create_texture(
            format=self._spy.Format.rgba32_float,
            width=settings.width,
            height=settings.height,
            usage=self._spy.TextureUsage.shader_resource | self._spy.TextureUsage.unordered_access,
            label=f"render_material_{_safe_name(entry.entry_id)}",
        )
        if entry.output.kind == "material":
            texture = self.visualizer.show_material(
                scene,
                material,
                samples_per_pixel=settings.samples_per_pixel,
                env_map_path=self.settings.radiance_ibl_path,
                result_texture=result_texture,
                sample_batch_size=settings.sample_batch_size,
                camera_size=(settings.width, settings.height),
                background_color=self.settings.background_color,
            )
        else:
            textures = self.visualizer.show_preview_properties(
                scene,
                material,
                property_textures={entry.output.name: result_texture},
                background_color=self.settings.background_color,
            )
            texture = textures[str(entry.output.name)]
        self._save_image(texture, image_path)


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)
