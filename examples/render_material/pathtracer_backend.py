# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pathtracer backend for render-material examples."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import falcor2 as f2
import slangpy as spy
from falcor2.editor.utils import create_device, save_image
from falcor2.rendernodes import PathTracerPipeline

from .obj_reader import ObjReader
from .material_properties import make_material_properties
from .render_material_manifest import RenderEntry, RenderSettings


def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _srgb8_to_linear(value: int) -> float:
    return _srgb_to_linear(value / 255.0)


# Match MaterialX RenderGLSL test-suite PNG background exactly.
MATERIALX_DEFAULT_SCREEN_COLOR_SRGB8 = (76, 76, 82)
MATERIALX_DEFAULT_SCREEN_COLOR = tuple(
    _srgb8_to_linear(value) for value in MATERIALX_DEFAULT_SCREEN_COLOR_SRGB8
)
# MaterialX value previews are wrapped as a BSDF, and the MX139 basic wrapper currently
# reports that BSDF reflection through the PathTracer's specular albedo guide.
PREVIEW_PROPERTY_GUIDES = {
    "albedo": "specular_albedo",
    "emission": "emission",
}


@dataclass
class PathTracerPreviewSettings:
    geometry_path: Path
    geometry_flip_v: bool = False
    geometry_uv_origin: f2.UVOrigin = f2.UVOrigin.upper_left
    device_type: str = "automatic"
    width: int = 512
    height: int = 512
    max_depth: int = 3
    enable_nee: bool = True
    enable_mis: bool = True
    enable_analytic_lights: bool = True
    enable_emissive_triangles: bool = True
    radiance_ibl_path: Path | None = None
    env_map_as_background: bool = False
    background_color: tuple[float, float, float] = MATERIALX_DEFAULT_SCREEN_COLOR
    tone_map: bool = False
    dump_generated_code: bool = False


class PathTracerPreviewRenderer:
    def __init__(self, settings: PathTracerPreviewSettings):
        self.settings = settings
        device_type = getattr(spy.DeviceType, settings.device_type)
        self.device = create_device(device_type=device_type)
        geometry_path = Path(settings.geometry_path)
        if geometry_path.suffix.lower() != ".obj":
            raise ValueError(f"Only OBJ preview geometry is supported in V1: {geometry_path}")
        importer_scene = ObjReader().read(
            geometry_path,
            flip_v=settings.geometry_flip_v,
            uv_origin=settings.geometry_uv_origin,
        )
        self.scene = f2.Scene.create(self.device, importer_scene)
        self.camera = self._create_camera(settings.width, settings.height)
        if settings.radiance_ibl_path is not None:
            env_entity = self.scene.create_entity()
            env_transform = f2.Transform()
            env_transform.scale = spy.float3(-1.0, 1.0, -1.0)
            env_entity.transform = env_transform
            env_map = env_entity.create_component(f2.EnvMapLight)
            env_map.env_map_path = settings.radiance_ibl_path
        self.pipeline = PathTracerPipeline.create(self.device)
        self._configure_pipeline(settings, settings.width, settings.height)
        self.scene.update()

    def render_entry(self, entry: RenderEntry, settings: RenderSettings, image_path: Path) -> None:
        if self.settings.dump_generated_code:
            entry = replace(
                entry,
                properties={
                    **entry.properties,
                    "debug_write_shader_path": str(image_path.with_suffix(".slang")),
                },
            )

        if (
            entry.output.kind == "preview_property"
            and entry.output.name not in PREVIEW_PROPERTY_GUIDES
        ):
            raise ValueError(f"Unsupported preview property: {entry.output.name}")
        if entry.output.kind not in {"material", "preview_property"}:
            raise ValueError(f"Unsupported render output kind: {entry.output.kind}")
        if entry.output.kind == "material" and self.settings.radiance_ibl_path is None:
            raise ValueError("Material renders require radianceIBLPath")

        material = self.scene.create_material(entry.material_class, make_material_properties(entry))
        _assign_preview_material(self.scene, material)
        self.scene.update()
        self.camera.width = settings.width
        self.camera.height = settings.height
        self.camera.recompute()
        preview_guide = (
            PREVIEW_PROPERTY_GUIDES[str(entry.output.name)]
            if entry.output.kind == "preview_property"
            else None
        )
        self._configure_pipeline(self.settings, settings.width, settings.height, preview_guide)
        self.pipeline.request_reset()
        if preview_guide is not None:
            self.pipeline.spp = 1
            _render_preview_guide(
                self.pipeline,
                self.scene,
                self.camera,
                preview_guide,
                settings.samples_per_pixel,
                self.settings.background_color,
                image_path,
            )
            return

        self.pipeline.spp = settings.samples_per_pixel
        image = self.pipeline(self.scene, self.camera)
        save_image(image, image_path)

    def _create_camera(self, width: int, height: int) -> f2.Camera:
        position = spy.float3(0.0, 0.0, 3.0)
        camera_entity = self.scene.create_entity()
        camera = camera_entity.create_component(f2.Camera)
        camera.width = width
        camera.height = height
        camera.fov_y = 45.0
        camera_transform = f2.Transform()
        camera_transform.translation = position
        camera_transform.rotation = spy.math.quat_from_look_at(
            -spy.math.normalize(position),
            spy.float3(0.0, 1.0, 0.0),
        )
        camera_entity.transform = camera_transform
        camera.recompute()
        self.scene.active_camera = camera
        return camera

    def _configure_pipeline(
        self,
        settings: PathTracerPreviewSettings,
        width: int,
        height: int,
        preview_guide: str | None = None,
    ) -> None:
        self.pipeline.path_tracer.max_depth = settings.max_depth
        self.pipeline.path_tracer.enable_nee = settings.enable_nee
        self.pipeline.path_tracer.enable_mis = settings.enable_mis
        self.pipeline.path_tracer.enable_analytic_lights = settings.enable_analytic_lights
        self.pipeline.path_tracer.enable_environment_light = settings.radiance_ibl_path is not None
        self.pipeline.path_tracer.enable_emissive_triangles = settings.enable_emissive_triangles
        self.pipeline.path_tracer.env_map_as_background = settings.env_map_as_background
        self.pipeline.path_tracer.background_color = spy.float3(*settings.background_color)
        self.pipeline.tone_map = settings.tone_map
        self.pipeline.output_spec = f2.ContainerSpec.texture2d(
            spy.Format.rgba32_float,
            (height, width),
        )
        guide_specs = {}
        if preview_guide is not None:
            guide_specs[preview_guide] = f2.ContainerSpec.texture2d(
                spy.Format.rgba32_float,
                (height, width),
            )
            guide_specs["geometry_id"] = f2.ContainerSpec.texture2d(
                spy.Format.rgba32_uint,
                (height, width),
            )
        self.pipeline.guide_output_specs = guide_specs


def _assign_preview_material(scene: f2.Scene, material: f2.Material) -> None:
    for component in scene.components:
        if isinstance(component, f2.GeometryInstance):
            component.materials = [material] * len(list(component.materials))

    for scene_material in list(scene.materials):
        if scene_material is not material:
            scene_material.remove()


def _render_preview_guide(
    pipeline: PathTracerPipeline,
    scene: f2.Scene,
    camera: f2.Camera,
    preview_guide: str,
    samples_per_pixel: int,
    background_color: tuple[float, float, float],
    image_path: Path,
) -> None:
    accumulated: np.ndarray | None = None
    last_guide: object | None = None
    for _ in range(samples_per_pixel):
        _image, guides = pipeline(scene, camera, output_guides=True)
        guide = guides.get(preview_guide)
        if guide is None:
            raise RuntimeError(f"PathTracerPipeline did not produce guide '{preview_guide}'")
        last_guide = guide
        data = _preview_guide_data_with_background(
            guide,
            guides.get("geometry_id"),
            background_color,
        )
        if data is None:
            save_image(guide, image_path)
            return
        if accumulated is None:
            accumulated = np.zeros(data.shape, dtype=np.float32)
        accumulated += data.astype(np.float32, copy=False)

    if accumulated is None:
        if last_guide is None:
            raise RuntimeError(f"PathTracerPipeline did not produce guide '{preview_guide}'")
        save_image(last_guide, image_path)
        return

    _save_preview_guide_data(accumulated / float(samples_per_pixel), image_path)


def _preview_guide_data_with_background(
    guide: object,
    geometry_id: object | None,
    background_color: tuple[float, float, float],
) -> np.ndarray | None:
    if not hasattr(guide, "to_numpy"):
        return None

    data = np.array(guide.to_numpy(), copy=True)
    if geometry_id is None or not hasattr(geometry_id, "to_numpy"):
        return data

    geometry = geometry_id.to_numpy()
    if data.ndim != 3 or data.shape[2] not in {3, 4} or geometry.ndim != 3 or geometry.shape[2] < 2:
        return data

    miss_mask = geometry[..., 1] == np.uint32(int(f2.GeometryInstanceID.invalid))
    if not miss_mask.any():
        return data

    if miss_mask.shape != data.shape[:2]:
        return data

    data[..., :3][miss_mask] = np.array(background_color, dtype=data.dtype)
    if data.shape[2] == 4:
        data[..., 3][miss_mask] = np.array(1.0, dtype=data.dtype)

    return data


def _save_preview_guide_data(data: np.ndarray, image_path: Path) -> None:
    pixel_format = spy.Bitmap.PixelFormat.rgba if data.shape[2] == 4 else spy.Bitmap.PixelFormat.rgb
    bitmap = spy.Bitmap(data=data, pixel_format=pixel_format, srgb_gamma=False)
    if image_path.suffix not in (".exr", ".hdr"):
        bitmap = bitmap.convert(component_type=spy.Bitmap.ComponentType.uint8, srgb_gamma=True)
    image_path.parent.mkdir(parents=True, exist_ok=True)
    bitmap.write(image_path)
