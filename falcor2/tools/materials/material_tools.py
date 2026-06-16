# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
import math
from os import PathLike
from pathlib import Path
from typing import Any, Literal, MutableMapping, Sequence
import hashlib

import falcor2 as f2
import numpy as np
import slangpy as spy

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PathInput = str | PathLike[str]
TextureFilteringMode = Literal["analytic", "lod0"]
DirectLightingMode = Literal["nee_only", "direct_hits_only", "both_50_50", "both_mis"]
MaterialAOVName = Literal[
    "beauty",
    "first_hit_non_delta_ibl",
    "secondary_transport",
]
PreviewPropertyName = Literal["albedo", "emission"]
DIRECT_LIGHTING_MODE_NEE_ONLY = 0
DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY = 1
DIRECT_LIGHTING_MODE_BOTH_50_50 = 2
DIRECT_LIGHTING_MODE_BOTH_MIS = 3
DEFAULT_DIRECT_LIGHTING_MODE: DirectLightingMode = "both_mis"
DIRECT_LIGHTING_MODE_CODES: dict[str, int] = {
    "nee_only": DIRECT_LIGHTING_MODE_NEE_ONLY,
    "direct_hits_only": DIRECT_LIGHTING_MODE_DIRECT_HITS_ONLY,
    "both_50_50": DIRECT_LIGHTING_MODE_BOTH_50_50,
    "both_mis": DIRECT_LIGHTING_MODE_BOTH_MIS,
}
DIRECT_LIGHTING_MODES: tuple[str, ...] = tuple(DIRECT_LIGHTING_MODE_CODES)
MATERIAL_AOV_NAMES: tuple[MaterialAOVName, ...] = (
    "beauty",
    "first_hit_non_delta_ibl",
    "secondary_transport",
)
PREVIEW_PROPERTY_NAMES: tuple[PreviewPropertyName, ...] = ("albedo", "emission")
DEFAULT_TEXTURE_FILTER_WIDTH_SCALE = 0.62
DEFAULT_MATERIAL_VISUALIZER_SAMPLE_BATCH_SIZE = 32
MATERIAL_VISUALIZER_PATH_LOG_LABELS = (
    "metadata",
    "jitter_sample",
    "jitter_and_camera_xy",
    "ray_origin_radius",
    "ray_dir_spread",
    "hit",
    "hit_ids",
    "position",
    "normal_geometric",
    "normal_shading",
    "wi_ws",
    "uv",
    "first_bounce_random",
    "env_dir",
    "env_weight",
    "env_pdf_bsdf_pdf_visible_weight",
    "bsdf_eval",
    "direct_contribution",
    "bsdf_sample_wo",
    "bsdf_sample_weight",
    "bsdf_sample_pdf_lobes",
    "next_ray_dir_spread",
    "radiance",
)
MATERIAL_VISUALIZER_PATH_LOG_FLOAT_COUNT = len(MATERIAL_VISUALIZER_PATH_LOG_LABELS) * 4


@dataclass
class AccumulationSession:
    """In-process GPU accumulation state for progressive material renders."""

    result_texture: spy.Texture
    sum_texture: spy.Texture
    residual_texture: spy.Texture
    completed_sample_count: int
    camera_size: tuple[int, int]
    pixel_offset: tuple[int, int]


MATERIALX_TEST_GEOMPROP_STREAM_IDS: tuple[tuple[str, int], ...] = (
    ("geompropvalue_integer", 1),
    ("geompropvalue_bool", 2),
    ("geompropvalue_boolean", 2),
    ("geompropvalue_float", 3),
    ("geompropvalue_vector2", 4),
    ("geompropvalue_vector3", 5),
    ("geompropvalue_vector4", 6),
    ("geompropvalue_color2", 7),
    ("geompropvalue_color3", 8),
    ("geompropvalue_color4", 9),
    ("myvalue", 10),
    ("texcoord_1", 11),
    ("color_0", 12),
    ("color_1", 13),
)
MATERIALX_TEST_GEOMPROP_NAMES: tuple[str, ...] = tuple(
    name for name, _ in MATERIALX_TEST_GEOMPROP_STREAM_IDS
)
MATERIALX_TEST_GEOMPROP_IDS: tuple[int, ...] = tuple(
    id for _, id in MATERIALX_TEST_GEOMPROP_STREAM_IDS
)


def direct_lighting_mode_code(mode: DirectLightingMode | str) -> int:
    try:
        return DIRECT_LIGHTING_MODE_CODES[str(mode)]
    except KeyError as exc:
        allowed = ", ".join(DIRECT_LIGHTING_MODES)
        raise ValueError(f"direct_lighting_mode must be one of {allowed}; got {mode!r}") from exc


def apply_materialx_test_geomprop_properties(props: f2.Properties) -> f2.Properties:
    props["mtlx_geomprop_names"] = list(MATERIALX_TEST_GEOMPROP_NAMES)
    props["mtlx_geomprop_ids"] = list(MATERIALX_TEST_GEOMPROP_IDS)
    return props


def _sample_batches(
    total_sample_count: int,
    sample_batch_size: int | None = None,
) -> tuple[tuple[int, int], ...]:
    total = int(total_sample_count)
    if total <= 0:
        raise ValueError("total_sample_count must be positive")

    max_batch_size = (
        DEFAULT_MATERIAL_VISUALIZER_SAMPLE_BATCH_SIZE
        if sample_batch_size is None
        else int(sample_batch_size)
    )
    if max_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive")

    batch_size = min(max_batch_size, total)
    return tuple(
        (offset, min(batch_size, total - offset)) for offset in range(0, total, batch_size)
    )


class MaterialVisualizer:
    def __init__(self, device: spy.Device) -> None:
        super().__init__()
        self.device = device
        self._prev_module: spy.Module | None = None
        self._prev_req_key: tuple[Any, ...] | None = None
        self._prev_scene: f2.Scene | None = None
        self._last_accumulation_session: AccumulationSession | None = None
        self._last_accumulation_resources: list[Any] = []
        self._accumulation_lifetime_guard: list[Any] = []

    def _keep_accumulation_resources(self, *resources: Any) -> None:
        self._accumulation_lifetime_guard.extend(resources)
        if len(self._accumulation_lifetime_guard) > 32:
            del self._accumulation_lifetime_guard[:-32]

    def create_accumulation_session(
        self,
        result_texture: spy.Texture | None = None,
        *,
        camera_size: Sequence[int] | None = None,
        pixel_offset: Sequence[int] = (0, 0),
        label: str = "material_visualizer_accumulation",
    ) -> AccumulationSession:
        result_texture = self._ensure_result_texture(result_texture)
        camera_size_tuple = self._validate_camera_size_for_texture(camera_size, result_texture)
        pixel_offset_tuple = self._validate_pixel_offset_tuple(
            pixel_offset, result_texture, camera_size_tuple
        )
        return AccumulationSession(
            result_texture=result_texture,
            sum_texture=self._create_result_texture(
                result_texture.width, result_texture.height, f"{label}_sum"
            ),
            residual_texture=self._create_result_texture(
                result_texture.width, result_texture.height, f"{label}_residual"
            ),
            completed_sample_count=0,
            camera_size=camera_size_tuple,
            pixel_offset=pixel_offset_tuple,
        )

    def resolve_accumulation(self, session: AccumulationSession) -> spy.Texture:
        if not isinstance(session, AccumulationSession):
            raise TypeError("resolve_accumulation() requires an AccumulationSession")
        return session.result_texture

    def accumulate_material(
        self,
        scene: f2.Scene,
        material: f2.Material,
        *,
        session: AccumulationSession,
        additional_samples: int,
        env_map_path: PathInput | None = None,
        sample_batch_size: int | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
        direct_lighting_mode: DirectLightingMode = DEFAULT_DIRECT_LIGHTING_MODE,
    ) -> spy.Texture:
        if not isinstance(session, AccumulationSession):
            raise TypeError("accumulate_material() requires an AccumulationSession")
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.accumulate_material() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.accumulate_material() requires a falcor2.Material")
        additional_samples = self._validate_positive_int(additional_samples, "additional_samples")
        sample_batches = _sample_batches(additional_samples, sample_batch_size)
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )
        direct_lighting_mode_value = direct_lighting_mode_code(direct_lighting_mode)

        self._ensure_env_map_light(scene, env_map_path)
        scene.update()
        if self._find_first_env_map_light(scene) is None:
            raise RuntimeError("MaterialVisualizer requires a Scene EnvMapLight")

        module = self._get_module(scene, material)
        render = module.sphere_tracing_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)

        camera_size_value = spy.uint2(session.camera_size[0], session.camera_size[1])
        pixel_offset_value = spy.uint2(session.pixel_offset[0], session.pixel_offset[1])
        for batch_offset, current_sample_count in sample_batches:
            completed_before_batch = session.completed_sample_count
            render.dispatch(
                thread_count=spy.uint3(
                    session.result_texture.width, session.result_texture.height, 1
                ),
                material=material,
                camera_size=camera_size_value,
                pixel_offset=pixel_offset_value,
                sample_index_offset=completed_before_batch,
                sample_count=current_sample_count,
                completed_sample_count=completed_before_batch,
                direct_lighting_mode=direct_lighting_mode_value,
                use_analytic_texture_filtering=use_analytic_texture_filtering,
                texture_filter_width_scale=texture_filter_width_scale,
                clear_result=completed_before_batch == 0 and batch_offset == 0,
                accumulation_sum=session.sum_texture,
                accumulation_residual=session.residual_texture,
                result=session.result_texture,
            )
            session.completed_sample_count += current_sample_count

        return session.result_texture

    def show_material(
        self,
        scene: f2.Scene,
        material: f2.Material,
        samples_per_pixel: int = 32,
        env_map_path: PathInput | None = None,
        result_texture: spy.Texture | None = None,
        sample_batch_size: int | None = None,
        camera_size: Sequence[int] | None = None,
        pixel_offset: Sequence[int] = (0, 0),
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
        aov_textures: MutableMapping[str, spy.Texture | None] | None = None,
        direct_lighting_mode: DirectLightingMode = DEFAULT_DIRECT_LIGHTING_MODE,
    ) -> spy.Texture:
        """
        Renders the given scene Material using sphere tracing and returns the result texture.
        If result_texture is provided, it is reused; otherwise a 512x512 texture is created.
        If no env_map_path is provided, the scene's first EnvMapLight is used or a default one is created.
        If aov_textures is provided, it is filled with the first-hit AOV textures written by this render.
        """
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_material() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_material() requires a falcor2.Material")
        if aov_textures is not None:
            aov_beauty = aov_textures.get("beauty")
            if (
                result_texture is not None
                and aov_beauty is not None
                and aov_beauty is not result_texture
            ):
                raise ValueError(
                    "aov_textures['beauty'] must match result_texture when both are provided"
                )
            if result_texture is None and aov_beauty is not None:
                result_texture = aov_beauty
        result_texture = self._ensure_result_texture(result_texture)

        if aov_textures is None:
            session = self.create_accumulation_session(
                result_texture,
                camera_size=camera_size,
                pixel_offset=pixel_offset,
            )
            self._last_accumulation_session = session
            self._last_accumulation_resources = []
            self._keep_accumulation_resources(session)
            texture = self.accumulate_material(
                scene,
                material,
                session=session,
                additional_samples=samples_per_pixel,
                env_map_path=env_map_path,
                sample_batch_size=sample_batch_size,
                texture_filtering=texture_filtering,
                texture_filter_width_scale=texture_filter_width_scale,
                direct_lighting_mode=direct_lighting_mode,
            )
            self.device.wait()
            return texture

        self._show_material_aovs(
            scene,
            material,
            samples_per_pixel=samples_per_pixel,
            env_map_path=env_map_path,
            result_texture=result_texture,
            sample_batch_size=sample_batch_size,
            camera_size=camera_size,
            pixel_offset=pixel_offset,
            texture_filtering=texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            aov_textures=aov_textures,
            direct_lighting_mode=direct_lighting_mode,
        )
        self._last_accumulation_session = None
        self.device.wait()
        return result_texture

    def _show_material_aovs(
        self,
        scene: f2.Scene,
        material: f2.Material,
        *,
        samples_per_pixel: int,
        env_map_path: PathInput | None,
        result_texture: spy.Texture,
        sample_batch_size: int | None,
        camera_size: Sequence[int] | None,
        pixel_offset: Sequence[int],
        texture_filtering: TextureFilteringMode,
        texture_filter_width_scale: float,
        aov_textures: MutableMapping[str, spy.Texture | None],
        direct_lighting_mode: DirectLightingMode,
    ) -> None:
        samples_per_pixel = self._validate_positive_int(samples_per_pixel, "samples_per_pixel")
        sample_batches = _sample_batches(samples_per_pixel, sample_batch_size)
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )
        direct_lighting_mode_value = direct_lighting_mode_code(direct_lighting_mode)

        self._ensure_env_map_light(scene, env_map_path)
        scene.update()
        if self._find_first_env_map_light(scene) is None:
            raise RuntimeError("MaterialVisualizer requires a Scene EnvMapLight")

        module = self._get_module(scene, material)
        aovs = self._ensure_material_aov_textures(aov_textures, result_texture)
        camera_size_tuple = self._validate_camera_size_for_texture(camera_size, result_texture)
        pixel_offset_tuple = self._validate_pixel_offset_tuple(
            pixel_offset, result_texture, camera_size_tuple
        )
        camera_size_value = spy.uint2(camera_size_tuple[0], camera_size_tuple[1])
        pixel_offset_value = spy.uint2(pixel_offset_tuple[0], pixel_offset_tuple[1])

        beauty_sum = self._create_result_texture(
            result_texture.width, result_texture.height, "material_visualizer_aov_beauty_sum"
        )
        beauty_residual = self._create_result_texture(
            result_texture.width,
            result_texture.height,
            "material_visualizer_aov_beauty_residual",
        )
        first_hit_sum = self._create_result_texture(
            result_texture.width, result_texture.height, "material_visualizer_aov_first_hit_sum"
        )
        first_hit_residual = self._create_result_texture(
            result_texture.width,
            result_texture.height,
            "material_visualizer_aov_first_hit_residual",
        )
        secondary_sum = self._create_result_texture(
            result_texture.width, result_texture.height, "material_visualizer_aov_secondary_sum"
        )
        secondary_residual = self._create_result_texture(
            result_texture.width,
            result_texture.height,
            "material_visualizer_aov_secondary_residual",
        )
        self._last_accumulation_resources = [
            beauty_sum,
            beauty_residual,
            first_hit_sum,
            first_hit_residual,
            secondary_sum,
            secondary_residual,
        ]
        self._keep_accumulation_resources(*self._last_accumulation_resources)

        render = module.sphere_tracing_first_hit_aov_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        completed_sample_count = 0
        for batch_offset, current_sample_count in sample_batches:
            render.dispatch(
                thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
                material=material,
                camera_size=camera_size_value,
                pixel_offset=pixel_offset_value,
                sample_index_offset=completed_sample_count,
                sample_count=current_sample_count,
                completed_sample_count=completed_sample_count,
                direct_lighting_mode=direct_lighting_mode_value,
                use_analytic_texture_filtering=use_analytic_texture_filtering,
                texture_filter_width_scale=texture_filter_width_scale,
                clear_result=completed_sample_count == 0 and batch_offset == 0,
                beauty_sum=beauty_sum,
                beauty_residual=beauty_residual,
                first_hit_non_delta_ibl_sum=first_hit_sum,
                first_hit_non_delta_ibl_residual=first_hit_residual,
                secondary_transport_sum=secondary_sum,
                secondary_transport_residual=secondary_residual,
                result=result_texture,
                first_hit_non_delta_ibl=aovs["first_hit_non_delta_ibl"],
                secondary_transport=aovs["secondary_transport"],
            )
            completed_sample_count += current_sample_count

    def trace_path_log(
        self,
        scene: f2.Scene,
        material: f2.Material,
        *,
        pixel: Sequence[int],
        sample_index: int = 0,
        max_bounces: int = 2,
        direct_lighting_mode: DirectLightingMode = DEFAULT_DIRECT_LIGHTING_MODE,
        no_jitter: bool = False,
        camera_size: Sequence[int] = (512, 512),
        env_map_path: PathInput | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
        capture_debug_print: bool = False,
    ) -> dict[str, Any]:
        """
        Trace one material sphere pixel and return the first-bounce path diagnostics.

        The trace follows material's own camera, sample-generator consumption,
        texture filtering, light sampling, and MIS so diagnostics stay on the
        release MaterialX rendering path.
        """
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.trace_path_log() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.trace_path_log() requires a falcor2.Material")
        if len(pixel) != 2:
            raise ValueError("pixel must have exactly two components")
        pixel_x = int(pixel[0])
        pixel_y = int(pixel[1])
        if pixel_x < 0 or pixel_y < 0:
            raise ValueError("pixel components must be non-negative")
        pixel_value = spy.uint2(pixel_x, pixel_y)
        sample_index_value = self._validate_nonnegative_int(sample_index, "sample_index")
        max_bounces_value = self._validate_nonnegative_int(max_bounces, "max_bounces")
        camera_size_value = self._validate_camera_size(camera_size)
        if pixel_value.x >= camera_size_value.x or pixel_value.y >= camera_size_value.y:
            raise ValueError("pixel must fit inside camera_size")
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )
        direct_lighting_mode_value = direct_lighting_mode_code(direct_lighting_mode)

        self._ensure_env_map_light(scene, env_map_path)
        scene.update()
        if self._find_first_env_map_light(scene) is None:
            raise RuntimeError("MaterialVisualizer requires a Scene EnvMapLight")

        module = self._get_module(scene, material)
        path_log = spy.Tensor.zeros(
            self.device,
            (MATERIAL_VISUALIZER_PATH_LOG_FLOAT_COUNT,),
            dtype=float,
        )
        trace = module.material_visualizer_path_log_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        trace.dispatch(
            thread_count=spy.uint3(1, 1, 1),
            material=material,
            max_bounces=max_bounces_value,
            camera_size=camera_size_value,
            pixel=pixel_value,
            sample_index=sample_index_value,
            direct_lighting_mode=direct_lighting_mode_value,
            no_jitter=bool(no_jitter),
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            enable_debug_print=bool(capture_debug_print),
            path_log=path_log,
        )
        values = path_log.to_numpy().astype(np.float32, copy=False).reshape((-1, 4))
        result: dict[str, Any] = {
            label: tuple(float(component) for component in values[index])
            for index, label in enumerate(MATERIAL_VISUALIZER_PATH_LOG_LABELS)
        }
        if capture_debug_print:
            result["debug_print"] = self.device.flush_print_to_string()
        return result

    def show_scene_material(
        self,
        scene: f2.Scene,
        material: f2.Material,
        samples_per_pixel: int = 32,
        env_map_path: PathInput | None = None,
        result_texture: spy.Texture | None = None,
        sample_batch_size: int | None = None,
        camera_size: Sequence[int] | None = None,
        pixel_offset: Sequence[int] = (0, 0),
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
        direct_lighting_mode: DirectLightingMode = DEFAULT_DIRECT_LIGHTING_MODE,
    ) -> spy.Texture:
        """
        Renders a material through the scene material system instead of binding the
        concrete material object directly. This path can use uniform material header
        flags, such as geometry_thin_walled, before creating the material instance.
        """
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_scene_material() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_scene_material() requires a falcor2.Material")
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )
        direct_lighting_mode_value = direct_lighting_mode_code(direct_lighting_mode)
        samples_per_pixel = self._validate_positive_int(samples_per_pixel, "samples_per_pixel")
        sample_batches = _sample_batches(samples_per_pixel, sample_batch_size)

        self._ensure_env_map_light(scene, env_map_path)
        scene.update()
        if self._find_first_env_map_light(scene) is None:
            raise RuntimeError("MaterialVisualizer requires a Scene EnvMapLight")

        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)
        camera_size_tuple = self._validate_camera_size_for_texture(camera_size, result_texture)
        pixel_offset_tuple = self._validate_pixel_offset_tuple(
            pixel_offset, result_texture, camera_size_tuple
        )
        camera_size_value = spy.uint2(camera_size_tuple[0], camera_size_tuple[1])
        pixel_offset_value = spy.uint2(pixel_offset_tuple[0], pixel_offset_tuple[1])
        accumulation_sum = self._create_result_texture(
            result_texture.width, result_texture.height, "material_visualizer_scene_sum"
        )
        accumulation_residual = self._create_result_texture(
            result_texture.width, result_texture.height, "material_visualizer_scene_residual"
        )
        self._last_accumulation_session = None
        self._last_accumulation_resources = [accumulation_sum, accumulation_residual]
        self._keep_accumulation_resources(*self._last_accumulation_resources)
        render = module.sphere_tracing_scene_material_kernel.type_conformances(
            scene.requirements.type_conformances
        ).write(self._bind_scene)
        completed_sample_count = 0
        for batch_offset, current_sample_count in sample_batches:
            render.dispatch(
                thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
                material_id=self._material_id(material),
                camera_size=camera_size_value,
                pixel_offset=pixel_offset_value,
                sample_index_offset=completed_sample_count,
                sample_count=current_sample_count,
                completed_sample_count=completed_sample_count,
                direct_lighting_mode=direct_lighting_mode_value,
                use_analytic_texture_filtering=use_analytic_texture_filtering,
                texture_filter_width_scale=texture_filter_width_scale,
                clear_result=completed_sample_count == 0 and batch_offset == 0,
                accumulation_sum=accumulation_sum,
                accumulation_residual=accumulation_residual,
                result=result_texture,
            )
            completed_sample_count += current_sample_count

        self.device.wait()
        return result_texture

    def show_preview_properties(
        self,
        scene: f2.Scene,
        material: f2.Material,
        property_textures: MutableMapping[str, spy.Texture | None] | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> dict[PreviewPropertyName, spy.Texture]:
        """
        Renders material preview properties on the same preview sphere used by show_material.
        The properties are collected once per pixel, then written into separate albedo and
        emission textures so callers can choose the appropriate preview channel afterward.
        """
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_preview_properties() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError(
                "MaterialVisualizer.show_preview_properties() requires a falcor2.Material"
            )
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)
        properties = self._ensure_preview_property_textures(property_textures)

        render = module.preview_properties_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(properties["albedo"].width, properties["albedo"].height, 1),
            material=material,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            albedo_result=properties["albedo"],
            emission_result=properties["emission"],
        )

        return properties

    def show_albedo(
        self,
        scene: f2.Scene,
        material: f2.Material,
        result_texture: spy.Texture | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> spy.Texture:
        """
        Renders the material albedo on the same preview sphere used by show_material.
        If result_texture is provided, it is reused; otherwise a 512x512 texture is created.
        """
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_albedo() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_albedo() requires a falcor2.Material")
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)

        render = module.albedo_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
            material=material,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            result=result_texture,
        )

        return result_texture

    def show_dense_direct(
        self,
        scene: f2.Scene,
        material: f2.Material,
        direct_light_count: int = 64,
        result_texture: spy.Texture | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> spy.Texture:
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_dense_direct() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_dense_direct() requires a falcor2.Material")
        if direct_light_count <= 0:
            raise ValueError("direct_light_count must be positive")
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)

        render = module.dense_direct_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
            material=material,
            direct_light_count=direct_light_count,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            result=result_texture,
        )

        return result_texture

    def show_dense_env_direct(
        self,
        scene: f2.Scene,
        material: f2.Material,
        direct_light_count: int = 64,
        use_env_grid: bool = False,
        env_grid_size: int = 64,
        use_env_importance: bool = False,
        use_env_tile: bool = False,
        env_width: int = 1,
        env_height: int = 1,
        env_tile_start: int = 0,
        env_tile_count: int = 1,
        env_map_path: PathInput | None = None,
        flip_camera_y: bool = False,
        flip_env_x: bool = False,
        flip_env_y: bool = False,
        flip_env_z: bool = False,
        result_texture: spy.Texture | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> spy.Texture:
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_dense_env_direct() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError(
                "MaterialVisualizer.show_dense_env_direct() requires a falcor2.Material"
            )
        if direct_light_count <= 0:
            raise ValueError("direct_light_count must be positive")
        if env_grid_size <= 0:
            raise ValueError("env_grid_size must be positive")
        if env_width <= 0 or env_height <= 0:
            raise ValueError("env_width and env_height must be positive")
        if env_tile_start < 0:
            raise ValueError("env_tile_start must be non-negative")
        if env_tile_count <= 0:
            raise ValueError("env_tile_count must be positive")
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        self._ensure_env_map_light(scene, env_map_path)
        scene.update()
        if self._find_first_env_map_light(scene) is None:
            raise RuntimeError("MaterialVisualizer requires a Scene EnvMapLight")
        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)

        render = module.dense_env_direct_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
            material=material,
            direct_light_count=direct_light_count,
            use_env_grid=use_env_grid,
            env_grid_size=env_grid_size,
            use_env_importance=use_env_importance,
            use_env_tile=use_env_tile,
            env_width=env_width,
            env_height=env_height,
            env_tile_start=env_tile_start,
            env_tile_count=env_tile_count,
            flip_camera_y=flip_camera_y,
            flip_env_x=flip_env_x,
            flip_env_y=flip_env_y,
            flip_env_z=flip_env_z,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            result=result_texture,
        )

        return result_texture

    def show_explicit_direct(
        self,
        scene: f2.Scene,
        material: f2.Material,
        direct_lights: np.ndarray,
        flip_camera_y: bool = False,
        flip_light_x: bool = False,
        flip_light_y: bool = False,
        flip_light_z: bool = False,
        result_texture: spy.Texture | None = None,
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> spy.Texture:
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_explicit_direct() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_explicit_direct() requires a falcor2.Material")
        light_data = self._validate_explicit_direct_lights(direct_lights)
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)
        light_buffer = self.device.create_buffer(
            usage=spy.BufferUsage.shader_resource,
            data=np.ascontiguousarray(light_data.reshape((-1, 4)), dtype=np.float32),
        )

        render = module.explicit_direct_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
            material=material,
            direct_lights=light_buffer,
            direct_light_count=light_data.shape[0],
            flip_camera_y=flip_camera_y,
            flip_light_x=flip_light_x,
            flip_light_y=flip_light_y,
            flip_light_z=flip_light_z,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            result=result_texture,
        )

        return result_texture

    def show_eval_plot(
        self,
        scene: f2.Scene,
        material: f2.Material,
        result_texture: spy.Texture | None = None,
        view_direction_ws: Sequence[float] = (0.0, 0.0, 1.0),
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> spy.Texture:
        if not isinstance(scene, f2.Scene):
            raise TypeError("MaterialVisualizer.show_eval_plot() requires a falcor2.Scene")
        if not isinstance(material, f2.Material):
            raise TypeError("MaterialVisualizer.show_eval_plot() requires a falcor2.Material")
        view_direction_ws_value = self._validate_float3_direction(
            view_direction_ws, "view_direction_ws"
        )
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)
        result_texture = self._ensure_result_texture(result_texture)

        render = module.eval_plot_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        render.dispatch(
            thread_count=spy.uint3(result_texture.width, result_texture.height, 1),
            material=material,
            view_direction_ws=view_direction_ws_value,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            result=result_texture,
        )

        return result_texture

    def full_sphere_eval_sample_diagnostics(
        self,
        scene: f2.Scene,
        material: f2.Material,
        *,
        width: int = 128,
        height: int = 64,
        sample_count: int = 1_000_000,
        pdf_integration_res: int = 4,
        samples_per_thread: int = 1024,
        seed: int = 1234,
        view_direction_ws: Sequence[float] = (0.0, 0.0, 1.0),
        texture_filtering: TextureFilteringMode = "analytic",
        texture_filter_width_scale: float = DEFAULT_TEXTURE_FILTER_WIDTH_SCALE,
    ) -> dict[str, np.ndarray | int | float]:
        if not isinstance(scene, f2.Scene):
            raise TypeError(
                "MaterialVisualizer.full_sphere_eval_sample_diagnostics() requires a falcor2.Scene"
            )
        if not isinstance(material, f2.Material):
            raise TypeError(
                "MaterialVisualizer.full_sphere_eval_sample_diagnostics() requires a falcor2.Material"
            )
        width = self._validate_positive_int(width, "width")
        height = self._validate_positive_int(height, "height")
        sample_count = self._validate_positive_int(sample_count, "sample_count")
        pdf_integration_res = self._validate_positive_int(
            pdf_integration_res, "pdf_integration_res"
        )
        samples_per_thread = self._validate_positive_int(samples_per_thread, "samples_per_thread")
        seed = self._validate_nonnegative_int(seed, "seed")
        view_direction_ws_value = self._validate_float3_direction(
            view_direction_ws, "view_direction_ws"
        )
        use_analytic_texture_filtering = self._use_analytic_texture_filtering(texture_filtering)
        texture_filter_width_scale = self._validate_texture_filter_width_scale(
            texture_filter_width_scale
        )

        scene.update()
        module = self._get_module(scene, material)

        eval_texture = self._create_result_texture(
            width, height, "material_visualizer_full_sphere_eval"
        )
        pdf_texture = self._create_result_texture(
            width, height, "material_visualizer_full_sphere_pdf"
        )
        eval_pdf_kernel = module.full_sphere_eval_pdf_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        eval_pdf_kernel.dispatch(
            thread_count=spy.uint3(width, height, 1),
            material=material,
            view_direction_ws=view_direction_ws_value,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            pdf_integration_res=pdf_integration_res,
            eval_result=eval_texture,
            pdf_result=pdf_texture,
        )

        histogram = spy.Tensor.zeros(self.device, (height, width), dtype="uint")
        counters = spy.Tensor.zeros(self.device, (6,), dtype="uint")
        thread_count = (sample_count + samples_per_thread - 1) // samples_per_thread
        delta_weight_sum_luma = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        delta_weight_max_luma = spy.Tensor.zeros(self.device, (thread_count,), dtype=float)
        sample_kernel = module.full_sphere_sample_histogram_kernel.type_conformances(
            self._type_conformances_for_tracer(scene)
        ).write(self._bind_scene)
        sample_kernel.dispatch(
            thread_count=spy.uint3(thread_count, 1, 1),
            material=material,
            sample_count=sample_count,
            samples_per_thread=samples_per_thread,
            seed=seed,
            view_direction_ws=view_direction_ws_value,
            use_analytic_texture_filtering=use_analytic_texture_filtering,
            texture_filter_width_scale=texture_filter_width_scale,
            histogram=histogram,
            counters=counters,
            delta_weight_sum_luma=delta_weight_sum_luma,
            delta_weight_max_luma=delta_weight_max_luma,
        )

        return {
            "eval_rgb": eval_texture.to_numpy()[..., :3].astype(np.float32).copy(),
            "pdf": pdf_texture.to_numpy()[..., 0].astype(np.float32).copy(),
            "histogram": histogram.to_numpy().astype(np.uint32, copy=False).copy(),
            "counters": counters.to_numpy().astype(np.uint32, copy=False).copy(),
            "delta_weight_sum_luma": float(np.sum(delta_weight_sum_luma.to_numpy())),
            "delta_weight_max_luma": float(np.max(delta_weight_max_luma.to_numpy())),
            "sample_count": sample_count,
            "samples_per_thread": samples_per_thread,
            "seed": seed,
        }

    def _bind_scene(self, cursor: Any) -> None:
        if self._prev_scene is None:
            raise RuntimeError("MaterialVisualizer has no scene to bind")
        self._prev_scene.bind(cursor)

    @staticmethod
    def _material_id(material: f2.Material) -> int:
        collection_index = int(material.collection_index)
        if collection_index < 0:
            raise ValueError("MaterialVisualizer requires a material owned by the scene")
        return collection_index + 1

    @staticmethod
    def _use_analytic_texture_filtering(texture_filtering: TextureFilteringMode) -> bool:
        if texture_filtering == "analytic":
            return True
        if texture_filtering == "lod0":
            return False
        raise ValueError("texture_filtering must be 'analytic' or 'lod0'")

    @staticmethod
    def _validate_texture_filter_width_scale(texture_filter_width_scale: float) -> float:
        value = float(texture_filter_width_scale)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError("texture_filter_width_scale must be finite and greater than zero")
        return value

    @staticmethod
    def _validate_positive_int(value: int, name: str) -> int:
        result = int(value)
        if result <= 0:
            raise ValueError(f"{name} must be positive")
        return result

    @staticmethod
    def _validate_nonnegative_int(value: int, name: str) -> int:
        result = int(value)
        if result < 0:
            raise ValueError(f"{name} must be non-negative")
        return result

    @staticmethod
    def _validate_camera_size(value: Sequence[int]) -> spy.uint2:
        if len(value) != 2:
            raise ValueError("camera_size must have exactly two components")
        width = int(value[0])
        height = int(value[1])
        if width <= 0 or height <= 0:
            raise ValueError("camera_size components must be positive")
        return spy.uint2(width, height)

    @staticmethod
    def _validate_camera_size_for_texture(
        camera_size: Sequence[int] | None, result_texture: spy.Texture
    ) -> tuple[int, int]:
        if camera_size is None:
            return (int(result_texture.width), int(result_texture.height))
        if len(camera_size) != 2:
            raise ValueError("camera_size must have exactly two components")
        width = int(camera_size[0])
        height = int(camera_size[1])
        if width <= 0 or height <= 0:
            raise ValueError("camera_size components must be positive")
        return (width, height)

    @staticmethod
    def _validate_pixel_offset_tuple(
        pixel_offset: Sequence[int],
        result_texture: spy.Texture,
        camera_size: tuple[int, int],
    ) -> tuple[int, int]:
        if len(pixel_offset) != 2:
            raise ValueError("pixel_offset must have exactly two components")
        x = int(pixel_offset[0])
        y = int(pixel_offset[1])
        if x < 0 or y < 0:
            raise ValueError("pixel_offset components must be non-negative")
        if (
            x + int(result_texture.width) > camera_size[0]
            or y + int(result_texture.height) > camera_size[1]
        ):
            raise ValueError("pixel_offset plus result texture size must fit inside camera_size")
        return (x, y)

    @staticmethod
    def _validate_float3_direction(value: Sequence[float], name: str) -> spy.float3:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (3,):
            raise ValueError(f"{name} must have exactly three components")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values")
        length = float(np.linalg.norm(array))
        if length <= 0.0:
            raise ValueError(f"{name} must be non-zero")
        array = array / length
        return spy.float3(float(array[0]), float(array[1]), float(array[2]))

    @staticmethod
    def _validate_explicit_direct_lights(direct_lights: np.ndarray) -> np.ndarray:
        light_data = np.asarray(direct_lights, dtype=np.float32)
        if light_data.ndim == 2 and light_data.shape[1] == 8:
            reshaped = np.zeros((light_data.shape[0], 2, 4), dtype=np.float32)
            reshaped[:, 0, :3] = light_data[:, :3]
            reshaped[:, 1, :3] = light_data[:, 4:7]
            light_data = reshaped
        if light_data.ndim != 3 or light_data.shape[1:] != (2, 4):
            raise ValueError("direct_lights must have shape (N, 2, 4) or (N, 8)")
        if light_data.shape[0] <= 0:
            raise ValueError("direct_lights must contain at least one light")
        if not np.all(np.isfinite(light_data[:, :, :3])):
            raise ValueError("direct_lights contains non-finite directions or radiance")
        return np.ascontiguousarray(light_data, dtype=np.float32)

    def _ensure_result_texture(self, result_texture: spy.Texture | None) -> spy.Texture:
        if result_texture is not None:
            return result_texture
        return self._create_result_texture(512, 512, "material_visualizer_result")

    def _create_result_texture(self, width: int, height: int, label: str) -> spy.Texture:
        return self.device.create_texture(
            format=spy.Format.rgba32_float,
            width=width,
            height=height,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
            label=label,
        )

    def _ensure_material_aov_textures(
        self, aov_textures: MutableMapping[str, spy.Texture | None], result_texture: spy.Texture
    ) -> dict[MaterialAOVName, spy.Texture]:
        unknown_names = sorted(set(aov_textures) - set(MATERIAL_AOV_NAMES))
        if unknown_names:
            raise ValueError(f"Unknown MaterialVisualizer AOV name(s): {', '.join(unknown_names)}")

        aov_textures["beauty"] = result_texture
        result: dict[MaterialAOVName, spy.Texture] = {"beauty": result_texture}
        for name in MATERIAL_AOV_NAMES:
            texture = aov_textures.get(name)
            if texture is None:
                texture = self._create_result_texture(
                    result_texture.width,
                    result_texture.height,
                    f"material_visualizer_{name}",
                )
                aov_textures[name] = texture
            if texture.width != result_texture.width or texture.height != result_texture.height:
                raise ValueError(
                    f"MaterialVisualizer AOV '{name}' must match the result texture dimensions"
                )
            result[name] = texture
        return result

    def _ensure_preview_property_textures(
        self, property_textures: MutableMapping[str, spy.Texture | None] | None
    ) -> dict[PreviewPropertyName, spy.Texture]:
        if property_textures is None:
            property_textures = {}
        unknown_names = sorted(set(property_textures) - set(PREVIEW_PROPERTY_NAMES))
        if unknown_names:
            raise ValueError(
                f"Unknown MaterialVisualizer preview property name(s): {', '.join(unknown_names)}"
            )

        albedo = property_textures.get("albedo")
        if albedo is None:
            albedo = self._create_result_texture(512, 512, "material_visualizer_preview_albedo")
            property_textures["albedo"] = albedo

        result: dict[PreviewPropertyName, spy.Texture] = {"albedo": albedo}
        for name in PREVIEW_PROPERTY_NAMES:
            texture = property_textures.get(name)
            if texture is None:
                texture = self._create_result_texture(
                    albedo.width,
                    albedo.height,
                    f"material_visualizer_preview_{name}",
                )
                property_textures[name] = texture
            if texture.width != albedo.width or texture.height != albedo.height:
                raise ValueError(
                    f"MaterialVisualizer preview property '{name}' must match the albedo "
                    "texture dimensions"
                )
            result[name] = texture
        return result

    @staticmethod
    def _type_conformances_for_tracer(scene: f2.Scene) -> list[Any]:
        return [
            type_conformance
            for type_conformance in scene.requirements.type_conformances
            if getattr(type_conformance, "interface_name", "") != "IMaterial"
        ]

    def _get_module(self, scene: f2.Scene, material: f2.Material) -> spy.Module:
        reqs = scene.requirements
        req_modules = [spy.Module(scene.render_module)] + [
            spy.Module(module) for module in reqs.modules
        ]
        req_key = (
            material.slang_type_name,
            tuple(str(module) for module in reqs.modules),
            tuple(str(type_conformance) for type_conformance in reqs.type_conformances),
        )
        self._prev_scene = scene
        if self._prev_module is not None and self._prev_req_key == req_key:
            return self._prev_module

        import_lines = [
            f"import {module.name};"
            for module in req_modules
            if module.name and "/" not in module.name and "\\" not in module.name
        ]
        link_time_source = "\n".join(
            import_lines
            + [
                "import falcor2.render;",
                "import falcor2.utils;",
                f"export struct Material : IMaterial = {material.slang_type_name};",
                "export struct SampleGenerator : ISampleGenerator = TinyUniformSampleGenerator;",
            ]
        )
        link_time_hash = hashlib.sha256(link_time_source.encode()).hexdigest()
        link_time_types = self.device.load_module_from_source(
            "material_link_time_types_" + link_time_hash,
            link_time_source,
        )

        self._prev_module = spy.Module.load_from_file(
            self.device,
            "falcor2.tools.materials.material_sphere_tracer",
            link=req_modules + [spy.Module(link_time_types)],
        )
        self._prev_req_key = req_key
        return self._prev_module

    @staticmethod
    def _resolve_path(path: PathInput) -> Path:
        return Path(path).expanduser().resolve(strict=False)

    @classmethod
    def _paths_equal(cls, a: PathInput, b: PathInput) -> bool:
        return cls._resolve_path(a) == cls._resolve_path(b)

    @staticmethod
    def _set_env_map_light_visualizer_transform(env_light: f2.EnvMapLight) -> None:
        transform = f2.Transform()
        transform.scale = spy.float3(-1.0, 1.0, -1.0)
        env_light.entity.transform = transform

    @staticmethod
    def _find_first_env_map_light(scene: f2.Scene) -> f2.EnvMapLight | None:
        for component in scene.components:
            if isinstance(component, f2.EnvMapLight):
                return component
        return None

    def _ensure_env_map_light(
        self, scene: f2.Scene, env_map_path: PathInput | None
    ) -> f2.EnvMapLight:
        env_light = self._find_first_env_map_light(scene)
        requested_path = self._resolve_path(env_map_path) if env_map_path else None

        if requested_path is None and env_light is not None:
            return env_light

        if requested_path is None:
            requested_path = self._resolve_path(
                PROJECT_ROOT / "external/MaterialX/resources/Lights/san_giuseppe_bridge.hdr"
            )

        if env_light is None:
            env_entity = scene.create_entity()
            env_light = env_entity.create_component(f2.EnvMapLight)
        elif env_light.env_map_path and self._paths_equal(
            Path(env_light.env_map_path), requested_path
        ):
            self._set_env_map_light_visualizer_transform(env_light)
            return env_light

        env_light.env_map_path = requested_path
        self._set_env_map_light_visualizer_transform(env_light)
        return env_light
