# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference path-tracer render node and guide-output management."""

from enum import IntEnum
from typing import Any, Optional

import slangpy as spy
from slangpy import CommandEncoder, Device, float3

import falcor2 as f2
from falcor2.reflection import reflected, reflected_property
from falcor2.rendergraph import (
    ContainerSpec,
    RenderNode,
    Container,
    OutputOperation,
    OutputPrelude,
)

from falcor2.editor.scene_shader import SceneShaderHelper

REFERENCE_MODULE_PATH = "falcor2/rendernodes/reference_pathtracer.slang"
WRITE_GUIDE_INTERFACE = "IWriteGuide"
MAX_PATH_DEPTH = 0xFF


class SchedulingMode(IntEnum):
    """Path scheduling implementation."""

    simple = 0
    ser = 1


class VisibilityRayMode(IntEnum):
    """Visibility-ray traversal implementation."""

    ray_query = 0
    trace_ray = 1


@reflected
class ReferencePathTracerNode(RenderNode):
    def __init__(self, device: Device):
        """Create a reference path tracer node and reflect its guide outputs."""
        super().__init__()
        self._device = device
        self._scene_shader = SceneShaderHelper(device)
        self._pathtracer_module = spy.Module.load_from_file(self._device, REFERENCE_MODULE_PATH)
        self._output_spec = ContainerSpec.auto()
        self._module = None
        self._prelude = OutputPrelude.create(self._pathtracer_module, WRITE_GUIDE_INTERFACE)
        self._render_func = None
        self._render_func_constants = None
        self._scene = None
        self._output = None
        self._previous_camera_uniforms: f2.CameraUniforms | None = None
        self._previous_camera_dims: tuple[int, int] | None = None
        self._guide_specs: dict[str, ContainerSpec | None] = {
            name: None for name in self._prelude.specs
        }
        self._guides: dict[str, Any | None] = {name: None for name in self._prelude.specs}
        self._guide_outputs_needing_clear: set[str] = set()
        self._light_sampler = f2.PowerLightSampler()
        self._enable_nee = False
        self._enable_mis = True
        self._enable_interior_tracking = True
        self._enable_nested_interiors = False
        self._enable_homogeneous_media = True
        self._enable_russian_roulette = True
        self._max_depth = 3
        self._rr_depth = 3
        self._enable_depth_of_field = True
        self._use_background_color = False
        self._background_color = float3(0.0, 0.0, 0.0)
        self._scheduling_mode = SchedulingMode.simple
        self._visibility_ray_mode = (
            VisibilityRayMode.ray_query
            if self._device.has_feature(spy.Feature.ray_query)
            else VisibilityRayMode.trace_ray
        )
        self._constants = {}
        self._settings = {}
        self.constants_changed()
        self.settings_changed()

    @classmethod
    def create(cls, device: Device) -> "ReferencePathTracerNode":
        """Create a reference path tracer node for ``device``."""
        return cls(device)

    def reset(self) -> None:
        """Reset temporal state used by motion vectors and NaN checks."""
        self._previous_camera_uniforms = None
        self._previous_camera_dims = None
        self.reset_nan_checks()

    def reset_nan_checks(self) -> None:
        """Clear the NaN count guide before its next dispatch."""
        if self._guides.get("nan_count") is not None:
            self._guide_outputs_needing_clear.add("nan_count")

    def constants_changed(self) -> None:
        """Refresh shader specialization constants after a constant changes."""
        self._constants = {
            "ENABLE_NEE": self._enable_nee,
            "ENABLE_MIS": self._enable_mis,
            "ENABLE_INTERIOR_TRACKING": self._enable_interior_tracking,
            "ENABLE_NESTED_INTERIORS": self._enable_nested_interiors,
            "ENABLE_HOMOGENEOUS_MEDIA": self._enable_homogeneous_media,
            "ENABLE_RUSSIAN_ROULETTE": self._enable_russian_roulette,
            "ENABLE_DEPTH_OF_FIELD": self._enable_depth_of_field,
            "SCHEDULING_MODE": int(self._scheduling_mode),
            "VISIBILITY_RAY_MODE": int(self._visibility_ray_mode),
        }
        self._render_func = None
        self._render_func_constants = None

    def settings_changed(self) -> None:
        """Refresh runtime shader settings after a setting changes."""
        self._settings = {
            "max_depth": self._max_depth,
            "rr_depth": self._rr_depth,
            "use_background_color": self._use_background_color,
            "background_color": self._background_color,
        }

    @property
    def output_spec(self) -> ContainerSpec:
        """Container specification for the color output."""
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value: ContainerSpec):
        """Set the color output container specification."""
        self._output_spec = value

    @reflected_property(
        object_factories=(
            f2.UniformLightSampler,
            f2.PowerLightSampler,
            f2.HierarchicalLightSampler,
        ),
        ui_label="Light sampler",
        ui_group="Sampling",
    )
    def light_sampler(self) -> f2.LightSampler:
        """Host-side strategy used to sample all scene lights."""
        return self._light_sampler

    @light_sampler.setter
    def light_sampler(self, value: f2.LightSampler) -> None:
        """Select the host-side scene light sampling strategy."""
        if not isinstance(value, f2.LightSampler):
            raise TypeError("light_sampler must be a LightSampler.")
        self._light_sampler = value
        self._render_func = None
        self._render_func_constants = None

    @property
    def guide_output_specs(self) -> dict[str, ContainerSpec | None]:
        """Container specifications for optional guide outputs."""
        return self._guide_specs

    @guide_output_specs.setter
    def guide_output_specs(self, value: dict[str, ContainerSpec | None]):
        """Set guide output specs, ignoring names not declared by the guide interface."""
        self._guide_specs = {name: value.get(name) for name in self._prelude.specs}

    @reflected_property(ui_label="Next-event estimation", ui_group="Sampling")
    def enable_nee(self) -> bool:
        """Whether next-event estimation is enabled."""
        return self._enable_nee

    @enable_nee.setter
    def enable_nee(self, value: bool):
        """Enable or disable next-event estimation."""
        self._enable_nee = value
        self.constants_changed()

    @reflected_property(
        ui_label="Multiple importance sampling",
        ui_group="Sampling",
        ui_enable_if=lambda path_tracer: path_tracer.enable_nee,
    )
    def enable_mis(self) -> bool:
        """Whether multiple importance sampling is enabled."""
        return self._enable_mis

    @enable_mis.setter
    def enable_mis(self, value: bool):
        """Enable or disable multiple importance sampling."""
        self._enable_mis = value
        self.constants_changed()

    @reflected_property(ui_label="Interior tracking", ui_group="Media")
    def enable_interior_tracking(self) -> bool:
        """Whether closed-surface interiors are tracked for solid dielectric transmission."""
        return self._enable_interior_tracking

    @enable_interior_tracking.setter
    def enable_interior_tracking(self, value: bool) -> None:
        """Enable or disable closed-surface interior tracking."""
        self._enable_interior_tracking = value
        self.constants_changed()

    @reflected_property(
        ui_label="Nested interiors",
        ui_group="Media",
        ui_enable_if=lambda path_tracer: path_tracer.enable_interior_tracking,
    )
    def enable_nested_interiors(self) -> bool:
        """Whether priority-based interfaces between nested interiors are handled."""
        return self._enable_nested_interiors

    @enable_nested_interiors.setter
    def enable_nested_interiors(self, value: bool) -> None:
        """Enable or disable priority-based nested-interior handling."""
        self._enable_nested_interiors = value
        self.constants_changed()

    @reflected_property(
        ui_label="Homogeneous media",
        ui_group="Media",
        ui_enable_if=lambda path_tracer: path_tracer.enable_interior_tracking,
    )
    def enable_homogeneous_media(self) -> bool:
        """Whether homogeneous absorption and scattering are handled."""
        return self._enable_homogeneous_media

    @enable_homogeneous_media.setter
    def enable_homogeneous_media(self, value: bool) -> None:
        """Enable or disable homogeneous absorption and scattering."""
        self._enable_homogeneous_media = value
        self.constants_changed()

    @reflected_property(ui_label="Russian roulette", ui_group="Sampling")
    def enable_russian_roulette(self) -> bool:
        """Whether Russian roulette path termination is enabled."""
        return self._enable_russian_roulette

    @enable_russian_roulette.setter
    def enable_russian_roulette(self, value: bool) -> None:
        """Enable or disable Russian roulette path termination."""
        self._enable_russian_roulette = value
        self.constants_changed()

    @reflected_property(
        value_range=(1, MAX_PATH_DEPTH),
        ui_label="Maximum depth",
        ui_group="Sampling",
    )
    def max_depth(self) -> int:
        """Maximum path depth."""
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int) -> None:
        """Set the maximum path depth."""
        self._max_depth = max(1, min(value, MAX_PATH_DEPTH))
        self.settings_changed()

    @reflected_property(
        value_range=(1, MAX_PATH_DEPTH),
        ui_label="Russian roulette depth",
        ui_group="Sampling",
        ui_enable_if=lambda path_tracer: path_tracer.enable_russian_roulette,
    )
    def rr_depth(self) -> int:
        """Path depth at which Russian roulette starts."""
        return self._rr_depth

    @rr_depth.setter
    def rr_depth(self, value: int) -> None:
        """Set the path depth at which Russian roulette starts."""
        self._rr_depth = max(1, min(value, MAX_PATH_DEPTH))
        self.settings_changed()

    @reflected_property(ui_label="Depth of field", ui_group="Camera")
    def enable_depth_of_field(self) -> bool:
        """Whether cameras with a nonzero aperture use stochastic depth of field."""
        return self._enable_depth_of_field

    @enable_depth_of_field.setter
    def enable_depth_of_field(self, value: bool) -> None:
        """Enable or disable stochastic thin-lens camera rays."""
        self._enable_depth_of_field = value
        self.constants_changed()

    @reflected_property(ui_label="Use background color", ui_group="Background")
    def use_background_color(self) -> bool:
        """Whether a constant color is used instead of the environment map as background."""
        return self._use_background_color

    @use_background_color.setter
    def use_background_color(self, value: bool) -> None:
        """Enable or disable using a constant background color."""
        self._use_background_color = value
        self.settings_changed()

    @reflected_property(
        ui_label="Background color",
        ui_group="Background",
        ui_enable_if=lambda path_tracer: path_tracer.use_background_color,
    )
    def background_color(self) -> float3:
        """Constant color used when ``use_background_color`` is enabled."""
        return self._background_color

    @background_color.setter
    def background_color(self, value: float3) -> None:
        """Set the constant background color."""
        self._background_color = value
        self.settings_changed()

    @reflected_property(
        ui_label="Scheduling",
        ui_group="Advanced",
        ui_enable_if=lambda path_tracer: path_tracer._device.has_feature(
            spy.Feature.shader_execution_reordering
        ),
    )
    def scheduling_mode(self) -> SchedulingMode:
        """Selected path scheduling implementation."""
        return self._scheduling_mode

    @scheduling_mode.setter
    def scheduling_mode(self, value: SchedulingMode):
        """Select the path scheduling implementation."""
        mode = SchedulingMode(value)
        if mode == SchedulingMode.ser and not self._device.has_feature(
            spy.Feature.shader_execution_reordering
        ):
            raise RuntimeError("SER scheduling is not supported by this device.")
        self._scheduling_mode = mode
        self.constants_changed()

    @reflected_property(
        ui_label="Visibility rays",
        ui_group="Advanced",
        ui_enable_if=lambda path_tracer: path_tracer._device.has_feature(spy.Feature.ray_query),
    )
    def visibility_ray_mode(self) -> VisibilityRayMode:
        """Selected visibility-ray traversal implementation."""
        return self._visibility_ray_mode

    @visibility_ray_mode.setter
    def visibility_ray_mode(self, value: VisibilityRayMode):
        """Select the visibility-ray traversal implementation."""
        mode = VisibilityRayMode(value)
        if mode == VisibilityRayMode.ray_query and not self._device.has_feature(
            spy.Feature.ray_query
        ):
            raise RuntimeError("Ray-query visibility is not supported by this device.")
        self._visibility_ray_mode = mode
        self.constants_changed()

    def _clear_guide_outputs(
        self,
        command_encoder: CommandEncoder,
        guide_outputs: dict[str, Any | None],
        iteration: int,
    ):
        """Clear guide outputs using their reflected operation and clear values."""
        for name, output in guide_outputs.items():
            if output is None:
                continue
            spec = self._prelude.specs[name]
            if (
                spec.operation == OutputOperation.increment
                and iteration != 0
                and name not in self._guide_outputs_needing_clear
            ):
                continue
            Container.clear(output, clear_value=spec.clear_value, command_encoder=command_encoder)
            self._guide_outputs_needing_clear.discard(name)

    def _get_module(self, scene: f2.Scene) -> Any:
        """Return the scene-specialized module and invalidate cached dispatch if needed."""
        module = self._scene_shader.get_module(scene, self._pathtracer_module)
        if module is not self._module:
            self._module = module
            self._render_func = None
            self._render_func_constants = None
        return self._module

    def _get_render_func(self, module: Any, write_guide: dict[str, Any] | None = None) -> Any:
        """Return a cached render function specialized for settings and guide outputs."""
        constants = dict(self._constants)
        write_guide = write_guide or {}

        # The render function is specialized by shader constants and by the generated
        # guide prelude. The prelude signature includes target types, so changing
        # which guides are written invalidates this cached dispatch object.
        render_func_constants = (
            constants,
            self._prelude.signature(write_guide),
            self._light_sampler.slang_type_name,
            self._light_sampler.shader_generation,
        )
        if self._render_func is None or self._render_func_constants != render_func_constants:
            assert self._scene

            scatter_ray_desc = f2.SceneRayTracingSetup.RayDesc()
            scatter_ray_desc.name = "scatter"
            scatter_ray_desc.has_miss = True
            scatter_ray_desc.has_closest_hit = True
            scatter_ray_desc.has_any_hit = self._scene.requirements.requires_opacity_evaluation
            ray_descs = [scatter_ray_desc]

            if self._visibility_ray_mode == VisibilityRayMode.trace_ray:
                visibility_ray_desc = f2.SceneRayTracingSetup.RayDesc()
                visibility_ray_desc.name = "visibility"
                visibility_ray_desc.has_miss = True
                visibility_ray_desc.has_any_hit = (
                    self._scene.requirements.requires_opacity_evaluation
                )
                ray_descs.append(visibility_ray_desc)

            rt_setup = f2.SceneRayTracingSetup.create(
                self._scene,
                ray_descs,
            )

            # Generate a prelude that selects the light sampler and implements
            # IWriteGuide for the requested guide targets.
            prelude = ""
            prelude += self._light_sampler.shader_specialization_source
            prelude += (
                "export struct LightSampler : ILightSampler = "
                f"{self._light_sampler.slang_type_name};\n"
            )
            prelude += self._prelude.generate(write_guide)
            render_func = module.render.constants(constants).prelude(prelude)

            # Attach scene binding and ray tracing dispatch metadata.
            self._render_func = (
                render_func.type_conformances(self._scene.requirements.type_conformances)
                .write(self._scene_shader.bind_scene)
                .ray_tracing(
                    hit_groups=rt_setup.hit_groups,
                    hit_group_names=rt_setup.sbt_hit_group_names,
                    miss_entry_points=rt_setup.sbt_miss_entry_points,
                    max_recursion=(
                        2 if self._visibility_ray_mode == VisibilityRayMode.trace_ray else 1
                    ),
                    max_ray_payload_size=128,
                    flags=rt_setup.pipeline_flags,
                )
            )
            self._render_func_constants = render_func_constants
        return self._render_func

    def _fix_format(self, format: spy.Format) -> spy.Format:
        """Return a writable format for the current device backend."""
        # This is a work around for now to prevent us selecting default formats that CUDA can't write to
        if self._device.desc.type == spy.DeviceType.cuda:
            if format == spy.Format.r16_float:
                return spy.Format.r32_float
            if format == spy.Format.rg16_float:
                return spy.Format.rg32_float
            if format == spy.Format.rgba16_float:
                return spy.Format.rgba32_float
            if format == spy.Format.rgba16_uint:
                return spy.Format.rgba32_uint
        return format

    def _default_color_format(self) -> spy.Format:
        """Return the default color target format for the current backend."""
        if self._device.desc.type == spy.DeviceType.cuda:
            return spy.Format.rgba32_float
        return spy.Format.rgba16_float

    def _resolve_color_output_spec(self, camera: f2.Camera) -> ContainerSpec:
        """Resolve the color output spec using the camera dimensions as fallback."""
        fallback = ContainerSpec.texture2d(
            self._default_color_format(),
            (camera.height, camera.width),
        )
        return self._output_spec.resolved(fallback)

    def _get_output(self, spec: ContainerSpec) -> Any:
        """Create or reuse the color output container for the resolved spec."""
        self._output = Container.create_temp(self._device, spec, current=self._output)
        return self._output

    def _resolve_guide_output_spec(
        self,
        name: str,
        spec: ContainerSpec,
        width: int,
        height: int,
    ) -> ContainerSpec:
        """Resolve one guide output spec using its reflected default format."""
        fallback = ContainerSpec.texture2d(
            self._fix_format(self._prelude.specs[name].format),
            (height, width),
        )
        return spec.resolved(fallback)

    def _get_guide_outputs(self, width: int, height: int) -> dict[str, Any | None]:
        """Create, reuse, or disable guide output containers for this frame."""
        outputs: dict[str, Any | None] = {}
        for name, spec in self._guide_specs.items():
            if spec is None:
                self._guides[name] = None
                self._guide_outputs_needing_clear.discard(name)
                outputs[name] = None
                continue
            resolved = self._resolve_guide_output_spec(name, spec, width, height)
            previous = self._guides.get(name)
            output = Container.create_temp(
                self._device,
                resolved,
                current=previous,
            )
            if output is not previous:
                self._guide_outputs_needing_clear.add(name)
            self._guides[name] = output
            outputs[name] = output
        return outputs

    def _make_write_guide_targets(self, guide_outputs: dict[str, Any | None]) -> dict[str, Any]:
        """Build the non-null guide target map consumed by the output prelude."""
        return {
            name: output
            for name, output in guide_outputs.items()
            if name in self._prelude.specs and output is not None
        }

    def _render(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        color: Any,
        iteration: int,
        guide_outputs: dict[str, Any | None] | None = None,
        subpixel_offset: Any = spy.float2(0.0, 0.0),
        subpixel_random_jitter: float = 1.0,
        cmd: Optional[CommandEncoder] = None,
    ):
        """Render into ``color`` and any requested guide outputs."""
        self._scene = scene
        render_dims = Container.dims(color)
        if len(render_dims) != 2:
            raise ValueError("ReferencePathTracerNode output must be two-dimensional.")
        render_height, render_width = render_dims
        current_dims = (int(render_width), int(render_height))

        # The module is scene-specialized, while the generated prelude is specialized
        # only by the write-guide target dictionary below.
        self._light_sampler.update(scene, cmd)
        module = self._get_module(scene)
        guide_outputs = guide_outputs or {name: None for name in self._prelude.specs}
        write_guide = self._make_write_guide_targets(guide_outputs)
        func = self._get_render_func(module, write_guide)

        # Guide buffers are caller-visible outputs. Clear them before dispatch so
        # pixels that are not written by a disabled path have deterministic values.
        render_cmd = cmd
        temp_cmd = None
        if any(output is not None for output in guide_outputs.values()):
            if render_cmd is None:
                temp_cmd = self._device.create_command_encoder()
                render_cmd = temp_cmd
            self._clear_guide_outputs(render_cmd, guide_outputs, iteration)

        render_camera = camera.calc_uniforms(current_dims[0], current_dims[1])
        current_camera_uniforms = render_camera
        if self._previous_camera_dims == current_dims:
            previous_camera = self._previous_camera_uniforms
        else:
            previous_camera = None
        previous_camera = previous_camera or current_camera_uniforms

        # Guide resources are bound per call rather than cached on the render
        # function, because the target containers are frame-local and can change.
        def bind_dispatch(cursor: Any) -> None:
            self._prelude.bind(cursor, write_guide)
            cursor["light_sampler"] = self._light_sampler
            path_tracer_settings = cursor["g_path_tracer"]["settings"]
            for name, value in self._settings.items():
                path_tracer_settings[name] = value
            trace_context = cursor["g_trace_path_context"]
            trace_context["current_camera"] = current_camera_uniforms
            trace_context["previous_camera"] = previous_camera

        func.write(bind_dispatch).call(
            ray_sampler=render_camera,
            previous_camera=previous_camera,
            output=Container.to_render_layout(color),
            iteration=iteration,
            subpixel_offset=subpixel_offset,
            subpixel_random_jitter=subpixel_random_jitter,
            _append_to=render_cmd,
        )
        self._previous_camera_uniforms = current_camera_uniforms
        self._previous_camera_dims = current_dims
        if temp_cmd is not None:
            self._device.submit_command_buffer(temp_cmd.finish())

    def _exec(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        iteration: int = 0,
        subpixel_offset: Any = spy.float2(0.0, 0.0),
        subpixel_random_jitter: float = 1.0,
        cmd: Optional[CommandEncoder] = None,
    ) -> tuple[Any, dict[str, Any | None]]:
        """Render a frame and return the color output plus guide outputs."""
        color_spec = self._resolve_color_output_spec(camera)
        output = self._get_output(color_spec)
        render_dims = Container.dims(output)
        if len(render_dims) != 2:
            raise ValueError("ReferencePathTracerNode output must be two-dimensional.")
        render_height, render_width = render_dims
        guide_outputs = self._get_guide_outputs(render_width, render_height)
        self._render(
            scene,
            camera,
            output,
            iteration=iteration,
            guide_outputs=guide_outputs,
            subpixel_offset=subpixel_offset,
            subpixel_random_jitter=subpixel_random_jitter,
            cmd=cmd,
        )
        return output, dict(guide_outputs)
