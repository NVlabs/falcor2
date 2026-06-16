# SPDX-License-Identifier: Apache-2.0
"""Reference path-tracer render node and guide-output management."""

from typing import Any, Optional

import slangpy as spy
from slangpy import CommandEncoder, Device, float3

import falcor2 as f2
from falcor2.rendergraph import (
    ContainerSpec,
    RenderNode,
    Container,
    OutputPrelude,
)

from falcor2.editor.scene_shader import SceneShaderHelper

REFERENCE_MODULE_PATH = "falcor2/rendernodes/reference_pathtracer.slang"
WRITE_GUIDE_INTERFACE = "IWriteGuide"


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
        self._previous_camera_uniforms: dict[str, Any] | None = None
        self._previous_camera_dims: tuple[int, int] | None = None
        self._guide_specs: dict[str, ContainerSpec | None] = {
            name: None for name in self._prelude.specs
        }
        self._guides: dict[str, Any | None] = {name: None for name in self._prelude.specs}
        self._enable_nee = False
        self._enable_mis = True
        self._max_depth = 3
        self._enable_analytic_lights = True
        self._enable_environment_light = True
        self._enable_emissive_triangles = True
        self._env_map_as_background = True
        self._background_color = float3(0.0, 0.0, 0.0)
        self._constants = {}
        self.settings_changed()

    @classmethod
    def create(cls, device: Device) -> "ReferencePathTracerNode":
        """Create a reference path tracer node for ``device``."""
        return cls(device)

    def reset(self) -> None:
        """Reset previous-camera history used for motion-vector guide outputs."""
        self._previous_camera_uniforms = None
        self._previous_camera_dims = None

    def settings_changed(self):
        """Refresh shader specialization constants after a setting changes."""
        self._constants = {
            "ENABLE_NEE": self._enable_nee,
            "ENABLE_MIS": self._enable_mis,
            "MAX_DEPTH": self._max_depth,
            "ENABLE_ANALYTIC_LIGHTS": self._enable_analytic_lights,
            "ENABLE_ENVIRONMENT_LIGHT": self._enable_environment_light,
            "ENABLE_EMISSIVE_TRIANGLES": self._enable_emissive_triangles,
            "ENV_MAP_AS_BACKGROUND": self._env_map_as_background,
            "BACKGROUND_COLOR": self._background_color,
        }
        self._render_func = None
        self._render_func_constants = None

    @property
    def output_spec(self) -> ContainerSpec:
        """Container specification for the color output."""
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value: ContainerSpec):
        """Set the color output container specification."""
        self._output_spec = value

    @property
    def guide_output_specs(self) -> dict[str, ContainerSpec | None]:
        """Container specifications for optional guide outputs."""
        return self._guide_specs

    @guide_output_specs.setter
    def guide_output_specs(self, value: dict[str, ContainerSpec | None]):
        """Set guide output specs, ignoring names not declared by the guide interface."""
        self._guide_specs = {name: value.get(name) for name in self._prelude.specs}

    @property
    def enable_nee(self) -> bool:
        """Whether next-event estimation is enabled."""
        return self._enable_nee

    @enable_nee.setter
    def enable_nee(self, value: bool):
        """Enable or disable next-event estimation."""
        self._enable_nee = value
        self.settings_changed()

    @property
    def enable_mis(self) -> bool:
        """Whether multiple importance sampling is enabled."""
        return self._enable_mis

    @enable_mis.setter
    def enable_mis(self, value: bool):
        """Enable or disable multiple importance sampling."""
        self._enable_mis = value
        self.settings_changed()

    @property
    def max_depth(self) -> int:
        """Maximum path depth."""
        return self._max_depth

    @max_depth.setter
    def max_depth(self, value: int):
        """Set the maximum path depth."""
        self._max_depth = value
        self.settings_changed()

    @property
    def enable_analytic_lights(self) -> bool:
        """Whether analytic lights contribute to rendering."""
        return self._enable_analytic_lights

    @enable_analytic_lights.setter
    def enable_analytic_lights(self, value: bool):
        """Enable or disable analytic light contribution."""
        self._enable_analytic_lights = value
        self.settings_changed()

    @property
    def enable_environment_light(self) -> bool:
        """Whether environment lighting contributes to rendering."""
        return self._enable_environment_light

    @enable_environment_light.setter
    def enable_environment_light(self, value: bool):
        """Enable or disable environment light contribution."""
        self._enable_environment_light = value
        self.settings_changed()

    @property
    def enable_emissive_triangles(self) -> bool:
        """Whether emissive triangles contribute to rendering."""
        return self._enable_emissive_triangles

    @enable_emissive_triangles.setter
    def enable_emissive_triangles(self, value: bool):
        """Enable or disable emissive triangle contribution."""
        self._enable_emissive_triangles = value
        self.settings_changed()

    @property
    def env_map_as_background(self) -> bool:
        """Whether the environment map is visible as the background."""
        return self._env_map_as_background

    @env_map_as_background.setter
    def env_map_as_background(self, value: bool):
        """Enable or disable using the environment map as the background."""
        self._env_map_as_background = value
        self.settings_changed()

    @property
    def background_color(self) -> float3:
        """Fallback background color when the environment map is not shown."""
        return self._background_color

    @background_color.setter
    def background_color(self, value: float3):
        """Set the fallback background color."""
        self._background_color = value
        self.settings_changed()

    def _bind_scene(self, cursor: Any):
        """Bind scene resources into the render call cursor."""
        self._scene_shader.bind_scene(cursor)

    def _clear_guide_outputs(
        self, command_encoder: CommandEncoder, guide_outputs: dict[str, Any | None]
    ):
        """Clear enabled guide outputs using their reflected clear values."""
        for name, output in guide_outputs.items():
            if output is None:
                continue
            spec = self._prelude.specs[name]
            Container.clear(output, clear_value=spec.clear_value, command_encoder=command_encoder)

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
        )
        if self._render_func is None or self._render_func_constants != render_func_constants:
            assert self._scene

            ray_desc = f2.SceneRayTracingSetup.RayDesc()
            ray_desc.name = "intersect"
            ray_desc.has_miss = True
            ray_desc.has_closest_hit = True
            rt_setup = f2.SceneRayTracingSetup.create(self._scene, [ray_desc])

            # Generate a prelude that implements IWriteGuide for the requested guide
            # targets, then attach scene binding and ray tracing dispatch metadata.
            render_func = module.render.constants(constants).prelude(
                self._prelude.generate(write_guide)
            )

            self._render_func = (
                render_func.type_conformances(self._scene.requirements.type_conformances)
                .write(self._bind_scene)
                .ray_tracing(
                    hit_groups=rt_setup.hit_groups,
                    hit_group_names=rt_setup.sbt_hit_group_names,
                    miss_entry_points=rt_setup.sbt_miss_entry_points,
                    max_recursion=3,
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
                outputs[name] = None
                continue
            resolved = self._resolve_guide_output_spec(name, spec, width, height)
            self._guides[name] = Container.create_temp(
                self._device,
                resolved,
                current=self._guides.get(name),
            )
            outputs[name] = self._guides[name]
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
            self._clear_guide_outputs(render_cmd, guide_outputs)

        render_camera = camera.calc_uniforms(current_dims[0], current_dims[1])
        current_camera_uniforms = dict(render_camera.get_uniforms())
        if self._previous_camera_dims == current_dims:
            previous_camera = self._previous_camera_uniforms
        else:
            previous_camera = None
        previous_camera = previous_camera or current_camera_uniforms

        # Guide resources are bound per call rather than cached on the render
        # function, because the target containers are frame-local and can change.
        func.write(lambda cursor: self._prelude.bind(cursor, write_guide)).call(
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

    def forward(
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
