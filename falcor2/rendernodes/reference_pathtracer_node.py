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
    IndexedOutput,
    OutputOperation,
    OutputPrelude,
)

from falcor2.editor.scene_shader import SceneShaderHelper
from falcor2.utils.cuda_workarounds import ensure_optix_cuda_stack
from falcor2.utils.sample_generators import SampleGeneratorType, bind_sample_generator

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


class VisibilityRayPlacement(IntEnum):
    """Visibility order within closest-hit processing."""

    early = 0
    late = 1


@reflected
class ReferencePathTracerNode(RenderNode):
    def __init__(self, device: Device):
        """Create a reference path tracer node and reflect its guide outputs."""
        super().__init__()
        self._device = device
        ensure_optix_cuda_stack(device)
        self._scene_shader = SceneShaderHelper(device)
        self._pathtracer_module = spy.Module.load_from_file(self._device, REFERENCE_MODULE_PATH)
        self._output_spec = ContainerSpec.auto()
        self._module = None
        self._prelude = OutputPrelude.create(self._pathtracer_module, WRITE_GUIDE_INTERFACE)
        self._fixed_guide_names = tuple(self._prelude.specs)
        self._indexed_outputs: tuple[IndexedOutput, ...] = ()
        self._render_func = None
        self._render_func_key = None
        self._scene = None
        self._scene_requirements: f2.SceneRequirements | None = None
        self._scene_capability_policy = f2.SceneCapabilityPolicy.required
        self._scene_capabilities: f2.SceneCapabilities | None = None
        self._output = None
        self._previous_camera_uniforms: f2.CameraUniforms | None = None
        self._previous_camera_dims: tuple[int, int] | None = None
        self._guide_specs: dict[str, ContainerSpec | None] = {
            name: None for name in self._fixed_guide_names
        }
        self._guides: dict[str, Any | None] = {name: None for name in self._fixed_guide_names}
        self._guide_outputs_needing_clear: set[str] = set()
        self._aov_registry: f2.MaterialAOVRegistry | None = None
        self._aov_validated_fixed: set[str] = set()
        self._aov_output_generation = -1
        # Name -> (ID, component count).
        self._aov_output_info: dict[str, tuple[int, int]] = {}
        self._light_sampler = f2.PowerLightSampler()
        self._sample_generator = SampleGeneratorType.uniform
        self.pixel_filter: f2.ReconstructionFilter = f2.GaussianReconstructionFilter()
        self._enable_nee = True
        self._enable_mis = True
        self._enable_nested_interiors = False
        self._enable_homogeneous_media = True
        self._enable_russian_roulette = True
        self._max_depth = 6
        self._rr_depth = 3
        self._enable_depth_of_field = True
        self._enable_triangle_backface_culling = False
        self._use_background_color = False
        self._background_color = float3(0.0, 0.0, 0.0)
        self._enable_precise_ray_offsets = False
        self._scheduling_mode = SchedulingMode.simple
        self._visibility_ray_mode = (
            VisibilityRayMode.ray_query
            if self._device.has_feature(spy.Feature.ray_query)
            else VisibilityRayMode.trace_ray
        )
        self._visibility_ray_placement = (
            VisibilityRayPlacement.early
            if self._visibility_ray_mode == VisibilityRayMode.ray_query
            else VisibilityRayPlacement.late
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
            "ENABLE_NESTED_INTERIORS": self._enable_nested_interiors,
            "ENABLE_HOMOGENEOUS_MEDIA": self._enable_homogeneous_media,
            "ENABLE_RUSSIAN_ROULETTE": self._enable_russian_roulette,
            "ENABLE_DEPTH_OF_FIELD": self._enable_depth_of_field,
            "ENABLE_TRIANGLE_BACKFACE_CULLING": self._enable_triangle_backface_culling,
            "ENABLE_PRECISE_RAY_OFFSETS": self._enable_precise_ray_offsets,
            "SCHEDULING_MODE": int(self._scheduling_mode),
            "VISIBILITY_RAY_MODE": int(self._visibility_ray_mode),
            "VISIBILITY_RAY_PLACEMENT": int(self._visibility_ray_placement),
        }
        self._render_func = None
        self._render_func_key = None

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

    @reflected_property(ui_label="Sample generator", ui_group="Sampling")
    def sample_generator(self) -> SampleGeneratorType:
        """Sequential random sample generator; QMC keeps a stable pixel scrambling seed."""
        return self._sample_generator

    @sample_generator.setter
    def sample_generator(self, value: SampleGeneratorType):
        self._sample_generator = SampleGeneratorType(value)
        self._render_func = None
        self._render_func_key = None

    @reflected_property(ui_label="Scene capabilities", ui_group="Advanced")
    def scene_capability_policy(self) -> f2.SceneCapabilityPolicy:
        """Select all, required, or specified scene capabilities for rendering."""
        return self._scene_capability_policy

    @scene_capability_policy.setter
    def scene_capability_policy(self, value: f2.SceneCapabilityPolicy) -> None:
        self._scene_capability_policy = value

    @property
    def scene_capabilities(self) -> f2.SceneCapabilities | None:
        """Caller-specified capabilities, if any.

        Assigning a mask selects the specified policy; None restores required.
        Rendering rejects selections that omit scene or consumer requirements.
        """
        return self._scene_capabilities

    @scene_capabilities.setter
    def scene_capabilities(self, value: f2.SceneCapabilities | None) -> None:
        if value is not None and not isinstance(value, f2.SceneCapabilities):
            raise TypeError("scene_capabilities must be SceneCapabilities or None.")
        self._scene_capabilities = value
        self._scene_capability_policy = (
            f2.SceneCapabilityPolicy.specified
            if value is not None
            else f2.SceneCapabilityPolicy.required
        )

    @property
    def capabilities(self) -> f2.SceneCapabilities | None:
        """Capabilities of the last prepared rendering configuration, or None before preparation."""
        return (
            self._scene_requirements.capabilities if self._scene_requirements is not None else None
        )

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
        self._render_func_key = None

    @property
    def guide_output_specs(self) -> dict[str, ContainerSpec | None]:
        """Container specifications for optional guide outputs."""
        return self._guide_specs

    @guide_output_specs.setter
    def guide_output_specs(self, value: dict[str, ContainerSpec | None]) -> None:
        """Set fixed guide and material-authored AOV output specifications."""
        guide_specs = {name: value.get(name) for name in self._fixed_guide_names}
        guide_specs.update(
            (name, spec) for name, spec in value.items() if name not in self._prelude.specs
        )
        self._guide_specs = guide_specs
        self._prune_guide_outputs()

    def _prune_guide_outputs(self) -> None:
        """Drop storage for names removed through assignment or in-place edits."""
        for name in self._guides.keys() - self._guide_specs.keys():
            del self._guides[name]
            self._guide_outputs_needing_clear.discard(name)

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

    @reflected_property(
        ui_label="Nested interiors",
        ui_group="Media",
        ui_enable_if=lambda path_tracer: path_tracer.capabilities is None
        or path_tracer.capabilities.has(f2.SceneCapabilityFlags.solid_transmission),
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
        ui_enable_if=lambda path_tracer: path_tracer.capabilities is None
        or path_tracer.capabilities.has(f2.SceneCapabilityFlags.solid_transmission),
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

    @reflected_property(ui_label="Triangle backface culling", ui_group="Advanced")
    def enable_triangle_backface_culling(self) -> bool:
        """Whether rays cull triangle backfaces on instances whose materials permit it."""
        return self._enable_triangle_backface_culling

    @enable_triangle_backface_culling.setter
    def enable_triangle_backface_culling(self, value: bool) -> None:
        """Enable or disable hardware triangle backface culling for this renderer."""
        self._enable_triangle_backface_culling = value
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

    @reflected_property(ui_label="Precise ray offsets", ui_group="Advanced")
    def enable_precise_ray_offsets(self) -> bool:
        """Use conservative triangle ray offsets to preserve visibility across scene scales.

        Adds shading cost and is disabled by default. Other geometry and sampled light
        endpoints use standard offsets. Changing this option recompiles the shader.
        """
        return self._enable_precise_ray_offsets

    @enable_precise_ray_offsets.setter
    def enable_precise_ray_offsets(self, value: bool) -> None:
        self._enable_precise_ray_offsets = value
        self.constants_changed()

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

    @reflected_property(ui_label="Visibility placement", ui_group="Advanced")
    def visibility_ray_placement(self) -> VisibilityRayPlacement:
        """Selected placement of visibility traversal within closest-hit processing."""
        return self._visibility_ray_placement

    @visibility_ray_placement.setter
    def visibility_ray_placement(self, value: VisibilityRayPlacement) -> None:
        """Select placement of visibility traversal within closest-hit processing."""
        self._visibility_ray_placement = VisibilityRayPlacement(value)
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
        module, requirements = self._scene_shader.prepare(
            scene,
            self._pathtracer_module,
            policy=self._scene_capability_policy,
            specified_capabilities=self._scene_capabilities,
            material_aovs=tuple(output.name for output in self._indexed_outputs),
        )
        self._scene_requirements = requirements
        if module is not self._module:
            self._module = module
            self._render_func = None
            self._render_func_key = None
        return self._module

    def _get_render_func(
        self,
        module: Any,
        write_guide: dict[str, Any] | None = None,
    ) -> Any:
        """Return a cached render function specialized for settings and guide outputs."""
        constants = dict(self._constants)
        write_guide = write_guide or {}

        # The dispatch key includes shader constants, the guide prelude (including
        # target types), light-sampler specialization, and scene pipeline requirements.
        render_func_key = (
            constants,
            self._prelude.signature(write_guide, self._indexed_outputs),
            self._light_sampler.slang_type_name,
            self._sample_generator,
            self._light_sampler.shader_generation,
            self._scene_requirements,
        )
        if self._render_func is None or self._render_func_key != render_func_key:
            assert self._scene
            assert self._scene_requirements is not None

            has_opacity = self._scene_requirements.capabilities.has(
                f2.SceneCapabilityFlags.opacity_evaluation
            )
            scatter_ray_desc = f2.SceneRayTracingSetup.RayDesc()
            scatter_ray_desc.name = "scatter"
            scatter_ray_desc.has_miss = True
            scatter_ray_desc.has_closest_hit = True
            scatter_ray_desc.has_any_hit = has_opacity
            ray_descs = [scatter_ray_desc]

            if self._visibility_ray_mode == VisibilityRayMode.trace_ray:
                visibility_ray_desc = f2.SceneRayTracingSetup.RayDesc()
                visibility_ray_desc.name = "visibility"
                visibility_ray_desc.has_miss = True
                visibility_ray_desc.has_any_hit = has_opacity
                ray_descs.append(visibility_ray_desc)

            rt_setup = f2.SceneRayTracingSetup.create(
                self._scene,
                ray_descs,
                requirements=self._scene_requirements,
            )
            ray_tracing_options: dict[str, Any] = {
                "hit_groups": rt_setup.hit_groups,
                "hit_group_names": rt_setup.sbt_hit_group_names,
                "miss_entry_points": rt_setup.sbt_miss_entry_points,
                "callable_entry_points": [],
                "max_recursion": (
                    2 if self._visibility_ray_mode == VisibilityRayMode.trace_ray else 1
                ),
                "max_ray_payload_size": 128,
                "max_attribute_size": 8,
                "flags": rt_setup.pipeline_flags,
            }
            pipeline_signature = {
                **ray_tracing_options,
                "hit_groups": [
                    (
                        group.hit_group_name,
                        group.closest_hit_entry_point,
                        group.any_hit_entry_point,
                        group.intersection_entry_point,
                    )
                    for group in rt_setup.hit_groups
                ],
                "flags": int(rt_setup.pipeline_flags),
            }

            # Generate a prelude that selects the light sampler and implements
            # IWriteGuide for the requested guide targets.
            # SlangPy's pipeline/SBT cache hashes generated code without ray-tracing
            # metadata. Include it here so distinct setups can share the scene module.
            prelude = f"// Ray-tracing setup: {pipeline_signature!r}\n"
            prelude += f"export struct SampleGenerator : ISampleGenerator = {self._sample_generator.slang_type_name};\n"
            prelude += self._light_sampler.shader_specialization_source
            prelude += (
                "export struct LightSampler : ILightSampler = "
                f"{self._light_sampler.slang_type_name};\n"
            )
            prelude += self._prelude.generate(write_guide, self._indexed_outputs)
            render_func = module.render.constants(constants).prelude(prelude)

            # Attach scene binding and ray tracing dispatch metadata.
            self._render_func = (
                render_func.type_conformances(self._scene_requirements.type_conformances)
                .write(self._scene_shader.bind_scene)
                .ray_tracing(**ray_tracing_options)
            )
        self._render_func_key = render_func_key
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

    def _get_guide_outputs(self, width: int, height: int, scene: f2.Scene) -> dict[str, Any | None]:
        """Resolve names in a prepared scene, then create or reuse selected targets."""
        self._prune_guide_outputs()
        outputs: dict[str, Any | None] = {}
        registry = None
        aov_info: dict[str, tuple[int, int]] | None = None
        for name, spec in self._guide_specs.items():
            if spec is None:
                self._guides[name] = None
                self._guide_outputs_needing_clear.discard(name)
                outputs[name] = None
                continue
            # Acquire catalog metadata only when at least one guide is enabled.
            if registry is None:
                registry = scene.material_aov_registry
                if registry is not self._aov_registry:
                    self._aov_registry = registry
                    self._aov_output_info.clear()
                    self._indexed_outputs = ()
                    self._aov_output_generation = -1
                generation = registry.output_generation
                if generation != self._aov_output_generation:
                    self._aov_validated_fixed.clear()
                    self._aov_output_generation = generation
            if name in self._prelude.specs:
                if name not in self._aov_validated_fixed:
                    if self._aov_output_generation and registry.has_output(name):
                        raise RuntimeError(
                            f"Guide output {name!r} is ambiguous: it is both a fixed guide and "
                            "a material-provided AOV."
                        )
                    self._aov_validated_fixed.add(name)
                default_format = self._prelude.specs[name].format
            else:
                info = self._aov_output_info.get(name)
                if info is None:
                    declaration = registry.find_output(name)
                    if declaration is None:
                        raise RuntimeError(
                            f"Guide output {name!r} is neither a fixed guide nor registered "
                            "in the current scene. Prepare the scene before rendering."
                        )
                    info = (declaration.id, declaration.component_count)
                if aov_info is None:
                    aov_info = {}
                aov_info[name] = info
                default_format = OutputPrelude.indexed_output_format(info[1])
            resolved = spec.resolved(
                ContainerSpec.texture2d(self._fix_format(default_format), (height, width))
            )
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
        # Commit binding metadata only after every requested name resolves.
        if aov_info != self._aov_output_info and (aov_info or self._aov_output_info):
            self._aov_output_info = aov_info or {}
            self._indexed_outputs = ()
        return outputs

    def _make_write_guide_targets(
        self,
        guide_outputs: dict[str, Any | None],
    ) -> dict[str, Any]:
        """Return fixed targets and retain indexed descriptors while their actual targets are unchanged."""
        fixed = {
            name: output
            for name, output in guide_outputs.items()
            if name in self._prelude.specs and output is not None
        }
        indexed_count = (
            sum(
                name in self._aov_output_info and output is not None
                for name, output in guide_outputs.items()
            )
            if self._aov_output_info
            else 0
        )
        if indexed_count != len(self._indexed_outputs) or any(
            guide_outputs.get(output.name) is not output.target for output in self._indexed_outputs
        ):
            self._indexed_outputs = tuple(
                IndexedOutput(
                    id=self._aov_output_info[name][0],
                    name=name,
                    component_count=self._aov_output_info[name][1],
                    target=output,
                )
                for name, output in guide_outputs.items()
                if name in self._aov_output_info and output is not None
            )
        return fixed

    def _render(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        color: Any,
        iteration: int,
        guide_outputs: dict[str, Any | None] | None = None,
        pixel_offset: Any = spy.float2(0.0, 0.0),
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
        guide_outputs = guide_outputs or {name: None for name in self._fixed_guide_names}
        write_guide = self._make_write_guide_targets(guide_outputs)
        indexed_outputs = self._indexed_outputs
        module = self._get_module(scene)
        func = self._get_render_func(module, write_guide)

        # Guide buffers are caller-visible outputs. Clear them before dispatch so
        # pixels that are not written by a disabled path have deterministic values.
        render_cmd = cmd
        temp_cmd = None
        if any(output is not None for output in guide_outputs.values()):
            if render_cmd is None:
                temp_cmd = self._device.create_command_encoder()
                render_cmd = temp_cmd
            self._clear_guide_outputs(render_cmd, write_guide, iteration)
            self._prelude.clear_indexed(indexed_outputs, command_encoder=render_cmd)

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
            self._prelude.bind(cursor, write_guide, indexed_outputs)
            cursor["light_sampler"] = self._light_sampler
            bind_sample_generator(cursor, self._device, self._sample_generator)
            path_tracer_settings = cursor["g_path_tracer"]["settings"]
            for name, value in self._settings.items():
                path_tracer_settings[name] = value
            trace_context = cursor["g_trace_path_context"]
            trace_context["current_camera"] = current_camera_uniforms
            trace_context["previous_camera"] = previous_camera

        func.write(bind_dispatch).call(
            ray_sampler=render_camera,
            pixel_filter=self.pixel_filter,
            previous_camera=previous_camera,
            output=Container.to_render_layout(color),
            iteration=iteration,
            pixel_offset=pixel_offset,
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
        pixel_offset: Any = spy.float2(0.0, 0.0),
        cmd: Optional[CommandEncoder] = None,
    ) -> tuple[Any, dict[str, Any | None]]:
        """Render a prepared scene; call scene.update() after scene edits.

        pixel_offset shifts the pixel's sampling footprint in input pixels, before
        the UV Y flip. Callers own accumulation and must reset it after persistent
        filter edits or a change to the fixed offset, but not for temporal jitter.
        """
        color_spec = self._resolve_color_output_spec(camera)
        output = self._get_output(color_spec)
        render_dims = Container.dims(output)
        if len(render_dims) != 2:
            raise ValueError("ReferencePathTracerNode output must be two-dimensional.")
        render_height, render_width = render_dims
        guide_outputs = self._get_guide_outputs(render_width, render_height, scene)
        self._render(
            scene,
            camera,
            output,
            iteration=iteration,
            guide_outputs=guide_outputs,
            pixel_offset=pixel_offset,
            cmd=cmd,
        )
        return output, dict(guide_outputs)
