# SPDX-License-Identifier: Apache-2.0
"""High-level path-tracing pipeline with optional DLSS Ray Reconstruction."""

import slangpy as spy
from typing import Any, Optional

import falcor2 as f2
from falcor2.rendergraph import ContainerSpec, RenderNode
from falcor2.rendernodes.accumulator_node import AccumulatorNode
from falcor2.rendernodes.dlss_ray_recon_node import DLSSRayReconNode
from falcor2.rendernodes.reference_pathtracer_node import ReferencePathTracerNode
from falcor2.rendernodes.tonemapper_node import TonemapperNode
from falcor2.utils.jitter import frame_jitter


class PathTracerPipeline(RenderNode):
    """Reference path-tracing pipeline with accumulation, tonemapping, and DLSS-RR."""

    def __init__(self, device: spy.Device):
        """Create the pipeline nodes and initialize frame state."""
        super().__init__()
        self.device = device
        self.path_tracer = ReferencePathTracerNode.create(device)
        self.accumulator = AccumulatorNode.create(device)
        self.tonemapper = TonemapperNode.create(device)
        self.dlss_rr = DLSSRayReconNode.create(device)
        self.enable_dlss_rr = False
        self.tone_map = True
        self._output_spec = ContainerSpec.auto()
        self._guide_output_specs = dict(self.path_tracer.guide_output_specs)
        self._iteration = 0
        self._last_scene_update_generation = None
        self._last_camera_uniforms = None
        self._pending_reset = False
        self._spp = 1
        self._last_dlss_key: tuple[int, int, int, int, Any] | None = None
        self._last_enable_dlss_rr = self.enable_dlss_rr

    @classmethod
    def create(cls, device: spy.Device) -> "PathTracerPipeline":
        """Create a path-tracer pipeline for ``device``."""
        return cls(device)

    @property
    def spp(self) -> int:
        """Number of samples to render per ``forward()`` call in accumulation mode."""
        return self._spp

    @spp.setter
    def spp(self, value: int):
        self._spp = value

    @property
    def output_spec(self) -> ContainerSpec:
        """Final pipeline output specification."""
        return self._output_spec

    @output_spec.setter
    def output_spec(self, value: ContainerSpec):
        self._output_spec = value
        self.request_reset()

    @property
    def guide_output_specs(self) -> dict[str, ContainerSpec | None]:
        """Guide output specs used by the normal non-DLSS path."""
        return self._guide_output_specs

    @guide_output_specs.setter
    def guide_output_specs(self, value: dict[str, ContainerSpec | None]):
        self._guide_output_specs = {
            name: value.get(name) for name in self.path_tracer.guide_output_specs
        }
        self.request_reset()

    def _needs_reset(self, scene: f2.Scene, camera: Optional[f2.Camera]):
        """Return true when scene/camera/pending state invalidates accumulation."""
        needs_reset = self._pending_reset

        if scene.update_generation != self._last_scene_update_generation:
            needs_reset = True
        self._last_scene_update_generation = scene.update_generation

        if camera is not None:
            cam_uniforms = camera.get_uniforms()
            if cam_uniforms != self._last_camera_uniforms:
                needs_reset = True
                self._last_camera_uniforms = cam_uniforms

        return needs_reset

    def request_reset(self):
        """Request a history reset on the next rendered frame."""
        self._pending_reset = True

    def _reset_accumulation(self, cmd: spy.CommandEncoder | None = None):
        """Reset accumulated color history for the normal path."""
        # TODO: We don't want to reset path tracer necessarily (as it is tracking prec camera - need to figure this out)
        self._iteration = 0
        # self.path_tracer.reset()
        self.accumulator.reset(cmd=cmd)
        self._pending_reset = False

    def _reset_dlss_rr(self):
        """Reset DLSS-RR temporal state tracked by the pipeline."""
        self._iteration = 0
        # TODO: We don't want to reset path tracer necessarily (as it is tracking prec camera - need to figure this out)
        # self.path_tracer.reset()
        self.dlss_rr.reset()
        self._pending_reset = False

    def _resolve_final_output_size(self, camera: f2.Camera) -> tuple[int, int]:
        """Resolve the final target size from the pipeline output spec."""
        fallback = ContainerSpec.texture2d(
            spy.Format.rgba16_float,
            (camera.height, camera.width),
        )
        resolved = self._output_spec.resolved(fallback)
        if not isinstance(resolved.dims, tuple) or len(resolved.dims) != 2:
            raise ValueError("PathTracerPipeline output dimensions must be two-dimensional.")
        target_height, target_width = resolved.dims
        return int(target_width), int(target_height)

    def _render_iteration(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        cmd: spy.CommandEncoder | None = None,
    ) -> tuple[Any, dict[str, Any | None]]:
        """Render one accumulated path-tracer iteration and return color plus guides."""
        scene.update()

        if self._needs_reset(scene, camera):
            self._reset_accumulation(cmd=cmd)

        # In normal mode the path tracer is configured from pipeline-owned specs
        # every frame, so stale DLSS internal specs cannot leak across mode changes.
        self.path_tracer.output_spec = self._output_spec
        self.path_tracer.guide_output_specs = self._guide_output_specs
        image, guides = self.path_tracer(scene, camera, iteration=self._iteration, cmd=cmd)
        image = self.accumulator(image, cmd=cmd)
        self._iteration += 1

        return image, guides

    def _render_dlss_rr_iteration(
        self,
        scene: f2.Scene,
        camera: f2.Camera,
        cmd: spy.CommandEncoder | None = None,
    ) -> tuple[Any, dict[str, Any | None]]:
        """Render one DLSS-RR frame and return final color plus low-resolution guides."""
        scene.update()

        # TODO: Need to figure out how to reset when scene changes but not camera
        # if self._needs_reset(scene, None):
        #     self._reset_dlss_rr()

        target_width, target_height = self._resolve_final_output_size(camera)
        color_spec, guide_specs = self.dlss_rr.get_optimal_specs(target_width, target_height)
        if not isinstance(color_spec.dims, tuple) or len(color_spec.dims) != 2:
            raise ValueError("DLSSRR optimal color dimensions must be two-dimensional.")
        render_height, render_width = color_spec.dims

        # DLSS history is keyed by the render/target sizes and quality mode. This
        # reset must happen before guide generation so motion vectors line up.
        dlss_key = (
            int(render_width),
            int(render_height),
            target_width,
            target_height,
            self.dlss_rr.quality,
        )
        if dlss_key != self._last_dlss_key:
            self._iteration = 0
            self.path_tracer.reset()
            self.dlss_rr.reset()
            self._last_dlss_key = dlss_key

        jitter_x, jitter_y = frame_jitter(self._iteration)

        # DLSS mode asks the path tracer for internal low-resolution color and
        # guide textures. The pipeline final output spec remains owned by dlss_rr.
        self.path_tracer.output_spec = color_spec
        self.path_tracer.guide_output_specs = guide_specs  # type: ignore
        color, guides = self.path_tracer(
            scene,
            camera,
            iteration=self._iteration,
            subpixel_offset=spy.float2(jitter_x, jitter_y),
            subpixel_random_jitter=0.0,
            cmd=cmd,
        )

        # DLSS evaluates to the final target size and format requested from the
        # pipeline, while consuming the internal path-tracer inputs above.
        self.dlss_rr.output_spec = self._output_spec
        output = self.dlss_rr(
            color,
            guides,
            camera,
            target_width,
            target_height,
            jitter_offset=spy.float2(-jitter_x, -jitter_y),
            cmd=cmd,
        )

        self._iteration += 1
        output_guides_dict = dict(guides)
        output_guides_dict["color"] = color
        return output, output_guides_dict

    def forward(
        self,
        scene: f2.Scene,
        camera: Optional[f2.Camera] = None,
        cmd: spy.CommandEncoder | None = None,
        output_guides: bool = False,
    ) -> Any:
        """Render the current frame.

        If ``output_guides`` is true, return ``(image, guides)`` instead of only
        the final image. In DLSS mode, ``guides`` also includes ``"color"`` for
        the low-resolution DLSS input color.
        """
        camera = camera or (scene and scene.active_camera)
        if camera is None:
            return None

        if self.enable_dlss_rr != self._last_enable_dlss_rr:
            self.request_reset()
            self._last_enable_dlss_rr = self.enable_dlss_rr

        if self.enable_dlss_rr:
            image, guides = self._render_dlss_rr_iteration(
                scene,
                camera,
                cmd=cmd,
            )
        else:
            image, guides = self._render_iteration(
                scene,
                camera,
                cmd=cmd,
            )
            for _ in range(1, self.spp):
                image, guides = self._render_iteration(
                    scene,
                    camera,
                    cmd=cmd,
                )

        if self.tone_map:
            image = self.tonemapper(image, cmd=cmd)

        if output_guides:
            return image, guides
        return image
