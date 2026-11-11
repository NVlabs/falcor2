# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import IntEnum
import logging
from pathlib import Path
from typing import Any, Callable, Optional
from uuid import uuid4
from weakref import ReferenceType, ref

import slangpy as spy
import falcor2 as f2

from falcor2.editor.scene_shader import SceneShaderHelper
from falcor2.editor.graph_settings import GraphSettingsPanel
from falcor2.editor.utils import save_image
from falcor2.rendergraph import is_torch_tensor, torch_to_slangpy
from falcor2.rendergraph.render_node import RenderNode
from falcor2.streaming.config import StreamingConfig


_active_editor: ReferenceType["Editor"] | None = None
_logger = logging.getLogger(__name__)


def _get_active_editor() -> Optional["Editor"]:
    return _active_editor() if _active_editor is not None else None


def _set_active_editor(editor: "Editor") -> None:
    global _active_editor
    _active_editor = ref(editor)


def _clear_active_editor(editor: "Editor") -> None:
    global _active_editor
    if _get_active_editor() is editor:
        _active_editor = None


class RenderMode(IntEnum):
    path_traced = 0
    shaded = 1
    flat_shaded = 2
    geometry_type = 3
    geometry_instance_id = 4
    primitive_index = 5
    pos_ws = 6
    normal_ws = 7
    front_facing = 8
    texcoord = 9
    shading_normal = 10
    shading_tangent = 11
    shading_bitangent = 12


@dataclass
class EditorConfig:
    """Editor presentation options.

    Disable window for offscreen rendering. With streaming enabled, width and
    height fix the complete video frame size; the docked viewport can be smaller.
    """

    width: int = 1280
    height: int = 720
    title: str = "Falcor2"
    vsync: bool = True
    mcp: bool = True
    window: bool = True
    streaming: StreamingConfig | None = None


@dataclass
class FrameOutput:
    width: int
    height: int
    color_target: spy.Texture


class Editor:
    def __init__(
        self,
        device: spy.Device,
        config: Optional[EditorConfig] = None,
        scene: Optional[f2.Scene] = None,
    ):
        super().__init__()
        self.device = device
        self.config = config or EditorConfig()

        if self.config.width < 1 or self.config.height < 1:
            raise ValueError("Editor dimensions must be positive")
        if self.config.streaming is not None:
            self.config.streaming._server_config(self.config.width, self.config.height)
        self.window = None
        self.surface = None
        if self.config.window:
            self.window = spy.Window(
                width=self.config.width,
                height=self.config.height,
                title=self.config.title,
                resizable=self.config.streaming is None,
            )
            self.surface = self.device.create_surface(self.window)
            self.surface.configure(
                width=self.config.width,
                height=self.config.height,
                vsync=self.config.vsync,
            )

        self._output: Optional[FrameOutput] = None
        self._cmd = None
        self._image = None
        self._closed = False
        self._shutdown_complete = False
        self._stream = None
        self._composite = None
        self._capture_image = None
        self._swapchain_image = None
        self._rendered_frames = 0
        self._screenshot_requested = False
        self._tev_requested = False
        self._ui_frame_active = False
        self._needs_render = False
        self._dt = 0.0
        self._timer = spy.Timer()
        self._render_mode = RenderMode.path_traced

        # ImGui UI context
        self._ui_context = spy.ui.Context(device)
        self.profiler = spy.Profiler()
        self.profiler.enabled = False
        self.show_profiler = False
        self._profile_frame: Optional[Any] = None

        # C++ camera controller (Unreal Editor-style with orbit, dolly, etc.)
        self._camera_controller = f2.ui.CameraController()
        self._camera: Optional[f2.Camera] = None

        # Scene editor components
        self._scene_editor = f2.ui.SceneEditor()
        self._render_node: ReferenceType[RenderNode] | None = None
        self._graph_settings_panel = GraphSettingsPanel(
            self._scene_editor,
            self._get_render_node,
        )
        self._scene_editor.camera_controller = self._camera_controller
        self._scene_editor.add_help_section(
            "Application",
            [
                ("F2", "Save PNG and EXR to screenshots/"),
                ("F3", "Send viewport image to tev"),
                ("F5", "Toggle scene editor"),
                ("F6", "Toggle profiler"),
                ("F12", "Cycle render mode (D3D12/Vulkan)"),
            ],
        )
        self._scene_picker = f2.ui.ScenePicker(device)
        self._selection_overlay = f2.ui.SelectionOverlay(device)
        self._interaction_controller = f2.ui.SceneInteractionController(
            self._scene_editor,
            self._scene_picker,
            self._selection_overlay,
        )
        self._interaction_controller.pointer_capture_callback = self._on_camera_capture
        self._scene: Optional[f2.Scene] = None

        # Scene shader helper for debug modes
        self._scene_shader: Optional[SceneShaderHelper] = None
        self._viewer_module = None
        self._present_module = None
        self._mcp_bridge: Optional[Any] = None

        # User callbacks
        self.on_keyboard_event: Optional[Callable[..., Any]] = None
        self.on_mouse_event: Optional[Callable[..., Any]] = None

        if self.window is not None:
            self.window.on_keyboard_event = self._on_local_keyboard_event
            self.window.on_mouse_event = self._on_local_mouse_event
            self.window.on_resize = self._on_resize

        # Set initial scene.
        self.scene = scene
        try:
            if self.config.streaming is not None:
                from falcor2.streaming.editor import EditorStream

                self._stream = EditorStream(
                    self.device,
                    self.config.streaming,
                    self.config.width,
                    self.config.height,
                    self._on_keyboard_event,
                    self._on_mouse_event,
                    self._cancel_interaction,
                )
            self._start_mcp_bridge()
            if self.streaming_url is not None:
                print(f"Open {self.streaming_url} to control the editor.", flush=True)
        except Exception:
            self.close()
            raise
        _set_active_editor(self)

    @classmethod
    def create(
        cls,
        device: spy.Device,
        config: Optional[EditorConfig] = None,
        scene: Optional[f2.Scene] = None,
    ) -> "Editor":
        return cls(device, config, scene)

    @property
    def scene(self) -> Optional[f2.Scene]:
        return self._scene

    @scene.setter
    def scene(self, value: Optional[f2.Scene]):
        if value is self._scene:
            return
        self._scene = value
        self._scene_editor.scene = value
        self._interaction_controller.scene = value
        self._camera = value and value.active_camera
        self._scene_editor.request_initial_camera_framing()

    @property
    def output(self) -> Optional[FrameOutput]:
        return self._output

    @property
    def streaming_url(self) -> str | None:
        """Browser URL, including the configured token, while streaming is enabled."""
        return self._stream.url if self._stream is not None else None

    @property
    def frame_texture(self) -> spy.Texture | None:
        """Complete offscreen frame, in sRGB, for headless or streaming editors.

        The editor reuses this texture each frame. Local-only editors return None.
        """
        return self._composite

    @property
    def cmd(self):
        return self._cmd

    @property
    def needs_render(self) -> bool:
        return self._needs_render

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def render_mode(self) -> RenderMode:
        return self._render_mode

    @render_mode.setter
    def render_mode(self, value: RenderMode):
        mode = value if isinstance(value, RenderMode) else RenderMode(value)
        if mode != RenderMode.path_traced and self.device.info.type == spy.DeviceType.cuda:
            raise ValueError("Editor debug render modes require D3D12 or Vulkan")
        if mode != self._render_mode:
            self._render_mode = mode

    def close(self):
        self._closed = True
        if self._shutdown_complete:
            return
        self._shutdown_complete = True
        _clear_active_editor(self)
        try:
            if self._stream is not None:
                self._stream.close()
        finally:
            self._stream = None
            self._stop_mcp_bridge()
            if self._ui_frame_active:
                # Finish ImGui even when the application closes mid-frame.
                assert self._image is not None
                self._ui_frame_active = False
                if self._cmd is None:
                    self._cmd = self.device.create_command_encoder()
                self._ui_context.end_frame(self._image, self._cmd)
            self._cmd = self._image = self._swapchain_image = None
            self._needs_render = False
            self._end_profile_frame()
            self._wait_for_device()
            if self.surface is not None:
                self.surface.unconfigure()
                self.surface = None
            if self.window is not None:
                self.window.close()
                self.window.on_keyboard_event = None
                self.window.on_mouse_event = None
                self.window.on_resize = None
                self.window = None

    def _set_render_node(self, node: RenderNode) -> None:
        self._render_node = ref(node)

    def _get_render_node(self) -> RenderNode | None:
        return self._render_node() if self._render_node is not None else None

    def _on_camera_capture(self, captured: bool) -> None:
        if self.window is not None:
            remote = self._stream is not None and self._stream.controls_input
            self.window.cursor_mode = (
                spy.CursorMode.disabled if captured and not remote else spy.CursorMode.normal
            )

    def _cancel_interaction(self) -> None:
        self._interaction_controller.cancel_pointer_owner()
        # Clear ImGui's held local state when control changes hands.
        for key in spy.KeyCode:
            event = spy.KeyboardEvent()
            event.type, event.key = spy.KeyboardEventType.key_release, key
            self._ui_context.handle_keyboard_event(event)
        for button in (spy.MouseButton.left, spy.MouseButton.middle, spy.MouseButton.right):
            event = spy.MouseEvent()
            event.type, event.button = spy.MouseEventType.button_up, button
            self._ui_context.handle_mouse_event(event)

    def update(self) -> bool:
        """Core frame driver. Returns False when the editor is closed."""
        if self._closed:
            self.close()
            return False
        if self._image is not None or self._cmd is not None:
            raise RuntimeError("Editor.update() called before the previous frame was presented.")

        # Timing
        self._dt = self._timer.elapsed_s()
        self._timer.reset()

        # Process window events
        if self.window is not None:
            self.window.process_events()
        if self._stream is not None:
            self._stream.update(self._streaming_stats())
        if (self.window is not None and self.window.should_close()) or self._closed:
            self.close()
            return False

        self._update_mcp_bridge()

        scene = self._scene

        # Advance animation playback (driven by transport UI play/pause).
        self._scene_editor.update_playback(self._dt)

        # Clear a deleted selection before Scene.update() destroys its entity.
        self._interaction_controller.update_selection_overlay()

        # Update scene and check for changes
        if scene is not None:
            scene.update()

        # Acquire swapchain image
        if self.surface is None or self._stream is not None:
            self._image = self._ensure_composite()
            if self.surface is not None and self.surface.config:
                self._swapchain_image = self.surface.acquire_next_image()
        else:
            self._image = self.surface.acquire_next_image() if self.surface.config else None
        if self._image is None:
            self._output = None
            self._cmd = None
            self._needs_render = False
            return True

        self._begin_profile_frame()

        # Begin ImGui frame (pass window when camera is not captured to allow ImGui to manage the cursor).
        remote = self._stream is not None and self._stream.controls_input
        ui_window = (
            None if remote or self._interaction_controller.has_pointer_capture() else self.window
        )
        self._ui_context.begin_frame(self._image.width, self._image.height, ui_window)
        self._ui_frame_active = True

        # Determine viewport render resolution.
        # When scene editor is visible, use the docked viewport panel size.
        # Otherwise render at full swapchain resolution.
        if self._scene_editor.visible:
            vp = self._scene_editor.viewport_state
            vp_width = int(vp.size.x) if vp.size.x >= 1 else self._image.width
            vp_height = int(vp.size.y) if vp.size.y >= 1 else self._image.height
        else:
            vp_width = self._image.width
            vp_height = self._image.height

        self._update_camera(vp_width, vp_height)

        # Create frame output and command encoder
        self._cmd = self.device.create_command_encoder()
        self._output = self._ensure_frame_output(
            vp_width,
            vp_height,
            self._output,
        )

        # Determine if user needs to render
        if self._render_mode == RenderMode.path_traced:
            self._needs_render = True
        else:
            # Debug mode: editor renders it directly
            self._render_debug()
            self._present_frame()
            self._needs_render = False

        return True

    def run(self, renderer: Callable[[Optional[f2.Scene]], Any]) -> None:
        """Render requested frames with renderer(scene) until the editor closes.

        Always close the editor on completion or failure. The caller retains
        ownership of the device. Custom loops can use update() and present().
        """
        try:
            while self.update():
                if self.needs_render:
                    self.present(renderer(self._scene))
        finally:
            self.close()

    def present(self, image: Any):
        assert self._cmd

        if image is None:
            frame_output = self._require_frame_output()
            self._cmd.clear_texture_float(frame_output.color_target)
            self._present_frame()
            return

        if isinstance(image, spy.Texture):
            self.present_texture(image)
            return

        frame_output = self._require_frame_output()

        # Work around for our blit function not handling float tensors yet.
        if is_torch_tensor(image):
            image = torch_to_slangpy(self.device, image, "float4")
        elif not isinstance(image, spy.Tensor):
            raise TypeError(f"Unsupported image type for Editor.present(): {type(image)!r}")

        self._cmd.set_buffer_state(image.storage, spy.ResourceState.shader_resource)

        self._get_present_module().blit_tensor_to_texture(
            spy.grid((frame_output.height, frame_output.width)),
            spy.int2(frame_output.width, frame_output.height),
            spy.int2(0, 0),
            False,
            image,
            frame_output.color_target,
            _append_to=self._cmd,
        )
        self._present_frame()

    def present_texture(self, texture: spy.Texture):
        frame_output = self._require_frame_output()
        if texture.width != frame_output.width or texture.height != frame_output.height:
            raise ValueError(
                "Presented texture dimensions must match the current editor output size."
            )
        assert self._cmd
        self._cmd.blit(frame_output.color_target, texture)
        self._present_frame()

    def _present_frame(self):
        """Finalize and present the active frame."""
        if self._output is None or self._cmd is None or self._image is None:
            self._end_profile_frame()
            return

        fps = 1.0 / self._dt if self._dt > 0 else 0.0

        # Preserve the rendered image before selection outlines are composited.
        export_texture = None
        if self._screenshot_requested or self._tev_requested:
            export_texture = self.device.create_texture(
                format=self._output.color_target.format,
                width=self._output.width,
                height=self._output.height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource,
                label="editor_image_export",
            )
            self._cmd.copy_texture(
                export_texture,
                0,
                0,
                spy.uint3(0),
                self._output.color_target,
                0,
                0,
                spy.uint3(0),
            )

        # Update selection state before drawing the overlay so this render pass uses
        # the latest selection observed by the interaction controller.
        self._interaction_controller.update_selection_overlay()

        # Scene editor GPU passes (only when editor is visible).
        if self._scene_editor.visible and self._scene is not None and self._camera is not None:
            self._scene_picker.render(self._cmd, self._scene, self._camera)
            self._selection_overlay.draw_overlay(
                self._cmd,
                self._output.color_target,
                self._scene_picker.geometry_instance_id_texture,
                self._scene,
                self._camera,
            )

        # Render UI.
        if self._composite is not None:
            self._cmd.clear_texture_float(self._image, clear_value=spy.float4(0, 0, 0, 1))
        if self._scene_editor.visible:
            # Docked editor mode: outliner, inspector, toolbar, viewport panel.
            self._scene_editor.editor_ui()
            self._scene_editor.viewport_ui(self._output.color_target, fps)
        else:
            # Fullscreen mode: blit color target to swapchain.
            self._cmd.blit(self._image, self._output.color_target)
            f2.ui.draw_viewport_overlay(
                spy.math.float2(0, 0),
                spy.math.float2(self._image.width, self._image.height),
                fps,
            )

        # Submit the profiler after the scene editor has created its dockspace.
        if self.show_profiler:
            spy.ui.render_profiler_window(self.profiler)

        # Finalize ImGui and render to swapchain.
        self._ui_frame_active = False
        self._ui_context.end_frame(self._image, self._cmd)

        # Submit and present
        if self._stream is not None:
            # Raw compatible-format copy preserves sRGB bytes for the encoder.
            assert self._capture_image is not None
            self._cmd.copy_texture(
                self._capture_image,
                0,
                0,
                spy.uint3(0),
                self._image,
                0,
                0,
                spy.uint3(0),
            )
        present_image = self._swapchain_image if self._composite is not None else self._image
        if present_image is not None:
            if self._composite is not None:
                self._cmd.blit(present_image, self._image)
            self._cmd.set_texture_state(present_image, spy.ResourceState.present)
        self.device.submit_command_buffer(self._cmd.finish())
        self._end_profile_frame()
        self.profiler.tick()
        del self._image
        self._image = None
        self._cmd = None
        if self._stream is not None:
            from falcor2.streaming import InputTransfer

            assert self._capture_image is not None
            self._stream.capture(self._capture_image, InputTransfer.srgb)
        if present_image is not None:
            assert self.surface is not None
            self.surface.present()
        self._swapchain_image = None
        if self._composite is not None:
            # Bound application work, including CUDA's host submission fences.
            self.device.wait_for_idle()
        self._rendered_frames += 1
        self._needs_render = False

        if export_texture is not None:
            self._export_image(export_texture)

    def _export_image(self, texture: spy.Texture) -> None:
        """Export a submitted viewport snapshot, keeping failures out of the render loop."""
        screenshot_requested = self._screenshot_requested
        tev_requested = self._tev_requested
        self._screenshot_requested = self._tev_requested = False

        if screenshot_requested:
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
                path = Path("screenshots") / f"screenshot-{timestamp}-{uuid4().hex}"
                for suffix in (".exr", ".png"):
                    output_path = path.with_suffix(suffix)
                    save_image(texture, output_path)
                    print(f"Saved viewport image to {output_path.resolve()}", flush=True)
            except Exception:
                _logger.exception("Failed to save viewport image")

        if tev_requested:
            try:
                # The texture overload reads back now; tev sends the owned bitmap asynchronously.
                spy.tev.show_async(texture)
            except Exception:
                _logger.exception("Failed to send viewport image to tev")

    def _ensure_composite(self) -> spy.Texture:
        if self._composite is None:
            usage = spy.TextureUsage.shader_resource
            usage |= (
                spy.TextureUsage.render_target
                if self.device.has_feature(spy.Feature.rasterization)
                else spy.TextureUsage.unordered_access
            )
            self._composite = self.device.create_texture(
                format=spy.Format.rgba8_unorm_srgb,
                width=self.config.width,
                height=self.config.height,
                mip_count=1,
                usage=usage,
                label="editor_composite",
            )
            if self._stream is not None:
                self._capture_image = self.device.create_texture(
                    format=spy.Format.rgba8_unorm,
                    width=self.config.width,
                    height=self.config.height,
                    mip_count=1,
                    usage=spy.TextureUsage.shader_resource,
                    label="editor_capture",
                )
        return self._composite

    def _streaming_stats(self) -> dict[str, Any]:
        result = {
            "mode": "editor",
            "rendered": self._rendered_frames,
            "render_mode": self._render_mode.name,
            "editor_visible": self._scene_editor.visible,
        }
        if self._camera is not None:
            pos = self._camera_controller.transform.translation
            result["camera_position"] = [pos.x, pos.y, pos.z]
        return result

    def _begin_profile_frame(self) -> None:
        if self._profile_frame is not None:
            raise RuntimeError("A profiler frame is already active.")
        frame = spy.profile_frame("frame")
        frame.__enter__()
        self._profile_frame = frame

    def _end_profile_frame(self) -> None:
        frame = self._profile_frame
        self._profile_frame = None
        if frame is not None:
            frame.__exit__(None, None, None)

    def _wait_for_device(self) -> None:
        self.device.wait()
        self.profiler.tick()

    def _update_camera(self, width: int, height: int) -> None:
        self._camera = self._scene_editor.update_camera(self._dt, width, height)
        if self._camera is None:
            self._interaction_controller.cancel_pointer_owner()

    def _render_debug(self):
        """Dispatch debug shaders using SceneShaderHelper."""
        assert self._output is not None
        assert self._scene is not None
        assert self._camera is not None

        self._get_viewer_function(self._scene, "render_debug_no_ids").dispatch(
            thread_count=spy.uint3(self._camera.width, self._camera.height, 1),
            camera=self._camera,
            render_mode=int(self._render_mode),
            color_output=self._output.color_target,
        )

    def _get_scene_shader(self) -> SceneShaderHelper:
        if self._scene_shader is None:
            self._scene_shader = SceneShaderHelper(self.device)
        return self._scene_shader

    def _get_present_module(self):
        if self._present_module is None:
            self._present_module = spy.Module(self.device.load_module("falcor2.editor"))
        return self._present_module

    def _require_frame_output(self) -> FrameOutput:
        if self._output is None or self._cmd is None:
            raise RuntimeError("Editor.present() requires an active frame after Editor.update().")
        return self._output

    def _get_viewer_function(self, scene: f2.Scene, name: str):
        helper = self._get_scene_shader()
        module, requirements = helper.prepare(scene, "falcor2.editor")
        return (
            getattr(module, name)
            .type_conformances(requirements.type_conformances)
            .write(helper.bind_scene)
        )

    def _ensure_frame_output(
        self,
        width: int,
        height: int,
        frame_output: Optional[FrameOutput],
    ) -> FrameOutput:
        color_target = None if frame_output is None else frame_output.color_target
        if color_target is None or color_target.width != width or color_target.height != height:
            color_target = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=width,
                height=height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="editor_color",
            )

        return FrameOutput(
            width=width,
            height=height,
            color_target=color_target,
        )

    # -- MCP bridge -----------------------------------------------------------

    def _start_mcp_bridge(self) -> None:
        if not self.config.mcp or self._mcp_bridge is not None:
            return

        from falcor2.mcp.editor_bridge import create_editor_bridge

        bridge = create_editor_bridge(self)
        bridge.start()
        self._mcp_bridge = bridge

    def _update_mcp_bridge(self) -> None:
        if self._mcp_bridge is not None:
            self._mcp_bridge.update()

    def _stop_mcp_bridge(self) -> None:
        bridge = self._mcp_bridge
        self._mcp_bridge = None
        if bridge is not None:
            bridge.stop()

    # -- Event handlers --------------------------------------------------------

    def _on_local_keyboard_event(self, event: spy.KeyboardEvent) -> None:
        if self._stream is None or not self._stream.controls_input:
            self._on_keyboard_event(event)

    def _on_local_mouse_event(self, event: spy.MouseEvent) -> None:
        if self._stream is None or not self._stream.controls_input:
            self._on_mouse_event(event)

    def _cycle_render_mode(self):
        if self.device.info.type == spy.DeviceType.cuda:
            return
        modes = list(RenderMode)
        next_index = (modes.index(self._render_mode) + 1) % len(modes)
        self.render_mode = modes[next_index]

    def _on_keyboard_event(self, event: spy.KeyboardEvent):
        # Always feed ImGui first so its input state stays coherent.
        imgui_consumed = self._ui_context.handle_keyboard_event(event)

        # Forward to interaction controller for camera movement/modifier tracking.
        camera_consumed = self._interaction_controller.handle_keyboard_event(event)

        # These application function keys must work even while ImGui owns focus.
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.f2:
                self._screenshot_requested = True
                return
            if event.key == spy.KeyCode.f3:
                self._tev_requested = True
                return
            if event.key == spy.KeyCode.f5:
                self._scene_editor.visible = not self._scene_editor.visible
                return
            if event.key == spy.KeyCode.f6:
                self._toggle_profiler()
                return
            if event.key == spy.KeyCode.f12:
                self._cycle_render_mode()
                return

        if imgui_consumed or camera_consumed:
            return

        if self._scene_editor.visible and self._scene_editor.handle_keyboard_shortcut(event):
            return

        # Remaining app hotkeys.
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.space:
                if not self._scene_editor.visible:
                    self._scene_editor.playing = not self._scene_editor.playing
                return
        if self.on_keyboard_event is not None:
            self.on_keyboard_event(event)

    def _toggle_profiler(self) -> None:
        self.show_profiler = not self.show_profiler
        if self.show_profiler and not self.profiler.enabled:
            self.profiler.enabled = True

    def _on_mouse_event(self, event: spy.MouseEvent):
        # Always feed ImGui so it maintains consistent state.
        imgui_consumed = self._ui_context.handle_mouse_event(event)

        # Forward to interaction controller while it owns the pointer, or when
        # the event is eligible for viewport interaction.
        interaction_consumed = False
        if (
            self._interaction_controller.has_pointer_capture()
            or not self._scene_editor.visible
            or self._scene_editor.is_viewport_interactive()
        ):
            interaction_consumed = self._interaction_controller.handle_mouse_event(event)

        if imgui_consumed or interaction_consumed:
            return

        if self.on_mouse_event is not None:
            self.on_mouse_event(event)

    def _on_resize(self, width: int, height: int):
        if self.surface is None:
            return
        self.device.wait()
        if width <= 0 or height <= 0:
            self.surface.unconfigure()
            return

        self.surface.configure(width=width, height=height, vsync=self.config.vsync)
