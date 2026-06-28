# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Optional

import slangpy as spy
import falcor2 as f2

from falcor2.editor.scene_shader import SceneShaderHelper
from falcor2.rendergraph import is_torch_tensor, torch_to_slangpy


class RenderMode(IntEnum):
    path_traced = 0
    shaded = 1
    flat_shaded = 2
    geometry_type = 3
    geometry_instance_id = 4
    primitive_index = 5
    position_ws = 6
    normal_ws = 7
    front_facing = 8
    texcoord = 9
    shading_normal = 10
    shading_tangent = 11
    shading_bitangent = 12


@dataclass
class EditorConfig:
    width: int = 1280
    height: int = 720
    title: str = "Falcor2"
    vsync: bool = True


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

        self.window = spy.Window(
            width=self.config.width,
            height=self.config.height,
            title=self.config.title,
            resizable=True,
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
        self._needs_render = False
        self._dt = 0.0
        self._timer = spy.Timer()
        self._render_mode = RenderMode.path_traced

        # ImGui UI context
        self._ui_context = spy.ui.Context(device)

        # C++ camera controller (Unreal Editor-style with orbit, dolly, etc.)
        self._camera_controller = f2.ui.CameraController()
        self._camera: Optional[f2.Camera] = None

        # Scene editor components
        self._scene_editor = f2.ui.SceneEditor()
        self._scene_editor.camera_controller = self._camera_controller
        self._scene_editor.add_help_section(
            "Application",
            [
                ("F5", "Toggle scene editor"),
                ("F12", "Cycle render mode"),
                ("F", "Focus camera on selection"),
                ("Escape", "Close application"),
            ],
        )
        self._scene_picker = f2.ui.ScenePicker(device)
        self._selection_overlay = f2.ui.SelectionOverlay(device)
        self._interaction_controller = f2.ui.SceneInteractionController(
            self._scene_editor,
            self._scene_picker,
            self._selection_overlay,
            self._camera_controller,
        )
        self._interaction_controller.pointer_capture_callback = self._on_camera_capture
        self._scene: Optional[f2.Scene] = None

        # Scene shader helper for debug modes
        self._scene_shader: Optional[SceneShaderHelper] = None
        self._viewer_module = None
        self._present_module = None

        # User callbacks
        self.on_keyboard_event: Optional[Callable[..., Any]] = None
        self.on_mouse_event: Optional[Callable[..., Any]] = None

        self.window.on_keyboard_event = self._on_keyboard_event
        self.window.on_mouse_event = self._on_mouse_event
        self.window.on_resize = self._on_resize

        # Set initial scene.
        self.scene = scene

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
        if self._camera is not None:
            self._camera_controller.transform = f2.Transform(
                self._camera.entity.world_from_object_matrix
            )

    @property
    def output(self) -> Optional[FrameOutput]:
        return self._output

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
        if mode != self._render_mode:
            self._render_mode = mode

    def close(self):
        self._closed = True
        self.device.wait()
        self.window.close()

    def _on_camera_capture(self, captured: bool) -> None:
        self.window.cursor_mode = spy.CursorMode.disabled if captured else spy.CursorMode.normal

    def update(self) -> bool:
        """Core frame driver. Returns False when the window is closed."""
        if self._image is not None or self._cmd is not None:
            raise RuntimeError("Editor.update() called before the previous frame was presented.")

        # Timing
        self._dt = self._timer.elapsed_s()
        self._timer.reset()

        # Process window events
        self.window.process_events()
        if self.window.should_close() or self._closed:
            self.device.wait()
            return False

        scene = self._scene

        # Advance animation playback (driven by transport UI play/pause).
        self._scene_editor.update_playback(self._dt)

        # Update scene and check for changes
        if scene is not None:
            scene.update()

        # Detect active camera changes (e.g., camera removed and replaced).
        camera = scene and scene.active_camera
        if camera is not self._camera:
            self._camera = camera
            if camera is not None:
                self._camera_controller.transform = f2.Transform(
                    camera.entity.world_from_object_matrix
                )

        # Update camera controller
        if self._camera_controller.update(self._dt):
            if camera is not None:
                camera.entity.set_world_transform(self._camera_controller.transform)

        # Check surface
        if not self.surface.config:
            self._output = None
            self._cmd = None
            self._needs_render = False
            return True

        # Acquire swapchain image
        self._image = self.surface.acquire_next_image()
        if self._image is None:
            self._output = None
            self._cmd = None
            self._needs_render = False
            return True

        # Begin ImGui frame (pass window when camera is not captured to allow ImGui to manage the cursor).
        ui_window = None if self._interaction_controller.has_pointer_capture() else self.window
        self._ui_context.begin_frame(self._image.width, self._image.height, ui_window)

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

        if camera is not None:
            # Handle resize
            if vp_width != camera.width or vp_height != camera.height:
                camera.width = vp_width
                camera.height = vp_height

        self._scene_editor.set_active_camera(camera)

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

    def run(self, renderer: Any):
        """Convenience method for the default render loop."""
        while self.update():
            if self.needs_render:
                image = renderer(self._scene)
                self.present(image)

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
            return

        fps = 1.0 / self._dt if self._dt > 0 else 0.0

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

        # Update selection overlay state (checks version counter).
        self._interaction_controller.update_selection_overlay()

        # Render UI.
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

        # Finalize ImGui and render to swapchain.
        self._ui_context.end_frame(self._image, self._cmd)

        # Submit and present
        self._cmd.set_texture_state(self._image, spy.ResourceState.present)
        self.device.submit_command_buffer(self._cmd.finish())
        del self._image
        self._image = None
        self._cmd = None
        self.surface.present()
        self._needs_render = False

    def _render_debug(self):
        """Dispatch debug shaders using SceneShaderHelper."""
        assert self._output is not None
        assert self._scene is not None
        assert self._camera is not None

        self._get_viewer_function(self._scene, "render_debug_no_ids").dispatch(
            thread_count=spy.uint3(self._camera.width, self._camera.height, 1),
            camera=self._camera.get_uniforms(),
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
        module = helper.get_module(scene, "falcor2.editor")
        return (
            getattr(module, name)
            .type_conformances(scene.requirements.type_conformances)
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

    # -- Event handlers --------------------------------------------------------

    def _cycle_render_mode(self):
        modes = list(RenderMode)
        next_index = (modes.index(self._render_mode) + 1) % len(modes)
        self.render_mode = modes[next_index]

    def _on_keyboard_event(self, event: spy.KeyboardEvent):
        # Always feed ImGui first so its input state stays coherent.
        imgui_consumed = self._ui_context.handle_keyboard_event(event)

        # Forward to interaction controller for camera movement/modifier tracking.
        camera_consumed = self._interaction_controller.handle_keyboard_event(event)

        if imgui_consumed or camera_consumed:
            return

        if self._scene_editor.visible and self._scene_editor.handle_keyboard_shortcut(event):
            return

        # Remaining app hotkeys.
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.close()
                return
            elif event.key == spy.KeyCode.space:
                if not self._scene_editor.visible:
                    self._scene_editor.playing = not self._scene_editor.playing
                return
            elif event.key == spy.KeyCode.f12:
                self._cycle_render_mode()
                return
            elif event.key == spy.KeyCode.f5:
                self._scene_editor.visible = not self._scene_editor.visible
                return
            elif event.key == spy.KeyCode.f and self._camera is not None:
                self._interaction_controller.focus_on_selection(self._camera)
                return

        if self.on_keyboard_event is not None:
            self.on_keyboard_event(event)

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
        self.device.wait()
        if width <= 0 or height <= 0:
            self.surface.unconfigure()
            return

        self.surface.configure(width=width, height=height, vsync=self.config.vsync)
