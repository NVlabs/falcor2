# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import slangpy as spy
import numpy as np
from falcor2.minitracer.geometry import GeometryInstance
from falcor2.minitracer.scene import Scene, DirtyFlag
from falcor2.minitracer.camera_controller import CameraController


class TransformWindow(spy.ui.Window):
    def __init__(self, parent: spy.ui.Screen, instance: GeometryInstance):
        super().__init__(
            parent, "Transform", position=spy.float2(10, 600), size=spy.float2(420, 200)
        )
        self._instance = instance
        self._name = spy.ui.Text(parent=self, text=instance.geometry.name)
        print(f"Selected: {instance.geometry.name}")
        self._position = spy.ui.DragFloat3(
            parent=self,
            label="Position",
            value=instance.transform.pos,
            speed=0.1,
            callback=lambda _: self._update_transform(),
        )
        self._rotation = spy.ui.DragFloat3(
            parent=self,
            label="Rotation",
            value=spy.math.degrees(spy.math.euler_angles(instance.transform.rot)),
            speed=0.1,
            callback=lambda _: self._update_transform(),
        )
        self._scale = spy.ui.DragFloat3(
            parent=self,
            label="Scale",
            value=instance.transform.scale,
            speed=0.1,
            callback=lambda _: self._update_transform(),
        )

    def _update_transform(self):
        self._instance.transform.pos = self._position.value
        self._instance.transform.rot = spy.math.quat_from_euler_angles(
            spy.math.radians(self._rotation.value)
        )
        self._instance.transform.scale = self._scale.value
        self._instance.scene.mark_dirty(DirtyFlag.transform)


class Viewport:

    def __init__(self, device: spy.Device, window: spy.Window, screen: spy.ui.Screen, scene: Scene):
        super().__init__()
        self._device = device
        self._window = window
        self._screen = screen
        self._scene = scene

        self._transform_window: Optional[TransformWindow] = None

        # Get camera from scene
        self._camera = scene.cameras[0]

        # Create camera controller
        self._camera_controller = CameraController(self._camera)
        self._camera_controller.on_disable_cursor = lambda: setattr(
            self._window, "cursor_mode", spy.CursorMode.disabled
        )
        self._camera_controller.on_enable_cursor = lambda: setattr(
            self._window, "cursor_mode", spy.CursorMode.normal
        )

        self._id_texture: spy.Texture = None  # type: ignore
        self._output_texture: spy.Texture = None  # type: ignore

        self._id_mouse_pos = device.create_buffer(size=4, usage=spy.BufferUsage.unordered_access)

        self._module = spy.Module(device.load_module("falcor2.minitracer.ui"))

        self._mouse_pos = spy.float2()
        self._update_selected_object = False
        self._selected_id = 0xFFFFFFFF

    @property
    def output_texture(self):
        return self._output_texture

    def render(self, width: int, height: int, render_texture: spy.Texture):

        if (
            not self._id_texture
            or self._id_texture.width != width
            or self._id_texture.height != height
        ):
            self._id_texture = self._device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.r8_uint,
                width=width,
                height=height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="id_texture",
            )
        if (
            not self._output_texture
            or self._output_texture.width != width
            or self._output_texture.height != height
        ):
            self._output_texture = self._device.create_texture(
                type=spy.TextureType.texture_2d,
                format=spy.Format.rgba32_float,
                width=width,
                height=height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="output_texture",
            )

        self._module.render_geometry_instance_id.dispatch(
            thread_count=spy.uint3(width, height, 1),
            camera=self._camera.get_uniforms(),
            mouse_pos=spy.uint2(int(max(self._mouse_pos.x, 0)), int(max(self._mouse_pos.y, 0))),
            id_texture=self._id_texture,
            id_mouse_pos=self._id_mouse_pos,
            vars={"g_scene": self._scene.get_uniforms()},
        )

        if self._update_selected_object:
            self._selected_id = int(self._id_mouse_pos.to_numpy().view(dtype=np.uint32)[0])
            self._update_selected_object = False
            self._selected_entity = (
                self._scene.geometry_instances[self._selected_id]
                if self._selected_id != 0xFFFFFFFF
                else None
            )
            # Recreating transform window is bit of a workaround
            # around weird behavior with widget callbacks when showing/hiding imgui windows.
            if self._transform_window:
                self._screen.remove_child(self._transform_window)
                self._transform_window = None
            if self._selected_entity:
                self._transform_window = TransformWindow(self._screen, self._selected_entity)

        self._module.render_overlay.dispatch(
            thread_count=spy.uint3(width, height, 1),
            scene=self._scene.get_uniforms(),
            camera=self._camera.get_uniforms(),
            selected_id=self._selected_id,
            id_texture=self._id_texture,
            render_texture=render_texture,
            output_texture=self._output_texture,
        )

    def update(self, dt: float) -> bool:
        if self._camera_controller.update(dt):
            self._camera_controller.camera.recompute()
            return True
        return False

    def handle_keyboard_event(self, event: spy.KeyboardEvent) -> bool:
        self._camera_controller.handle_keyboard_event(event)
        return False

    def handle_mouse_event(self, event: spy.MouseEvent) -> bool:
        self._camera_controller.handle_mouse_event(event)

        if event.is_button_down():
            if event.button == spy.MouseButton.left:
                self._update_selected_object = True
        if event.is_move():
            self._mouse_pos = event.pos

        return False
