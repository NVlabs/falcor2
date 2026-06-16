# SPDX-License-Identifier: Apache-2.0

import math
from os import PathLike
from pathlib import Path
from typing import Optional, Union, Callable

import numpy as np

import slangpy as spy
from slangpy.core.function import FunctionNode

from falcor2.minitracer.camera import Camera
from falcor2.minitracer.accumulator import Accumulator
from falcor2.minitracer.pathtracer import PathTracer
from falcor2.minitracer.scene import Scene, DirtyFlag
from falcor2.minitracer.importerutils import load_usd_scene, load_gltf_scene

import falcor2.minitracer.slangpyextensions  # type: ignore


def get_slang_include_paths():
    """
    Get core slang include paths required to compile MiniTracer shaders.
    """
    return [
        Path(__file__).parent.parent.parent / "slang",
        spy.SHADER_PATH,
    ]


def create_device(
    device_type: spy.DeviceType = spy.DeviceType.automatic,
    additional_include_paths: list[PathLike] = [],
) -> spy.Device:
    """
    Helper to creates a new SlangPy device configured correctly for use with MiniTracer
    with the most basic options.
    """
    return spy.Device(
        type=device_type,
        compiler_options={
            "include_paths": get_slang_include_paths()
            + [Path(p) for p in additional_include_paths],
        },
    )


class Renderer:
    """
    Simpler wrapper that combines the device, a path tracer and an accumulator to provide an
    interface for rendering images and guides.
    """

    def __init__(self, device: spy.Device):
        """
        Basic initialization of the renderer.
        """
        super().__init__()
        self.device = device
        self.path_tracer = PathTracer(self.device)
        self.accumulator: Optional[Accumulator] = None
        self.tone_map = True

    @property
    def max_depth(self) -> int:
        """
        Get the maximum path tracing depth.
        """
        return self.path_tracer.max_depth

    @max_depth.setter
    def max_depth(self, value: int):
        """Set the maximum path tracing depth."""
        self.path_tracer.max_depth = value

    @property
    def enable_nee(self) -> bool:
        """
        Check if next event estimation is enabled.
        """
        return self.path_tracer.enable_nee

    @enable_nee.setter
    def enable_nee(self, value: bool):
        """Enable or disable next event estimation."""
        self.path_tracer.enable_nee = value

    @property
    def enable_mis(self) -> bool:
        """
        Check if multiple importance sampling is enabled.
        """
        return self.path_tracer.enable_mis

    @enable_mis.setter
    def enable_mis(self, value: bool):  #
        """Enable or disable multiple importance sampling."""
        self.path_tracer.enable_mis = value

    @property
    def enable_emissive_triangles(self) -> bool:
        """
        Check if emissive triangles are enabled.
        """
        return self.path_tracer.enable_emissive_triangles

    @enable_emissive_triangles.setter
    def enable_emissive_triangles(self, value: bool):
        """Enable or disable emissive triangles."""
        self.path_tracer.enable_emissive_triangles = value

    @property
    def enable_env_map(self) -> bool:
        """
        Check if environment mapping is enabled.
        """
        return self.path_tracer.enable_env_map

    @enable_env_map.setter
    def enable_env_map(self, value: bool):
        """Enable or disable environment mapping."""
        self.path_tracer.enable_env_map = value

    @property
    def env_map_as_background(self) -> bool:
        """
        Check if the environment map is used as the background.
        """
        return self.path_tracer.env_map_as_background

    @env_map_as_background.setter
    def env_map_as_background(self, value: bool):
        """Set whether to use the environment map as the background."""
        self.path_tracer.env_map_as_background = value

    @property
    def background_color(self) -> spy.float3:
        """
        Get the background color (when env map not used as background).
        """
        return self.path_tracer.background_color

    @background_color.setter
    def background_color(self, value: spy.float3):
        """Set the background color (when env map not used as background)."""
        self.path_tracer.background_color = value

    @property
    def use_raytracing_pipeline(self) -> bool:
        """
        Enable/disable ray tracing pipeline (default: inline on d3d/vk, rtp on cuda).
        """
        return self.path_tracer.use_raytracing_pipeline

    @use_raytracing_pipeline.setter
    def use_raytracing_pipeline(self, value: bool):
        """Enable/disable ray tracing pipeline (default: inline on d3d/vk, rtp on cuda)."""
        self.path_tracer.use_raytracing_pipeline = value

    def render(self, scene: Scene, camera: Camera, spp: int = 1) -> spy.Tensor:
        """
        Render an image of the given scene from the given camera.
        """

        scene.build()

        cmd = self.device.create_command_encoder()

        # Ensure the accumulator is the right size + reset.
        if self.accumulator and (
            self.accumulator.width != camera.width or self.accumulator.height != camera.height
        ):
            self.accumulator = None
        if not self.accumulator:
            self.accumulator = Accumulator(self.device, camera.width, camera.height)

        self.accumulator.reset(cmd)

        output_tensor = camera.create_output_tensor()

        sample = 0
        dispatches_per_submit = 16
        submit_counter = 0
        while spp > 16:
            self.path_tracer.render(
                scene=scene,
                camera=camera,
                color=output_tensor,
                spp=16,
                iteration=int(math.ceil(sample / 16)),
                cmd=cmd,
            )
            self.accumulator.update(output_tensor, cmd)
            sample += 16
            spp -= 16

            submit_counter += 1
            if submit_counter >= dispatches_per_submit:
                self.device.submit_command_buffer(cmd.finish())
                cmd = self.device.create_command_encoder()
                submit_counter = 0

        if spp > 0:
            self.path_tracer.render(
                scene=scene,
                camera=camera,
                color=output_tensor,
                spp=spp,
                iteration=int(math.ceil(sample / spp)),
                cmd=cmd,
            )
            self.accumulator.update(output_tensor, cmd)
            sample += spp

            submit_counter += 1
            if submit_counter >= dispatches_per_submit:
                self.device.submit_command_buffer(cmd.finish())
                cmd = self.device.create_command_encoder()
                submit_counter = 0

        self.accumulator.output(output_tensor, cmd)

        self.device.submit_command_buffer(cmd.finish())

        if self.tone_map:
            self.path_tracer.apply_tonemap(output_tensor, output_tensor)

        return output_tensor

    def render_guides(self, scene: Scene, camera: Camera, spp: int = 1) -> dict[str, spy.Tensor]:
        """
        Render guide images of the given scene from the given camera and outputs in
        a dictionary with keys: "depth", "albedo", "position", "normal", "uv".
        """

        scene.build()

        cmd = self.device.create_command_encoder()

        depth = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")
        albedo = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        position = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        normal = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        tangent = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        shading_normal = spy.Tensor.zeros(
            self.device, (camera.height, camera.width), dtype="float3"
        )
        uv = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float2")
        specular = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        metallic = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")
        roughness = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")

        self.path_tracer.render_guides(
            scene=scene,
            camera=camera,
            spp=spp,
            depth=depth,
            albedo=albedo,
            position=position,
            normal=normal,
            tangent=tangent,
            shading_normal=shading_normal,
            uv=uv,
            specular=specular,
            metallic=metallic,
            roughness=roughness,
            cmd=cmd,
        )

        if cmd:
            self.device.submit_command_buffer(cmd.finish())

        return {
            "depth": depth,
            "albedo": albedo,
            "position": position,
            "normal": normal,
            "uv": uv,
            "tangent": tangent,
            "shading_normal": shading_normal,
            "specular": specular,
            "metallic": metallic,
            "roughness": roughness,
        }

    def load_module(self, path: PathLike[str]) -> None:
        """
        Add a slang module plugin to the path tracer.

        Args:
            path: Path to the slang module file
        """
        return self.path_tracer.load_module(path)

    def find_render_function(
        self,
        module: spy.Module,
        func_name: str,
    ) -> FunctionNode:
        """
        Find a rendering function in the path tracer module, specializing it appropriately for the current
        ray tracing mode (inline vs ray tracing pipeline).
        """
        return self.path_tracer.find_render_function(module, func_name)

    def call(
        self,
        func: FunctionNode,
        scene: Scene,
        **kwargs,
    ):
        """
        Calls the specified function with the given scene and keyword arguments. The function must currently
        have been retrieved via `find_render_function`.
        """
        scene.build()
        return self.path_tracer.call(
            func,
            scene=scene,
            **kwargs,
        )


class Viewer:
    """
    Simple viewer for interactively displaying a scene, camera and renderer in a window.
    Provides basic mouse controls for camera rotation and zoom.
    """

    def __init__(self, scene: Scene, camera: Camera, renderer: Renderer, spp: int = 32):
        """
        Initialize the viewer with a scene, camera, and renderer.

        Args:
            scene: The scene to display
            camera: The camera to render from
            renderer: The renderer to use
            spp: Samples per pixel for rendering
        """
        super().__init__()
        self.scene = scene
        self.camera = camera
        self.renderer = renderer
        self.spp = spp

        # Setup slangpy window and device
        self.window = spy.Window(width=1024, height=1024, title="Falcor2 Viewer", resizable=False)
        self.device = renderer.device  # Use the device from the renderer

        # Setup surface
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        # Hook window events
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        self.on_keyboard_event_callbacks: list[Callable[[spy.KeyboardEvent], bool]] = []

        # Internal state
        self.output_texture: spy.Texture = None  # type: ignore (will be set when needed)
        self.reset_accumulation = True
        self.frame = 0
        self.last_mouse_pos: spy.float2 = spy.float2(0, 0)

        # Camera control state
        self.is_camera_moving = False
        self.mouse_drag_start_pos: spy.float2 = spy.float2(0, 0)
        self.camera_start_pos: spy.float3 = spy.float3(0, 0, 0)
        self.camera_start_rot: spy.quatf = spy.quatf(0, 0, 0, 1)

        # Configure camera size to match window
        self.camera.width = self.window.width
        self.camera.height = self.window.height  # Create output tensor for rendering
        self.output_tensor = self.camera.create_output_tensor()

        # Create guide output tensors
        self.depth = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")
        self.albedo = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        self.position = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        self.normal = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        self.tangent = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        self.shading_normal = spy.Tensor.zeros(
            self.device, (camera.height, camera.width), dtype="float3"
        )
        self.uv = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float2")
        self.specular = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float3")
        self.metallic = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")
        self.roughness = spy.Tensor.zeros(self.device, (camera.height, camera.width), dtype="float")
        self.guides = {
            "depth": self.depth,
            "albedo": self.albedo,
            "position": self.position,
            "normal": self.normal,
            "tangent": self.tangent,
            "shading_normal": self.shading_normal,
            "uv": self.uv,
            "specular": self.specular,
            "metallic": self.metallic,
            "roughness": self.roughness,
        }

        # Ensure renderer has an accumulator of the right size
        if self.renderer.accumulator and (
            self.renderer.accumulator.width != self.camera.width
            or self.renderer.accumulator.height != self.camera.height
        ):
            self.renderer.accumulator = None
        if not self.renderer.accumulator:
            self.renderer.accumulator = Accumulator(
                self.device, self.camera.width, self.camera.height
            )

        self.views = ["beauty"] + list(self.guides.keys())
        self.current_view = 0

        # Hot reload hook so accumulator can reset when shaders change.
        self.device.register_shader_hot_reload_callback(lambda device: self.on_hot_reload())

    def register_keyboard_event_callback(self, callback: Callable[[spy.KeyboardEvent], bool]):
        """
        Register a callback for keyboard events. The callback should return True if it handled the event, False otherwise.

        Args:
            callback: The callback function to register
        """
        self.on_keyboard_event_callbacks.append(callback)

    def on_hot_reload(self):
        """Handle shader hot-reload events."""
        self.reset_accumulation = True

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        """Handle keyboard events."""
        for callback in self.on_keyboard_event_callbacks:
            if callback(event):
                return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.space:
                self.current_view = (self.current_view + 1) % len(self.views)
                print(f"Switched to view: {self.views[self.current_view]}")
                self.reset_accumulation = True
            elif event.key == spy.KeyCode.o:
                save_image(self.output_tensor, "output.png")
                for guide_name, guide_tensor in self.guides.items():
                    save_image(guide_tensor, f"output_{guide_name}.png")

    def on_mouse_event(self, event: spy.MouseEvent):
        """Handle mouse events for camera controls."""
        if event.is_button_down() and event.button == spy.MouseButton.left:
            self.is_camera_moving = True
            self.mouse_drag_start_pos = self.last_mouse_pos
            self.camera_start_pos = self.camera.transform.pos
            self.camera_start_rot = self.camera.transform.rot
            self.reset_accumulation = True
        elif (
            event.is_button_up() and event.button == spy.MouseButton.left and self.is_camera_moving
        ):
            self.is_camera_moving = False
            self.reset_accumulation = True
        elif event.is_move():
            self.last_mouse_pos = event.pos
        if self.is_camera_moving:
            self._update_camera_rotation(self.last_mouse_pos)
        if event.is_scroll():
            self._update_camera_zoom(event.scroll.y)

    def _update_camera_rotation(self, current_mouse_pos: spy.float2):
        """Trivial orbit camera rotation based on mouse movement."""
        mouse_delta = spy.float2(
            current_mouse_pos.x - self.mouse_drag_start_pos.x,
            current_mouse_pos.y - self.mouse_drag_start_pos.y,
        )
        sensitivity = 0.005
        yaw_delta = -mouse_delta.x * sensitivity
        pitch_delta = -mouse_delta.y * sensitivity
        orig_camera_to_origin = -self.camera_start_pos
        orig_distance = spy.math.length(orig_camera_to_origin)
        orig_forward = spy.math.normalize(orig_camera_to_origin)
        orig_yaw = spy.math.atan2(orig_forward.x, orig_forward.z)
        orig_pitch = spy.math.asin(orig_forward.y)
        new_yaw = orig_yaw + yaw_delta
        new_pitch_value = orig_pitch + pitch_delta
        new_pitch = max(-1.5, min(1.5, new_pitch_value))
        new_forward = spy.float3(
            spy.math.cos(new_pitch) * spy.math.sin(new_yaw),
            spy.math.sin(new_pitch),
            spy.math.cos(new_pitch) * spy.math.cos(new_yaw),
        )
        self.camera.transform.pos = -orig_distance * new_forward
        self.camera.transform.rot = spy.math.quat_from_look_at(
            -spy.math.normalize(self.camera.transform.pos), spy.float3(0, 1, 0)
        )
        self.reset_accumulation = True

    def _update_camera_zoom(self, wheel_delta: float):
        """Update camera zoom based on mouse wheel."""
        zoom_sensitivity = 0.1
        zoom_factor = 1.0 + (wheel_delta * zoom_sensitivity)
        current_distance = spy.math.length(self.camera.transform.pos)
        new_distance_value = current_distance / zoom_factor
        new_distance = max(0.1, min(100.0, new_distance_value))
        direction = spy.math.normalize(self.camera.transform.pos)
        self.camera.transform.pos = direction * new_distance
        self.reset_accumulation = True

    def on_resize(self, width: int, height: int):
        """Handle window resize events."""
        self.device.wait()
        self.surface.configure(width=max(width, 1), height=max(height, 1))
        # TODO: Update camera dimensions and reset accumulation

    def render_output(self, output: spy.Tensor):
        """Render path tracer output to the window screen."""
        # Get next frame buffer image
        image = self.surface.acquire_next_image()
        if image is None:
            return

        # Create or recreate output texture if size changed
        if (
            self.output_texture is None
            or self.output_texture.width != image.width
            or self.output_texture.height != image.height
        ):
            self.output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=image.width,
                height=image.height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="viewer_output_texture",
            )

        # Resample the output texture to the swapchain image with tone mapping
        self.renderer.path_tracer.resample(
            output,
            self.output_texture,
            output_pos=spy.uint2(0),
            output_size=spy.uint2(image.width, image.height),
            tone_map=self.renderer.tone_map and self.current_view == 0,
        )

        # Blit to screen
        command_encoder = self.device.create_command_encoder()
        command_encoder.blit(image, self.output_texture)

        # Present
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        self.device.submit_command_buffer(command_encoder.finish())
        del image
        self.surface.present()

    def update(self) -> bool:
        """
        Update the viewer for one frame.

        Returns:
            True if the viewer should continue running, False if it should close
        """
        if self.window.should_close():
            return False

        # Process window events
        self.window.process_events()

        # Check if scene needs rebuilding
        if self.scene.is_dirty(DirtyFlag.any):
            self.reset_accumulation = True

        self.scene.build()

        # Create command encoder for rendering
        cmd = self.device.create_command_encoder()

        # Reset accumulation if needed
        if self.reset_accumulation:
            if self.renderer.accumulator:
                self.renderer.accumulator.reset(cmd)
            self.frame = 0
            self.reset_accumulation = False

        # Render frame

        # Beauty view
        self.renderer.path_tracer.render(
            scene=self.scene,
            camera=self.camera,
            color=self.output_tensor,
            spp=self.spp if not self.is_camera_moving else 8,
            iteration=self.frame,
            cmd=cmd,
        )

        # Guide views
        self.renderer.path_tracer.render_guides(
            scene=self.scene,
            camera=self.camera,
            spp=self.spp if not self.is_camera_moving else 8,
            cmd=cmd,
            **self.guides,
        )

        # Update accumulator
        if self.renderer.accumulator:
            self.renderer.accumulator.update_and_output(self.output_tensor, self.output_tensor, cmd)

        # Submit rendering commands
        if cmd:
            self.device.submit_command_buffer(cmd.finish())

        # Render to screen
        if self.current_view == 0:
            self.render_output(self.output_tensor)
        else:
            guide_name = self.views[self.current_view]
            self.render_output(self.guides[guide_name])

        self.frame += 1

        return True

    def run(self) -> None:
        """
        Run the viewer loop until the window is closed.
        """
        while self.update():
            pass

        # Cleanup
        self.device.wait()


def create_viewer(scene: Scene, camera: Camera, renderer: Renderer, spp: int = 32) -> Viewer:
    """
    Create a new viewer instance.

    Args:
        scene: The scene to display
        camera: The camera to render from
        renderer: The renderer to use
        spp: Samples per pixel for rendering

    Returns:
        A new Viewer instance
    """
    return Viewer(scene, camera, renderer, spp)


def create_renderer(device: spy.Device) -> Renderer:
    """
    Create a new renderer instance.
    """
    return Renderer(device)


def create_scene(device: spy.Device) -> Scene:
    """
    Create an empty scene instance.
    """
    return Scene(device)


def load_scene(
    device: spy.Device,
    path: Union[str, PathLike[str]],
    rescale_to: Optional[float] = None,
    append_to_scene: Optional[Scene] = None,
    scene_from_world: Optional[spy.float4x4] = None,
) -> Scene:
    """
    Load a scene from a USD or GLTF file. File type is determined by extension.
    """
    path_obj = Path(path)

    if path_obj.suffix.lower() in [".gltf", ".glb"]:
        scene = load_gltf_scene(
            device,
            path_obj,
            rescale_to=rescale_to,
            append_to_scene=append_to_scene,
            scene_from_world=scene_from_world,
        )
    elif path_obj.suffix.lower() in [".usd", ".usda", ".usdc"]:
        scene = load_usd_scene(
            device,
            path_obj,
            rescale_to=rescale_to,
            append_to_scene=append_to_scene,
            scene_from_world=scene_from_world,
        )
    else:
        raise ValueError(f"Unsupported scene file format: {path_obj.suffix}")

    return scene


def save_image(tensor: spy.Tensor, path: Union[str, PathLike[str]]) -> None:
    """
    Save a tensor as an image file. The file format is inferred from the file extension.
    Supported formats are those supported by SlangPy Bitmap, including PNG, JPEG, BMP,
    EXR, HDR, etc. Tensor is expected to either:
    - 2D with shape (H,W), and type float/float2/float3/float4
    - 3D with shape (H,W,C) where C is 1, 2, 3 or 4, and type float
    """

    path = Path(path)
    data = tensor.to_numpy()

    if len(data.shape) == 2:
        # convert to 3 channels, all with the same value
        data = np.stack([data] * 3, axis=-1)
        pixel_format = spy.Bitmap.PixelFormat.rgb
    elif len(data.shape) == 3:
        if data.shape[2] == 1:
            # convert to 3 channels, all with the same value
            data = np.stack([data[:, :, 0]] * 3, axis=-1)
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 2:
            # add extra channel of 0s to make it 3 channels
            data = np.concatenate(
                [data, np.zeros((data.shape[0], data.shape[1], 1), dtype=data.dtype)], axis=-1
            )
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 3:
            # already 3 channel rgb
            pixel_format = spy.Bitmap.PixelFormat.rgb
        elif data.shape[2] == 4:
            # already 4 channel rgba
            pixel_format = spy.Bitmap.PixelFormat.rgba
        else:
            raise ValueError(f"Unsupported number of channels: {data.shape[2]}")
    else:
        raise ValueError(f"Unsupported tensor shape: {data.shape}")

    # for none-hdr formats, convert to 8-bit
    if path.suffix != ".exr" and path.suffix != ".hdr":
        if data.dtype == np.float16 or data.dtype == np.float32 or data.dtype == np.float64:
            data = np.clip(data, 0.0, 1.0)
            data = np.pow(data, 1.0 / 2.2)
            data = (data * 255).astype(np.uint8)

    # create and write bitmap
    bitmap = spy.Bitmap(data=data, pixel_format=pixel_format)
    bitmap.write(path)
