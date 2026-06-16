# SPDX-License-Identifier: Apache-2.0

from typing_extensions import Any, Optional, Union

from pathlib import Path
from datetime import datetime
import numpy as np
import slangpy as spy
import sys

sys.path.append(str(Path(__file__).parent))

from falcor2.minitracer.camera import Camera
from falcor2.minitracer.accumulator import Accumulator
from falcor2.minitracer.pathtracer import PathTracer
from falcor2.minitracer.scene import Scene, DirtyFlag
from testscenes.emissiontest import emission_test_scene
from testscenes.difftest import diff_test_scene
from falcor2.minitracer.properties import PropertiesWindow
from falcor2.minitracer.viewport import Viewport
from falcor2.utils.optimizer.gradient_descent import GradientDescentOptimizer
from falcor2.utils.optimizer.adam import AdamOptimizer

import falcor2.minitracer.slangpyextensions  # type: ignore

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# spy.set_dump_generated_shaders(True)
# spy.set_dump_slang_intermediates(True)


class App:
    def __init__(self):
        super().__init__()

        # Setup slangpy window and device
        self.window = spy.Window(width=1920, height=1080, title="PathTracer", resizable=False)
        self.device = spy.Device(
            # type=spy.DeviceType.cuda,
            enable_debug_layers=False,
            # enable_compilation_reports=True,
            compiler_options={
                "include_paths": [
                    Path(__file__).parent.parent.parent / "slang",
                    spy.SHADER_PATH,
                ],
                # "debug_info": spy.SlangDebugInfoLevel.standard,
            },
            # enable_cuda_interop=True,
        )

        # Setup surface
        self.surface = self.device.create_surface(self.window)
        self.surface.configure(width=self.window.width, height=self.window.height)

        # Setup UI
        self.ui = spy.ui.Context(self.device)
        self.properties_window = PropertiesWindow(self.ui.screen)

        # Hook window events
        self.window.on_keyboard_event = self.on_keyboard_event
        self.window.on_mouse_event = self.on_mouse_event
        self.window.on_resize = self.on_resize

        # Setup internal state
        self.output_texture: spy.Texture = None  # type: ignore (will be set immediately)
        self.training = False
        self.reset_accumulation = False
        self.debug_gradient_texture = 0

        # Create path tracer
        self.path_tracer = PathTracer(self.device)

        # Setup properties window
        self.properties_window.add_properties(self.path_tracer, "Path Tracer")

        # Reset accumulation when path tracer properties change
        def path_tracer_property_changed(property: Any, value: Any):
            self.reset_accumulation = True

        self.path_tracer.add_properties_listener(path_tracer_property_changed)

        # Create a scene
        self.scene = emission_test_scene(self.device)
        # self.scene = emission_test_scene(self.device)
        # self.scene = load_usd_scene(self.device, DATA_DIR / "assets/cornell-box/usdpreviewsurface/cornell-box.usda")
        self.camera = self.scene.cameras[0]

        # Configure camera
        self.camera.width = self.window.width
        self.camera.height = self.window.height

        # Create framebuffer & accumulator
        self.output_tensor = self.camera.create_output_tensor()
        self.accumulator = Accumulator(self.device, self.camera.width, self.camera.height)

        # Create viewport
        self.viewport = Viewport(self.device, self.window, self.ui.screen, self.scene)

        # Refresh GPU data
        self.scene.build()

        self.timer = spy.Timer()
        self.frame = 0

        # Register hot reload callbacks to reset accumulation
        self.device.register_shader_hot_reload_callback(lambda x: self.set_reset_accumulation())

    def on_keyboard_event(self, event: spy.KeyboardEvent):
        if self.ui.handle_keyboard_event(event):
            return
        if self.viewport.handle_keyboard_event(event):
            return

        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.escape:
                self.window.close()
            elif event.key == spy.KeyCode.f1:
                if self.output_texture:
                    spy.tev.show_async(self.output_texture, name=datetime.now().isoformat())
            elif event.key == spy.KeyCode.f2:
                if self.output_texture:
                    bitmap = self.output_texture.to_bitmap()
                    bitmap.convert(
                        spy.Bitmap.PixelFormat.rgb,
                        spy.Bitmap.ComponentType.uint8,
                        srgb_gamma=True,
                    ).write_async("screenshot.png")
            elif event.key == spy.KeyCode.l:
                self.path_tracer.enable_nee = not self.path_tracer.enable_nee
            elif event.key == spy.KeyCode.m:
                self.path_tracer.enable_mis = not self.path_tracer.enable_mis
            elif event.key == spy.KeyCode.t:
                if not self.training:
                    self.start_training()
                else:
                    self.stop_training()
            elif event.key == spy.KeyCode.right_bracket:
                self.debug_gradient_texture = self.debug_gradient_texture + 1
                self.reset_accumulation = True
                print(f"Debug gradient texture: {self.debug_gradient_texture}")
            elif event.key == spy.KeyCode.left_bracket:
                self.debug_gradient_texture = self.debug_gradient_texture - 1
                self.reset_accumulation = True
                print(f"Debug gradient texture: {self.debug_gradient_texture}")

    def on_mouse_event(self, event: spy.MouseEvent):
        if self.ui.handle_mouse_event(event):
            return
        if self.viewport.handle_mouse_event(event):
            return

    def on_resize(self, width: int, height: int):
        self.device.wait()
        self.surface.configure(width=max(width, 1), height=max(height, 1))

    def set_reset_accumulation(self):
        self.reset_accumulation = True

    def show_in_tev(self, tensor: spy.Tensor, name: str, apply_tonemap: bool = True):
        if apply_tonemap:
            tonemapped = spy.Tensor.zeros_like(tensor)
            self.path_tracer.apply_tonemap(tensor, tonemapped)
            tensor = tonemapped
        spy.tev.show(spy.Bitmap(tensor.to_numpy()), name=name)

    # Renders path tracer output to screen
    def render_output(self, output: spy.Tensor):

        # Attempt to get next frame buffer image, recalculating
        # output texture if changes size.
        image = self.surface.acquire_next_image()
        if image is None:
            return
        if (
            self.output_texture == None
            or self.output_texture.width != image.width
            or self.output_texture.height != image.height
        ):
            self.output_texture = self.device.create_texture(
                format=spy.Format.rgba32_float,
                width=image.width,
                height=image.height,
                mip_count=1,
                usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access,
                label="output_texture",
            )

        # Resample the output texture to the swapchain image
        self.path_tracer.resample(
            output,
            self.output_texture,
            output_pos=spy.uint2(0),
            output_size=spy.uint2(image.width, image.height),
            tone_map=True,
        )

        # TODO: CUDA doesn't currently support UI + viewport rendering
        if self.device.info.type == spy.DeviceType.cuda:
            # Blit output texture
            command_encoder = self.device.create_command_encoder()
            command_encoder.blit(image, self.output_texture)
        else:
            # Render viewport UI
            self.viewport.render(
                width=image.width, height=image.height, render_texture=self.output_texture
            )

            # Blit + render UI
            command_encoder = self.device.create_command_encoder()
            command_encoder.blit(image, self.viewport.output_texture)
            self.ui.begin_frame(image.width, image.height)
            self.ui.end_frame(image, command_encoder)

        # Present
        command_encoder.set_texture_state(image, spy.ResourceState.present)
        self.device.submit_command_buffer(command_encoder.finish())
        del image
        self.surface.present()
        self.frame += 1

    # Normal path tracer update
    def update_normal(self):
        if self.scene.is_dirty(DirtyFlag.any):
            self.reset_accumulation = True

        self.scene.build()

        cmd = self.device.create_command_encoder()
        # cmd = None

        if self.reset_accumulation:
            self.accumulator.reset(cmd)
            self.frame = 0
            self.reset_accumulation = False

        self.path_tracer.render(
            scene=self.scene,
            camera=self.camera,
            color=self.output_tensor,
            spp=8,
            iteration=self.frame,
            cmd=cmd,
        )

        self.accumulator.update_and_output(self.output_tensor, self.output_tensor, cmd)

        if cmd:
            self.device.submit_command_buffer(cmd.finish())

        self.render_output(self.output_tensor)

    def start_training(self):
        print("Starting training...")

        self.iteration = 0

        # self.loss_func = spy.Module.load_from_file(self.device, "falcor2/utils.slang").find_function("l2_loss<float, 4>")
        self.loss_func = spy.Module.load_from_file(self.device, "falcor2/utils.slang").test_loss
        self.clamp_value_func = self.path_tracer._module.clamp_value
        self.random_value_func = self.path_tracer._module.random_value

        print("Render target image...")
        self.target_image = self.camera.create_output_tensor()
        self.path_tracer.render_accumulated(self.scene, self.camera, self.target_image, 8, 128)
        self.show_in_tev(self.target_image, "Target")

        # Setup loss tensor
        width = self.camera.width
        height = self.camera.height
        self.loss = spy.Tensor.zeros(self.device, (height, width), dtype="float").with_grads()
        self.loss.grad.copy_from_numpy(
            np.ones_like(self.loss.to_numpy()) * (1.0 / (width * height))
        )

        # Setup predicted image tensor
        self.pred_image = self.camera.create_output_tensor().with_grads()

        # Setup trainable parameters
        params = self.scene_params = self.scene.get_parameters()
        params["grey/albedo"].require_grads()
        params["grey/specular"].require_grads()
        params["grey/roughness"].require_grads()
        params["grey/normal"].require_grads()
        params.write_to_scene()

        # Initialize trainable parameters with random values
        for i, param in enumerate(params.values()):
            if param.has_grads:
                self.random_value_func(
                    spy.RandFloatArg(param.value_range[0], param.value_range[1], seed=i * 123),
                    param.tensor,
                )

        # Initialize optimizer
        self.optimizer = AdamOptimizer(self.device, learning_rate=0.01)
        # self.optimizer = GradientDescentOptimizer(self.device, learning_rate=0.00001)
        self.optimizer.initialize(params.tensors_with_grads())

        self.optimizer.zero_grads()

        self.training = True

    def stop_training(self):
        print("Stopping training...")
        self.training = False

    def update_training(self):

        CHECKPOINT_ITERATIONS = 100

        print(f"Training iteration {self.iteration}...")

        if self.scene.is_dirty(DirtyFlag.any):
            self.reset_accumulation = True
        self.scene.build()

        if self.iteration % CHECKPOINT_ITERATIONS == 0:
            print(f"Render checkpoint image (iteration={self.iteration})...")
            self.path_tracer.render_accumulated(self.scene, self.camera, self.pred_image, 8, 128)
            self.show_in_tev(self.pred_image, name=f"Checkpoint (iteration={self.iteration})")

        cmd = self.device.create_command_encoder()
        # cmd = None

        # Forward pass
        self.path_tracer.render(
            scene=self.scene,
            camera=self.camera,
            color=self.pred_image,
            spp=8,
            iteration=self.iteration,
            cmd=cmd,
        )

        # Calculate loss
        self.loss_func(self.pred_image, self.target_image, _result=self.loss, _append_to=cmd)

        # Calculate gradients
        self.pred_image.grad.clear(cmd)
        self.loss_func.bwds(self.pred_image, self.target_image, self.loss, _append_to=cmd)

        # Perform training step
        # Use a different random sequence for backward pass (iteration + 1000)
        self.path_tracer.render_bwd(
            scene=self.scene,
            camera=self.camera,
            color=self.pred_image,
            spp=8,
            iteration=self.iteration + 100,
            cmd=cmd,
        )
        self.reset_accumulation = False

        self.optimizer.step(cmd=cmd)

        # Clamp parameters
        for param in self.scene_params.values():
            if param.has_grads:
                self.clamp_value_func(
                    param.tensor, param.value_range[0], param.value_range[1], _append_to=cmd
                )

        self.iteration += 1

        if cmd:
            self.device.submit_command_buffer(cmd.finish())

        self.render_output(self.pred_image)

    # Main loop handles window events, updates path tracer and
    # then renders to screen.
    def main_loop(self):
        while not self.window.should_close():
            dt = self.timer.elapsed_s()
            self.timer.reset()

            self.window.process_events()

            if self.viewport.update(dt):
                self.reset_accumulation = True

            if self.training:
                self.update_training()
            else:
                self.update_normal()

        self.device.wait()


if __name__ == "__main__":
    app = App()
    app.main_loop()
