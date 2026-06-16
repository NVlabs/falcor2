# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import falcor2
import slangpy as spy
from falcor2.editor import create_device, create_torch_device, load_scene, save_image
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline

OUTPUT_CONTAINER = "torch"  # Options: "tensor", "texture", "torch"

if OUTPUT_CONTAINER == "torch":
    device = create_torch_device()
else:
    device = create_device()

scene = load_scene(
    device,
    "data/assets/kronos/EmissiveStrengthTest/glTF/EmissiveStrengthTest.gltf",
)

# Add env map
env_entity = scene.create_entity()
env_map = env_entity.create_component(falcor2.EnvMapLight)
env_map["env_map_path"] = "data/assets/envmaps/aerodynamics_workshop_512.hdr"

# Add camera
position = spy.float3(0, 0.2, 20)
camera_entity = scene.create_entity()
camera = camera_entity.create_component(falcor2.Camera)
camera.width = 1024
camera.height = 1024
camera.fov_y = 45
camera_transform = falcor2.Transform()
camera_transform.translation = position
camera_transform.rotation = spy.math.quat_from_look_at(
    -spy.math.normalize(position),
    spy.float3(0, 1, 0),
)
camera_entity.transform = camera_transform
camera.recompute()

# Create a pipeline, and choose an output type
pipeline = PathTracerPipeline.create(device)
if OUTPUT_CONTAINER == "texture":
    pipeline.output_spec = ContainerSpec.texture2d(format=spy.Format.rgba32_float)
elif OUTPUT_CONTAINER == "torch":
    pipeline.output_spec = ContainerSpec.torch(format=spy.Format.rgba32_float)
else:
    pipeline.output_spec = ContainerSpec.tensor(format=spy.Format.rgba32_float)

# Configure some path tracer options
pipeline.path_tracer.max_depth = 3
pipeline.path_tracer.enable_nee = True
pipeline.path_tracer.enable_mis = True

# Accumulate 32 frames
for _ in range(32):
    image = pipeline(scene, camera)

# Convert torch tensor to a slangpy type to make saving easy
if OUTPUT_CONTAINER == "torch":
    image = spy.Tensor.from_torch(device, image.permute(1, 2, 0).contiguous(), dtype=spy.float4)

# Save output
save_image(image, f"output/pathtrace_{OUTPUT_CONTAINER}.png")
print(f"Saved output/pathtrace_{OUTPUT_CONTAINER}.png from {type(image).__name__}")
