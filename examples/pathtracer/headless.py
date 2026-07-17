# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import slangpy as spy
from falcor2.editor import create_device, create_torch_device, load_scene, save_image
from falcor2.rendergraph import ContainerSpec
from falcor2.rendernodes import PathTracerPipeline

OUTPUT_CONTAINER = "torch"  # Options: "tensor", "texture", "torch"
SCENE_PATH = "data/scenes/emissive-strength.py"

if OUTPUT_CONTAINER == "torch":
    device = create_torch_device()
else:
    device = create_device()

scene = load_scene(device, SCENE_PATH)
camera = scene.active_camera
if camera is None:
    raise RuntimeError(f"Scene file does not contain an active camera: {SCENE_PATH}")

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
