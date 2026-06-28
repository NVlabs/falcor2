# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import falcor2.minitracer.tools as mt
import slangpy as spy
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def main(input_path: str):

    device = mt.create_device()

    path = Path(input_path)

    if not path.exists():
        print(f"Input file does not exist: {path}")
        return

    # Load scene + create camera, env map and renderer
    print(f"Loading scene from: {path}")
    scene = mt.load_scene(device, path, rescale_to=0.85)

    # Add env map (and optionally adjust brightness)
    envmap = scene.create_env_map(DATA_DIR / "assets/envmaps/aerodynamics_workshop_512.hdr")
    envmap.scaling_factor = spy.float3(1)

    # Create camera and renderer
    camera = scene.create_camera(1024, 1024, 45)
    renderer = mt.create_renderer(device)

    # Various config options for the renderer
    # Next-event estimation (default=true)
    renderer.enable_nee = True
    # Multiple importance sampling (default=true)
    renderer.enable_mis = True
    # Whether to include emissive triangles lighting (default=true)
    renderer.enable_emissive_triangles = True
    # Whether to include env map lighting (default=true)
    renderer.enable_env_map = True
    # Whether to use the env map as a background (default=true)
    renderer.env_map_as_background = False
    # Background color if not using env map (default=black)
    renderer.background_color = spy.float3(0.0, 0.0, 0.0)
    # Max path depth (default=3)
    renderer.max_depth = 3
    # Whether to apply simple ACES tone mapping (default=True)
    renderer.tone_map = True

    # Initial camera position
    radius = 1.8
    camera.transform.pos = spy.float3(radius, 0.2, 0)
    camera.transform.rot = spy.math.quat_from_look_at(
        -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
    )

    # Enable this to add a ground plane
    # box = scene.create_box(spy.float3(10, 0.01, 10))
    # box.transform.pos = spy.float3(0, -0.425, 0)
    # grey_mat = scene.create_material("default")
    # grey_mat.albedo = scene.create_texture(16,16,spy.float3(0.25, 0.25, 0.25))
    # box.material = grey_mat

    # Ensure output dir exists
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)

    # Option 1: Create / run interactive viewer.
    if True:
        viewer = mt.create_viewer(scene, camera, renderer)
        viewer.run()

    # Option 2: Viewer with custom update loop.:
    if False:
        while viewer.update():
            # Could do custom processing here
            pass

    # Option 3: Render some frames to files
    if False:
        for i in range(8):

            angle = (i / 8.0) * 3.14159 * 2.0 + 0.01
            radius = 1.8
            camera.transform.pos = spy.float3(
                radius * spy.math.sin(angle), 0.2, radius * spy.math.cos(angle)
            )
            camera.transform.rot = spy.math.quat_from_look_at(
                -spy.math.normalize(camera.transform.pos), spy.float3(0, 1, 0)
            )
            image = renderer.render(scene, camera, spp=1024)

            # Save to PNG (SRGB). For HDR, use .exr.
            mt.save_image(image, f"output/{i:03d}.png")

            # Enable this to render out guide images
            guides = renderer.render_guides(scene, camera, spp=64)
            for name, guide in guides.items():
                mt.save_image(guide, f"output/{i:03d}-{name}.png")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python example.py <path/to/scene>")
    else:
        main(sys.argv[1])
