# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from slangpy import Device, float3
from falcor2.minitracer.scene import Scene

from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def emission_test_scene(device: Device) -> Scene:
    scene = Scene(device)

    bricks_diffuse = scene.load_texture(
        DATA_DIR / "assets/textures/Bricks075A/Bricks075A_1K.diffuse.jpg"
    )
    bricks_normal = scene.load_normal_map(
        DATA_DIR / "assets/textures/Bricks075A/Bricks075A_1K.normal.jpg"
    )
    bricks_roughness = scene.load_texture(
        DATA_DIR / "assets/textures/Bricks075A/Bricks075A_1K.roughness.jpg"
    )
    bricks_metallic = scene.create_texture(16, 16, float3(0.1))
    bricks_specular = scene.create_texture(16, 16, float3(0.1))

    red_texture = scene.create_texture(16, 16, float3(1.0, 0.0, 0.0))
    green_texture = scene.create_texture(16, 16, float3(0.0, 1.0, 0.0))
    blue_texture = scene.create_texture(16, 16, float3(0.0, 0.0, 1.0))

    grey_material = scene.create_material("grey")
    grey_material.albedo = bricks_diffuse
    grey_material.emission = scene.black_texture
    grey_material.specular = bricks_specular
    grey_material.metallic = bricks_metallic
    grey_material.roughness = bricks_roughness
    grey_material.normal = bricks_normal

    mirror_material = scene.create_material("mirror")
    mirror_material.albedo = scene.white_texture
    mirror_material.emission = scene.black_texture
    mirror_material.specular = scene.white_texture
    mirror_material.metallic = scene.white_texture
    mirror_material.roughness = scene.create_texture(16, 16, float3(0.2))
    mirror_material.normal = bricks_normal
    mirror_material.normal_map_strength = 0.05

    red_emissive_material = scene.create_material("red_emissive")
    red_emissive_material.albedo = scene.black_texture
    red_emissive_material.emission = red_texture
    red_emissive_material.emission_strength = 10

    floor = scene.create_box(float3(100, 0.5, 100))
    floor.transform.pos = float3(0, -2, 0.5)
    floor.material = mirror_material

    box1 = scene.create_box(float3(1, 2, 1))
    box1.transform.pos = float3(-4, -1, 0)
    box1.material = grey_material

    box2 = scene.create_box(float3(1, 2, 1))
    box2.transform.pos = float3(0, -1, 0)
    box2.material = red_emissive_material

    box3 = scene.create_box(float3(1, 2, 1))
    box3.transform.pos = float3(4, -1, 0)
    box3.material = grey_material

    camera = scene.create_camera(128, 128, 70)
    camera.transform.pos = float3(0, 0, 4)

    env_map = scene.create_env_map(DATA_DIR / "assets/envmaps/brown_photostudio_02_2k.hdr")
    env_map.scaling_factor = float3(0.25)

    return scene
