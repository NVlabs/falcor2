# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# pyright: reportArgumentType=false

from __future__ import annotations

import argparse

from falcor2.pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

import falcor2 as f2
import slangpy as spy
from falcor2.editor import (
    Editor,
    EditorConfig,
    EditorArgumentParser,
    create_device,
)
from falcor2.rendernodes import PathTracerPipeline

DESCRIPTION = "USD Stage PathTracer Example"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = EditorArgumentParser(
        description=DESCRIPTION,
        editor_defaults=EditorConfig(
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            title=DESCRIPTION,
            vsync=False,
        ),
    )
    return parser.parse_args(argv)


def create_material(
    stage: Usd.Stage,
    path: str,
    color: tuple[float, float, float],
) -> UsdShade.Material:
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def create_mesh(
    stage: Usd.Stage,
    path: str,
    points: list[tuple[float, float, float]],
    face_vertex_counts: list[int],
    face_vertex_indices: list[int],
    material: UsdShade.Material,
) -> None:
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
    mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)


def create_box(
    stage: Usd.Stage,
    path: str,
    minimum: tuple[float, float, float],
    maximum: tuple[float, float, float],
    material: UsdShade.Material,
) -> None:
    x0, y0, z0 = minimum
    x1, y1, z1 = maximum
    faces = [
        (0, 3, 2, 1),
        (4, 5, 6, 7),
        (0, 4, 7, 3),
        (1, 2, 6, 5),
        (0, 1, 5, 4),
        (3, 7, 6, 2),
    ]
    create_mesh(
        stage,
        path,
        [
            (x0, y0, z0),
            (x1, y0, z0),
            (x1, y1, z0),
            (x0, y1, z0),
            (x0, y0, z1),
            (x1, y0, z1),
            (x1, y1, z1),
            (x0, y1, z1),
        ],
        [4] * len(faces),
        [index for face in faces for index in face],
        material,
    )


def create_stage() -> Usd.Stage:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Looks")

    white = create_material(stage, "/World/Looks/White", (0.75, 0.75, 0.75))
    red = create_material(stage, "/World/Looks/Red", (0.75, 0.08, 0.05))
    blue = create_material(stage, "/World/Looks/Blue", (0.05, 0.15, 0.8))

    create_mesh(
        stage,
        "/World/Floor",
        [(-3, 0, -3), (-3, 0, 2), (3, 0, 2), (3, 0, -3)],
        [4],
        [0, 1, 2, 3],
        white,
    )
    create_mesh(
        stage,
        "/World/BackWall",
        [(-3, 0, -3), (3, 0, -3), (3, 3, -3), (-3, 3, -3)],
        [4],
        [0, 1, 2, 3],
        white,
    )
    create_box(stage, "/World/RedBox", (-1.5, 0, -1.2), (-0.2, 1.1, 0.1), red)
    create_box(stage, "/World/BlueBox", (0.35, 0, -2.0), (1.55, 1.8, -0.7), blue)

    light = UsdLux.SphereLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(1800.0)
    light.CreateRadiusAttr(0.2)
    light.AddTranslateOp().Set(Gf.Vec3d(0.0, 2.6, 0.5))

    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    camera.CreateFocalLengthAttr(35.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(20.25)
    camera.AddTranslateOp().Set(Gf.Vec3d(0.0, 1.45, 5.5))

    return stage


def main() -> None:
    args = parse_args()
    device_type = getattr(spy.DeviceType, args.device_type)
    device = create_device(device_type=device_type, enable_cuda_interop=True)
    scene = f2.Scene.from_stage(device, create_stage())

    pipeline = PathTracerPipeline.create(device)
    pipeline.path_tracer.max_depth = 3
    pipeline.path_tracer.enable_nee = True
    pipeline.path_tracer.enable_mis = True
    pipeline.path_tracer.use_background_color = False
    pipeline.tone_map = True

    editor = Editor.create(
        device,
        config=args.editor_config,
        scene=scene,
    )
    editor.run(lambda scene: pipeline(scene, delta_time=editor.dt))


if __name__ == "__main__":
    main()
