# SPDX-License-Identifier: Apache-2.0
"""Generate minimal animated USD test files for unit tests."""

import os
import math


def generate_linear_translation(output_dir: str) -> None:
    """A cube translating linearly along X from 0 to 5 over 2 seconds (3 keyframes at 0, 1, 2s)."""
    from pxr import Usd, UsdGeom, Gf, Sdf

    path = os.path.join(output_dir, "usd_anim_linear_translation.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(48)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/Cube/Mesh")

    # Simple cube vertices
    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Animation: translate along X from 0 to 5 over 2 seconds
    # 3 keyframes at time codes 0, 24, 48 (= 0s, 1s, 2s at 24 fps)
    xform_op = xform.AddTranslateOp()
    xform_op.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0))
    xform_op.Set(Gf.Vec3d(2.5, 0.0, 0.0), Usd.TimeCode(24))
    xform_op.Set(Gf.Vec3d(5.0, 0.0, 0.0), Usd.TimeCode(48))

    stage.Save()
    print(f"  Generated: {path}")


def generate_rotation_animation(output_dir: str) -> None:
    """A cube rotating 90 degrees around Y over 1 second (2 keyframes)."""
    from pxr import Usd, UsdGeom, Gf, Sdf

    path = os.path.join(output_dir, "usd_anim_rotation.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/Cube/Mesh")

    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Use a full 4x4 matrix xform op so we get rotation through matrix decomposition.
    # Identity at t=0, 90 deg Y rotation at t=24 (1s).
    xform_op = xform.AddTransformOp()
    xform_op.Set(Gf.Matrix4d(1.0), Usd.TimeCode(0))

    angle_rad = math.radians(90)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    # Rotation around Y axis (column-major in USD):
    # fmt: off
    rot_matrix = Gf.Matrix4d(
        c, 0, -s, 0,
        0, 1, 0, 0,
        s, 0, c, 0,
        0, 0, 0, 1,
    )
    # fmt: on
    xform_op.Set(rot_matrix, Usd.TimeCode(24))

    stage.Save()
    print(f"  Generated: {path}")


def generate_multi_channel(output_dir: str) -> None:
    """A cube with translation, rotation, and scale animation (3 channels)."""
    from pxr import Usd, UsdGeom, Gf, Sdf

    path = os.path.join(output_dir, "usd_anim_multi_channel.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(72)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/Cube/Mesh")

    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Use separate translate, rotate, scale ops with time samples.
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0, 0, 0), Usd.TimeCode(0))
    translate_op.Set(Gf.Vec3d(1, 2, 3), Usd.TimeCode(48))  # at 2s

    rotate_op = xform.AddRotateYOp()
    rotate_op.Set(0.0, Usd.TimeCode(0))
    rotate_op.Set(90.0, Usd.TimeCode(48))  # at 2s

    scale_op = xform.AddScaleOp()
    scale_op.Set(Gf.Vec3f(1, 1, 1), Usd.TimeCode(0))
    scale_op.Set(Gf.Vec3f(2, 2, 2), Usd.TimeCode(72))  # at 3s

    stage.Save()
    print(f"  Generated: {path}")


def generate_multi_node(output_dir: str) -> None:
    """Two nodes: parent translates, child rotates."""
    from pxr import Usd, UsdGeom, Gf, Sdf

    path = os.path.join(output_dir, "usd_anim_multi_node.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(24)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # Parent xform with translation animation.
    parent = UsdGeom.Xform.Define(stage, "/Parent")
    translate_op = parent.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0, 0, 0), Usd.TimeCode(0))
    translate_op.Set(Gf.Vec3d(10, 0, 0), Usd.TimeCode(24))

    # Child xform with rotation animation.
    child = UsdGeom.Xform.Define(stage, "/Parent/Child")
    rotate_op = child.AddRotateYOp()
    rotate_op.Set(0.0, Usd.TimeCode(0))
    rotate_op.Set(90.0, Usd.TimeCode(24))

    mesh = UsdGeom.Mesh.Define(stage, "/Parent/Child/Mesh")
    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    stage.Save()
    print(f"  Generated: {path}")


def generate_no_animation(output_dir: str) -> None:
    """A static cube with no animation (control test)."""
    from pxr import Usd, UsdGeom, Gf

    path = os.path.join(output_dir, "usd_anim_static.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(24.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cube")
    xform.AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))

    mesh = UsdGeom.Mesh.Define(stage, "/Cube/Mesh")
    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    stage.Save()
    print(f"  Generated: {path}")


def generate_timecode_scaling(output_dir: str) -> None:
    """A cube with animation at 30 fps to test time code normalization."""
    from pxr import Usd, UsdGeom, Gf

    path = os.path.join(output_dir, "usd_anim_30fps.usda")
    stage = Usd.Stage.CreateNew(path)
    stage.SetTimeCodesPerSecond(30.0)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(60)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.y)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    xform = UsdGeom.Xform.Define(stage, "/Cube")
    mesh = UsdGeom.Mesh.Define(stage, "/Cube/Mesh")

    points = [
        (-0.5, -0.5, -0.5),
        (0.5, -0.5, -0.5),
        (0.5, 0.5, -0.5),
        (-0.5, 0.5, -0.5),
        (-0.5, -0.5, 0.5),
        (0.5, -0.5, 0.5),
        (0.5, 0.5, 0.5),
        (-0.5, 0.5, 0.5),
    ]
    face_vertex_counts = [4, 4, 4, 4, 4, 4]
    # fmt: off
    face_vertex_indices = [
        0, 1, 2,
        3, 4, 5,
        6, 7, 0,
        4, 7, 3,
        1, 5, 6,
        2, 0, 1,
        5, 4, 3,
        2, 6, 7,
    ]
    # fmt: on
    mesh.GetPointsAttr().Set([Gf.Vec3f(*p) for p in points])
    mesh.GetFaceVertexCountsAttr().Set(face_vertex_counts)
    mesh.GetFaceVertexIndicesAttr().Set(face_vertex_indices)

    # Animation: translate along X, at 30fps => 60 time codes = 2 seconds
    translate_op = xform.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0), Usd.TimeCode(0))
    translate_op.Set(Gf.Vec3d(3.0, 0.0, 0.0), Usd.TimeCode(30))  # 1s
    translate_op.Set(Gf.Vec3d(6.0, 0.0, 0.0), Usd.TimeCode(60))  # 2s

    stage.Save()
    print(f"  Generated: {path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "..", "data", "assets", "animation_test")
    os.makedirs(output_dir, exist_ok=True)

    print("Generating animated USD test files...")
    generate_linear_translation(output_dir)
    generate_rotation_animation(output_dir)
    generate_multi_channel(output_dir)
    generate_multi_node(output_dir)
    generate_no_animation(output_dir)
    generate_timecode_scaling(output_dir)
    print("Done.")
