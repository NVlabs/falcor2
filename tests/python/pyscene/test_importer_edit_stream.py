# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import shutil
import sys
import textwrap
from collections.abc import Iterator
from pathlib import Path

import pytest
import slangpy as spy
import falcor2 as f2
import falcor2.testing.helpers as helpers


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA = PROJECT_ROOT / "data"
CORNELL_BOX = DATA / "assets" / "cornell-box" / "usdpreviewsurface" / "cornell-box.usda"
BOX_GLB = DATA / "assets" / "kronos" / "Box" / "glTF-Binary" / "Box.glb"
BOX_MATERIAL_NAME = "Red"
ENV_MAP = DATA / "assets" / "envmaps" / "aerodynamics_workshop_512.hdr"
DATA_CHECKER_MATERIAL_SCENE = PROJECT_ROOT / "data" / "scenes" / "checker-material.py"
DATA_SCENES_DIR = DATA_CHECKER_MATERIAL_SCENE.parent
if str(DATA_SCENES_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_SCENES_DIR))

from checker_material_support import CheckerMaterial  # noqa: E402


def write_python_scene(path: Path, source: str) -> Path:
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clear_current_importer() -> Iterator[None]:
    f2.Importer.clear_current()
    yield
    f2.Importer.clear_current()


def test_collection_bindings_record_camera_node_and_environment(tmp_path: Path) -> None:
    importer = f2.Importer.create()
    env_path = tmp_path / "environment.hdr"
    env_path.touch()

    camera = importer.cameras.create(
        name="Camera",
        focal_length=35.0,
        fstop=4.0,
        sensor_size_mm=36.0,
        enable_depth_of_field=True,
        focus_distance=2.0,
        projection=f2.ImporterCamera.Projection.orthographic,
        fov_direction=f2.ImporterCamera.FOVDirection.horizontal,
    )
    importer.nodes.create(name="CameraNode", camera=camera)
    importer.env.set(path=env_path, exposure=5.0, name="Environment")

    scene = importer.build_importer_scene()
    assert len(scene.cameras) == 1
    assert len(scene.nodes) == 2
    assert len(scene.lights) == 1
    assert scene.cameras[0].focal_length == 35.0
    assert scene.cameras[0].fstop == 4.0
    assert scene.cameras[0].sensor_size_mm == 36.0
    assert scene.cameras[0].enable_depth_of_field is True
    assert scene.cameras[0].focus_distance == 2.0
    assert scene.cameras[0].projection == f2.ImporterCamera.Projection.orthographic
    assert scene.nodes[0].name == "CameraNode"
    assert scene.nodes[0].camera_index == 0
    assert Path(scene.lights[0].env_map_path) == env_path.resolve()
    assert scene.lights[0].exposure == 5.0


def test_node_collection_create_camera_creates_camera_and_node() -> None:
    importer = f2.Importer.create()
    transform = f2.Transform()
    transform.translation = spy.float3(1.0, 2.0, 3.0)

    node = importer.nodes.create_camera(
        name="View",
        transform=transform.matrix,
        focal_length=35.0,
        fstop=4.0,
        sensor_size_mm=36.0,
        enable_depth_of_field=True,
        focus_distance=2.0,
        depth_range=spy.float2(0.1, 500.0),
        projection=f2.ImporterCamera.Projection.orthographic,
        fov_direction=f2.ImporterCamera.FOVDirection.horizontal,
    )
    importer.nodes.create(name="Child", parent=node)

    assert isinstance(node, f2.ImporterNodeSelector)
    scene = importer.build_importer_scene()
    assert len(scene.cameras) == 1
    assert len(scene.nodes) == 2
    assert scene.cameras[0].name == "View"
    assert scene.cameras[0].focal_length == 35.0
    assert scene.cameras[0].fstop == 4.0
    assert scene.cameras[0].sensor_size_mm == 36.0
    assert scene.cameras[0].enable_depth_of_field is True
    assert scene.cameras[0].focus_distance == 2.0
    assert scene.cameras[0].depth_range == spy.float2(0.1, 500.0)
    assert scene.cameras[0].projection == f2.ImporterCamera.Projection.orthographic
    assert scene.cameras[0].fov_direction == f2.ImporterCamera.FOVDirection.horizontal
    assert scene.nodes[0].name == "View"
    assert scene.nodes[0].camera_index == 0
    assert scene.nodes[0].transform == transform.matrix
    assert scene.nodes[1].name == "Child"
    assert scene.nodes[1].parent == 0


def test_node_collection_create_camera_fov_uses_defaults_and_derives_focal_length() -> None:
    importer = f2.Importer.create()

    node = importer.nodes.create_camera_fov(fov_degrees=45.0, sensor_size_mm=24.0)

    assert isinstance(node, f2.ImporterNodeSelector)
    scene = importer.build_importer_scene()
    assert len(scene.cameras) == 1
    assert len(scene.nodes) == 1
    assert scene.cameras[0].name == "Camera"
    assert scene.cameras[0].focal_length == pytest.approx(
        24.0 / (2.0 * math.tan(math.radians(45.0) * 0.5))
    )
    assert scene.cameras[0].fstop == 8.0
    assert scene.cameras[0].sensor_size_mm == 24.0
    assert scene.cameras[0].enable_depth_of_field is False
    assert scene.nodes[0].name == "Camera"
    assert scene.nodes[0].camera_index == 0
    assert scene.nodes[0].transform == spy.float4x4.identity()


def test_importer_create_accepts_import_options() -> None:
    options = f2.ImportOptions()
    assert options.recompute_normals is False
    assert options.force_rectangle_lights_to_geometry is False
    assert options.force_disk_lights_to_geometry is False
    assert options.force_sphere_lights_to_geometry is False

    options.recompute_normals = True
    options.force_rectangle_lights_to_geometry = True
    options.force_disk_lights_to_geometry = True
    assert options.force_rectangle_lights_to_geometry
    assert options.force_disk_lights_to_geometry
    assert not options.force_sphere_lights_to_geometry
    importer = f2.Importer.create(default_import_options=options)
    importer.import_asset(CORNELL_BOX)

    scene = importer.build_importer_scene()
    assert len(scene.meshes) > 0


def test_material_collection_validates_input() -> None:
    importer = f2.Importer.create()
    with pytest.raises(RuntimeError, match="name must not be empty"):
        importer.materials.find("")
    with pytest.raises(RuntimeError, match="type must not be empty"):
        importer.materials["Material"].replace("")
    with pytest.raises(TypeError, match="cls must inherit from Material"):
        importer.materials["Material"].replace(dict)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_material_replacement_builds_live_scene(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    importer = f2.Importer.create()
    importer.import_asset(BOX_GLB)

    props = f2.Properties()
    props["roughness_factor"] = 0.25
    importer.materials[BOX_MATERIAL_NAME].replace("StandardMaterial", props)

    scene = f2.Scene.from_importer_scene(device, importer.build_importer_scene())
    material = scene.materials.find(BOX_MATERIAL_NAME)
    assert isinstance(material, f2.StandardMaterial)
    assert float(material.roughness_factor) == pytest.approx(0.25)

    instances = [
        component for component in scene.components if isinstance(component, f2.GeometryInstance)
    ]
    assert instances
    assert any(material in list(instance.materials) for instance in instances)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_material_replacement_accepts_native_material_class(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)
    importer = f2.Importer.create()
    importer.import_asset(BOX_GLB)

    props = f2.Properties()
    props["roughness_factor"] = 0.375
    importer.materials[BOX_MATERIAL_NAME].replace(f2.StandardMaterial, props)

    scene = f2.Scene.from_importer_scene(device, importer.build_importer_scene())
    material = scene.materials.find(BOX_MATERIAL_NAME)
    assert isinstance(material, f2.StandardMaterial)
    assert float(material.roughness_factor) == pytest.approx(0.375)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_from_importer_runs_scene_loaded_callback_once(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    importer = f2.Importer.create()
    camera = importer.cameras.create_fov(name="RecordedCamera", fov_degrees=45.0)
    importer.nodes.create(name="RecordedCameraNode", camera=camera)
    callbacks_run: list[f2.Scene] = []
    importer.on_scene_loaded(lambda scene: callbacks_run.append(scene))

    importer.build_importer_scene()
    assert callbacks_run == []

    scene = f2.Scene.from_importer(device, importer)

    assert callbacks_run == [scene]
    assert scene.active_camera is not None
    assert scene.active_camera.name == "RecordedCamera"
    assert scene.active_camera.fov_y == pytest.approx(45.0)
    assert scene.entities.find("RecordedCameraNode") is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_explicit_importer_scene_preparation_adds_best_view_camera_only_when_missing(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    importer = f2.Importer.create()

    importer_scene = importer.build_importer_scene()
    importer_scene.add_default_camera_best_view(50.0, 4.0 / 3.0)
    scene = f2.Scene.from_importer_scene(device, importer_scene)

    assert len(scene.components.find_all(type=f2.Camera)) == 1
    assert scene.active_camera is not None
    assert scene.entities.find("AutoCamera_BestView") is not None

    authored_importer = f2.Importer.create()
    camera = authored_importer.cameras.create_fov(name="AuthoredCamera")
    authored_importer.nodes.create(name="AuthoredCameraNode", camera=camera)

    authored_importer_scene = authored_importer.build_importer_scene()
    if not authored_importer_scene.cameras:
        authored_importer_scene.add_default_camera_best_view()
    authored_scene = f2.Scene.from_importer_scene(device, authored_importer_scene)

    assert len(authored_scene.components.find_all(type=f2.Camera)) == 1
    assert authored_scene.entities.find("AuthoredCameraNode") is not None
    assert authored_scene.entities.find("AutoCamera_BestView") is None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_append_supports_importer_scene_and_importer(
    device_type: spy.DeviceType,
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)

    data_importer = f2.Importer.create()
    data_importer.nodes.create(name="DataNode")
    scene.append(data_importer.build_importer_scene())
    assert scene.entities.find("DataNode") is not None

    callback_count = 0
    recipe_importer = f2.Importer.create()
    recipe_importer.nodes.create(name="RecipeNode")

    def on_loaded(loaded_scene: f2.Scene) -> None:
        nonlocal callback_count
        callback_count += 1
        assert loaded_scene.entities.find("DataNode") is not None
        assert loaded_scene.entities.find("RecipeNode") is not None

    recipe_importer.on_scene_loaded(on_loaded)
    scene.append(recipe_importer)

    assert callback_count == 1


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_append_python_path_runs_callback_after_existing_content(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    scene = f2.Scene.create(device)
    existing = scene.create_material(f2.StandardMaterial)
    existing.name = "ExistingMaterial"
    scene_path = write_python_scene(
        workspace_tmp_path / "append_scene.py",
        """
        import falcor2 as f2

        imp = f2.Importer.get()
        imp.nodes.create(name="AppendedNode")

        def finish(scene: f2.Scene) -> None:
            assert scene.materials.find("ExistingMaterial") is not None
            assert scene.entities.find("AppendedNode") is not None
            material = scene.create_material(f2.StandardMaterial)
            material.name = "CallbackMaterial"

        imp.on_scene_loaded(finish)
        """,
    )

    scene.append(scene_path)

    assert scene.materials.find("ExistingMaterial") is existing
    assert scene.materials.find("CallbackMaterial") is not None
    assert scene.entities.find("AppendedNode") is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_load_executes_python_scene_file(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    scene_path = write_python_scene(
        workspace_tmp_path / "camera_scene.py",
        """
        from falcor2 import Importer

        imp = Importer.get()
        camera = imp.cameras.create_fov(name="Camera")
        imp.nodes.create(name="CameraNode", camera=camera)
        """,
    )

    scene = f2.Scene.load(device, scene_path)

    assert len(scene.entities) == 1
    assert scene.entities[0].name == "CameraNode"
    assert len(scene.components) == 1
    assert scene.active_camera is not None
    assert scene.active_camera.fov_y == pytest.approx(70.0)


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_create_resolves_source_relative_assets_from_unrelated_cwd(
    device_type: spy.DeviceType,
    workspace_tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scene_dir = workspace_tmp_path / "relative_scene"
    other_dir = workspace_tmp_path / "other"
    scene_dir.mkdir()
    other_dir.mkdir()
    shutil.copyfile(BOX_GLB, scene_dir / "Box.glb")
    shutil.copyfile(ENV_MAP, scene_dir / "environment.hdr")
    scene_path = write_python_scene(
        scene_dir / "scene.py",
        """
        from falcor2 import Importer

        imp = Importer.get()
        imp.import_asset("Box.glb")
        imp.env.set("environment.hdr", name="RelativeEnv")
        """,
    ).resolve()
    expected_env_path = (scene_dir / "environment.hdr").resolve()
    monkeypatch.chdir(other_dir)

    device = helpers.get_device(device_type)
    scene = f2.Scene.load(device, scene_path)
    env_map = scene.components.find(type=f2.EnvMapLight)

    assert len(scene.geometries) > 0
    assert env_map is not None
    assert Path(env_map.env_map_path) == expected_env_path


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_loaded_hook_can_create_active_camera(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    scene_path = write_python_scene(
        workspace_tmp_path / "hook_camera_scene.py",
        """
        import falcor2 as f2

        imp = f2.Importer.get()

        def finish(scene: f2.Scene) -> None:
            entity = scene.create_entity()
            entity.name = "HookCameraEntity"
            camera = entity.create_component(f2.Camera)
            camera.name = "HookCamera"
            scene.active_camera = camera

        imp.on_scene_loaded(finish)
        """,
    )

    scene = f2.Scene.load(device, scene_path)

    assert len(scene.entities) == 1
    assert scene.entities[0].name == "HookCameraEntity"
    assert scene.active_camera is not None
    assert scene.active_camera.name == "HookCamera"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_create_python_exception_restores_current_importer(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    previous = f2.Importer.get()
    previous.cameras.create(name="BeforeFailure")
    failing_scene_path = write_python_scene(
        workspace_tmp_path / "failing_scene.py",
        """
        from falcor2 import Importer

        Importer.get().cameras.create(name="LeakedFailureCamera")
        raise RuntimeError("pyscene boom")
        """,
    )

    with pytest.raises(RuntimeError, match="pyscene boom"):
        f2.Scene.load(device, failing_scene_path)

    restored_scene = f2.Importer.get().build_importer_scene()
    assert len(restored_scene.cameras) == 1
    assert restored_scene.cameras[0].name == "BeforeFailure"
    assert len(restored_scene.nodes) == 0

    f2.Importer.clear_current()
    clean_scene_path = write_python_scene(
        workspace_tmp_path / "clean_scene.py",
        """
        from falcor2 import Importer

        imp = Importer.get()
        camera = imp.cameras.create(name="CleanCamera")
        imp.nodes.create(name="CleanCameraNode", camera=camera)
        """,
    )

    clean_scene = f2.Scene.load(device, clean_scene_path)
    assert len(clean_scene.entities) == 1
    assert clean_scene.entities[0].name == "CleanCameraNode"

    post_load_scene = f2.Importer.get().build_importer_scene()
    assert len(post_load_scene.cameras) == 0


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_create_skips_main_guard(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    guarded_scene_path = write_python_scene(
        workspace_tmp_path / "guarded_scene.py",
        """
        if __name__ == "__main__":
            raise RuntimeError("main guard must not run during embedded loading")
        """,
    )

    scene = f2.Scene.load(device, guarded_scene_path)

    assert scene is not None


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_scene_loaded_hook_exception_restores_current_importer(
    device_type: spy.DeviceType, workspace_tmp_path: Path
) -> None:
    device = helpers.get_device(device_type)
    previous = f2.Importer.get()
    previous.cameras.create(name="BeforeCallbackFailure")
    failing_scene_path = write_python_scene(
        workspace_tmp_path / "failing_callback_scene.py",
        """
        import falcor2 as f2

        imp = f2.Importer.get()
        imp.cameras.create(name="LeakedCallbackFailureCamera")

        def finish(scene: f2.Scene) -> None:
            raise RuntimeError("pyscene callback boom")

        imp.on_scene_loaded(finish)
        """,
    )

    with pytest.raises(RuntimeError, match="pyscene callback boom"):
        f2.Scene.load(device, failing_scene_path)

    restored_scene = f2.Importer.get().build_importer_scene()
    assert len(restored_scene.cameras) == 1
    assert restored_scene.cameras[0].name == "BeforeCallbackFailure"


@pytest.mark.parametrize("device_type", helpers.DEFAULT_DEVICE_TYPES)
def test_data_checker_material_python_scene_loads(device_type: spy.DeviceType) -> None:
    device = helpers.get_device(device_type)

    assert DATA_CHECKER_MATERIAL_SCENE.exists()
    scene = f2.Scene.load(device, DATA_CHECKER_MATERIAL_SCENE)

    assert scene.active_camera is not None
    assert scene.components.find(type=f2.EnvMapLight) is not None
    material = scene.materials.find(BOX_MATERIAL_NAME)
    assert isinstance(material, CheckerMaterial)
    assert material.scale == pytest.approx(12.0)

    instances = [
        component for component in scene.components if isinstance(component, f2.GeometryInstance)
    ]
    assert instances
    assert any(material in list(instance.materials) for instance in instances)
