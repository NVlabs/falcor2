# SPDX-License-Identifier: Apache-2.0

import falcor2.minitracer.tools as mt
import slangpy as spy
from pathlib import Path
from falcor2.minitracer.scene import Scene, DirtyFlag, Material, GeometryInstance
import PIL.Image as Image
import numpy as np
from typing import Any
import json

DATA_DIR = Path(__file__).parent.parent.parent / "data"


def get_material_data(
    materials_path: Path,
    side: float,
    res: int,
    tesselation_res: int,
) -> dict[str, dict[str, Any]]:

    if not materials_path.exists():
        raise FileNotFoundError(f"Materials path does not exist: {materials_path}")

    buffers = ["basecolor", "specular", "roughness", "metallic", "normal", "height"]

    data = {}

    # Iterate through materials in the specified folder.
    # Each material can be a subfolder containing maps.
    for material_cache_folder in materials_path.iterdir():
        if not material_cache_folder.is_dir():
            continue

        material = material_cache_folder.stem
        print(f'Loading "{material}" from dataset...')
        material_data: dict[str, Image.Image | dict[Any, Any]] = {}
        for key in buffers:
            for ext in ["png", "jpg"]:
                filepath = material_cache_folder / f"{key}.{ext}"
                if filepath.exists():
                    material_data[key] = Image.open(filepath)
                    break
            if key not in material_data:
                # the specular map is optional
                assert key in ["specular"]
        try:
            with open(material_cache_folder / "metadata.json") as f:
                metadata = json.load(f)
        except:
            metadata = {}

        data[material] = {}
        for key in material_data:
            if not isinstance(material_data[key], Image.Image):
                continue
            if key == "normal":
                continue
            img: Image.Image = material_data[key]  # type: ignore
            target_res = tesselation_res if key == "height" else res
            if img.size[0] > target_res:
                img = img.resize((target_res, target_res), resample=Image.Resampling.BICUBIC)
            np_img: np.ndarray = None  # type: ignore
            if img.mode == "I":
                raise Exception("--- We shouldn't be here! ---")
                np_img = np.array(img, dtype=np.uint16).astype(np.float32) / np.iinfo(np.uint16).max
            elif img.mode == "I;16":
                np_img = np.array(img).astype(np.float32) / np.iinfo(np.uint16).max
            else:
                np_img = np.array(img).astype(np.float32) / 255.0
            if np_img.ndim == 3 and np_img.shape[2] == 4:
                np_img = np_img[..., :3]
            if np_img.ndim == 3 and key in ["metallic"]:
                np_img = np_img.mean(axis=2)
            if key == "normal":
                np_img = np_img * 2.0 - 1.0
            elif key == "height":
                height_mean = metadata.get("height_mean", np_img.mean())
                height_factor = metadata.get("height_factor", 10.0 / 255) * side
                np_img = (np_img - height_mean) * height_factor
            data[material][key] = np_img
    return data


def create_material(scene: Scene, name: str, data: dict[Any, Any]) -> Material:
    device = scene.device
    material = scene.create_material(name)
    material.albedo = spy.Tensor.from_numpy(device, data["basecolor"])
    if "specular" in data:
        material.specular = spy.Tensor.from_numpy(device, data["specular"])
    else:
        material.specular = scene.white_texture
    material.roughness = spy.Tensor.from_numpy(device, data["roughness"][..., None])
    material.metallic = spy.Tensor.from_numpy(device, data["metallic"][..., None])
    if "normal" in data:
        material.normal = spy.Tensor.from_numpy(device, data["normal"])
    material.double_sided = True
    return material


def generate_triangle_indices(N: int) -> np.ndarray:
    """
    Generates indices for a triangulated (N, N) grid.

    Args:
        N: The grid dimension (number of vertices per side).

    Returns:
        indices: (Num_Triangles, 3) int32 array.
                Winding is CCW (Positive Z).
    """
    # 1. Create a grid of vertex indices (0 to N^2 - 1)
    #    We reshape it to (N, N) to easily slice rows/cols
    idx_grid = np.arange(N * N, dtype=np.int32).reshape(N, N)

    # 2. Get the "Bottom-Left" corner index for every valid quad cell.
    #    A grid of N vertices has (N-1) quads.
    #    We select everything except the last row and last column.
    #    (Assuming indexing='xy' where row index increases with Y)
    base_indices = idx_grid[:-1, :-1].flatten()

    # 3. Calculate corner offsets relative to the base index
    #    k = Bottom-Left
    k = base_indices

    #    k_right    = Bottom-Right (k + 1)
    #    k_up       = Top-Left     (k + N)
    #    k_up_right = Top-Right    (k + N + 1)

    k_right = k + 1
    k_up = k + N
    k_up_right = k + N + 1

    # 4. Form the two triangles for each quad
    t1 = np.stack([k, k_up, k_right], axis=1)
    t2 = np.stack([k_right, k_up, k_up_right], axis=1)

    # 5. Interleave or Concatenate them
    #    Stacking them creates shape (Num_Quads, 2, 3)
    #    Reshape flattens them into a simple list of triangles (Num_Quads*2, 3)
    triangles = np.stack([t1, t2], axis=1).reshape(-1, 3)

    return triangles


def make_grid_geometry_instance(
    scene: Scene, res: int = 1024, side: float = 1.0
) -> GeometryInstance:
    # Create grid of positions
    x = np.linspace(-0.5, 0.5, res, dtype=np.float32) * side
    y = np.linspace(-0.5, 0.5, res, dtype=np.float32) * side
    xx, yy = np.meshgrid(x, y, indexing="xy")
    zz = np.zeros_like(xx)
    positions = np.stack([xx, yy, zz], axis=-1)

    # Create UV coordinates
    u = np.linspace(0.0, 1.0, res, dtype=np.float32)
    v = np.linspace(0.0, 1.0, res, dtype=np.float32)
    uu, vv = np.meshgrid(u, v, indexing="xy")
    uv_coords = np.stack([uu, vv], axis=-1)

    P = positions.reshape(-1, 3)
    N = np.tile(np.array([0, 0, 1], dtype=np.float32), (P.shape[0], 1))
    T = np.tile(np.array([1, 0, 0], dtype=np.float32), (P.shape[0], 1))
    B = np.tile(np.array([0, 1, 0], dtype=np.float32), (P.shape[0], 1))
    color = np.tile(np.array([1, 1, 1], dtype=np.float32), (P.shape[0], 1))
    uv0 = uv_coords.reshape(-1, 2)

    indices = generate_triangle_indices(res)

    geometry = scene.alloc_geometry()
    geometry.alloc(len(P), len(indices) * 3)
    cursor = geometry.vertex_cursor()

    cursor.write_from_numpy(
        {
            "position": P,
            "normal": N,
            "tangent": T,
            "bitangent": B,
            "color": color,
            "uv": np.hstack([uv0, uv0]),
        },
        unchecked_copy=True,
    )
    cursor.apply()

    cursor = geometry.index_cursor()
    cursor.write_from_numpy(indices.flatten(), unchecked_copy=True)
    cursor.apply()

    geometry_instance = scene.alloc_geometry_instance()
    geometry_instance.geometry = geometry

    return geometry_instance


class ExtendedMaterial:
    def __init__(self, device: spy.Device, material: Material, data: dict[Any, Any]):
        super().__init__()
        self.material = material
        if "height" in data:
            self.height_mean = float(data["height"].mean())
            self.height_min = float(data["height"].min())
            self.height_max = float(data["height"].max())
            self.height = spy.Tensor.from_numpy(device, data["height"])
        else:
            self.height = None


class MultimatGeometry:
    def __init__(
        self,
        scene: Scene,
        materials_path: Path,
        image_res: int = 512,
        tesselation_res: int = 512,
        side: float = 1.0,
    ):
        super().__init__()
        self.module = spy.Module(
            scene.device.load_module("falcor2.minitracer.examples.example_animated")
        )
        self.tesselation_res = tesselation_res
        self.side = side
        self.scene = scene

        self.geometry_instance = make_grid_geometry_instance(scene, tesselation_res, side)
        self.geometry = self.geometry_instance.geometry

        material_data = get_material_data(
            materials_path=materials_path,
            side=side,
            res=image_res,
            tesselation_res=tesselation_res,
        )

        self.materials: list[tuple[str, ExtendedMaterial]] = []
        for name, data in material_data.items():
            base_material = create_material(scene, name, data)
            self.materials.append((name, ExtendedMaterial(scene.device, base_material, data)))

        self.local_material_index = 0
        self.active_material = self.materials[0]
        self.geometry_instance.material = self.active_material[1].material
        self.auto_advance_counter = 0
        self.auto_advance_step = 0
        self._update_displacement()
        pass

    def change(self):
        self.local_material_index = (self.local_material_index + 1) % len(self.materials)
        self.active_material = self.materials[self.local_material_index]
        self.geometry_instance.material = self.active_material[1].material
        self.auto_advance_counter = 0
        self._update_displacement()

    # Called every frame, just to give chance to auto-update
    def update(self):
        self.auto_advance_counter += self.auto_advance_step
        if self.auto_advance_counter > 30:
            self.change()

    def toggle_autoadvance(self):
        self.auto_advance_step = 1 - self.auto_advance_step

    def _update_displacement(self):
        if self.active_material[1].height is not None:
            self.module.apply_heightfield(
                index2d=spy.grid((self.tesselation_res, self.tesselation_res)),
                vertices=self.scene.vertex_buffers[self.geometry.vertex_buffer].buffer.storage,
                vertex_start=self.geometry.vertex_start,
                dimensions=spy.uint2(self.tesselation_res, self.tesselation_res),
                heightfield=self.active_material[1].height,
                height_offset=-self.active_material[1].height_mean,
            )
        else:
            # If there's no height map, reset vertex positions to the original flat grid
            self.module.reset_heightfield(
                index2d=spy.grid((self.tesselation_res, self.tesselation_res)),
                vertices=self.scene.vertex_buffers[self.geometry.vertex_buffer].buffer.storage,
                vertex_start=self.geometry.vertex_start,
                dimensions=spy.uint2(self.tesselation_res, self.tesselation_res),
            )

        self.module.recalc_normals(
            index2d=spy.grid((self.tesselation_res, self.tesselation_res)),
            vertices=self.scene.vertex_buffers[self.geometry.vertex_buffer].buffer.storage,
            vertex_start=self.geometry.vertex_start,
            dimensions=spy.uint2(self.tesselation_res, self.tesselation_res),
        )

        # Sync to make sure we won't be updating BLAS/TLAS while
        # they're still used in the in-flight GPU work
        self.module.device.wait()
        self.scene.mark_dirty(DirtyFlag.geometry)


class AnimatedApp:
    def __init__(self):
        super().__init__()
        pass

    def main(self):

        device = mt.create_device()

        # Load scene + create camera, env map and renderer
        scene = Scene(device)
        self.animated_geometries: list[Any] = []

        materials_path = Path(DATA_DIR / "assets/matsynth_small")
        self.animated_geometries.append(
            MultimatGeometry(scene, materials_path=materials_path, side=4.0)
        )

        # Add env map (and optionally adjust brightness)
        envmap = scene.create_env_map(DATA_DIR / "assets/envmaps/brown_photostudio_02_2k.hdr")
        envmap.scaling_factor = spy.float3(1)

        # Create camera and renderer
        camera = scene.create_camera(128, 128, 70)
        camera.transform.pos = spy.float3(0, 0, 4)
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
        renderer.env_map_as_background = True
        # Background color if not using env map (default=black)
        renderer.background_color = spy.float3(0.0, 0.0, 0.0)
        # Max path depth (default=3)
        renderer.max_depth = 3
        # Whether to apply simple ACES tone mapping (default=True)
        renderer.tone_map = True

        # Option 2: Viewer with custom update loop.:
        viewer = mt.create_viewer(scene, camera, renderer)
        viewer.register_keyboard_event_callback(self.on_keyboard_event)
        while viewer.update():
            for animated_geometry in self.animated_geometries:
                animated_geometry.update()
            pass

    def on_keyboard_event(self, event: spy.KeyboardEvent) -> bool:
        """Handle keyboard events."""
        if event.type == spy.KeyboardEventType.key_press:
            if event.key == spy.KeyCode.space:
                for animated_geometry in self.animated_geometries:
                    animated_geometry.toggle_autoadvance()
                return True
            if event.key == spy.KeyCode.c:
                for animated_geometry in self.animated_geometries:
                    animated_geometry.change()
                return True
        return False


if __name__ == "__main__":
    app = AnimatedApp()
    app.main()
