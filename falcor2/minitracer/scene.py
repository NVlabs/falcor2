# SPDX-License-Identifier: Apache-2.0

from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
import numpy.typing as npt

import slangpy as spy
from slangpy import (
    AccelerationStructureInstanceFlags,
    Buffer,
    AccelerationStructure,
    AccelerationStructureBuildFlags,
    AccelerationStructureInstanceDesc,
    AccelerationStructureBuildInputTriangles,
    AccelerationStructureBuildDesc,
    AccelerationStructureBuildMode,
    BufferUsage,
    uint3,
    float3,
    float4,
    float3x4,
    BufferOffsetPair,
    Tensor,
    Bitmap,
)
from slangpy.core.packedarg import PackedArg

from enum import Flag

from falcor2.minitracer.camera import Camera
from falcor2.minitracer.env_map import EnvMap
from falcor2.minitracer.geometry import Geometry, GeometryInstance

from slangpy import Module, Tensor
from slangpy.reflection import SlangType

from falcor2 import ImporterMesh
from falcor2.minitracer.material import Material

if TYPE_CHECKING:
    from slangpy import Device


class DirtyFlag(Flag):
    none = 0
    entity = 1 << 0
    geometry = 1 << 1
    geometry_instance = 1 << 3
    material = 1 << 4
    transform = 1 << 5
    env_map = 1 << 6
    textures = 1 << 7
    any = 0xFFFFFFFF


class LinearBufferAllocator:
    def __init__(self, device: "Device", element_type: "SlangType", initial_capacity: int):
        super().__init__()
        self.buffer = Tensor.empty(device, (initial_capacity,), element_type)
        self.usage = 0

    def alloc(self, count: int = 1):
        # Fail if the count would exceed the buffer size.
        if self.usage + count > self.buffer.element_count:
            return -1

        res = self.usage
        self.usage += count
        return res


class SceneParameter:
    """
    Represents a scene parameter that is backed by a tensor.
    """

    def __init__(
        self, scene: "Scene", tensor: Tensor, set_tensor_cb: Any, value_range: tuple[float, float]
    ):
        super().__init__()
        self._scene = scene
        self._tensor = tensor
        self._set_tensor_cb = set_tensor_cb
        self._value_range = value_range
        self._changed = False

    def require_grads(self):
        self.tensor = self.tensor.with_grads()

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @tensor.setter
    def tensor(self, value: Tensor):
        self._tensor = value
        self._changed = True

    @property
    def has_grads(self) -> bool:
        return self._tensor.grad_out is not None

    @property
    def value_range(self) -> tuple[float, float]:
        return self._value_range

    def _write_change(self):
        self._set_tensor_cb(self._tensor)


class SceneParameters:
    """
    A collection of scene parameters.
    """

    def __init__(self, scene: "Scene"):
        super().__init__()
        self._scene = scene
        self._dict: dict[str, SceneParameter] = {}

        for material in scene.materials:
            material.get_scene_parameters(self)

    def write_to_scene(self):
        changed = False
        for param in self.values():
            if param._changed:
                param._write_change()
                changed = True
        if changed:
            self._scene.mark_dirty(DirtyFlag.textures)

    def _add(
        self,
        key: str,
        tensor: Tensor,
        write_change: Any,
        range: tuple[float, float] = (float("-inf"), float("inf")),
    ):
        if key in self._dict:
            raise KeyError(f"Parameter '{key}' already exists in scene parameters.")
        self._dict[key] = SceneParameter(self._scene, tensor, write_change, range)

    def __getitem__(self, key: str) -> SceneParameter:
        if key in self._dict:
            return self._dict[key]
        else:
            raise KeyError(f"Parameter '{key}' not found in scene parameters.")

    def keys(self) -> list[str]:
        return list(self._dict.keys())

    def values(self) -> list[SceneParameter]:
        return list(self._dict.values())

    def items(self) -> list[tuple[str, SceneParameter]]:
        return list(self._dict.items())

    def tensors_with_grads(self) -> list[Tensor]:
        return [param.tensor for param in self._dict.values() if param.has_grads]


class Scene:

    MAX_MATERIAL_COUNT = 200
    # keep in sync with constant in scene/core.slang
    MAX_TEXTURE_COUNT = 256
    MAX_INSTANCE_COUNT = 8000
    MAX_EMISSIVE_TRIANGLE_COUNT = 1024 * 1024

    def __init__(self, device: "Device"):
        super().__init__()
        self.device = device
        self.geometry_instances: list[GeometryInstance] = []
        self.geometries: list[Geometry] = []
        self.materials: list[Material] = []
        self.cameras: list[Camera] = []
        self.env_map: Optional[EnvMap] = None

        self.blas_geometry_descriptors: list[AccelerationStructureBuildInputTriangles] = []
        self.blas_acceleration_structures: list[AccelerationStructure] = []
        self.blas_buffer: Buffer = None  # type: ignore
        self.blas_scratch_buffer: Buffer = None  # type: ignore

        self.tlas_instance_descriptors: list[AccelerationStructureInstanceDesc] = []
        self.tlas: AccelerationStructure = None  # type: ignore
        self.tlas_buffer: Buffer = None  # type: ignore
        self.tlas_scratch_buffer: Buffer = None  # type: ignore

        self.scene_module = Module(device.load_module("falcor2.minitracer.scene"))
        self.utils_module = Module(device.load_module("falcor2.utils"))

        self.float4x4_type = self.scene_module.layout.require_type_by_name("float4x4")

        # Geometry data layout
        self.vertex_struct_type = self.scene_module.layout.require_type_by_name("Vertex")
        self.index_struct_type = self.scene_module.layout.require_type_by_name("uint")

        # Basic type layouts
        self.f3x4_identity_buffer = Tensor.empty(
            device, (1,), self.scene_module.layout.require_type_by_name("float3x4")
        )

        # Write identity matrix to buffer
        c = self.f3x4_identity_buffer.cursor()
        c[0].write(float3x4.identity())
        c.apply()

        # Layouts for general material data.
        self.material_data_layout = self.scene_module.layout.require_type_by_name("MaterialData")

        # Geometry data layouts.
        self.geometry_instance_struct_type = self.scene_module.layout.require_type_by_name(
            "GeometryInstance"
        )
        self.emissive_geometry_instance_struct_type = self.scene_module.layout.require_type_by_name(
            "EmissiveGeometryInstance"
        )
        self.emissive_triangle_struct_type = self.scene_module.layout.require_type_by_name(
            "EmissiveTriangle"
        )
        self.emissive_flux_struct_type = self.scene_module.layout.require_type_by_name(
            "EmissiveFlux"
        )

        # Init all geometry buffers.
        self.vertex_buffers: list[LinearBufferAllocator] = []
        self.index_buffers: list[LinearBufferAllocator] = []

        self.geometry_instance_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_INSTANCE_COUNT,), self.geometry_instance_struct_type
        )
        self.emissive_geometry_instance_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_INSTANCE_COUNT,), self.emissive_geometry_instance_struct_type
        )
        self.emissive_triangle_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_EMISSIVE_TRIANGLE_COUNT,), self.emissive_triangle_struct_type
        )
        self.emissive_flux_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_EMISSIVE_TRIANGLE_COUNT,), self.emissive_flux_struct_type
        )

        self.transform_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_INSTANCE_COUNT,), self.float4x4_type
        )
        self.inverse_transpose_transform_buffer: Tensor = Tensor.empty(
            device, (Scene.MAX_INSTANCE_COUNT,), self.float4x4_type
        )

        self.emissive_triangle_count = 0

        self.dirty_flags = DirtyFlag.none

        # Material buffers
        self.material_data = Tensor.empty(
            self.device,
            (Scene.MAX_MATERIAL_COUNT,),
            self.material_data_layout,
            usage=BufferUsage.shader_resource,
        )

        # Create a black texture to use as dummy
        dummy_tensor = np.zeros((1, 1, 4), dtype=np.float32)
        dummy_texture = Tensor.from_numpy(self.device, dummy_tensor)

        # Textures (basically tensors)
        self.textures: list[Tensor] = [dummy_texture] * Scene.MAX_TEXTURE_COUNT
        self.texture_ids: dict[Tensor, int] = {}

        self.black_texture = self.create_texture(16, 16, float3(0.0, 0.0, 0.0))
        self.white_texture = self.create_texture(16, 16, float3(1.0, 1.0, 1.0))
        self.def_norm_texture = self.create_texture(16, 16, float3(0, 0, 1))

        # Create a dummy material for init
        self.dummy_material = self.alloc_material("dummy")

        # Cached packed arg of the scene parameter block.
        self._packed_arg: Optional[PackedArg] = None

    def alloc_geometry(self):
        res = Geometry(self, len(self.geometries))
        self.geometries.append(res)
        self.mark_dirty(DirtyFlag.geometry)
        return res

    def alloc_geometry_instance(self):
        res = GeometryInstance(self, len(self.geometry_instances))
        self.geometry_instances.append(res)
        self.mark_dirty(DirtyFlag.geometry_instance)
        return res

    def alloc_material(self, name: str):
        res = Material(self, name, len(self.materials))
        self.materials.append(res)
        self.mark_dirty(DirtyFlag.material)
        return res

    def alloc_camera(self):
        res = Camera(self, len(self.cameras))
        self.cameras.append(res)
        self.mark_dirty(DirtyFlag.entity)
        return res

    def alloc_env_map(self):
        if self.env_map is not None:
            raise RuntimeError("Cannot allocate more than 1 environment map.")
        self.env_map = EnvMap(self)
        self.mark_dirty(DirtyFlag.env_map)
        return self.env_map

    def alloc_vertices(self, count: int):
        if count == 0:
            return 0, 0

        for i, x in enumerate(self.vertex_buffers):
            idx = x.alloc(count)
            if idx >= 0:
                return i, idx

        size = 10000000
        if count > size:
            size = count

        new_buffer = LinearBufferAllocator(self.device, self.vertex_struct_type, size)
        self.vertex_buffers.append(new_buffer)
        start = new_buffer.alloc(count)
        assert start >= 0
        return len(self.vertex_buffers) - 1, start

    def alloc_indices(self, count: int):
        if count == 0:
            return 0, 0

        for i, x in enumerate(self.index_buffers):
            idx = x.alloc(count)
            if idx >= 0:
                return i, idx

        size = 10000000
        if count > size:
            size = count

        new_buffer = LinearBufferAllocator(self.device, self.index_struct_type, size)
        self.index_buffers.append(new_buffer)
        start = new_buffer.alloc(count)
        assert start >= 0
        return len(self.index_buffers) - 1, start

    def vertex_cursor(self, buffer: int, start: int, count: int):
        return self.vertex_buffers[buffer].buffer.cursor(start, count)

    def index_cursor(self, buffer: int, start: int, count: int):
        return self.index_buffers[buffer].buffer.cursor(start, count)

    def get_uniforms(self, bind_emissive_triangles: bool = True) -> dict[str, Any]:
        textures = [x for x in self.textures]
        texture_grads = [x.grad_out if x.grad_out else {} for x in self.textures]
        texture_grads_enabled = [1 if x.grad_out else 0 for x in self.textures]
        uniforms = {
            "_type": "Scene",
            "tlas": self.tlas,
            "geometry_instances": self.geometry_instance_buffer.storage,
            "vertices": self.vertex_buffers[0].buffer.storage,
            "indices": self.index_buffers[0].buffer.storage,
            "transforms": self.transform_buffer.storage,
            "inverse_transpose_transforms": self.inverse_transpose_transform_buffer.storage,
            "materials": self.material_data.storage,
            "textures": textures,
            "texture_grads": texture_grads,
            "texture_grads_enabled": texture_grads_enabled,
            "env_map": self.env_map.get_uniforms() if self.env_map else {},
        }
        if bind_emissive_triangles:
            uniforms["emissive_triangles"] = {
                "triangle_count": self.emissive_triangle_count,
                "geometry_instances": self.emissive_geometry_instance_buffer.storage,
                "triangles": self.emissive_triangle_buffer.storage,
                "fluxes": self.emissive_flux_buffer.storage,
            }
        return uniforms

    def get_this(self) -> PackedArg:
        if self._packed_arg is None:
            self._packed_arg = spy.pack(self.scene_module, self.get_uniforms())
        return self._packed_arg

    def create_box(self, size: float3):
        vertices = np.array(
            [
                # position(3), normal(3), tangent(3), bitangent(3), color(3), uv0(2), uv1(2)
                # bottom
                [-0.5, -0.5, -0.5, 0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-0.5, -0.5, +0.5, 0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [+0.5, -0.5, +0.5, 0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [+0.5, -0.5, -0.5, 0, -1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # top
                [-0.5, +0.5, +0.5, 0, +1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [-0.5, +0.5, -0.5, 0, +1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [+0.5, +0.5, -0.5, 0, +1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [+0.5, +0.5, +0.5, 0, +1, 0, 1, 0, 0, 0, 0, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # left
                [-0.5, +0.5, -0.5, 0, 0, -1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [-0.5, -0.5, -0.5, 0, 0, -1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [+0.5, -0.5, -0.5, 0, 0, -1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [+0.5, +0.5, -0.5, 0, 0, -1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                # right
                [+0.5, +0.5, +0.5, 0, 0, +1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [+0.5, -0.5, +0.5, 0, 0, +1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [-0.5, -0.5, +0.5, 0, 0, +1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-0.5, +0.5, +0.5, 0, 0, +1, 1, 0, 0, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                # back
                [-0.5, +0.5, +0.5, -1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [-0.5, -0.5, +0.5, -1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [-0.5, -0.5, -0.5, -1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [-0.5, +0.5, -0.5, -1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                # front
                [+0.5, +0.5, -0.5, +1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0],
                [+0.5, -0.5, -0.5, +1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [+0.5, -0.5, +0.5, +1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0],
                [+0.5, +0.5, +0.5, +1, 0, 0, 0, 0, 1, 0, 1, 0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )
        vertices[:, 0:3] *= [size[0], size[1], size[2]]

        indices = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 6, 5],
                [4, 7, 6],
                [8, 10, 9],
                [8, 11, 10],
                [12, 14, 13],
                [12, 15, 14],
                [16, 18, 17],
                [16, 19, 18],
                [20, 22, 21],
                [20, 23, 22],
            ],
            dtype=np.uint32,
        )

        geometry = self.alloc_geometry()
        geometry.alloc(len(vertices), len(indices) * 3)
        cursor = geometry.vertex_cursor()
        cursor.write_from_numpy(
            {
                "position": vertices[:, 0:3],
                "normal": vertices[:, 3:6],
                "tangent": vertices[:, 6:9],
                "bitangent": vertices[:, 9:12],
                "color": vertices[:, 12:15],
                "uv": vertices[:, 15:19],
            },
            unchecked_copy=True,
        )
        cursor.apply()

        cursor = geometry.index_cursor()
        cursor.write_from_numpy(indices.flatten(), unchecked_copy=True)
        cursor.apply()

        geometry_instance = self.alloc_geometry_instance()
        geometry_instance.geometry = geometry

        return geometry_instance

    def create_camera(self, width: int, height: int, fov_degrees: float):
        camera = self.alloc_camera()
        camera.width = width
        camera.height = height
        camera.fov_degrees = fov_degrees
        camera.recompute()
        return camera

    def create_env_map(self, path: Union[str, PathLike[str]]):
        env_map = self.alloc_env_map()
        env_map.texture = self.load_texture(Path(path))
        return env_map

    def create_material(self, name: str):
        return self.alloc_material(name)

    def create_texture(self, width: int, height: int, color: Union[float3, float4]) -> Tensor:
        data = np.ones((height, width, 4), dtype=np.float32)
        data[:, :, 0:1] = color.x
        data[:, :, 1:2] = color.y
        data[:, :, 2:3] = color.z
        if isinstance(color, float3):
            data[:, :, 3:4] = 1.0
        else:
            data[:, :, 3:4] = color.w
        return Tensor.from_numpy(self.device, data)

    def import_mesh(self, mesh: ImporterMesh):
        geometry = self.alloc_geometry()
        geometry.import_mesh(mesh)
        return geometry

    def import_mesh_instance(self, mesh: ImporterMesh):
        geometry = self.import_mesh(mesh)
        geometry_instance = self.alloc_geometry_instance()
        geometry_instance.geometry = geometry
        return geometry_instance

    def load_texture(
        self, path_or_data: Union[str, PathLike, npt.NDArray], convert_srgb: bool = True
    ) -> Tensor:

        if isinstance(path_or_data, np.ndarray):
            bitmap = Bitmap.load_from_numpy(path_or_data)
        else:
            bitmap = Bitmap.load_from_file(path_or_data)

        srgb = bitmap.srgb_gamma
        if convert_srgb and srgb:
            srgb = False

        bitmap = bitmap.convert(
            pixel_format=Bitmap.PixelFormat.rgba,
            component_type=Bitmap.ComponentType.float32,
            srgb_gamma=srgb,
        )

        data = np.asarray(bitmap, copy=False)
        return Tensor.from_numpy(self.device, data)

    def load_normal_map(self, path_or_data: Union[str, PathLike, npt.NDArray]) -> Tensor:
        if isinstance(path_or_data, np.ndarray):
            bitmap = Bitmap.load_from_numpy(path_or_data)
        else:
            bitmap = Bitmap.load_from_file(path_or_data)

        bitmap = bitmap.convert(
            pixel_format=Bitmap.PixelFormat.rgba, component_type=Bitmap.ComponentType.float32
        )

        data = np.asarray(bitmap, copy=False)
        data = data * 2.0 - 1.0
        dataN = np.linalg.norm(data, axis=2, keepdims=True)
        data = data / dataN
        return Tensor.from_numpy(self.device, data)

    def get_parameters(self) -> SceneParameters:
        return SceneParameters(self)

    def mark_dirty(self, flag: DirtyFlag):
        self.dirty_flags |= flag

    def is_dirty(self, flag: DirtyFlag):
        return self.dirty_flags & flag != DirtyFlag.none

    def _get_texture_id(self, texture: Tensor) -> int:
        if not texture in self.texture_ids:
            # check 2d texture with 3 channels
            if len(texture.shape) != 3:
                raise ValueError(f"Invalid texture shape: {texture.shape}")
            if texture.shape[2] < 1 or texture.shape[2] > 4:
                raise ValueError(f"Invalid texture shape: {texture.shape}")
            id = len(self.texture_ids)
            if id >= Scene.MAX_TEXTURE_COUNT:
                raise ValueError(
                    f"Maximum number of textures ({Scene.MAX_TEXTURE_COUNT}) exceeded."
                )
            self.textures[id] = texture
            self.texture_ids[texture] = id
        return self.texture_ids[texture]

    def build(self):

        # Check if we need to reassign texture IDs.
        if self.is_dirty(DirtyFlag.textures):
            self.texture_ids = {}
            self.mark_dirty(DirtyFlag.material)
            self.mark_dirty(DirtyFlag.env_map)

        # Write material data to material buffer.
        if self.is_dirty(DirtyFlag.material):
            # print("update materials")
            material_data_cursor = self.material_data.cursor()
            for mat in self.materials:
                mat.write(material_data_cursor)
            material_data_cursor.apply()

        # Generate BLAS inputs for all geometry groups
        if self.is_dirty(DirtyFlag.geometry):
            # print("update BLAS")
            blas_inputs = [x.create_blas_build_inputs() for x in self.geometries]

            # Get prebuild info for all BLAS inputs
            blas_prebuild = [self.device.get_acceleration_structure_sizes(x) for x in blas_inputs]

            def align_to(x: int, alignment: int = 256):
                return (x + alignment - 1) & ~(alignment - 1)

            # Calculate size/offsets for scratch and result buffers for all BLAS inputs
            scratch_buffer_offset = [0 for x in blas_prebuild]
            for i in range(1, len(scratch_buffer_offset)):
                scratch_buffer_offset[i] = align_to(
                    scratch_buffer_offset[i - 1] + blas_prebuild[i - 1].scratch_size
                )
            scratch_buffer_size = scratch_buffer_offset[-1] + blas_prebuild[-1].scratch_size

            # Allocate scratch buffer and BLAS buffer.
            if not self.blas_scratch_buffer or self.blas_scratch_buffer.size < scratch_buffer_size:
                self.blas_scratch_buffer = self.device.create_buffer(
                    scratch_buffer_size,
                    usage=BufferUsage.unordered_access,
                    label="blas_scratch_buffer",
                )

            # Create empty BLAS structures.
            blas_structures = [
                self.device.create_acceleration_structure(size=prebuild.acceleration_structure_size)
                for prebuild in blas_prebuild
            ]

            # Build BLAS structures.
            command_encoder = self.device.create_command_encoder()
            for blas_input, scratch_offset, blas_structure in zip(
                blas_inputs, scratch_buffer_offset, blas_structures
            ):
                command_encoder.build_acceleration_structure(
                    desc=blas_input,
                    dst=blas_structure,
                    src=None,
                    scratch_buffer=BufferOffsetPair(self.blas_scratch_buffer, scratch_offset),
                )
            self.device.submit_command_buffer(command_encoder.finish())

            self.blases = blas_structures

        if self.is_dirty(DirtyFlag.geometry_instance | DirtyFlag.transform | DirtyFlag.geometry):
            # print("update TLAS")
            # Write buffers for instances and transforms
            instances = self.geometry_instances
            geometry_instance_cursor = self.geometry_instance_buffer.cursor()
            transform_cursor = self.transform_buffer.cursor()
            inverse_transpose_transform_cursor = self.inverse_transpose_transform_buffer.cursor()
            for inst in instances:
                inst.write(
                    geometry_instance_cursor,
                    transform_cursor,
                    inverse_transpose_transform_cursor,
                )
            geometry_instance_cursor.apply()
            transform_cursor.apply()
            inverse_transpose_transform_cursor.apply()

            # Build/filter descriptors
            rt_instance_list = self.device.create_acceleration_structure_instance_list(
                len(instances)
            )
            for i, inst in enumerate(instances):
                rt_instance_list.write(
                    i,
                    {
                        "transform": inst.transform.to_float3x4(),
                        "instance_id": inst.index,
                        "instance_mask": 0xFF,
                        "instance_contribution_to_hit_group_index": 0,
                        "flags": AccelerationStructureInstanceFlags.none,
                        "acceleration_structure": self.blases[inst.geometry.index].handle,
                    },
                )

            # Setup TLAS input
            tlas_build_inputs = AccelerationStructureBuildDesc()
            tlas_build_inputs.mode = AccelerationStructureBuildMode.build
            tlas_build_inputs.flags = AccelerationStructureBuildFlags.none
            tlas_build_inputs.inputs = [rt_instance_list.build_input_instances()]

            # Get prebuild info
            tlas_prebuild_info = self.device.get_acceleration_structure_sizes(tlas_build_inputs)

            # Allocate scratch and result buffers
            if (
                not self.tlas_scratch_buffer
                or self.tlas_scratch_buffer.size < tlas_prebuild_info.scratch_size
            ):
                self.tlas_scratch_buffer = self.device.create_buffer(
                    size=tlas_prebuild_info.scratch_size,
                    usage=BufferUsage.unordered_access,
                    label="tlas_scratch_buffer",
                )

            # Create TLAS
            self.tlas = self.device.create_acceleration_structure(
                size=tlas_prebuild_info.acceleration_structure_size,
                label="tlas",
            )

            # Build TLAS
            command_encoder = self.device.create_command_encoder()
            command_encoder.build_acceleration_structure(
                desc=tlas_build_inputs,
                dst=self.tlas,
                src=None,
                scratch_buffer=self.tlas_scratch_buffer,
            )
            self.device.submit_command_buffer(command_encoder.finish())

        if self.is_dirty(
            DirtyFlag.material
            | DirtyFlag.geometry
            | DirtyFlag.geometry_instance
            | DirtyFlag.transform
        ):
            # print("update emissive triangles")
            self.build_emissive_triangles()

        if self.is_dirty(DirtyFlag.env_map):
            if self.env_map is not None:
                self.env_map.build_importance_map()

        # Eagerly reset packed arg to force re-creation
        if self.dirty_flags != DirtyFlag.none:
            self._packed_arg = None

        self.dirty_flags = DirtyFlag.none

    def build_emissive_triangles(self):
        # Count number of emissive triangle for each geometry instance.
        geometry_instance_count = len(self.geometry_instances)
        self.scene_module.setup_emissive_geometry_triangle_count.dispatch(
            thread_count=uint3(geometry_instance_count, 1, 1),
            geometry_instance_count=geometry_instance_count,
            emissive_geometry_instances=self.emissive_geometry_instance_buffer.storage,
            vars={"g_scene": self.get_uniforms(bind_emissive_triangles=False)},
        )
        # Perform prefix sum to get triangle starts.
        # TODO: this needs to be done on the GPU
        cursor = self.emissive_geometry_instance_buffer.cursor()
        self.emissive_triangle_count = 0
        self.max_emissive_triangle_per_geometry_count = 0
        for i in range(geometry_instance_count):
            cursor[i].write({"triangle_start": self.emissive_triangle_count})
            triangle_count = cursor[i].read()["triangle_count"]  # type: ignore
            self.max_emissive_triangle_per_geometry_count = max(
                self.max_emissive_triangle_per_geometry_count, triangle_count
            )
            self.emissive_triangle_count += triangle_count
        cursor.apply()

        if self.emissive_triangle_count > 0:
            # Write emissive triangles.
            self.scene_module.setup_emissive_triangles.dispatch(
                thread_count=uint3(
                    geometry_instance_count, self.max_emissive_triangle_per_geometry_count, 1
                ),
                geometry_instance_count=geometry_instance_count,
                emissive_geometry_instances=self.emissive_geometry_instance_buffer.storage,
                emissive_triangles=self.emissive_triangle_buffer.storage,
                vars={"g_scene": self.get_uniforms(bind_emissive_triangles=False)},
            )
            # Update the emissive triangles.
            self.scene_module.update_emissive_triangles.dispatch(
                thread_count=uint3(self.emissive_triangle_count, 1, 1),
                triangle_count=self.emissive_triangle_count,
                emissive_geometry_instances=self.emissive_geometry_instance_buffer.storage,
                emissive_triangles=self.emissive_triangle_buffer.storage,
                emissive_fluxes=self.emissive_flux_buffer.storage,
                vars={"g_scene": self.get_uniforms(bind_emissive_triangles=False)},
            )
