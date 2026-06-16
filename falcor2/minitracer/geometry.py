# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING

from slangpy import (
    AccelerationStructureBuildDesc,
    AccelerationStructureBuildFlags,
    AccelerationStructureBuildMode,
    BufferCursor,
    Format,
    AccelerationStructureBuildInputTriangles,
    AccelerationStructureGeometryFlags,
    BufferOffsetPair,
    IndexFormat,
)
from slangpy.math import transpose, inverse
from falcor2.minitracer.posrotscale import PosRotScale
from falcor2 import ImporterMesh, ImporterSemantic

if TYPE_CHECKING:
    from falcor2.minitracer.scene import Scene

import numpy as np


# A mesh, represented by a BLAS. Has no actual representation in the scene,
# as it's all compacted into the geometry instance.
class Geometry:
    def __init__(self, scene: "Scene", index: int):
        super().__init__()
        self.scene = scene
        self.index = index
        self.name = f"Geometry {index}"

    def alloc(self, max_vertices: int, max_indices: int):
        vbuffer, vstart = self.scene.alloc_vertices(max_vertices)
        ibuffer, istart = self.scene.alloc_indices(max_indices)
        self.vertex_buffer = vbuffer
        self.vertex_start = vstart
        self.vertex_count = max_vertices
        self.index_buffer = ibuffer
        self.index_start = istart
        self.index_count = max_indices

    def vertex_cursor(self):
        return self.scene.vertex_cursor(self.vertex_buffer, self.vertex_start, self.vertex_count)

    def index_cursor(self):
        return self.scene.index_cursor(self.index_buffer, self.index_start, self.index_count)

    def write(self, cursor: BufferCursor):
        cursor[self.index].write(
            {
                "vertex_buffer": self.vertex_buffer,
                "vertex_start": self.vertex_start,
                "vertex_count": self.vertex_count,
                "index_buffer": self.index_buffer,
                "index_start": self.index_start,
                "index_count": self.index_count,
            }
        )

    def create_blas_descriptor(self):

        vbuffer = self.scene.vertex_buffers[self.vertex_buffer].buffer.storage
        ibuffer = self.scene.index_buffers[self.index_buffer].buffer.storage
        vstride = self.scene.vertex_struct_type.buffer_layout.stride
        istride = self.scene.index_struct_type.buffer_layout.stride

        blas_geometry_desc = AccelerationStructureBuildInputTriangles()
        blas_geometry_desc.flags = AccelerationStructureGeometryFlags.none

        blas_geometry_desc.pre_transform_buffer = BufferOffsetPair(
            self.scene.f3x4_identity_buffer.storage, 0
        )

        blas_geometry_desc.index_format = IndexFormat.uint32
        blas_geometry_desc.vertex_format = Format.rgb32_float
        blas_geometry_desc.vertex_stride = vstride
        blas_geometry_desc.index_count = self.index_count
        blas_geometry_desc.vertex_count = self.vertex_count
        blas_geometry_desc.index_buffer = BufferOffsetPair(ibuffer, self.index_start * istride)
        blas_geometry_desc.vertex_buffers = [BufferOffsetPair(vbuffer, self.vertex_start * vstride)]

        return blas_geometry_desc

    def create_blas_build_inputs(self):
        blas_build_inputs = AccelerationStructureBuildDesc()
        blas_build_inputs.mode = AccelerationStructureBuildMode.build
        blas_build_inputs.flags = AccelerationStructureBuildFlags.none
        blas_build_inputs.inputs = [self.create_blas_descriptor()]
        return blas_build_inputs

    def import_mesh(self, importer_mesh: ImporterMesh):
        self.alloc(importer_mesh.vertex_count, len(importer_mesh.subgeometries[0].indices) * 3)
        cursor = self.vertex_cursor()

        # UVs on GPU are stored as an array of float2, one for each UV set, so they
        # need to be gathered together into a single array of shape (N, uvcount, 2).
        uvcount = 0
        for field in cursor.element_type.fields:
            if field.name == "uv":
                uvcount = field.type.element_count
                break
        uvarrays: list[Optional[np.ndarray]] = [None] * uvcount
        for attr in importer_mesh.attributes:
            if attr.semantic == ImporterSemantic.tex_coord:
                uvarrays[attr.index] = importer_mesh.get_stream(attr)
        for i in range(0, uvcount):
            if uvarrays[i] is None:
                uvarrays[i] = np.zeros((importer_mesh.vertex_count, 2), dtype=np.float32)
        uvs = np.stack(uvarrays, axis=1)

        # HACK: write_from_numpy needs some work, as it doesn't handle
        # non-contiguous or >2D arrays properly and thus breaks with uvs.
        # For now, reshape the uvs to a 2D array of bytes, where D0 is
        # vertex index and D1 is the flattened uv data.
        uvs = np.ascontiguousarray(uvs)
        uvs = uvs.view(np.uint8).reshape(-1, uvcount * 2 * 4)

        # Read colors, and default to white if none exist
        colors = importer_mesh.colors
        if colors is None:
            colors = np.ones((importer_mesh.vertex_count, 3), dtype=np.float32)
        colors = colors[:, :3]  # Currently only support RGB vertex colors (not RGBA)

        cursor.write_from_numpy(
            {
                "position": importer_mesh.positions,
                "normal": importer_mesh.normals,
                "tangent": importer_mesh.tangents,
                "bitangent": np.zeros_like(importer_mesh.normals),  # Not used
                "uv": uvs,
                "color": colors,
            },
            unchecked_copy=True,
        )
        cursor.apply()

        cursor = self.index_cursor()
        cursor.write_from_numpy(
            importer_mesh.subgeometries[0].indices_numpy.flatten(), unchecked_copy=True
        )
        cursor.apply()


# Instance of given geometry. Represented by an instance in the TLAS,
# and a GeometryInstance in the scene.
class GeometryInstance:
    def __init__(self, scene: "Scene", index: int):
        super().__init__()
        self.scene = scene
        self.index = index
        self.geometry = Geometry(scene, -1)
        self.transform = PosRotScale()
        self.material = scene.dummy_material

    def write(
        self,
        cursor: BufferCursor,
        transform_cursor: BufferCursor,
        inverse_transpose_cursor: BufferCursor,
    ):

        assert self.geometry.index != -1
        cursor[self.index].write(
            {
                "flags": 0,
                "material": self.material.index,
                "vertex_buffer": self.geometry.vertex_buffer,
                "vertex_start": self.geometry.vertex_start,
                "vertex_count": self.geometry.vertex_count,
                "index_buffer": self.geometry.index_buffer,
                "index_start": self.geometry.index_start,
                "index_count": self.geometry.index_count,
            }
        )
        t = self.transform.to_float4x4()
        transform_cursor[self.index].write(t)
        inverse_transpose_cursor[self.index].write(transpose(inverse(t)))
