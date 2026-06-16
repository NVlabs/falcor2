# SPDX-License-Identifier: Apache-2.0

"""
Utility functions to convert from ImporterScene to minitracer Scene.
This replaces the separate gltf_importer.py and usd_importer.py files.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
from slangpy import Device, Tensor, float3, float3x4, float4x4, float4
import slangpy as spy

import falcor2
from falcor2.minitracer.posrotscale import PosRotScale
from falcor2.minitracer.scene import Scene
from falcor2.minitracer.material import Material, MaterialTexture
from falcor2.minitracer.geometry import Geometry

DATA_DIR = Path(__file__).parent.parent.parent.parent / "data"


def decompose_matrix(matrix: float4x4) -> PosRotScale:
    """Convert a 4x4 matrix to position, rotation, and scale components."""
    # Decompose the matrix
    scale = float3()
    orientation = spy.math.quatf()
    translation = float3()
    skew = float3()
    perspective = spy.float4()
    spy.math.decompose(matrix, scale, orientation, translation, skew, perspective)

    return PosRotScale(translation, orientation, scale)


def convert_importer_scene_to_scene(
    device: Device,
    importer_scene: falcor2.ImporterScene,
    path: Path,
    trivial_materials: bool = False,
    append_to_scene: Optional[Scene] = None,
    scene_from_world: Optional[float4x4] = None,
) -> Scene:
    """
    Convert an ImporterScene to a minitracer Scene.

    Args:
        device: SlangPy device
        importer_scene: The loaded ImporterScene from USD or GLTF
        path: Original file path (for debugging/logging)
        trivial_materials: If True, use simple white materials instead of full material processing

    Returns:
        Configured Scene ready for rendering
    """

    if append_to_scene is not None:
        scene = append_to_scene
    else:
        scene = Scene(device)

    if scene_from_world is None:
        scene_from_world = float4x4.identity()

    # Create materials
    start = time.time()
    materials: Dict[str, Material] = {}
    if trivial_materials or len(importer_scene.materials) == 0:
        white_material = scene.create_material("white_material")
        white_material.albedo = MaterialTexture(scene.create_texture(16, 16, float3(0.8, 0.8, 0.8)))
        materials["white_material"] = white_material
        print(f"Created white material in {time.time() - start:.2f} seconds")
    else:
        materials = load_full_materials(scene, importer_scene)
        print(f"Loaded materials in {time.time() - start:.2f} seconds")

    # Process meshes and nodes
    start = time.time()
    total_vertices = sum(mesh.vertex_count for mesh in importer_scene.meshes)
    total_indices = sum(
        len(subgeo.indices) * 3 for mesh in importer_scene.meshes for subgeo in mesh.subgeometries
    )
    total_subgeometries = sum(len(mesh.subgeometries) for mesh in importer_scene.meshes)
    print(
        f"Total meshes: {len(importer_scene.meshes)}, total vertices: {total_vertices}, "
        f"total subgeometries: {total_subgeometries}, total indices: {total_indices}, "
        f"collected in {time.time() - start:.2f} seconds"
    )

    # Create geometry for each SubGeometry of each Mesh, and store them in grouped by mesh
    mesh_geometries: list[list[Geometry]] = []
    for importer_mesh in importer_scene.meshes:
        geometries = []
        for subgeo in importer_mesh.subgeometries:

            # Note: inneficient as we're re-allocating all vertices for each subgeometry
            geometry = scene.alloc_geometry()
            geometry.alloc(importer_mesh.vertex_count, len(subgeo.indices) * 3)
            geometry.name = f"{importer_mesh.name}_{subgeo.name}"
            geometries.append(geometry)

            cursor = geometry.vertex_cursor()

            # UVs on GPU are stored as an array of float2, one for each UV set, so they
            # need to be gathered together into a single array of shape (N, uvcount, 2).
            uvcount = 0
            for field in cursor.element_type.fields:
                if field.name == "uv":
                    uvcount = field.type.element_count
                    break
            uvarrays: list[Optional[np.ndarray]] = [None] * uvcount
            for attr in importer_mesh.attributes:
                if attr.semantic == falcor2.ImporterSemantic.tex_coord:
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

            cursor = geometry.index_cursor()
            cursor.write_from_numpy(subgeo.indices_numpy.flatten(), unchecked_copy=True)
            cursor.apply()

        mesh_geometries.append(geometries)

    # Process nodes with meshes
    start = time.time()
    instances = {}

    print(f"Root nodes: {importer_scene.root_nodes}")
    print(f"Total nodes: {len(importer_scene.nodes)}")

    def process_node(node_index: int, world_from_parent: float4x4):
        """Recursively process nodes and their children."""
        node = importer_scene.nodes[node_index]
        print(f"Processing node {node_index}: {node.name}, mesh_index={node.mesh_index}")

        # Accumulate transform from parent
        world_from_local = spy.math.mul(world_from_parent, node.transform)

        # We only care about the first camera in the scene.
        if node.camera_index == 0 and len(scene.cameras) == 0:
            camera = scene.create_camera(128, 128, 70)
            camera.transform = decompose_matrix(world_from_local)

        # If this node has a mesh, create geometry instances for it
        if node.mesh_index >= 0 and node.mesh_index < len(importer_scene.meshes):
            importer_mesh = importer_scene.meshes[node.mesh_index]
            geometries = mesh_geometries[node.mesh_index]

            # Process each subgeometry in the mesh
            for idx, subgeo in enumerate(importer_mesh.subgeometries):
                # Get pre-created geometry
                geometry = geometries[idx]

                # Create geometry instance
                geometry_instance = scene.alloc_geometry_instance()
                geometry_instance.geometry = geometry
                geometry_instance.transform = decompose_matrix(world_from_local)

                # Assign material
                material_name = (
                    subgeo.material_name if subgeo.material_name in materials else "white_material"
                )
                if material_name not in materials:
                    material_name = "white_material"
                geometry_instance.material = materials[material_name]

                instance_name = f"{node.name}_{subgeo.name}"
                instances[instance_name] = geometry_instance

        # Process children
        for child_index in node.children:
            process_node(child_index, world_from_local)

        # Process prototype
        if node.prototype_index >= 0 and node.prototype_index < len(importer_scene.prototypes):
            prototype = importer_scene.prototypes[node.prototype_index]
            process_node(prototype.root_node, world_from_local)

    # Start processing from root nodes
    for root_node_index in importer_scene.root_nodes:
        process_node(root_node_index, scene_from_world)

    # Add default camera if no cameras were created during node processing.
    if len(scene.cameras) == 0:
        camera = scene.create_camera(128, 128, 70)
        camera.transform.pos = float3(0, 0, 5)

    print(f"Total processing time: {time.time() - start:.2f} seconds")
    return scene


# NOTE: SlangPy version is incoming but not merged yet. Delete this when native version is available.
def convert_3x4_to_4x4(matrix: float3x4) -> float4x4:
    """Convert a 3x4 matrix to a 4x4 matrix by adding [0,0,0,1] row."""
    m4x4 = float4x4()
    m4x4.set_row(0, matrix.get_row(0))
    m4x4.set_row(1, matrix.get_row(1))
    m4x4.set_row(2, matrix.get_row(2))
    m4x4.set_row(3, spy.float4(0, 0, 0, 1))
    return m4x4


def convert_value_to_rgb(param: Any) -> float3:
    if isinstance(param, float4):
        return float3(param.x, param.y, param.z)
    elif isinstance(param, float3):
        return param
    elif isinstance(param, float):
        return float3(param)
    elif isinstance(param, int):
        return float3(float(param))
    else:
        raise ValueError(f"Could not convert {type(param)} to float3: {param}")


def convert_value_to_rgba(param: Any) -> float4:
    if isinstance(param, float4):
        return param
    elif isinstance(param, float3):
        return float4(param.x, param.y, param.z, 1.0)
    elif isinstance(param, float):
        return float4(param, param, param, 1.0)
    elif isinstance(param, int):
        return float4(float(param), float(param), float(param), 1.0)
    else:
        raise ValueError(f"Could not convert {type(param)} to float4: {param}")


def create_texture(scene: Scene, importer_scene: falcor2.ImporterScene, param: Any) -> Tensor:
    if isinstance(param, falcor2.ImporterTexture):
        # Param is already a texture
        texture = param
        if texture.texture_data is not None and texture.texture_data.size > 0:
            return scene.load_texture(np.array(texture.texture_data))
        else:
            return scene.load_texture(Path(texture.texture_path))
    elif isinstance(param, int):
        # param is a texture index
        if param < 0 or param >= len(importer_scene.textures):
            raise ValueError(f"Invalid texture index: {param}")
        texture = importer_scene.textures[param]
        if texture.texture_data is not None and texture.texture_data.size > 0:
            return scene.load_texture(np.array(texture.texture_data))
        else:
            return scene.load_texture(Path(texture.texture_path))
    else:
        # Param is a value representing a colour
        return scene.create_texture(16, 16, convert_value_to_rgb(param))


def create_normal_map(scene: Scene, importer_scene: falcor2.ImporterScene, param: Any) -> Tensor:
    if isinstance(param, falcor2.ImporterTexture):
        # Param is already a texture
        texture = param
        if texture.texture_data is not None and texture.texture_data.size > 0:
            return scene.load_normal_map(np.array(texture.texture_data))
        else:
            return scene.load_normal_map(Path(texture.texture_path))
    elif isinstance(param, int):
        # param is a texture index
        if param < 0 or param >= len(importer_scene.textures):
            raise ValueError(f"Invalid texture index: {param}")
        texture = importer_scene.textures[param]
        if texture.texture_data is not None and texture.texture_data.size > 0:
            return scene.load_normal_map(np.array(texture.texture_data))
        else:
            return scene.load_normal_map(Path(texture.texture_path))
    else:
        raise ValueError(f"Unsupported normal map parameter type: {type(param)}")


class TextureCache:
    def __init__(self, scene: Scene, importer_scene: falcor2.ImporterScene):
        super().__init__()
        self.scene = scene
        self.importer_scene = importer_scene
        self.cache: Dict[Tuple[str, Any], Tensor] = {}

    def get_texture(
        self, cat: str, param: Any, convert_srgb: bool = True
    ) -> Tuple[Tensor, Optional[falcor2.ImporterTexture]]:
        """Get texture tensor and the original ImporterTexture if available."""
        if isinstance(param, falcor2.ImporterTexture):
            # Param is already a texture
            texture = param
            key = (cat, texture)
            if key not in self.cache:
                if texture.texture_data is not None and texture.texture_data.size > 0:
                    self.cache[key] = self.scene.load_texture(
                        np.array(texture.texture_data), convert_srgb
                    )
                else:
                    self.cache[key] = self.scene.load_texture(
                        Path(texture.texture_path), convert_srgb
                    )
            return self.cache[key], texture
        elif isinstance(param, int):
            # Param is a texture index - look up the texture and recurse
            if param < 0 or param >= len(self.importer_scene.textures):
                raise ValueError(f"Invalid texture index: {param}")
            return self.get_texture(cat, self.importer_scene.textures[param], convert_srgb)
        else:
            # Param is a string representing a colour - no ImporterTexture available
            key = (cat, param)
            if key not in self.cache:
                self.cache[key] = self.scene.create_texture(16, 16, convert_value_to_rgb(param))
            return self.cache[key], None

    def get_normal_map(self, param: Any) -> Tuple[Tensor, Optional[falcor2.ImporterTexture]]:
        if isinstance(param, falcor2.ImporterTexture):
            # Param is already a texture
            texture = param
            key = ("normal", texture)
            if key not in self.cache:
                if texture.texture_data is not None and texture.texture_data.size > 0:
                    self.cache[key] = self.scene.load_normal_map(np.array(texture.texture_data))
                else:
                    self.cache[key] = self.scene.load_normal_map(Path(texture.texture_path))
            return self.cache[key], texture
        elif isinstance(param, int):
            # Param is a texture index - look up the texture and recurse
            if param < 0 or param >= len(self.scene.textures):
                raise ValueError(f"Invalid texture index: {param}")
            return self.get_normal_map(self.importer_scene.textures[param])
        else:
            raise ValueError(f"Unsupported normal map parameter type: {type(param)}")


def load_full_materials(scene: Scene, importer_scene: falcor2.ImporterScene) -> Dict[str, Material]:
    """Load full materials from importer materials."""

    materials: Dict[str, Material] = {}
    texture_cache = TextureCache(scene, importer_scene)

    def get_material_texture_info(
        cat: str, importer_material: falcor2.ImporterMaterial, param: Any, convert_srgb: bool = True
    ) -> MaterialTexture:
        texture, importer_texture = texture_cache.get_texture(
            cat, importer_material.params[param], convert_srgb
        )

        offset = spy.float2(0.0, 0.0)
        scale = spy.float2(1.0, 1.0)
        rotation = 0.0
        texcoord = 0
        if f"{param}_offset" in importer_material.params:
            offset = importer_material.params.get_float2(f"{param}_offset")
        if f"{param}_scale" in importer_material.params:
            scale = importer_material.params.get_float2(f"{param}_scale")
        if f"{param}_rotation" in importer_material.params:
            rotation = importer_material.params.get_float(f"{param}_rotation")
        if f"{param}_texcoord" in importer_material.params:
            texcoord = importer_material.params.get_int(f"{param}_texcoord")

        # Extract wrap and filter properties from ImporterTexture if available
        if importer_texture is not None:
            wrap_s = importer_texture.wrap_s
            wrap_t = importer_texture.wrap_t
            # For filter, we'll use mag_filter as the primary choice since it's used for the visible texture sampling
            # If mag_filter is not available or is default, fall back to min_filter
            filter_mode = importer_texture.mag_filter
        else:
            # Use defaults when no ImporterTexture is available (e.g., for solid color textures)
            wrap_s = falcor2.TextureWrapMode.repeat
            wrap_t = falcor2.TextureWrapMode.repeat
            filter_mode = falcor2.TextureFilterMode.linear

        return MaterialTexture(
            texture, offset, scale, rotation, texcoord, wrap_s, wrap_t, filter_mode
        )

    def get_material_normalmap_info(
        importer_material: falcor2.ImporterMaterial, param: Any
    ) -> MaterialTexture:
        texture, importer_texture = texture_cache.get_normal_map(importer_material.params[param])

        offset = spy.float2(0.0, 0.0)
        scale = spy.float2(1.0, 1.0)
        rotation = 0.0
        texcoord = 0
        if f"{param}_offset" in importer_material.params:
            offset = importer_material.params.get_float2(f"{param}_offset")
        if f"{param}_scale" in importer_material.params:
            scale = importer_material.params.get_float2(f"{param}_scale")
        if f"{param}_rotation" in importer_material.params:
            rotation = importer_material.params.get_float(f"{param}_rotation")
        if f"{param}_texcoord" in importer_material.params:
            texcoord = importer_material.params.get_int(f"{param}_texcoord")

        # Extract wrap and filter properties from ImporterTexture if available
        if importer_texture is not None:
            wrap_s = importer_texture.wrap_s
            wrap_t = importer_texture.wrap_t
            # For filter, we'll use mag_filter as the primary choice
            filter_mode = importer_texture.mag_filter
        else:
            # Use defaults when no ImporterTexture is available
            wrap_s = falcor2.TextureWrapMode.repeat
            wrap_t = falcor2.TextureWrapMode.repeat
            filter_mode = falcor2.TextureFilterMode.linear

        return MaterialTexture(
            texture, offset, scale, rotation, texcoord, wrap_s, wrap_t, filter_mode
        )

    for importer_material in importer_scene.materials:
        material = scene.create_material(importer_material.name)
        if importer_material.params["_type"] == "usd_UsdPreviewSurface":
            material.albedo = MaterialTexture(scene.black_texture)
            if "diffuseColor" in importer_material.params:
                material.albedo = get_material_texture_info(
                    "srgb", importer_material, "diffuseColor"
                )
            if "emissiveColor" in importer_material.params:
                material.emission = get_material_texture_info(
                    "srgb", importer_material, "emissiveColor"
                )
            if "metallic" in importer_material.params:
                material.metallic = get_material_texture_info(
                    "lingray", importer_material, "metallic"
                )
            if "roughness" in importer_material.params:
                material.roughness = get_material_texture_info(
                    "lingray", importer_material, "roughness"
                )

        # GLTF PBR Metallic Roughness
        if importer_material.params["_type"] == "gltf_pbrMetallicRoughness":

            #
            if "baseColorFactor" in importer_material.params:
                baseColorFactor = importer_material.params.get_float4("baseColorFactor")
            else:
                baseColorFactor = spy.float4(1.0, 1.0, 1.0, 1.0)
            if "baseColorTexture" in importer_material.params:
                material.albedo = get_material_texture_info(
                    "srgb", importer_material, "baseColorTexture"
                )
                # Multiply the albedo texture by the baseColorFactor to get the final albedo
                scene.utils_module["multiply<float4>"](
                    material.albedo.texture, baseColorFactor, _result=material.albedo.texture
                )
            else:
                material.albedo, _ = texture_cache.get_texture("srgb", baseColorFactor)

            # Specular of 0.5 when multiplied by 0.08 in spec-diffuse brdf gives F0 of 0.04
            material.specular, _ = texture_cache.get_texture("srgb", spy.float3(0.5, 0.5, 0.5))

            if "metallicFactor" in importer_material.params:
                metallicFactor = importer_material.params.get_float("metallicFactor")
            else:
                metallicFactor = 1.0

            if "roughnessFactor" in importer_material.params:
                roughnessFactor = importer_material.params.get_float("roughnessFactor")
            else:
                roughnessFactor = 1.0

            if "metallicRoughnessTexture" in importer_material.params:
                metallicRoughness_texture = get_material_texture_info(
                    "lingray", importer_material, "metallicRoughnessTexture", False
                )

                mrtensor = metallicRoughness_texture.texture

                # Multiply the metallicRoughness texture by the factors to get the final colours
                scene.utils_module["multiply<float3>"](
                    mrtensor, spy.float3(0, roughnessFactor, metallicFactor), _result=mrtensor
                )

                metallic_tensor = metallicRoughness_texture.texture.view(
                    shape=(mrtensor.shape[0], mrtensor.shape[1], 1),
                    strides=mrtensor.strides,
                    offset=2,
                )
                roughness_tensor = metallicRoughness_texture.texture.view(
                    shape=(mrtensor.shape[0], mrtensor.shape[1], 1),
                    strides=mrtensor.strides,
                    offset=1,
                )

                material.metallic = MaterialTexture(
                    metallic_tensor,
                    metallicRoughness_texture.offset,
                    metallicRoughness_texture.scale,
                    metallicRoughness_texture.rotation,
                    metallicRoughness_texture.texcoord,
                )
                material.roughness = MaterialTexture(
                    roughness_tensor,
                    metallicRoughness_texture.offset,
                    metallicRoughness_texture.scale,
                    metallicRoughness_texture.rotation,
                    metallicRoughness_texture.texcoord,
                )
            else:
                material.metallic, _ = texture_cache.get_texture("lingray", metallicFactor)
                material.roughness, _ = texture_cache.get_texture("lingray", roughnessFactor)

        if importer_material.params["_type"] == "gltf_pbrSpecularGlossiness":
            material.type = 1

            # First read the 3 factors up front
            if "diffuseFactor" in importer_material.params:
                diffuseFactor = importer_material.params.get_float4("diffuseFactor")
            else:
                diffuseFactor = spy.float4(1.0, 1.0, 1.0, 1.0)
            if "diffuseTexture" in importer_material.params:
                material.albedo = get_material_texture_info(
                    "srgb", importer_material, "diffuseTexture"
                )
                scene.utils_module["multiply<float4>"](
                    material.albedo.texture, diffuseFactor, _result=material.albedo.texture
                )
            else:
                material.albedo, _ = texture_cache.get_texture("srgb", diffuseFactor)

            if "specularFactor" in importer_material.params:
                specularFactor = importer_material.params.get_float3("specularFactor")
            else:
                specularFactor = spy.float3(1.0, 1.0, 1.0)
            if "glossinessFactor" in importer_material.params:
                glossinessFactor = importer_material.params.get_float("glossinessFactor")
            else:
                glossinessFactor = 1.0

            if "specularGlossinessTexture" in importer_material.params:
                specularGlossiness_mat = get_material_texture_info(
                    "srgb", importer_material, "specularGlossinessTexture"
                )
                sgtexture = specularGlossiness_mat.texture

                # Multiply the specularGlossiness texture by the factors to get the final colours
                scene.utils_module["multiply<float4>"](
                    sgtexture, spy.float4(specularFactor, glossinessFactor), _result=sgtexture
                )

                specular_tensor = sgtexture.view(
                    shape=(sgtexture.shape[0], sgtexture.shape[1], 3),
                    strides=sgtexture.strides,
                    offset=0,
                )
                glossiness_tensor = sgtexture.view(
                    shape=(sgtexture.shape[0], sgtexture.shape[1], 1),
                    strides=sgtexture.strides,
                    offset=3,
                )

                material.specular = MaterialTexture(
                    specular_tensor,
                    specularGlossiness_mat.offset,
                    specularGlossiness_mat.scale,
                    specularGlossiness_mat.rotation,
                    specularGlossiness_mat.texcoord,
                )
                material.roughness = MaterialTexture(
                    glossiness_tensor,
                    specularGlossiness_mat.offset,
                    specularGlossiness_mat.scale,
                    specularGlossiness_mat.rotation,
                    specularGlossiness_mat.texcoord,
                )
            else:
                material.specular, _ = texture_cache.get_texture("srgb", specularFactor)
                material.roughness, _ = texture_cache.get_texture("lingray", glossinessFactor)

        # Properties common to both GLTF PBR types
        if importer_material.params["_type"] in [
            "gltf_pbrMetallicRoughness",
            "gltf_pbrSpecularGlossiness",
        ]:
            if "emissiveFactor" in importer_material.params:
                emissiveFactor = importer_material.params.get_float3("emissiveFactor")
            else:
                emissiveFactor = spy.float3(0.0, 0.0, 0.0)

            if "emissiveTexture" in importer_material.params:
                material.emission = get_material_texture_info(
                    "srgb", importer_material, "emissiveTexture"
                )
                # Multiply the emission texture by the emissiveFactor to get the final emission
                scene.utils_module["multiply<float3>"](
                    material.emission.texture, emissiveFactor, _result=material.emission.texture
                )
            else:
                material.emission, _ = texture_cache.get_texture("srgb", emissiveFactor)

            if "normalTexture" in importer_material.params:
                material.normal = get_material_normalmap_info(importer_material, "normalTexture")

            if "normalTexture_strength" in importer_material.params:
                material.normal_map_strength = importer_material.params.get_float(
                    "normalTexture_strength"
                )
            if "doubleSided" in importer_material.params:
                material.double_sided = importer_material.params.get_bool("doubleSided")
            if "alphaMode" in importer_material.params:
                material.alpha_mode = importer_material.params.get_enum(
                    "alphaMode", falcor2.AlphaMode
                )
            if "alphaCutoff" in importer_material.params:
                material.alpha_cutoff = importer_material.params.get_float("alphaCutoff")

        materials[importer_material.name] = material
    return materials


def load_gltf_scene(
    device: Device,
    path: Path,
    trivial_materials: bool = False,
    rescale_to: Optional[float] = None,
    append_to_scene: Optional[Scene] = None,
    scene_from_world: Optional[float4x4] = None,
) -> Scene:
    """Load a GLTF scene and return a Scene object ready for rendering."""
    path = path.absolute()

    gltf_importer = falcor2.GltfImporter()
    start = time.time()
    importer_scene = gltf_importer.load_scene(path)
    if rescale_to is not None:
        importer_scene.calculate_aabbs()
        importer_scene.rescale_to_fit(rescale_to)
    print(
        f"Loaded GLTF scene {path} in {time.time() - start:.2f} seconds, num textures = {len(importer_scene.textures)}"
    )

    return convert_importer_scene_to_scene(
        device,
        importer_scene,
        path,
        trivial_materials=trivial_materials,
        append_to_scene=append_to_scene,
        scene_from_world=scene_from_world,
    )


def load_usd_scene(
    device: Device,
    path: Path,
    trivial_materials: bool = False,
    rescale_to: Optional[float] = None,
    append_to_scene: Optional[Scene] = None,
    scene_from_world: Optional[float4x4] = None,
) -> Scene:
    """Load a USD scene and return a Scene object ready for rendering."""
    path = path.absolute()

    usd_importer = falcor2.UsdImporter()
    start = time.time()
    importer_scene = usd_importer.load_scene(path)
    if rescale_to is not None:
        importer_scene.calculate_aabbs()
        importer_scene.rescale_to_fit(rescale_to)
    print(f"Loaded USD scene in {time.time() - start:.2f} seconds")

    return convert_importer_scene_to_scene(
        device,
        importer_scene,
        path,
        trivial_materials=trivial_materials,
        append_to_scene=append_to_scene,
        scene_from_world=scene_from_world,
    )
