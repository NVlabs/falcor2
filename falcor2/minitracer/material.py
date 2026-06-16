# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Union

from slangpy import BufferCursor, Tensor, float2
import slangpy.math as spym
from falcor2 import TextureFilterMode, TextureWrapMode, AlphaMode

if TYPE_CHECKING:
    from falcor2.minitracer.scene import Scene, SceneParameters


class MaterialTexture:
    def __init__(
        self,
        texture: Tensor,
        offset: float2 = float2(),
        scale: float2 = float2(1.0, 1.0),
        rotation: float = 0.0,
        texcoord: int = 0,
        wrap_s: TextureWrapMode = TextureWrapMode.repeat,
        wrap_t: TextureWrapMode = TextureWrapMode.repeat,
        filter: TextureFilterMode = TextureFilterMode.linear,
    ):
        super().__init__()
        self.texture = texture
        self.offset = offset
        self.scale = scale
        self.rotation = rotation
        self.texcoord = texcoord
        self.wrap_s = wrap_s
        self.wrap_t = wrap_t
        self.filter = filter


TMaterialTextureRef = Union[MaterialTexture, Tensor]


class Material:
    def __init__(self, scene: "Scene", name: str, index: int):
        super().__init__()
        self.type = 0
        self.scene = scene
        self.name = name
        self.index = index
        self.albedo: TMaterialTextureRef = scene.white_texture
        self.specular: TMaterialTextureRef = scene.black_texture
        self.emission: TMaterialTextureRef = scene.black_texture
        self.metallic: TMaterialTextureRef = scene.black_texture
        self.roughness: TMaterialTextureRef = scene.white_texture
        self.normal: TMaterialTextureRef = scene.def_norm_texture
        self.double_sided = False
        self.alpha_mode = AlphaMode.opaque
        self.alpha_cutoff = 0.5

        # TODO[PathTracer]: These don't necessarily need to be differentiable
        # as they just increase/decrease the effect of the texture tensors.
        self.normal_map_strength = 1.0
        self.emission_strength = 1.0

    def get_scene_parameters(self, params: "SceneParameters"):
        for slot, range in [
            ("albedo", (0.0, 1.0)),
            ("specular", (0.0, 1.0)),
            ("emission", (0.0, float("inf"))),
            ("metallic", (0.0, 1.0)),
            ("roughness", (0.05, 1.0)),
            ("normal", (0.0, 1.0)),
        ]:
            name = self.name + "/" + slot
            params._add(
                name,
                getattr(self, slot),
                lambda tensor, self=self, slot=slot: setattr(self, slot, tensor),
                range=range,
            )

    def get_texture_uniforms(self, texture: TMaterialTextureRef):
        if isinstance(texture, MaterialTexture):
            return {
                "texture": self.scene._get_texture_id(texture.texture),
                "transform": spym.mul(
                    spym.matrix_from_translation_2d(texture.offset),
                    spym.mul(
                        spym.matrix_from_rotation_2d(-texture.rotation),  # TODO(ccummings): Why?
                        spym.matrix_from_scaling_2d(texture.scale),
                    ),
                ),
                "tex_coord": texture.texcoord,
                "wrap_s": texture.wrap_s.value,
                "wrap_t": texture.wrap_t.value,
                "filter": texture.filter.value,
            }
        elif isinstance(texture, Tensor):
            return {
                "texture": self.scene._get_texture_id(texture),
                "transform": spym.float3x3.identity(),
                "tex_coord": 0,
                "wrap_s": TextureWrapMode.repeat.value,
                "wrap_t": TextureWrapMode.repeat.value,
                "filter": TextureFilterMode.linear.value,
            }
        else:
            raise TypeError(f"Invalid texture type: {type(texture)}")

    def write(self, material_cursor: BufferCursor):
        material_cursor[self.index].write(
            {
                "type": self.type,
                "albedo": self.get_texture_uniforms(self.albedo),
                "specular": self.get_texture_uniforms(self.specular),
                "emission": self.get_texture_uniforms(self.emission),
                "metallic": self.get_texture_uniforms(self.metallic),
                "roughness": self.get_texture_uniforms(self.roughness),
                "normal": self.get_texture_uniforms(self.normal),
                "normal_map_strength": self.normal_map_strength,
                "emission_strength": self.emission_strength,
                "double_sided": self.double_sided,
                "alpha_mode": self.alpha_mode.value,
                "alpha_cutoff": self.alpha_cutoff,
            }
        )
