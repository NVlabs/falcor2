# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import slangpy as spy
from slangpy import float3, uint3
from slangpy.math import transpose, inverse

from falcor2.minitracer.posrotscale import PosRotScale

if TYPE_CHECKING:
    from falcor2.minitracer.scene import Scene


class EnvMap:
    def __init__(self, scene: "Scene"):
        super().__init__()
        self.scene = scene
        self.texture = scene.white_texture
        self.importance_map: Optional[spy.Texture] = None
        self.transform = PosRotScale()
        self.scaling_factor = float3(1.0)

    def get_uniforms(self):
        m = self.transform.to_float4x4()
        importance_map_dims = (
            uint3(
                self.importance_map.width,
                self.importance_map.height,
                self.importance_map.mip_count,
            )
            if self.importance_map
            else uint3(0)
        )
        d = {
            "texture_id": self.scene._get_texture_id(self.texture),
            "importance_map_dims": importance_map_dims,
            "ws_from_ls": m,
            "ls_from_ws": transpose(inverse(m)),
            "scaling_factor": self.scaling_factor,
        }
        # TODO: we should probably just allow None arguments to be passed to the shader
        if self.importance_map:
            d["importance_map"] = self.importance_map
        return d

    def build_importance_map(self):
        device = self.scene.device

        self.importance_map = device.create_texture(
            format=spy.Format.r32_float,
            width=512,
            height=512,
            mip_count=spy.ALL_MIPS,
            usage=spy.TextureUsage.shader_resource | spy.TextureUsage.unordered_access
            # TODO this is only needed for generate_mips
            | spy.TextureUsage.render_target,
        )

        self.scene.scene_module.build_importance_map.dispatch(
            thread_count=uint3(512, 512, 1),
            texture=self.texture,
            sample_count=16,
            importance_map=self.importance_map,
        )

        command_encoder = device.create_command_encoder()
        command_encoder.generate_mips(self.importance_map)
        device.submit_command_buffer(command_encoder.finish())

        # for mip in range(self.importance_map.mip_count):
        #     bitmap = self.importance_map.to_bitmap(mip=mip)
        #     spy.tev.show(bitmap, f"Importance Map (mip={mip})")
