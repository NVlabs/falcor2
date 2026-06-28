// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "nbdictionary.h"

#include "falcor2/render/texture_manager.h"

#include <sgl/device/device.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace falcor {

using SamplerConfig = TextureManager::SamplerConfig;

FALCOR_DICT_TO_DESC_BEGIN(SamplerConfig)
FALCOR_DICT_TO_DESC_FIELD(min_filter, sgl::TextureFilteringMode)
FALCOR_DICT_TO_DESC_FIELD(mag_filter, sgl::TextureFilteringMode)
FALCOR_DICT_TO_DESC_FIELD(mip_filter, sgl::TextureFilteringMode)
FALCOR_DICT_TO_DESC_FIELD(address_u, sgl::TextureAddressingMode)
FALCOR_DICT_TO_DESC_FIELD(address_v, sgl::TextureAddressingMode)
FALCOR_DICT_TO_DESC_FIELD(address_w, sgl::TextureAddressingMode)
FALCOR_DICT_TO_DESC_FIELD(border_color, float4)
FALCOR_DICT_TO_DESC_END()

} // namespace falcor

FALCOR_PY_EXPORT(render_texture_manager)
{
    using namespace falcor;

    nb::class_<TextureHandle> texture_handle(m, "TextureHandle", D(TextureHandle));
    texture_handle //
        .def(nb::init<>(), D(TextureHandle, TextureHandle))
        .def(nb::self == nb::self)
        .def(nb::self != nb::self)
        .def("is_valid", &TextureHandle::is_valid, D(TextureHandle, is_valid))
        .def("__bool__", &TextureHandle::is_valid, D(TextureHandle, is_valid))
        .def("is_finalized", &TextureHandle::is_finalized, D(TextureHandle, is_finalized))
        .def("is_udim", &TextureHandle::is_udim, D(TextureHandle, is_udim))
        .DEF_PROP_RO(TextureHandle, path)
        .DEF_PROP_RO(TextureHandle, udim_tiles)
        .DEF_PROP_RO(TextureHandle, texture)
        .DEF_PROP_RO(TextureHandle, sampler)
        .DEF_PROP_RO(TextureHandle, data)
        .def(
            "get_this",
            [](TextureHandle* self)
            {
                nb::dict result = NBDictionary::get_uniforms(*self);
                result["_type"] = "TextureHandle";
                return result;
            },
            "Returns the bindings for this texture handle."
        );

    // TODO(scene) there seems to be a bug how pydoc is generated for this nested class!
    nb::class_<TextureHandle::UDIMTile>(texture_handle, "UDIMTile", D(UDIMTile))
        .def_ro("tile_index", &TextureHandle::UDIMTile::tile_index, D(UDIMTile, tile_index))
        .def_ro("tile_u", &TextureHandle::UDIMTile::tile_u, D(UDIMTile, tile_u))
        .def_ro("tile_v", &TextureHandle::UDIMTile::tile_v, D(UDIMTile, tile_v))
        .def_ro("texture_handle", &TextureHandle::UDIMTile::texture_handle, D(UDIMTile, texture_handle));

    nb::class_<TextureManager, Object> texture_manager(m, "TextureManager", D(TextureManager));
    texture_manager //
        .def(nb::init<ref<sgl::Device>>(), "device"_a, D(TextureManager, TextureManager))
        .DEF_PROP_RO(TextureManager, device)
        .DEF_PROP_RO(TextureManager, default_sampler)
        .DEF_PROP_RO(TextureManager, stats)
        .def(
            "load_texture",
            [](TextureManager& self,
               std::filesystem::path path,
               ref<AssetResolver> resolver,
               std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler,
               bool generate_mips,
               bool srgb,
               sgl::TextureUsage usage,
               bool load_deferred)
            {
                return self.load_texture({
                    .path = std::move(path),
                    .resolver = std::move(resolver),
                    .sampler = std::move(sampler),
                    .generate_mips = generate_mips,
                    .srgb = srgb,
                    .usage = usage,
                    .load_deferred = load_deferred,
                });
            },
            "path"_a,
            "resolver"_a.none() = nb::none(),
            "sampler"_a.none() = nb::none(),
            "generate_mips"_a = TextureManager::LoadTextureDesc().generate_mips,
            "srgb"_a = TextureManager::LoadTextureDesc().srgb,
            "usage"_a = TextureManager::LoadTextureDesc().usage,
            "load_deferred"_a = TextureManager::LoadTextureDesc().load_deferred,
            D(TextureManager, load_texture)
        )
        .def(
            "register_texture",
            [](TextureManager& self,
               ref<sgl::Texture> texture,
               std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler)
            {
                return self.register_texture({
                    .texture = std::move(texture),
                    .sampler = std::move(sampler),
                });
            },
            "texture"_a,
            "sampler"_a.none() = nb::none(),
            D(TextureManager, register_texture)
        )
        .def("update", &TextureManager::update, D(TextureManager, update));

    nb::class_<TextureManager::SamplerConfig>(texture_manager, "SamplerConfig", D(TextureManager, SamplerConfig))
        .def(nb::init<>())
        .def(
            "__init__",
            [](TextureManager::SamplerConfig* self, nb::dict dict)
            {
                new (self) TextureManager::SamplerConfig(dict_to_SamplerConfig(dict));
            }
        )
        .DEF_RW_2(TextureManager, SamplerConfig, min_filter)
        .DEF_RW_2(TextureManager, SamplerConfig, mag_filter)
        .DEF_RW_2(TextureManager, SamplerConfig, mip_filter)
        .DEF_RW_2(TextureManager, SamplerConfig, address_u)
        .DEF_RW_2(TextureManager, SamplerConfig, address_v)
        .DEF_RW_2(TextureManager, SamplerConfig, address_w)
        .DEF_RW_2(TextureManager, SamplerConfig, border_color);

    nb::implicitly_convertible<nb::dict, TextureManager::SamplerConfig>();

    nb::class_<TextureManager::Stats>(texture_manager, "Stats", D(TextureManager, Stats))
        .DEF_RO_2(TextureManager, Stats, total_handle_count)
        .DEF_RO_2(TextureManager, Stats, regular_handle_count)
        .DEF_RO_2(TextureManager, Stats, udim_handle_count)
        .DEF_RO_2(TextureManager, Stats, total_texture_count)
        .DEF_RO_2(TextureManager, Stats, regular_texture_count)
        .DEF_RO_2(TextureManager, Stats, udim_texture_count)
        .DEF_RO_2(TextureManager, Stats, total_sampler_count);
}
