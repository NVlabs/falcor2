// SPDX-License-Identifier: Apache-2.0

#include "texture_manager.h"
#include "texture_manager_types.h"

#include "falcor2/core/types.h"

#include <sgl/device/device.h>
#include <sgl/device/sampler.h>
#include <sgl/device/command.h>
#include <sgl/device/cursor_utils.h>
#include <sgl/core/timer.h>
#include <sgl/core/string.h>

namespace falcor {

// -----------------------------------------------------------------------------
// TextureManager
// -----------------------------------------------------------------------------

TextureManager::TextureManager(ref<sgl::Device> device)
    : m_device(std::move(device))
{
    static_assert(TextureManager::INVALID_TEXTURE_HANDLE == shared::detail::INVALID_TEXTURE_HANDLE);

    m_texture_loader.reset(new sgl::TextureLoader(m_device));
    m_default_asset_resolver.reset(new AssetResolver());

    // Create default sampler.
    m_default_sampler = m_device->create_sampler({});
    m_default_sampler_id = get_or_create_sampler(m_default_sampler);
}

TextureManager::~TextureManager() { }

TextureHandle TextureManager::load_texture(const LoadTextureDesc& desc)
{
    // Use default resolver if none is specified.
    AssetResolver* resolver = desc.resolver ? desc.resolver : m_default_asset_resolver.get();

    std::filesystem::path path = desc.path;
    UDIMPaths udim_paths;

    // Check for UDIM token in the filename.
    std::string udim_filename = path.filename().string();
    auto udim_token_pos = udim_filename.find("<UDIM>");
    bool is_udim = (udim_token_pos != std::string::npos);

    // Resolve texture paths.
    if (is_udim) {
        std::string udim_filename_pattern = udim_filename;
        udim_filename_pattern.replace(udim_token_pos, 6, "[1-9][0-9][0-9][0-9]");
        std::vector<std::filesystem::path> paths
            = resolver->resolve_path_pattern(path.parent_path(), udim_filename_pattern, false);
        if (paths.empty()) {
            sgl::log_warn("Failed to resolve UDIM texture path \"{}\". No matching files found.", path);
            return {};
        }
        path = paths.front().parent_path() / udim_filename;
        for (const auto& udim_path : paths) {
            std::string filename = udim_path.filename().string();
            std::string tile_index_str = filename.substr(udim_token_pos, 4);
            uint32_t tile_index = static_cast<uint32_t>(std::stoi(tile_index_str));
            if (tile_index < 1001 || tile_index > 1999) {
                sgl::log_warn("Invalid UDIM tile index {} in file \"{}\".", tile_index, udim_path);
                continue;
            }
            udim_paths.push_back(
                UDIMPath{
                    .tile_index = tile_index,
                    .path = udim_path,
                }
            );
        }
        if (udim_paths.empty()) {
            sgl::log_warn("Failed to resolve UDIM texture path \"{}\". No valid UDIM tiles found.", path);
            return {};
        }
        udim_paths = normalize_udim_paths(udim_paths);
    } else {
        auto resolved_path = resolver->resolve_path(path);
        if (resolved_path.empty()) {
            sgl::log_warn("Failed to resolve texture path \"{}\".", path);
            return {};
        }
        path = resolved_path;
    }

    TextureOptions options{
        .generate_mips = desc.generate_mips,
        .srgb = desc.srgb,
        .usage = desc.usage,
    };

    if (is_udim) {
        // Get or create UDIM tile textures.
        std::vector<TextureID> udim_texture_ids;
        std::vector<HandleID> udim_handle_ids;
        UDIMTiles udim_tiles;
        for (const auto& udim_path : udim_paths) {
            TextureID texture_id = get_or_create_texture(udim_path.path, options);
            SamplerID sampler_id = get_or_create_sampler(desc.sampler);
            HandleID handle_id = get_or_create_handle(texture_id, sampler_id);
            udim_texture_ids.push_back(texture_id);
            udim_handle_ids.push_back(handle_id);
            udim_tiles.push_back(
                UDIMTile{
                    .tile_index = udim_path.tile_index,
                    .tile_u = (udim_path.tile_index - 1001) % 10,
                    .tile_v = (udim_path.tile_index - 1001) / 10,
                    .handle_id = handle_id,
                }
            );
        }
        if (desc.load_deferred) {
            for (TextureID texture_id : udim_texture_ids)
                if (!is_texture_loaded(texture_id))
                    m_pending_textures_to_load.push_back(texture_id);
            for (HandleID handle_id : udim_handle_ids)
                if (!is_handle_finalized(handle_id))
                    m_pending_handles_to_finalize.push_back(handle_id);
        } else {
            load_textures(udim_texture_ids);
            finalize_handles(udim_handle_ids);
        }
        // Create UDIM texture.
        TextureID texture_id = get_or_create_udim_texture(path, udim_paths, udim_tiles, options);
        SamplerID sampler_id = get_or_create_sampler(desc.sampler);
        HandleID handle_id = get_or_create_handle(texture_id, sampler_id);
        if (!is_handle_finalized(handle_id)) {
            if (desc.load_deferred)
                m_pending_udim_handles_to_finalize.push_back(handle_id);
            else
                finalize_udim_handle(handle_id);
        }
        return TextureHandle(this, handle_id);
    } else {
        TextureID texture_id = get_or_create_texture(path, options);
        if (!is_texture_loaded(texture_id)) {
            if (desc.load_deferred)
                m_pending_textures_to_load.push_back(texture_id);
            else
                load_texture(texture_id);
        }
        SamplerID sampler_id = get_or_create_sampler(desc.sampler);
        HandleID handle_id = get_or_create_handle(texture_id, sampler_id);
        if (!is_handle_finalized(handle_id)) {
            if (desc.load_deferred)
                m_pending_handles_to_finalize.push_back(handle_id);
            else
                finalize_handle(handle_id);
        }
        return TextureHandle(this, handle_id);
    }
}

TextureHandle TextureManager::register_texture(const RegisterTextureDesc& desc)
{
    if (!desc.texture)
        return {};
    TextureID texture_id = get_or_create_texture(desc.texture);
    SamplerID sampler_id = get_or_create_sampler(desc.sampler);
    HandleID handle_id = get_or_create_handle(texture_id, sampler_id);
    if (!is_handle_finalized(handle_id))
        finalize_handle(handle_id);
    return TextureHandle(this, handle_id);
}

void TextureManager::update()
{
    // Load pending textures.
    if (!m_pending_textures_to_load.empty()) {
        load_textures(m_pending_textures_to_load);
        m_pending_textures_to_load.clear();
    }

    // Finalize pending handles.
    if (!m_pending_handles_to_finalize.empty()) {
        finalize_handles(m_pending_handles_to_finalize);
        m_pending_handles_to_finalize.clear();
    }

    // Finalize pending handles for UDIM textures.
    // This needs to happen AFTER finalizing regular handles, since those are needed
    // for the UDIM indirection texture creation.
    if (!m_pending_udim_handles_to_finalize.empty()) {
        finalize_udim_handles(m_pending_udim_handles_to_finalize);
        m_pending_udim_handles_to_finalize.clear();
    }

    // Garbage collect unused resources.
    if (m_pending_garbage_collect) {
        run_garbage_collect();
        m_pending_garbage_collect = false;
    }

    // Update statistics if needed.
    if (m_stats_dirty) {
        m_stats = collect_stats();
        m_stats_dirty = false;
    }
}

void TextureManager::inc_handle_ref(HandleID handle_id)
{
    HandleEntry& handle_entry = get_handle_entry(handle_id);
    handle_entry.atomic_ref_count()++;
}

void TextureManager::dec_handle_ref(HandleID handle_id)
{
    HandleEntry& handle_entry = get_handle_entry(handle_id);
    uint32_t prev_ref_count = handle_entry.atomic_ref_count().fetch_sub(1);
    FALCOR_CHECK_GT(prev_ref_count, 0);
    if (prev_ref_count == 1)
        m_pending_garbage_collect = true;
}

bool TextureManager::is_handle_finalized(HandleID handle_id) const
{
    return get_handle_entry(handle_id).is_finalized;
}

bool TextureManager::is_handle_udim(HandleID handle_id) const
{
    const HandleEntry& handle_entry = get_handle_entry(handle_id);
    const TextureEntry& texture_entry = get_texture_entry(handle_entry.info.texture_id);
    return texture_entry.udim_tiles.size() > 0;
}

const std::filesystem::path* TextureManager::get_handle_path(HandleID handle_id) const
{
    const HandleEntry& handle_entry = get_handle_entry(handle_id);
    const TextureEntry& texture_entry = get_texture_entry(handle_entry.info.texture_id);
    return texture_entry.info ? &texture_entry.info->path : nullptr;
}

ref<sgl::Texture> TextureManager::get_handle_texture(HandleID handle_id) const
{
    const HandleEntry& handle_entry = get_handle_entry(handle_id);
    return get_texture_entry(handle_entry.info.texture_id).texture;
}

ref<sgl::Sampler> TextureManager::get_handle_sampler(HandleID handle_id) const
{
    const HandleEntry& handle_entry = get_handle_entry(handle_id);
    return get_sampler_entry(handle_entry.info.sampler_id).sampler;
}

uint32_t TextureManager::get_handle_data(HandleID handle_id) const
{
    return get_handle_entry(handle_id).data;
}

void TextureManager::finalize_handle(HandleID handle_id)
{
    HandleEntry& handle_entry = get_handle_entry(handle_id);
    if (handle_entry.is_finalized)
        return;

    const TextureEntry& texture_entry = get_texture_entry(handle_entry.info.texture_id);
    const SamplerEntry& sampler_entry = get_sampler_entry(handle_entry.info.sampler_id);

    uint32_t data = 0;
    uint32_t is_udim = texture_entry.udim_tiles.size() > 0;

    // CUDA doesn't support independent sampler objects.
    // CUDA texture objects always have a sampler associated to them.
    // We first need to create a texture view with the required sampler to get a combined texture/sampler descriptor.
    // On CUDA we use more bits for encoding the texture descriptor as the sampler descriptor doesn't need to be stored.
    if (m_device->type() == sgl::DeviceType::cuda) {
        handle_entry.texture_view = texture_entry.texture->create_view({
            .sampler = sampler_entry.sampler,
        });
        uint64_t texture_descriptor = handle_entry.texture_view->descriptor_handle_ro().value;
        FALCOR_CHECK(
            texture_descriptor <= shared::detail::TextureHandleFieldsCUDA::texture::MAX_VALUE,
            "Texture descriptor handle value exceeds maximum representable index."
        );
        shared::detail::TextureHandleFieldsCUDA::texture::insert(data, static_cast<uint32_t>(texture_descriptor));
        shared::detail::TextureHandleFieldsCUDA::udim::insert(data, is_udim);
    } else {
        uint64_t texture_descriptor = texture_entry.texture->descriptor_handle_ro().value;
        FALCOR_CHECK(
            texture_descriptor <= shared::detail::TextureHandleFieldsNonCUDA::texture::MAX_VALUE,
            "Texture descriptor handle value exceeds maximum representable index."
        );
        uint64_t sampler_descriptor = sampler_entry.sampler->descriptor_handle().value;
        FALCOR_CHECK(
            sampler_descriptor <= shared::detail::TextureHandleFieldsNonCUDA::sampler::MAX_VALUE,
            "Sampler descriptor handle value exceeds maximum representable index."
        );
        shared::detail::TextureHandleFieldsNonCUDA::texture::insert(data, static_cast<uint32_t>(texture_descriptor));
        shared::detail::TextureHandleFieldsNonCUDA::sampler::insert(data, static_cast<uint32_t>(sampler_descriptor));
        shared::detail::TextureHandleFieldsNonCUDA::udim::insert(data, is_udim);
    }

    handle_entry.data = data;
    handle_entry.is_finalized = true;
}

void TextureManager::finalize_handles(std::span<HandleID> handle_ids)
{
    for (HandleID handle_id : handle_ids)
        finalize_handle(handle_id);
}

void TextureManager::finalize_udim_handle(HandleID handle_id)
{
    HandleEntry& handle_entry = get_handle_entry(handle_id);
    if (handle_entry.is_finalized)
        return;

    TextureEntry& texture_entry = get_texture_entry(handle_entry.info.texture_id);
    if (!texture_entry.is_loaded) {
        texture_entry.texture = create_udim_indirection_texture(texture_entry.udim_tiles);
        texture_entry.is_loaded = true;
    }

    finalize_handle(handle_id);
}

void TextureManager::finalize_udim_handles(std::span<HandleID> handle_ids)
{
    for (HandleID handle_id : handle_ids)
        finalize_udim_handle(handle_id);
}

bool TextureManager::is_texture_loaded(TextureID texture_id) const
{
    return get_texture_entry(texture_id).is_loaded;
}

void TextureManager::load_texture(TextureID texture_id)
{
    TextureEntry& texture_entry = get_texture_entry(texture_id);
    if (texture_entry.is_loaded)
        return;

    sgl::Timer timer;

    FALCOR_ASSERT(texture_entry.info);
    texture_entry.texture = m_texture_loader->load_texture(
        texture_entry.info->path,
        sgl::TextureLoader::Options{
            .load_as_srgb = texture_entry.info->options.srgb,
            .generate_mips = texture_entry.info->options.generate_mips,
            .usage = texture_entry.info->options.usage,
        }
    );
    texture_entry.is_loaded = true;
    m_texture_to_texture_id.emplace(texture_entry.texture.get(), texture_id);
    notify_textures_loaded(std::span(&texture_id, 1), timer.elapsed_s());
}

void TextureManager::load_textures(std::span<const TextureID> texture_ids)
{
    std::vector<std::filesystem::path> paths;
    std::vector<sgl::TextureLoader::Options> options;
    std::vector<TextureEntry*> texture_entries;
    for (TextureID texture_id : texture_ids) {
        TextureEntry& texture_entry = get_texture_entry(texture_id);
        if (texture_entry.is_loaded)
            continue;
        FALCOR_ASSERT(texture_entry.info);
        paths.push_back(texture_entry.info->path);
        options.push_back(
            sgl::TextureLoader::Options{
                .load_as_srgb = texture_entry.info->options.srgb,
                .generate_mips = texture_entry.info->options.generate_mips,
                .usage = texture_entry.info->options.usage,
            }
        );
        texture_entries.push_back(&texture_entry);
    }

    if (paths.empty())
        return;

    sgl::Timer timer;

    std::vector<ref<sgl::Texture>> textures = m_texture_loader->load_textures(paths, options);

    FALCOR_ASSERT(textures.size() == texture_entries.size());
    for (size_t i = 0; i < textures.size(); ++i) {
        texture_entries[i]->texture = textures[i];
        texture_entries[i]->is_loaded = true;
        m_texture_to_texture_id.emplace(texture_entries[i]->texture.get(), texture_ids[i]);
    }

    notify_textures_loaded(texture_ids, timer.elapsed_s());
}

void TextureManager::notify_textures_loaded(std::span<const TextureID> texture_ids, double duration_s)
{
    size_t total_disk_size = 0;
    size_t total_device_size = 0;

    for (TextureID texture_id : texture_ids) {
        const TextureEntry& texture_entry = get_texture_entry(texture_id);
        total_disk_size += std::filesystem::file_size(texture_entry.info->path);
        total_device_size += texture_entry.texture->memory_usage().device;
        sgl::log_info(
            "Loaded texture \"{}\" (disk size: {}, device size: {})",
            texture_entry.info->path.string(),
            sgl::string::format_byte_size(std::filesystem::file_size(texture_entry.info->path)),
            sgl::string::format_byte_size(texture_entry.texture->memory_usage().device)
        );
    }

    sgl::log_info(
        "Loaded {} texture(s) (total disk size: {}, total device size: {}) in {}",
        texture_ids.size(),
        sgl::string::format_byte_size(total_disk_size),
        sgl::string::format_byte_size(total_device_size),
        sgl::string::format_duration(duration_s)
    );
}

TextureManager::UDIMPaths TextureManager::normalize_udim_paths(const UDIMPaths& paths)
{
    UDIMPaths normalized_paths = paths;
    std::sort(
        normalized_paths.begin(),
        normalized_paths.end(),
        [](const auto& a, const auto& b)
        {
            return a.tile_index < b.tile_index;
        }
    );
    return normalized_paths;
}

ref<sgl::Texture> TextureManager::create_udim_indirection_texture(const UDIMTiles& udim_tiles)
{
    // Figure out the size of the indirection texture.
    uint32_t width = 0;
    uint32_t height = 0;
    for (const auto& udim_tile : udim_tiles) {
        width = std::max(width, udim_tile.tile_u + 1);
        height = std::max(height, udim_tile.tile_v + 1);
    }
    FALCOR_ASSERT_GT(width, 0);
    FALCOR_ASSERT_GT(height, 0);
    // Create indirection data.
    std::vector<uint32_t> indirection_data(width * height, shared::detail::INVALID_TEXTURE_HANDLE);
    for (const auto& udim_tile : udim_tiles) {
        indirection_data[udim_tile.tile_v * width + udim_tile.tile_u] = get_handle_data(udim_tile.handle_id);
    }
    // Create indirection texture.
    sgl::SubresourceData init_data{
        .data = indirection_data.data(),
        .size = indirection_data.size() * sizeof(uint32_t),
        .row_pitch = width * sizeof(uint32_t),
        .slice_pitch = width * height * sizeof(uint32_t),
    };
    return m_device->create_texture({
        .type = sgl::TextureType::texture_2d,
        .format = sgl::Format::r32_uint,
        .width = width,
        .height = height,
        .usage = sgl::TextureUsage::shader_resource,
        .data = std::span(&init_data, 1),
    });
}

TextureManager::HandleID TextureManager::get_or_create_handle(TextureID texture_id, SamplerID sampler_id)
{
    FALCOR_ASSERT(texture_id != TextureID::invalid);
    FALCOR_ASSERT(sampler_id != SamplerID::invalid);

    HandleInfo info{
        .texture_id = texture_id,
        .sampler_id = sampler_id,
    };

    auto it = m_handle_info_to_handle_id.find(info);
    if (it != m_handle_info_to_handle_id.end())
        return it->second;

    HandleID id = allocate_handle();
    uint32_t index = static_cast<uint32_t>(id);
    m_handle_entries[index].info = info;
    m_handle_info_to_handle_id.emplace(info, id);
    return id;
}

TextureManager::HandleEntry& TextureManager::get_handle_entry(HandleID handle_id)
{
    uint32_t index = static_cast<uint32_t>(handle_id);
    FALCOR_ASSERT_LT(index, m_handle_entries.size());
    HandleEntry& entry = m_handle_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

const TextureManager::HandleEntry& TextureManager::get_handle_entry(HandleID handle_id) const
{
    uint32_t index = static_cast<uint32_t>(handle_id);
    FALCOR_ASSERT_LT(index, m_handle_entries.size());
    const HandleEntry& entry = m_handle_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

TextureManager::HandleID TextureManager::allocate_handle()
{
    uint32_t index = 0;
    if (m_free_handle_indices.empty()) {
        index = static_cast<uint32_t>(m_handle_entries.size());
        m_handle_entries.resize(index + 1);
    } else {
        index = m_free_handle_indices.back();
        m_free_handle_indices.pop_back();
    }
    m_handle_entries[index] = {
        .allocated = true,
    };
    m_stats_dirty = true;
    return HandleID{index};
}

void TextureManager::free_handle(HandleID handle_id)
{
    uint32_t index = static_cast<uint32_t>(handle_id);
    FALCOR_ASSERT_LT(index, m_handle_entries.size());
    HandleEntry& entry = m_handle_entries[index];
    FALCOR_ASSERT(entry.allocated);

    m_handle_info_to_handle_id.erase(entry.info);

    entry = {};
    m_free_handle_indices.push_back(index);
    m_stats_dirty = true;
}

TextureManager::TextureID TextureManager::get_or_create_texture(ref<sgl::Texture> texture)
{
    FALCOR_ASSERT(texture != nullptr);

    auto it = m_texture_to_texture_id.find(texture.get());
    if (it != m_texture_to_texture_id.end())
        return it->second;

    TextureID texture_id = allocate_texture();
    TextureEntry& texture_entry = m_texture_entries[static_cast<uint32_t>(texture_id)];
    texture_entry.texture = texture;
    texture_entry.is_loaded = true;
    m_texture_to_texture_id.emplace(texture.get(), texture_id);
    return texture_id;
}

TextureManager::TextureID
TextureManager::get_or_create_texture(const std::filesystem::path& path, const TextureOptions& options)
{
    TextureInfo info{
        .path = path,
        .options = options,
    };

    auto it = m_texture_info_to_texture_id.find(info);
    if (it != m_texture_info_to_texture_id.end())
        return it->second;

    TextureID texture_id = allocate_texture();
    TextureEntry& texture_entry = m_texture_entries[static_cast<uint32_t>(texture_id)];
    texture_entry.info = info;
    m_texture_info_to_texture_id.emplace(info, texture_id);

    return texture_id;
}

TextureManager::TextureID TextureManager::get_or_create_udim_texture(
    const std::filesystem::path& path,
    const UDIMPaths& udim_paths,
    const UDIMTiles& udim_tiles,
    const TextureOptions& options
)
{
    TextureInfo info{
        .path = path,
        .udim_paths = udim_paths,
        .options = options,
    };

    auto it = m_texture_info_to_texture_id.find(info);
    if (it != m_texture_info_to_texture_id.end())
        return it->second;

    TextureID texture_id = allocate_texture();
    TextureEntry& texture_entry = m_texture_entries[static_cast<uint32_t>(texture_id)];
    texture_entry.info = info;
    texture_entry.udim_tiles = udim_tiles;
    m_texture_info_to_texture_id.emplace(info, texture_id);

    return texture_id;
}

TextureManager::TextureEntry& TextureManager::get_texture_entry(TextureID texture_id)
{
    uint32_t index = static_cast<uint32_t>(texture_id);
    FALCOR_ASSERT_LT(index, m_texture_entries.size());
    TextureEntry& entry = m_texture_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

const TextureManager::TextureEntry& TextureManager::get_texture_entry(TextureID texture_id) const
{
    uint32_t index = static_cast<uint32_t>(texture_id);
    FALCOR_ASSERT_LT(index, m_texture_entries.size());
    const TextureEntry& entry = m_texture_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

TextureManager::TextureID TextureManager::allocate_texture()
{
    uint32_t index = 0;
    if (m_free_texture_indices.empty()) {
        index = static_cast<uint32_t>(m_texture_entries.size());
        m_texture_entries.resize(index + 1);
    } else {
        index = m_free_texture_indices.back();
        m_free_texture_indices.pop_back();
    }
    m_texture_entries[index] = {
        .allocated = true,
    };
    m_stats_dirty = true;
    return TextureID{index};
}

void TextureManager::free_texture(TextureID texture_id)
{
    uint32_t index = static_cast<uint32_t>(texture_id);
    FALCOR_ASSERT_LT(index, m_texture_entries.size());
    TextureEntry& entry = m_texture_entries[index];
    FALCOR_ASSERT(entry.allocated);

    if (entry.texture)
        m_texture_to_texture_id.erase(entry.texture.get());
    if (entry.info)
        m_texture_info_to_texture_id.erase(*entry.info);

    entry = {};
    m_free_texture_indices.push_back(index);
    m_stats_dirty = true;
}

TextureManager::SamplerID
TextureManager::get_or_create_sampler(std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler)
{
    if (std::holds_alternative<std::monostate>(sampler))
        return m_default_sampler_id;
    else if (std::holds_alternative<SamplerConfig>(sampler))
        return get_or_create_sampler(std::get<SamplerConfig>(sampler));
    else if (std::holds_alternative<ref<sgl::Sampler>>(sampler))
        return get_or_create_sampler(std::get<ref<sgl::Sampler>>(sampler));
    else
        FALCOR_UNREACHABLE();
}

TextureManager::SamplerID TextureManager::get_or_create_sampler(ref<sgl::Sampler> sampler)
{
    FALCOR_ASSERT(sampler != nullptr);

    auto it = m_sampler_to_sampler_id.find(sampler.get());
    if (it != m_sampler_to_sampler_id.end())
        return it->second;

    SamplerID sampler_id = allocate_sampler();
    SamplerEntry& sampler_entry = m_sampler_entries[static_cast<uint32_t>(sampler_id)];
    sampler_entry.sampler = sampler;
    m_sampler_to_sampler_id.emplace(sampler.get(), sampler_id);
    return sampler_id;
}

TextureManager::SamplerID TextureManager::get_or_create_sampler(const SamplerConfig& sampler_config)
{
    auto it = m_sampler_config_to_sampler_id.find(sampler_config);
    if (it != m_sampler_config_to_sampler_id.end())
        return it->second;

    SamplerID sampler_id = allocate_sampler();
    SamplerEntry& sampler_entry = m_sampler_entries[static_cast<uint32_t>(sampler_id)];
    sampler_entry.sampler = m_device->create_sampler({
        .min_filter = sampler_config.min_filter,
        .mag_filter = sampler_config.mag_filter,
        .mip_filter = sampler_config.mip_filter,
        .address_u = sampler_config.address_u,
        .address_v = sampler_config.address_v,
        .address_w = sampler_config.address_w,
        .border_color = sampler_config.border_color,
    });
    sampler_entry.sampler_config = sampler_config;
    m_sampler_config_to_sampler_id.emplace(sampler_config, sampler_id);
    return sampler_id;
}

TextureManager::SamplerEntry& TextureManager::get_sampler_entry(SamplerID sampler_id)
{
    uint32_t index = static_cast<uint32_t>(sampler_id);
    FALCOR_ASSERT_LT(index, m_sampler_entries.size());
    SamplerEntry& entry = m_sampler_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

const TextureManager::SamplerEntry& TextureManager::get_sampler_entry(SamplerID sampler_id) const
{
    uint32_t index = static_cast<uint32_t>(sampler_id);
    FALCOR_ASSERT_LT(index, m_sampler_entries.size());
    const SamplerEntry& entry = m_sampler_entries[index];
    FALCOR_ASSERT(entry.allocated);
    return entry;
}

TextureManager::SamplerID TextureManager::allocate_sampler()
{
    uint32_t index = 0;
    if (m_free_sampler_indices.empty()) {
        index = static_cast<uint32_t>(m_sampler_entries.size());
        m_sampler_entries.resize(index + 1);
    } else {
        index = m_free_sampler_indices.back();
        m_free_sampler_indices.pop_back();
    }
    m_sampler_entries[index] = {
        .allocated = true,
    };
    m_stats_dirty = true;
    return SamplerID{index};
}

void TextureManager::free_sampler(SamplerID sampler_id)
{
    uint32_t index = static_cast<uint32_t>(sampler_id);
    FALCOR_ASSERT_LT(index, m_sampler_entries.size());
    SamplerEntry& entry = m_sampler_entries[index];
    FALCOR_ASSERT(entry.allocated);

    if (entry.sampler)
        m_sampler_to_sampler_id.erase(entry.sampler.get());
    if (entry.sampler_config)
        m_sampler_config_to_sampler_id.erase(entry.sampler_config.value());

    entry = {};
    m_free_sampler_indices.push_back(index);
    m_stats_dirty = true;
}

void TextureManager::run_garbage_collect()
{
    // Reset marked resources.
    m_handles_marked.assign(m_handle_entries.size(), false);
    m_texture_marked.assign(m_texture_entries.size(), false);
    m_sampler_marked.assign(m_sampler_entries.size(), false);

    // Mark default sampler as used.
    m_sampler_marked[static_cast<uint32_t>(m_default_sampler_id)] = true;

    // Mark resources from active handles.
    for (size_t i = 0; i < m_handle_entries.size(); ++i) {
        HandleEntry& handle_entry = m_handle_entries[i];
        if (handle_entry.allocated && handle_entry.atomic_ref_count().load() > 0) {
            FALCOR_ASSERT(handle_entry.info.texture_id != TextureID::invalid);
            m_handles_marked[static_cast<uint32_t>(i)] = true;
            m_texture_marked[static_cast<uint32_t>(handle_entry.info.texture_id)] = true;
            const TextureEntry& texture_entry = m_texture_entries[static_cast<uint32_t>(handle_entry.info.texture_id)];
            // For UDIMs, mark all associated handles.
            for (const auto& udim_tile : texture_entry.udim_tiles) {
                m_handles_marked[static_cast<uint32_t>(udim_tile.handle_id)] = true;
                m_texture_marked[static_cast<uint32_t>(m_handle_entries[static_cast<uint32_t>(udim_tile.handle_id)]
                                                           .info.texture_id)]
                    = true;
            }
            FALCOR_ASSERT(handle_entry.info.sampler_id != SamplerID::invalid);
            m_sampler_marked[static_cast<uint32_t>(handle_entry.info.sampler_id)] = true;
        }
    }

    // Garbage collect handles.
    for (size_t i = 0; i < m_handle_entries.size(); ++i)
        if (!m_handles_marked[i] && m_handle_entries[i].allocated && m_handle_entries[i].atomic_ref_count().load() == 0)
            free_handle(HandleID{static_cast<uint32_t>(i)});

    // Garbage collect textures.
    for (size_t i = 0; i < m_texture_entries.size(); ++i)
        if (!m_texture_marked[i] && m_texture_entries[i].allocated)
            free_texture(TextureID{static_cast<uint32_t>(i)});

    // Garbage collect samplers.
    for (size_t i = 0; i < m_sampler_entries.size(); ++i)
        if (!m_sampler_marked[i] && m_sampler_entries[i].allocated)
            free_sampler(SamplerID{static_cast<uint32_t>(i)});

    // Check that all marked handles are still valid.
    for (size_t i = 0; i < m_handles_marked.size(); ++i) {
        if (m_handles_marked[i]) {
            FALCOR_ASSERT(m_handle_entries[i].allocated);
        }
    }

    // Check that all marked textures are still valid.
    for (size_t i = 0; i < m_texture_entries.size(); ++i) {
        if (m_texture_marked[i]) {
            FALCOR_ASSERT(m_texture_entries[i].allocated);
            FALCOR_ASSERT(m_texture_entries[i].texture != nullptr);
        }
    }

    // Check that all marked samplers are still valid.
    for (size_t i = 0; i < m_sampler_entries.size(); ++i) {
        if (m_sampler_marked[i]) {
            FALCOR_ASSERT(m_sampler_entries[i].allocated);
            FALCOR_ASSERT(m_sampler_entries[i].sampler != nullptr);
        }
    }
}

TextureManager::Stats TextureManager::collect_stats() const
{
    Stats stats;

    for (size_t i = 0; i < m_handle_entries.size(); ++i) {
        const HandleEntry& handle_entry = m_handle_entries[i];
        if (handle_entry.allocated) {
            const TextureEntry& texture_entry = m_texture_entries[static_cast<size_t>(handle_entry.info.texture_id)];
            stats.total_handle_count++;
            if (texture_entry.udim_tiles.size() > 0)
                stats.udim_handle_count++;
            else
                stats.regular_handle_count++;
        }
    }
    for (size_t i = 0; i < m_texture_entries.size(); ++i) {
        const TextureEntry& texture_entry = m_texture_entries[i];
        if (texture_entry.allocated) {
            stats.total_texture_count++;
            if (texture_entry.udim_tiles.size() > 0)
                stats.udim_texture_count++;
            else
                stats.regular_texture_count++;
        }
    }
    for (size_t i = 0; i < m_sampler_entries.size(); ++i) {
        const SamplerEntry& sampler_entry = m_sampler_entries[i];
        if (sampler_entry.allocated)
            stats.total_sampler_count++;
    }

    return stats;
}

// -----------------------------------------------------------------------------
// TextureHandle
// -----------------------------------------------------------------------------

TextureHandle::TextureHandle(TextureManager* texture_manager, TextureManager::HandleID handle_id)
    : m_texture_manager(ref<TextureManager>(texture_manager))
    , m_handle_id(handle_id)
{
    FALCOR_ASSERT(texture_manager);
    texture_manager->inc_handle_ref(m_handle_id);
}

TextureHandle::~TextureHandle()
{
    if (m_texture_manager)
        m_texture_manager->dec_handle_ref(m_handle_id);
}

TextureHandle::TextureHandle(const TextureHandle& other)
{
    m_texture_manager = other.m_texture_manager;
    m_handle_id = other.m_handle_id;
    if (m_texture_manager)
        m_texture_manager->inc_handle_ref(m_handle_id);
}

TextureHandle::TextureHandle(TextureHandle&& other)
{
    m_texture_manager = std::move(other.m_texture_manager);
    m_handle_id = other.m_handle_id;
    other.m_handle_id = TextureManager::HandleID::invalid;
}

TextureHandle& TextureHandle::operator=(const TextureHandle& other)
{
    if (this != &other) {
        if (m_texture_manager)
            m_texture_manager->dec_handle_ref(m_handle_id);
        m_texture_manager = other.m_texture_manager;
        m_handle_id = other.m_handle_id;
        if (m_texture_manager)
            m_texture_manager->inc_handle_ref(m_handle_id);
    }
    return *this;
}

TextureHandle& TextureHandle::operator=(TextureHandle&& other)
{
    if (this != &other) {
        if (m_texture_manager)
            m_texture_manager->dec_handle_ref(m_handle_id);
        m_texture_manager = std::move(other.m_texture_manager);
        m_handle_id = other.m_handle_id;
        other.m_handle_id = TextureManager::HandleID::invalid;
    }
    return *this;
}

bool TextureHandle::is_finalized() const
{
    return m_texture_manager ? m_texture_manager->is_handle_finalized(m_handle_id) : false;
}

bool TextureHandle::is_udim() const
{
    return m_texture_manager ? m_texture_manager->is_handle_udim(m_handle_id) : false;
}

const std::filesystem::path* TextureHandle::path() const
{
    return m_texture_manager ? m_texture_manager->get_handle_path(m_handle_id) : nullptr;
}

ref<sgl::Texture> TextureHandle::texture() const
{
    return m_texture_manager ? m_texture_manager->get_handle_texture(m_handle_id) : nullptr;
}

ref<sgl::Sampler> TextureHandle::sampler() const
{
    return m_texture_manager ? m_texture_manager->get_handle_sampler(m_handle_id) : nullptr;
}

std::vector<TextureHandle::UDIMTile> TextureHandle::udim_tiles() const
{
    std::vector<TextureHandle::UDIMTile> tiles;
    if (m_texture_manager) {
        const TextureManager::HandleEntry& handle_entry = m_texture_manager->get_handle_entry(m_handle_id);
        const TextureManager::TextureEntry& texture_entry
            = m_texture_manager->get_texture_entry(handle_entry.info.texture_id);
        for (const auto& udim_tile : texture_entry.udim_tiles) {
            tiles.push_back(
                UDIMTile{
                    .tile_index = udim_tile.tile_index,
                    .tile_u = udim_tile.tile_u,
                    .tile_v = udim_tile.tile_v,
                    .texture_handle = TextureHandle(m_texture_manager, udim_tile.handle_id),
                }
            );
        }
    }
    return tiles;
}

uint32_t TextureHandle::data() const
{
    return m_texture_manager ? m_texture_manager->get_handle_data(m_handle_id) : TextureManager::INVALID_TEXTURE_HANDLE;
}

FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<TextureHandle>());

} // namespace falcor
