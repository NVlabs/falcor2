// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/asset_resolver.h"

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/object.h"
#include "falcor2/core/types.h"

#include "falcor2/utils/idictionary.h"

#include <sgl/device/fwd.h>
#include <sgl/device/types.h>
#include <sgl/device/resource.h>
#include <sgl/device/shader_cursor.h>
#include <sgl/utils/texture_loader.h>

#include <atomic>
#include <vector>
#include <map>

namespace falcor {

class TextureHandle;

/// Manages textures and samplers for rendering.
///
/// The texture manager handles loading textures from files, registering existing textures,
/// and creating handles that can be used on the device. It supports deferred loading, UDIM textures,
/// and automatic garbage collection of unused resources.
///
/// ## Bindless Design
///
/// The texture manager uses a fully bindless design where textures and samplers are accessed
/// via descriptor indices rather than bound resource slots. This allows shaders to access any
/// texture without rebinding, enabling efficient rendering of scenes with many materials/textures.
///
/// Each TextureHandle is represented on the device as a 32-bit packed value. The bit layout
/// differs between CUDA and non-CUDA backends:
///
/// D3D12/Vulkan:
/// - Bits 0-19: Texture descriptor index (supports up to ~1M textures)
/// - Bits 20-29: Sampler descriptor index (supports up to ~1K samplers)
/// - Bit 31: UDIM flag (indicates the texture is a UDIM indirection texture)
///
/// CUDA:
/// - Bits 0-29: Combined texture/sampler descriptor index
/// - Bit 31: UDIM flag
///
/// CUDA does not support independent sampler objects - texture objects always have an
/// associated sampler. To handle this, the texture manager creates a texture view with
/// the required sampler settings, yielding a combined texture/sampler descriptor. This
/// allows more bits for the texture index since no separate sampler index is needed.
///
/// Shaders use this packed handle to look up the texture (and sampler on non-CUDA) from
/// global descriptor arrays, allowing texture sampling without any binding changes.
///
/// ## UDIM Support
///
/// UDIM textures are represented using a two-level indirection scheme:
/// 1. The main handle points to a small indirection texture (r32_uint format)
/// 2. Each texel in the indirection texture contains another packed handle for the actual tile
/// 3. Shaders use the UV coordinates to index into the indirection texture, then use the
///    resulting handle to sample the actual tile texture
///
/// ## Resource Lifetime
///
/// Resources are reference-counted through TextureHandle instances. When all handles to a
/// texture/sampler pair are released, the resources become eligible for garbage collection
/// during the next update() call. This allows efficient resource reuse while ensuring
/// resources stay alive as long as they're needed.
///
/// ## Thread Safety
///
/// TextureManager is currently not thread-safe.
///
class FALCOR_API TextureManager : public Object {
    FALCOR_OBJECT(TextureManager)
public:
    /// Configuration for a texture sampler.
    struct SamplerConfig {
        /// Minification filter.
        sgl::TextureFilteringMode min_filter{sgl::TextureFilteringMode::linear};
        /// Magnification filter.
        sgl::TextureFilteringMode mag_filter{sgl::TextureFilteringMode::linear};
        /// Mipmapping filter.
        sgl::TextureFilteringMode mip_filter{sgl::TextureFilteringMode::linear};
        /// Addressing mode for U coordinate.
        sgl::TextureAddressingMode address_u{sgl::TextureAddressingMode::wrap};
        /// Addressing mode for V coordinate.
        sgl::TextureAddressingMode address_v{sgl::TextureAddressingMode::wrap};
        /// Addressing mode for W coordinate.
        sgl::TextureAddressingMode address_w{sgl::TextureAddressingMode::wrap};
        /// Border color used when any address mode is clamp_to_border.
        float4 border_color{0.f};

        auto operator<=>(const SamplerConfig&) const = default;
    };

    /// Parameters for loading a texture from file.
    struct LoadTextureDesc {
        /// Path to the texture file.
        std::filesystem::path path;
        /// Asset resolver to use for resolving relative paths (optional).
        ref<AssetResolver> resolver;
        /// Sampler or sampler configuration (optional).
        std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler;
        /// Generate mipmaps for the texture.
        bool generate_mips{true};
        /// Treat the texture as sRGB.
        bool srgb{true};
        /// Texture usage flags.
        sgl::TextureUsage usage{sgl::TextureUsage::shader_resource};
        /// Defer texture loading until `update()` is called.
        bool load_deferred{false};
    };

    /// Parameters for registering a texture.
    struct RegisterTextureDesc {
        /// Texture to register.
        ref<sgl::Texture> texture;
        /// Sampler or sampler configuration (optional).
        std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler;
    };

    /// Statistics.
    struct Stats {
        /// Total number of texture handles.
        uint32_t total_handle_count{0};
        /// Number of texture handles representing regular (non-UDIM) textures.
        uint32_t regular_handle_count{0};
        /// Number of texture handles representing UDIM textures.
        uint32_t udim_handle_count{0};
        /// Total number textures.
        uint32_t total_texture_count{0};
        /// Number of regular (non-UDIM) textures.
        uint32_t regular_texture_count{0};
        /// Number of UDIM indirection textures.
        uint32_t udim_texture_count{0};
        /// Total number of samplers.
        uint32_t total_sampler_count{0};
    };

    TextureManager(ref<sgl::Device> device);
    ~TextureManager();

    /// Device associated with the texture manager.
    sgl::Device* device() const { return m_device; }

    /// Default texture sampler.
    sgl::Sampler* default_sampler() const { return m_default_sampler; }

    /// Statistics.
    /// These are only updated during `update()` calls.
    const Stats& stats() const { return m_stats; }

    /// Load a texture from file and create a texture handle for it.
    /// Uses the default sampler if no sampler is specified.
    /// If the texture/sampler is already registered, returns the existing handle.
    /// @param path Path to the texture file.
    /// @param resolver Asset resolver to use for resolving relative paths (optional).
    /// @param sampler Sampler or sampler configuration (optional).
    /// @param generate_mips Generate mipmaps for the texture.
    /// @param srgb Treat the texture as sRGB.
    /// @param usage Texture usage flags.
    /// @param load_deferred Defer texture loading until `update()` is called.
    /// @return Texture handle.
    TextureHandle load_texture(const LoadTextureDesc& desc);

    /// Register a texture and create a texture handle from it.
    /// Uses the default sampler if no sampler is specified.
    /// If the texture/sampler is already registered, returns the existing handle.
    /// @param texture Texture to register.
    /// @param sampler Sampler or sampler configuration (optional).
    /// @return Texture handle.
    TextureHandle register_texture(const RegisterTextureDesc& desc);

    /// Process pending work.
    /// This loads any deferred textures, finalizes pending handles, and runs garbage collection
    /// for resources that are no longer referenced.
    /// Should be called once per frame or after a batch of texture loading operations.
    void update();

private:
    static constexpr uint32_t INVALID_TEXTURE_HANDLE = 0xffffffff;

    enum class HandleID : uint32_t { invalid = 0xffffffff };
    enum class TextureID : uint32_t { invalid = 0xffffffff };
    enum class SamplerID : uint32_t { invalid = 0xffffffff };

    struct HandleInfo {
        TextureID texture_id{TextureID::invalid};
        SamplerID sampler_id{SamplerID::invalid};
        auto operator<=>(const HandleInfo&) const = default;
    };

    struct TextureOptions {
        bool generate_mips{true};
        bool srgb{true};
        sgl::TextureUsage usage{sgl::TextureUsage::shader_resource};
        auto operator<=>(const TextureOptions&) const = default;
    };

    struct UDIMPath {
        uint32_t tile_index;
        std::filesystem::path path;
        auto operator<=>(const UDIMPath&) const = default;
    };

    using UDIMPaths = std::vector<UDIMPath>;

    struct TextureInfo {
        std::filesystem::path path;
        UDIMPaths udim_paths;
        TextureOptions options;
        auto operator<=>(const TextureInfo&) const = default;
    };

    struct HandleEntry {
        bool allocated{false};
        /// Reference count. Should not be used directly but through `atomic_ref_count` accessor.
        uint32_t ref_count{0};
        HandleInfo info;
        /// Texture view used on CUDA to get combined texture/sampler descriptor.
        ref<sgl::TextureView> texture_view;
        uint32_t data{INVALID_TEXTURE_HANDLE};
        bool is_finalized{false};

        /// Returns an atomic reference to ref_count for thread-safe operations.
        std::atomic_ref<uint32_t> atomic_ref_count() { return std::atomic_ref(ref_count); }
    };

    struct UDIMTile {
        uint32_t tile_index;
        uint32_t tile_u;
        uint32_t tile_v;
        HandleID handle_id;
    };

    using UDIMTiles = std::vector<UDIMTile>;

    struct TextureEntry {
        bool allocated{false};
        std::optional<TextureInfo> info;
        ref<sgl::Texture> texture;
        UDIMTiles udim_tiles;
        bool is_loaded{false};
    };

    struct SamplerEntry {
        bool allocated{false};
        ref<sgl::Sampler> sampler;
        std::optional<SamplerConfig> sampler_config;
    };

    /// Increment handle reference count.
    void inc_handle_ref(HandleID handle_id);

    /// Decrement handle reference count.
    /// Handle is garbage collected in next call to update() if reference count reaches zero.
    void dec_handle_ref(HandleID handle_id);

    bool is_handle_finalized(HandleID handle_id) const;
    bool is_handle_udim(HandleID handle_id) const;
    const std::filesystem::path* get_handle_path(HandleID handle_id) const;
    ref<sgl::Texture> get_handle_texture(HandleID handle_id) const;
    ref<sgl::Sampler> get_handle_sampler(HandleID handle_id) const;
    uint32_t get_handle_data(HandleID handle_id) const;

    void finalize_handle(HandleID handle_id);
    void finalize_handles(std::span<HandleID> handle_ids);
    void finalize_udim_handle(HandleID handle_id);
    void finalize_udim_handles(std::span<HandleID> handle_ids);

    bool is_texture_loaded(TextureID texture_id) const;

    void load_texture(TextureID texture_id);
    void load_textures(std::span<const TextureID> texture_ids);
    void notify_textures_loaded(std::span<const TextureID> texture_ids, double duration_s);

    UDIMPaths normalize_udim_paths(const UDIMPaths& paths);
    ref<sgl::Texture> create_udim_indirection_texture(const UDIMTiles& udim_tiles);

    HandleID get_or_create_handle(TextureID texture_id, SamplerID sampler_id);
    HandleEntry& get_handle_entry(HandleID handle_id);
    const HandleEntry& get_handle_entry(HandleID handle_id) const;
    HandleID allocate_handle();
    void free_handle(HandleID handle_id);

    TextureID get_or_create_texture(ref<sgl::Texture> texture);
    TextureID get_or_create_texture(const std::filesystem::path& path, const TextureOptions& options);
    TextureID get_or_create_udim_texture(
        const std::filesystem::path& path,
        const UDIMPaths& udim_paths,
        const UDIMTiles& udim_tiles,
        const TextureOptions& options
    );
    TextureEntry& get_texture_entry(TextureID texture_id);
    const TextureEntry& get_texture_entry(TextureID texture_id) const;
    TextureID allocate_texture();
    void free_texture(TextureID texture_id);

    SamplerID get_or_create_sampler(std::variant<std::monostate, SamplerConfig, ref<sgl::Sampler>> sampler);
    SamplerID get_or_create_sampler(ref<sgl::Sampler> sampler);
    SamplerID get_or_create_sampler(const SamplerConfig& sampler_config);
    SamplerEntry& get_sampler_entry(SamplerID sampler_id);
    const SamplerEntry& get_sampler_entry(SamplerID sampler_id) const;
    SamplerID allocate_sampler();
    void free_sampler(SamplerID sampler_id);

    void run_garbage_collect();

    Stats collect_stats() const;

    ref<sgl::Device> m_device;
    ref<sgl::TextureLoader> m_texture_loader;
    /// Default asset resolver. Used when no resolver is specified to resolve relative texture paths.
    ref<AssetResolver> m_default_asset_resolver;

    ref<sgl::Sampler> m_default_sampler;
    SamplerID m_default_sampler_id;

    Stats m_stats;
    mutable bool m_stats_dirty{false};

    std::vector<HandleEntry> m_handle_entries;
    std::map<HandleInfo, HandleID> m_handle_info_to_handle_id;
    std::vector<uint32_t> m_free_handle_indices;

    std::vector<TextureEntry> m_texture_entries;
    std::map<sgl::Texture*, TextureID> m_texture_to_texture_id;
    std::map<TextureInfo, TextureID> m_texture_info_to_texture_id;
    std::vector<uint32_t> m_free_texture_indices;

    std::vector<SamplerEntry> m_sampler_entries;
    std::map<sgl::Sampler*, SamplerID> m_sampler_to_sampler_id;
    std::map<SamplerConfig, SamplerID> m_sampler_config_to_sampler_id;
    std::vector<uint32_t> m_free_sampler_indices;

    // Pending work to do in update().
    std::vector<TextureID> m_pending_textures_to_load;
    std::vector<HandleID> m_pending_handles_to_finalize;
    std::vector<HandleID> m_pending_udim_handles_to_finalize;
    bool m_pending_garbage_collect{false};

    // Garbage collection marks.
    std::vector<bool> m_handles_marked;
    std::vector<bool> m_texture_marked;
    std::vector<bool> m_sampler_marked;

    FALCOR_NON_COPYABLE_AND_MOVABLE(TextureManager);

    friend class TextureHandle;
};

/// Handle to a texture managed by TextureManager.
/// This is a reference-counted handle that automatically manages the lifetime of the underlying
/// texture/sampler resources. When all handles to a texture are destroyed, the texture becomes eligible
/// for garbage collection during the next TextureManager::update() call.
/// Handles can be copied and moved, and support both regular textures and UDIM texture sets.
class FALCOR_API TextureHandle {
public:
    FALCOR_STATIC_WRITE_TO_CURSOR(TextureHandle);

    /// Default constructor. Creates an invalid handle.
    TextureHandle() = default;
    /// Destructor.
    ~TextureHandle();

    /// Copy constructor.
    TextureHandle(const TextureHandle& other);
    /// Move constructor.
    TextureHandle(TextureHandle&& other);

    /// Copy assignment.
    TextureHandle& operator=(const TextureHandle& other);
    /// Move assignment.
    TextureHandle& operator=(TextureHandle&& other);

    /// Equality operator.
    bool operator==(const TextureHandle& other) const = default;
    /// Inequality operator.
    bool operator!=(const TextureHandle& other) const = default;

    /// Returns true if handle is valid.
    bool is_valid() const { return m_texture_manager != nullptr; }
    operator bool() const { return is_valid(); }

    /// Returns true if the handle is finalized and ready for use.
    /// This can be false if the handle was created with deferred loading.
    bool is_finalized() const;
    /// Returns true if this handle represents a UDIM texture.
    bool is_udim() const;

    /// Absolute path of the original texture file.
    /// Only available if the texture was loaded from a file.
    const std::filesystem::path* path() const;

    /// Texture resource.
    /// May not be available if the texture is not yet loaded.
    ref<sgl::Texture> texture() const;

    /// Sampler resource.
    ref<sgl::Sampler> sampler() const;

    /// Forward declaration of UDIM tile information.
    struct UDIMTile;

    /// List of UDIM tiles referenced by this texture.
    /// Each tile contains the tile index (e.g., 1001), UV coordinates, and a texture handle
    /// for accessing the individual tile texture.
    /// Empty if this is not a UDIM texture.
    std::vector<UDIMTile> udim_tiles() const;

    /// Packed handle data for use on the device.
    uint32_t data() const;

    /// Serialize the handle data to a dictionary.
    /// @param dict Dictionary to write to.
    void to_dictionary(IDictionary& dict) const { dict["data"] = data(); }

    /// Write the handle data to a shader cursor for GPU binding.
    /// @tparam TCursor Shader cursor type.
    /// @param cursor Shader cursor to write to.
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = data();
    }

private:
    TextureHandle(TextureManager* texture_manager, TextureManager::HandleID handle_id);

    ref<TextureManager> m_texture_manager;
    TextureManager::HandleID m_handle_id{TextureManager::HandleID::invalid};

    friend class TextureManager;
};

/// Information about a single UDIM tile in a UDIM texture.
struct TextureHandle::UDIMTile {
    /// UDIM tile index (e.g. 1001, 1002, etc.).
    uint32_t tile_index;
    /// Tile U coordinate (zero-based).
    uint32_t tile_u;
    /// Tile V coordinate (zero-based).
    uint32_t tile_v;
    /// Tile texture handle.
    TextureHandle texture_handle;
};

} // namespace falcor
