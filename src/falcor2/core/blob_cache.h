// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/object.h"

#include <sgl/core/fwd.h>
#include <sgl/core/crypto.h>

#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace falcor {

template<typename T>
using ref = sgl::ref<T>;

/// Error category for blob cache specific errors.
/// @return The process-wide blob cache error category.
FALCOR_API const std::error_category& blob_cache_category() noexcept;

/// File-based binary blob cache indexed by SHA1 digests.
/// Each cache key maps to a directory containing one or more named payloads.
/// The key directory timestamp is used for LRU eviction.
class FALCOR_API BlobCache : public Object {
    FALCOR_OBJECT(BlobCache)
public:
    static constexpr std::string_view DEFAULT_SUB_DATA_NAME = "blob";

    /// Key type used to identify cache entries.
    using Key = sgl::SHA1::Digest;

    /// Blob cache error codes returned through std::error_code.
    enum class Error {
        ok = 0,
        lock_timeout = 1,
        write_failed = 2,
        read_failed = 3,
        invalid_sub_data_name = 5,
    };

    struct Options {
        /// Root directory for the cache. Defaults to platform::app_data_directory() / "blob_cache".
        std::optional<std::filesystem::path> root_dir;
        /// Maximum total cache size in bytes (default 128 GB).
        size_t max_size = 128ULL * 1024 * 1024 * 1024;
        /// Trigger eviction when total size exceeds this percentage of max_size.
        uint32_t eviction_threshold_pct = 80;
        /// Evict down to this percentage of max_size.
        uint32_t eviction_target_pct = 60;
        /// Whether to flush file streams after writes (default false for performance).
        bool flush_after_write = false;
    };

    struct Stats {
        /// Total size in bytes across all live payload files.
        size_t total_size{0};
        /// Number of cache keys that contain at least one live payload.
        size_t entry_count{0};
        /// Configured cache size limit in bytes.
        size_t max_size{0};
    };

    class FALCOR_API WriteStream {
    public:
        WriteStream() = default;
        ~WriteStream();

        WriteStream(WriteStream&& other);
        WriteStream& operator=(WriteStream&& other);

        WriteStream(const WriteStream&) = delete;
        WriteStream& operator=(const WriteStream&) = delete;

        /// The writable temporary file stream owned by this pending write.
        ref<sgl::FileStream> stream() const& { return m_stream; }
        ref<sgl::FileStream> stream() && = delete;

        /// Publish the temporary file as the cache payload or throw std::system_error on failure.
        void commit();

        /// Try to publish the temporary file as the cache payload.
        /// @param ec Receives the failure reason when the commit fails.
        /// @return True when the payload was published successfully.
        bool try_commit(std::error_code& ec) noexcept;

        /// Cancel the pending write and remove the temporary file if it still exists.
        void abort() noexcept;

    private:
        friend class BlobCache;

        WriteStream(
            BlobCache* cache,
            Key key,
            std::string sub_data_name,
            std::filesystem::path tmp_path,
            ref<sgl::FileStream> stream
        );

        BlobCache* m_cache{nullptr};
        Key m_key{};
        std::string m_sub_data_name;
        std::filesystem::path m_tmp_path;
        ref<sgl::FileStream> m_stream;
    };

    /// Create a cache using the supplied options, or defaults when no options are provided.
    /// @param options Optional cache configuration.
    explicit BlobCache(std::optional<Options> options = std::nullopt);
    ~BlobCache();

    // --- Write ---

    /// Store data under key and optional sub-data name, replacing any existing payload.
    /// @param key Key identifying the cache entry.
    /// @param data Bytes to store in the cache.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    void
    write(const Key& key, std::span<const uint8_t> data, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Non-throwing write variant.
    /// @param key Key identifying the cache entry.
    /// @param data Bytes to store in the cache.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @param ec Receives the failure reason on validation, locking, or I/O failure.
    /// @return True when the payload was written successfully.
    bool try_write(
        const Key& key,
        std::span<const uint8_t> data,
        std::optional<std::string_view> sub_data_name,
        std::error_code& ec
    ) noexcept;

    // --- Write (streaming) ---

    /// Start a streaming write to a temporary file. The payload is visible only after WriteStream::commit().
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return A pending write stream that publishes the payload on commit.
    WriteStream write_stream(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    // --- Read ---

    /// Read an existing payload into out_data. The buffer size must exactly match the payload size.
    /// @param key Key identifying the cache entry.
    /// @param out_data Destination buffer. Its size must exactly match the payload size.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return True when the payload exists and was read into out_data.
    bool
    read(const Key& key, std::span<uint8_t> out_data, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Non-throwing fixed-buffer read variant.
    /// @param key Key identifying the cache entry.
    /// @param out_data Destination buffer. Its size must exactly match the payload size.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @param ec Receives the failure reason on validation or I/O failure.
    /// @return True when the payload exists and was read into out_data. Missing payloads or size mismatches return
    /// false without setting ec.
    bool try_read(
        const Key& key,
        std::span<uint8_t> out_data,
        std::optional<std::string_view> sub_data_name,
        std::error_code& ec
    ) noexcept;

    /// Read a payload into a new byte vector, or return std::nullopt when the payload is missing.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return The payload bytes, or std::nullopt when the payload is missing.
    std::optional<std::vector<uint8_t>>
    read(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Open an existing payload as a read-only file stream, or return nullptr when missing.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return A readable file stream for the payload, or nullptr when the payload is missing.
    ref<sgl::FileStream>
    read_as_file_stream(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Open a non-empty payload as a read-only memory-mapped stream, or return nullptr when missing or empty.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return A readable memory-mapped stream for the payload, or nullptr when the payload is missing or empty.
    ref<sgl::MemoryMappedFileStream>
    read_as_memory_mapped_file_stream(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    // --- Query ---

    /// Return true when a payload exists for key and optional sub-data name.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return True when the payload exists.
    bool contains(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Return the payload size in bytes, or std::nullopt when the payload is missing.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return The payload size in bytes, or std::nullopt when the payload is missing.
    std::optional<size_t> get_entry_size(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt);

    /// Return the filesystem path that would contain the payload. The payload may not exist.
    /// @param key Key identifying the cache entry.
    /// @param sub_data_name Optional payload name within the cache entry. Uses DEFAULT_SUB_DATA_NAME when omitted.
    /// @return Filesystem path for the requested payload.
    std::filesystem::path get_path(const Key& key, std::optional<std::string_view> sub_data_name = std::nullopt) const;

    // --- Maintenance ---

    /// Evict least-recently-used cache entries until the configured target size is reached.
    void evict();
    /// Remove all cache entries while preserving cache metadata such as the lock file.
    void clear();

    /// Scan the cache directory and return current cache statistics.
    /// @return Current cache statistics.
    Stats stats();

private:
    struct CacheEntry {
        std::filesystem::path key_dir;
        uint64_t total_size{0};
        std::filesystem::file_time_type last_access{};
    };

    struct ScanResult {
        uint64_t total_size{0};
        std::vector<CacheEntry> entries;
    };

    static std::string digest_to_hex(const Key& digest);
    static bool is_valid_sub_data_name(std::string_view sub_data_name) noexcept;

    std::filesystem::path get_key_dir(const Key& key) const;
    std::filesystem::path get_blob_path(const Key& key, std::string_view sub_data_name) const;
    std::filesystem::path get_temp_path(const Key& key, std::string_view sub_data_name) const;

    bool validate_sub_data_name(std::string_view sub_data_name, std::error_code& ec) const noexcept;
    ScanResult scan_entries(std::error_code& ec) const noexcept;
    bool entry_has_payload(const std::filesystem::path& key_dir) const noexcept;

    bool prepare_key_dir(const std::filesystem::path& key_dir, std::error_code& ec) noexcept;
    bool publish_temp_file(
        const Key& key,
        std::string_view sub_data_name,
        const std::filesystem::path& tmp_path,
        std::error_code& ec
    ) noexcept;
    void touch_key_dir(const std::filesystem::path& key_dir) noexcept;
    void remove_payload(const std::filesystem::path& key_dir, const std::filesystem::path& blob_path) noexcept;

    void cleanup_temp_files() noexcept;
    void maybe_evict() noexcept;

    Options m_options;
    std::filesystem::path m_root_dir;
    std::filesystem::path m_lock_path;
    std::mutex m_mutex; // in-process thread safety

    FALCOR_NON_COPYABLE(BlobCache);
    FALCOR_NON_MOVABLE(BlobCache);
};

/// Make std::error_code from BlobCache::Error.
/// @param e Blob cache error value.
/// @return std::error_code using blob_cache_category().
inline std::error_code make_error_code(BlobCache::Error e) noexcept
{
    return {static_cast<int>(e), blob_cache_category()};
}

} // namespace falcor

// Enable std::error_code creation from BlobCache::Error.
template<>
struct std::is_error_code_enum<falcor::BlobCache::Error> : std::true_type { };
