// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/blob_cache.h"
#include "falcor2/core/error.h"
#include "falcor2/core/platform.h"

#include <sgl/core/file_stream.h>
#include <sgl/core/logger.h>
#include <sgl/core/memory_mapped_file_stream.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <limits>
#include <random>
#include <string>

namespace falcor {

// ----------------------------------------------------------------------------
// Error category
// ----------------------------------------------------------------------------

namespace {

class BlobCacheCategoryImpl : public std::error_category {
public:
    const char* name() const noexcept override { return "blob_cache"; }

    std::string message(int ev) const override
    {
        switch (static_cast<BlobCache::Error>(ev)) {
        case BlobCache::Error::ok:
            return "success";
        case BlobCache::Error::lock_timeout:
            return "failed to acquire cache lock within timeout";
        case BlobCache::Error::write_failed:
            return "failed to write blob to cache";
        case BlobCache::Error::read_failed:
            return "failed to read blob from cache";
        case BlobCache::Error::invalid_sub_data_name:
            return "invalid blob cache sub-data name";
        default:
            return "unknown blob cache error";
        }
    }
};

std::string generate_random_suffix()
{
    static thread_local std::mt19937 rng(
        static_cast<unsigned>(std::chrono::steady_clock::now().time_since_epoch().count())
    );
    static constexpr char chars[] = "0123456789abcdef";
    std::string suffix;
    suffix.reserve(8);
    for (int i = 0; i < 8; ++i)
        suffix.push_back(chars[rng() % 16]);
    return suffix;
}

bool is_hex_string(std::string_view text, size_t size) noexcept
{
    if (text.size() != size)
        return false;

    for (char c : text) {
        if ((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))
            continue;
        return false;
    }
    return true;
}

bool is_temp_file_name(std::string_view filename) noexcept
{
    static constexpr std::string_view delimiter = ".tmp.";
    size_t pos = filename.rfind(delimiter);
    if (pos == std::string_view::npos)
        return false;

    return is_hex_string(filename.substr(pos + delimiter.size()), 8);
}

size_t saturating_to_size_t(uint64_t value) noexcept
{
    if (value > std::numeric_limits<size_t>::max())
        return std::numeric_limits<size_t>::max();
    return static_cast<size_t>(value);
}

std::optional<uintmax_t> query_regular_file_size(const std::filesystem::path& path, std::error_code& ec) noexcept
{
    ec.clear();

    std::error_code fs_ec;
    bool exists = std::filesystem::exists(path, fs_ec);
    if (fs_ec) {
        ec = fs_ec;
        return std::nullopt;
    }
    if (!exists)
        return std::nullopt;

    bool is_regular = std::filesystem::is_regular_file(path, fs_ec);
    if (fs_ec) {
        ec = fs_ec;
        return std::nullopt;
    }
    if (!is_regular)
        return std::nullopt;

    auto size = std::filesystem::file_size(path, fs_ec);
    if (fs_ec) {
        ec = fs_ec;
        return std::nullopt;
    }

    return size;
}

} // namespace

const std::error_category& blob_cache_category() noexcept
{
    static BlobCacheCategoryImpl instance;
    return instance;
}

// ----------------------------------------------------------------------------
// BlobCache
// ----------------------------------------------------------------------------

std::string BlobCache::digest_to_hex(const Key& digest)
{
    static constexpr char hex_digits[] = "0123456789abcdef";
    std::string hex;
    hex.reserve(40);
    for (auto b : digest) {
        hex.push_back(hex_digits[b >> 4]);
        hex.push_back(hex_digits[b & 0xf]);
    }
    return hex;
}

bool BlobCache::is_valid_sub_data_name(std::string_view sub_data_name) noexcept
{
    if (sub_data_name.empty() || sub_data_name == "." || sub_data_name == "..")
        return false;
    if (sub_data_name.find(".tmp.") != std::string_view::npos)
        return false;

    for (char c : sub_data_name) {
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_'
            || c == '-')
            continue;
        return false;
    }

    return true;
}

std::filesystem::path BlobCache::get_key_dir(const Key& key) const
{
    std::string hex = digest_to_hex(key);
    std::string prefix = hex.substr(0, 2);
    return m_root_dir / prefix / hex;
}

std::filesystem::path BlobCache::get_blob_path(const Key& key, std::string_view sub_data_name) const
{
    return get_key_dir(key) / std::string(sub_data_name);
}

std::filesystem::path BlobCache::get_temp_path(const Key& key, std::string_view sub_data_name) const
{
    return get_key_dir(key) / (std::string(sub_data_name) + ".tmp." + generate_random_suffix());
}

bool BlobCache::validate_sub_data_name(std::string_view sub_data_name, std::error_code& ec) const noexcept
{
    if (is_valid_sub_data_name(sub_data_name))
        return true;

    ec = make_error_code(Error::invalid_sub_data_name);
    return false;
}

BlobCache::BlobCache(std::optional<Options> options)
    : m_options(options.value_or(Options{}))
{
    if (m_options.root_dir.has_value()) {
        m_root_dir = *m_options.root_dir;
    } else {
        m_root_dir = platform::app_data_directory() / "blob_cache";
    }

    std::error_code ec;
    std::filesystem::create_directories(m_root_dir, ec);
    if (ec)
        FALCOR_THROW("BlobCache: failed to create root directory '{}': {}", m_root_dir.string(), ec.message());

    m_lock_path = m_root_dir / "cache.lock";

    cleanup_temp_files();
}

BlobCache::~BlobCache() = default;

BlobCache::WriteStream::WriteStream(
    BlobCache* cache,
    Key key,
    std::string sub_data_name,
    std::filesystem::path tmp_path,
    ref<sgl::FileStream> stream
)
    : m_cache(cache)
    , m_key(key)
    , m_sub_data_name(std::move(sub_data_name))
    , m_tmp_path(std::move(tmp_path))
    , m_stream(std::move(stream))
{
}

BlobCache::WriteStream::~WriteStream()
{
    abort();
}

BlobCache::WriteStream::WriteStream(WriteStream&& other)
    : m_cache(other.m_cache)
    , m_key(other.m_key)
    , m_sub_data_name(std::move(other.m_sub_data_name))
    , m_tmp_path(std::move(other.m_tmp_path))
    , m_stream(std::move(other.m_stream))
{
    other.m_cache = nullptr;
    other.m_tmp_path.clear();
}

BlobCache::WriteStream& BlobCache::WriteStream::operator=(WriteStream&& other)
{
    if (this != &other) {
        abort();

        m_cache = other.m_cache;
        m_key = other.m_key;
        m_sub_data_name = std::move(other.m_sub_data_name);
        m_tmp_path = std::move(other.m_tmp_path);
        m_stream = std::move(other.m_stream);

        other.m_cache = nullptr;
        other.m_tmp_path.clear();
    }

    return *this;
}

void BlobCache::WriteStream::commit()
{
    std::error_code ec;
    if (!try_commit(ec)) {
        if (!ec)
            ec = make_error_code(Error::write_failed);
        throw std::system_error(ec, "BlobCache::WriteStream::commit");
    }
}

bool BlobCache::WriteStream::try_commit(std::error_code& ec) noexcept
{
    ec.clear();

    try {
        if (!m_cache || !m_stream || m_tmp_path.empty()) {
            ec = make_error_code(Error::write_failed);
            return false;
        }

        if (!m_cache->validate_sub_data_name(m_sub_data_name, ec)) {
            abort();
            return false;
        }

        std::filesystem::path tmp_path = m_tmp_path;
        try {
            if (m_cache->m_options.flush_after_write)
                m_stream->flush();
            m_stream->close();
            m_stream.reset();
        } catch (...) {
            ec = make_error_code(Error::write_failed);
            abort();
            return false;
        }

        BlobCache* cache = m_cache;
        if (!cache->publish_temp_file(m_key, m_sub_data_name, tmp_path, ec)) {
            m_cache = nullptr;
            m_tmp_path.clear();
            return false;
        }

        m_cache = nullptr;
        m_tmp_path.clear();
        cache->maybe_evict();
        return true;
    } catch (...) {
        ec = make_error_code(Error::write_failed);
        abort();
        return false;
    }
}

void BlobCache::WriteStream::abort() noexcept
{
    auto stream = std::move(m_stream);
    auto tmp_path = std::move(m_tmp_path);
    m_cache = nullptr;
    m_tmp_path.clear();

    try {
        if (stream)
            stream->close();
    } catch (...) {
    }

    try {
        if (!tmp_path.empty()) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
        }
    } catch (...) {
    }
}

// ----------------------------------------------------------------------------
// Write
// ----------------------------------------------------------------------------

void BlobCache::write(const Key& key, std::span<const uint8_t> data, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);
    if (!try_write(key, data, resolved_sub_data_name, ec)) {
        if (!ec)
            ec = make_error_code(Error::write_failed);
        throw std::system_error(ec, "BlobCache::write");
    }
}

bool BlobCache::try_write(
    const Key& key,
    std::span<const uint8_t> data,
    std::optional<std::string_view> sub_data_name,
    std::error_code& ec
) noexcept
{
    ec.clear();
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        return false;

    std::filesystem::path tmp_path;
    try {
        auto key_dir = get_key_dir(key);
        if (!prepare_key_dir(key_dir, ec))
            return false;

        tmp_path = get_temp_path(key, resolved_sub_data_name);
        {
            std::ofstream file(tmp_path, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                ec = make_error_code(Error::write_failed);
                return false;
            }
            if (!data.empty())
                file.write(reinterpret_cast<const char*>(data.data()), static_cast<std::streamsize>(data.size()));
            if (m_options.flush_after_write)
                file.flush();
            file.close();
            if (!file.good()) {
                std::error_code remove_ec;
                std::filesystem::remove(tmp_path, remove_ec);
                ec = make_error_code(Error::write_failed);
                return false;
            }
        }

        if (!publish_temp_file(key, resolved_sub_data_name, tmp_path, ec))
            return false;

        maybe_evict();
        return true;
    } catch (...) {
        std::error_code remove_ec;
        if (!tmp_path.empty())
            std::filesystem::remove(tmp_path, remove_ec);
        ec = make_error_code(Error::write_failed);
        return false;
    }
}

// ----------------------------------------------------------------------------
// Write (streaming)
// ----------------------------------------------------------------------------

BlobCache::WriteStream BlobCache::write_stream(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    ec.clear();
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::write_stream");

    try {
        auto key_dir = get_key_dir(key);
        if (!prepare_key_dir(key_dir, ec))
            throw std::system_error(ec, "BlobCache::write_stream");

        auto tmp_path = get_temp_path(key, resolved_sub_data_name);
        auto stream = sgl::make_ref<sgl::FileStream>(tmp_path, sgl::FileStream::Mode::write);
        return WriteStream(this, key, std::string(resolved_sub_data_name), std::move(tmp_path), std::move(stream));
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        throw std::system_error(make_error_code(Error::write_failed), "BlobCache::write_stream");
    }
}

// ----------------------------------------------------------------------------
// Read
// ----------------------------------------------------------------------------

bool BlobCache::read(const Key& key, std::span<uint8_t> out_data, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);
    bool result = try_read(key, out_data, resolved_sub_data_name, ec);
    if (ec)
        throw std::system_error(ec, "BlobCache::read");
    return result;
}

bool BlobCache::try_read(
    const Key& key,
    std::span<uint8_t> out_data,
    std::optional<std::string_view> sub_data_name,
    std::error_code& ec
) noexcept
{
    ec.clear();
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        return false;

    std::filesystem::path key_dir;
    std::filesystem::path blob_path;
    try {
        key_dir = get_key_dir(key);
        blob_path = get_blob_path(key, resolved_sub_data_name);
        size_t size = out_data.size();

        std::error_code fs_ec;
        auto file_size = std::filesystem::file_size(blob_path, fs_ec);
        if (fs_ec)
            return false;

        if (file_size != size) {
            sgl::log_debug("BlobCache: blob '{}' has size {}, expected {}", blob_path.string(), file_size, size);
            return false;
        }

        std::ifstream file(blob_path, std::ios::binary);
        if (!file.is_open()) {
            sgl::log_debug("BlobCache: failed to open blob '{}', removing", blob_path.string());
            remove_payload(key_dir, blob_path);
            return false;
        }

        if (!out_data.empty())
            file.read(reinterpret_cast<char*>(out_data.data()), static_cast<std::streamsize>(out_data.size()));
        if (!file.good()) {
            sgl::log_debug("BlobCache: failed to read blob '{}', removing", blob_path.string());
            file.close();
            remove_payload(key_dir, blob_path);
            return false;
        }

        touch_key_dir(key_dir);
    } catch (...) {
        if (!blob_path.empty())
            remove_payload(key_dir, blob_path);
        return false;
    }

    return true;
}

std::optional<std::vector<uint8_t>> BlobCache::read(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::read");

    try {
        auto blob_path = get_blob_path(key, resolved_sub_data_name);

        std::error_code fs_ec;
        auto file_size = query_regular_file_size(blob_path, fs_ec);
        if (fs_ec)
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::read");
        if (!file_size.has_value())
            return std::nullopt;
        if (*file_size > std::numeric_limits<size_t>::max())
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::read");

        std::vector<uint8_t> out_data(static_cast<size_t>(*file_size));
        bool found = try_read(key, out_data, resolved_sub_data_name, ec);
        if (ec)
            throw std::system_error(ec, "BlobCache::read");
        if (!found)
            return std::nullopt;
        return out_data;
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        throw std::system_error(make_error_code(Error::read_failed), "BlobCache::read");
    }
}

ref<sgl::FileStream> BlobCache::read_as_file_stream(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::read_as_file_stream");

    std::filesystem::path key_dir;
    std::filesystem::path blob_path;
    try {
        key_dir = get_key_dir(key);
        blob_path = get_blob_path(key, resolved_sub_data_name);
        std::error_code fs_ec;
        auto size = query_regular_file_size(blob_path, fs_ec);
        if (fs_ec)
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::read_as_file_stream");
        if (!size.has_value())
            return nullptr;

        auto stream = sgl::make_ref<sgl::FileStream>(blob_path, sgl::FileStream::Mode::read);
        touch_key_dir(key_dir);
        return stream;
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        sgl::log_debug("BlobCache: failed to open blob '{}' as stream, removing", blob_path.string());
        if (!blob_path.empty())
            remove_payload(key_dir, blob_path);
        return nullptr;
    }
}

ref<sgl::MemoryMappedFileStream>
BlobCache::read_as_memory_mapped_file_stream(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::read_as_memory_mapped_file_stream");

    std::filesystem::path key_dir;
    std::filesystem::path blob_path;
    try {
        key_dir = get_key_dir(key);
        blob_path = get_blob_path(key, resolved_sub_data_name);
        std::error_code fs_ec;
        auto size = query_regular_file_size(blob_path, fs_ec);
        if (fs_ec)
            throw std::system_error(
                make_error_code(Error::read_failed),
                "BlobCache::read_as_memory_mapped_file_stream"
            );
        if (!size.has_value())
            return nullptr;
        if (*size == 0)
            return nullptr;

        auto stream = sgl::make_ref<sgl::MemoryMappedFileStream>(
            blob_path,
            sgl::MemoryMappedFile::WHOLE_FILE,
            sgl::MemoryMappedFile::AccessHint::sequential
        );
        touch_key_dir(key_dir);
        return stream;
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        sgl::log_debug("BlobCache: failed to memory-map blob '{}', removing", blob_path.string());
        if (!blob_path.empty())
            remove_payload(key_dir, blob_path);
        return nullptr;
    }
}

// ----------------------------------------------------------------------------
// Query
// ----------------------------------------------------------------------------

bool BlobCache::contains(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::contains");

    try {
        std::error_code fs_ec;
        auto size = query_regular_file_size(get_blob_path(key, resolved_sub_data_name), fs_ec);
        if (fs_ec)
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::contains");
        return size.has_value();
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        throw std::system_error(make_error_code(Error::read_failed), "BlobCache::contains");
    }
}

std::optional<size_t> BlobCache::get_entry_size(const Key& key, std::optional<std::string_view> sub_data_name)
{
    std::error_code ec;
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);

    if (!validate_sub_data_name(resolved_sub_data_name, ec))
        throw std::system_error(ec, "BlobCache::get_entry_size");

    try {
        auto blob_path = get_blob_path(key, resolved_sub_data_name);
        std::error_code fs_ec;
        auto size = query_regular_file_size(blob_path, fs_ec);
        if (fs_ec)
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::get_entry_size");
        if (!size.has_value())
            return std::nullopt;
        if (*size > std::numeric_limits<size_t>::max())
            throw std::system_error(make_error_code(Error::read_failed), "BlobCache::get_entry_size");
        return static_cast<size_t>(*size);
    } catch (const std::system_error&) {
        throw;
    } catch (...) {
        throw std::system_error(make_error_code(Error::read_failed), "BlobCache::get_entry_size");
    }
}

std::filesystem::path BlobCache::get_path(const Key& key, std::optional<std::string_view> sub_data_name) const
{
    std::string_view resolved_sub_data_name = sub_data_name.value_or(DEFAULT_SUB_DATA_NAME);
    if (!is_valid_sub_data_name(resolved_sub_data_name))
        throw std::system_error(make_error_code(Error::invalid_sub_data_name), "BlobCache::get_path");

    return get_blob_path(key, resolved_sub_data_name);
}

// ----------------------------------------------------------------------------
// Maintenance
// ----------------------------------------------------------------------------

void BlobCache::evict()
{
    std::lock_guard lock(m_mutex);

    try {
        platform::FileLock file_lock(m_lock_path);
        if (!file_lock.try_lock()) {
            sgl::log_warn("BlobCache: evict failed to acquire lock, skipping");
            return;
        }

        std::error_code ec;
        auto scan = scan_entries(ec);
        if (ec)
            return;

        uint64_t target_size = static_cast<uint64_t>(m_options.max_size) * m_options.eviction_target_pct / 100;
        if (scan.total_size <= target_size)
            return;

        std::sort(
            scan.entries.begin(),
            scan.entries.end(),
            [](const CacheEntry& a, const CacheEntry& b)
            {
                return a.last_access < b.last_access;
            }
        );

        uint64_t total_size = scan.total_size;
        for (const auto& entry : scan.entries) {
            if (total_size <= target_size)
                break;

            std::error_code remove_ec;
            std::filesystem::remove_all(entry.key_dir, remove_ec);
            if (remove_ec) {
                sgl::log_warn("BlobCache: failed to delete cache entry during eviction: {}", entry.key_dir.string());
                continue;
            }
            total_size -= std::min(entry.total_size, total_size);
        }
    } catch (...) {
        sgl::log_warn("BlobCache: evict failed");
    }
}

void BlobCache::clear()
{
    std::lock_guard lock(m_mutex);

    try {
        platform::FileLock file_lock(m_lock_path);
        if (!file_lock.try_lock()) {
            sgl::log_warn("BlobCache: clear failed to acquire lock, skipping");
            return;
        }

        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator(m_root_dir, ec);
             !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            const auto& path = it->path();
            if (path == m_lock_path)
                continue;

            std::error_code remove_ec;
            std::filesystem::remove_all(path, remove_ec);
            if (remove_ec)
                sgl::log_warn("BlobCache: failed to remove cache path '{}': {}", path.string(), remove_ec.message());
        }
        if (ec)
            sgl::log_warn("BlobCache: failed to iterate cache directory '{}': {}", m_root_dir.string(), ec.message());
    } catch (const std::exception& e) {
        sgl::log_warn("BlobCache: clear failed: {}", e.what());
    } catch (...) {
        sgl::log_warn("BlobCache: clear failed");
    }
}

BlobCache::Stats BlobCache::stats()
{
    std::lock_guard lock(m_mutex);

    Stats s;
    s.max_size = m_options.max_size;

    try {
        platform::FileLock file_lock(m_lock_path);
        file_lock.try_lock();

        std::error_code ec;
        auto scan = scan_entries(ec);
        if (!ec) {
            s.total_size = saturating_to_size_t(scan.total_size);
            s.entry_count = scan.entries.size();
        }
    } catch (...) {
    }

    return s;
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

BlobCache::ScanResult BlobCache::scan_entries(std::error_code& ec) const noexcept
{
    ec.clear();
    ScanResult result;

    try {
        std::error_code it_ec;
        for (auto prefix_it = std::filesystem::directory_iterator(m_root_dir, it_ec);
             !it_ec && prefix_it != std::filesystem::directory_iterator();
             prefix_it.increment(it_ec)) {
            std::error_code entry_ec;
            if (!prefix_it->is_directory(entry_ec) || entry_ec)
                continue;

            std::string prefix_name = prefix_it->path().filename().string();
            if (!is_hex_string(prefix_name, 2))
                continue;

            for (auto key_it = std::filesystem::directory_iterator(prefix_it->path(), entry_ec);
                 !entry_ec && key_it != std::filesystem::directory_iterator();
                 key_it.increment(entry_ec)) {
                std::error_code key_ec;
                if (!key_it->is_directory(key_ec) || key_ec)
                    continue;

                std::string key_name = key_it->path().filename().string();
                if (!is_hex_string(key_name, 40))
                    continue;

                CacheEntry cache_entry;
                cache_entry.key_dir = key_it->path();
                cache_entry.last_access = key_it->last_write_time(key_ec);
                if (key_ec) {
                    key_ec.clear();
                    continue;
                }

                for (auto payload_it = std::filesystem::directory_iterator(cache_entry.key_dir, key_ec);
                     !key_ec && payload_it != std::filesystem::directory_iterator();
                     payload_it.increment(key_ec)) {
                    std::error_code payload_ec;
                    if (!payload_it->is_regular_file(payload_ec) || payload_ec)
                        continue;

                    std::string payload_name = payload_it->path().filename().string();
                    if (is_temp_file_name(payload_name))
                        continue;

                    auto size = payload_it->file_size(payload_ec);
                    if (payload_ec || size > std::numeric_limits<uint64_t>::max()) {
                        payload_ec.clear();
                        continue;
                    }

                    uint64_t entry_size = static_cast<uint64_t>(size);
                    if (std::numeric_limits<uint64_t>::max() - cache_entry.total_size < entry_size)
                        cache_entry.total_size = std::numeric_limits<uint64_t>::max();
                    else
                        cache_entry.total_size += entry_size;
                }

                if (key_ec)
                    key_ec.clear();

                if (cache_entry.total_size == 0 && !entry_has_payload(cache_entry.key_dir))
                    continue;

                if (std::numeric_limits<uint64_t>::max() - result.total_size < cache_entry.total_size)
                    result.total_size = std::numeric_limits<uint64_t>::max();
                else
                    result.total_size += cache_entry.total_size;
                result.entries.push_back(std::move(cache_entry));
            }

            if (entry_ec)
                entry_ec.clear();
        }

        if (it_ec)
            ec = make_error_code(Error::read_failed);
    } catch (...) {
        ec = make_error_code(Error::read_failed);
        return ScanResult{};
    }

    return result;
}

bool BlobCache::entry_has_payload(const std::filesystem::path& key_dir) const noexcept
{
    try {
        std::error_code ec;
        for (auto it = std::filesystem::directory_iterator(key_dir, ec);
             !ec && it != std::filesystem::directory_iterator();
             it.increment(ec)) {
            std::error_code entry_ec;
            if (!it->is_regular_file(entry_ec) || entry_ec)
                continue;
            if (!is_temp_file_name(it->path().filename().string()))
                return true;
        }
    } catch (...) {
    }

    return false;
}

bool BlobCache::prepare_key_dir(const std::filesystem::path& key_dir, std::error_code& ec) noexcept
{
    try {
        auto prefix_dir = key_dir.parent_path();
        std::filesystem::create_directories(prefix_dir, ec);
        if (ec) {
            ec = make_error_code(Error::write_failed);
            return false;
        }

        std::error_code fs_ec;
        if (std::filesystem::exists(key_dir, fs_ec) && !std::filesystem::is_directory(key_dir, fs_ec)) {
            std::filesystem::remove(key_dir, fs_ec);
            if (fs_ec) {
                ec = make_error_code(Error::write_failed);
                return false;
            }
        }

        std::filesystem::create_directories(key_dir, ec);
        if (ec) {
            ec = make_error_code(Error::write_failed);
            return false;
        }

        return true;
    } catch (...) {
        ec = make_error_code(Error::write_failed);
        return false;
    }
}

bool BlobCache::publish_temp_file(
    const Key& key,
    std::string_view sub_data_name,
    const std::filesystem::path& tmp_path,
    std::error_code& ec
) noexcept
{
    ec.clear();

    try {
        auto key_dir = get_key_dir(key);
        auto blob_path = get_blob_path(key, sub_data_name);

        std::lock_guard lock(m_mutex);
        platform::FileLock file_lock(m_lock_path);
        if (!file_lock.try_lock()) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            ec = make_error_code(Error::lock_timeout);
            return false;
        }

        if (!prepare_key_dir(key_dir, ec)) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            return false;
        }

        std::error_code fs_ec;
        bool blob_exists = std::filesystem::exists(blob_path, fs_ec);
        if (fs_ec) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            ec = make_error_code(Error::write_failed);
            return false;
        }

        if (blob_exists)
            std::filesystem::remove(blob_path, fs_ec);
        if (fs_ec) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            ec = make_error_code(Error::write_failed);
            return false;
        }

        std::filesystem::rename(tmp_path, blob_path, fs_ec);
        if (fs_ec) {
            std::error_code remove_ec;
            std::filesystem::remove(tmp_path, remove_ec);
            ec = make_error_code(Error::write_failed);
            return false;
        }

        touch_key_dir(key_dir);
        return true;
    } catch (...) {
        std::error_code remove_ec;
        std::filesystem::remove(tmp_path, remove_ec);
        ec = make_error_code(Error::write_failed);
        return false;
    }
}

void BlobCache::touch_key_dir(const std::filesystem::path& key_dir) noexcept
{
    auto now = std::filesystem::file_time_type::clock::now();
    std::error_code ec;
    std::filesystem::last_write_time(key_dir, now, ec);
}

void BlobCache::remove_payload(const std::filesystem::path& key_dir, const std::filesystem::path& blob_path) noexcept
{
    try {
        std::error_code ec;
        std::filesystem::remove(blob_path, ec);
        if (!entry_has_payload(key_dir)) {
            ec.clear();
            std::filesystem::remove(key_dir, ec);
        }
    } catch (...) {
    }
}

void BlobCache::cleanup_temp_files() noexcept
{
    try {
        auto now = std::filesystem::file_time_type::clock::now();

        std::error_code ec;
        for (auto prefix_it = std::filesystem::directory_iterator(m_root_dir, ec);
             !ec && prefix_it != std::filesystem::directory_iterator();
             prefix_it.increment(ec)) {
            std::error_code entry_ec;

            if (!prefix_it->is_directory(entry_ec) || entry_ec)
                continue;

            std::string prefix_name = prefix_it->path().filename().string();
            if (!is_hex_string(prefix_name, 2))
                continue;

            for (auto key_it = std::filesystem::directory_iterator(prefix_it->path(), entry_ec);
                 !entry_ec && key_it != std::filesystem::directory_iterator();
                 key_it.increment(entry_ec)) {
                std::error_code key_ec;
                std::string key_name = key_it->path().filename().string();
                if (!is_hex_string(key_name, 40))
                    continue;

                if (!key_it->is_directory(key_ec) || key_ec)
                    continue;

                for (auto payload_it = std::filesystem::directory_iterator(key_it->path(), key_ec);
                     !key_ec && payload_it != std::filesystem::directory_iterator();
                     payload_it.increment(key_ec)) {
                    std::string filename = payload_it->path().filename().string();
                    if (!is_temp_file_name(filename))
                        continue;

                    std::error_code time_ec;
                    auto write_time = payload_it->last_write_time(time_ec);
                    if (!time_ec && (now - write_time) > std::chrono::minutes(1)) {
                        std::error_code remove_ec;
                        std::filesystem::remove(payload_it->path(), remove_ec);
                    }
                }
            }
        }
    } catch (...) {
    }
}

void BlobCache::maybe_evict() noexcept
{
    try {
        bool needs_eviction = false;
        {
            std::lock_guard lock(m_mutex);

            platform::FileLock file_lock(m_lock_path);
            if (!file_lock.try_lock())
                return;

            std::error_code ec;
            auto scan = scan_entries(ec);
            if (ec)
                return;

            uint64_t threshold = static_cast<uint64_t>(m_options.max_size) * m_options.eviction_threshold_pct / 100;
            needs_eviction = scan.total_size >= threshold;
        }

        if (needs_eviction)
            evict();
    } catch (...) {
    }
}

} // namespace falcor
