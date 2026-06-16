// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/blob_cache.h"

#include <sgl/core/crypto.h>
#include <sgl/core/file_stream.h>
#include <sgl/core/memory_mapped_file_stream.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <thread>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("BlobCache");

static BlobCache::Key make_digest(const std::string& data)
{
    return sgl::SHA1(data).digest();
}

static BlobCache::Options make_test_options()
{
    BlobCache::Options opts;
    opts.root_dir = falcor::testing::get_case_temp_directory() / "blob_cache";
    opts.max_size = 1024 * 1024;
    opts.eviction_threshold_pct = 80;
    opts.eviction_target_pct = 60;
    return opts;
}

static std::filesystem::path key_dir_for_path(const std::filesystem::path& payload_path)
{
    return payload_path.parent_path();
}

static void set_entry_time(const std::filesystem::path& payload_path, std::filesystem::file_time_type time)
{
    std::filesystem::last_write_time(payload_path, time);
    std::filesystem::last_write_time(payload_path.parent_path(), time);
}

TEST_CASE("key-only APIs use the default sub-data payload")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("default_payload_test");
    std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};

    cache.write(key, data);

    auto path = cache.get_path(key);
    CHECK(path.filename().string() == std::string(BlobCache::DEFAULT_SUB_DATA_NAME));
    CHECK(cache.get_path(key, std::nullopt) == path);
    CHECK(std::filesystem::is_regular_file(path));

    auto out = cache.read(key);
    REQUIRE(out.has_value());
    CHECK(*out == data);
    CHECK(cache.contains(key));
    CHECK(cache.contains(key, std::nullopt));

    auto size = cache.get_entry_size(key);
    REQUIRE(size.has_value());
    CHECK(*size == data.size());

    auto out_null = cache.read(key, std::nullopt);
    REQUIRE(out_null.has_value());
    CHECK(*out_null == data);
}

TEST_CASE("constructor accepts optional options and stats use nested type")
{
    auto opts = make_test_options();
    BlobCache cache(std::optional<BlobCache::Options>{opts});

    BlobCache::Stats s = cache.stats();
    CHECK(s.max_size == opts.max_size);
    CHECK(s.entry_count == 0);
    CHECK(s.total_size == 0);
}

TEST_CASE("write and read raw memory")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("raw_memory_test");
    std::vector<uint8_t> data(256);
    std::iota(data.begin(), data.end(), uint8_t(0));

    cache.write(key, std::span<const uint8_t>(data.data(), data.size()));

    std::vector<uint8_t> out(256);
    bool found = cache.read(key, std::span<uint8_t>(out.data(), out.size()));
    CHECK(found);
    CHECK(out == data);
}

TEST_CASE("fixed-buffer read requires exact blob size without deleting payloads")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto small_key = make_digest("fixed_read_too_small");
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    cache.write(small_key, data);
    auto small_path = cache.get_path(small_key);

    std::vector<uint8_t> too_small(4);
    std::error_code ec;
    bool found = cache.try_read(small_key, std::span<uint8_t>(too_small.data(), too_small.size()), std::nullopt, ec);
    CHECK_FALSE(found);
    CHECK(!ec);
    CHECK(std::filesystem::exists(small_path));
    CHECK(cache.contains(small_key));

    auto large_key = make_digest("fixed_read_too_large");
    cache.write(large_key, data);
    auto large_path = cache.get_path(large_key);

    std::vector<uint8_t> too_large(6);
    found = cache.try_read(large_key, std::span<uint8_t>(too_large.data(), too_large.size()), std::nullopt, ec);
    CHECK_FALSE(found);
    CHECK(!ec);
    CHECK(std::filesystem::exists(large_path));
    CHECK(cache.contains(large_key));

    std::vector<uint8_t> out(data.size());
    found = cache.try_read(large_key, std::span<uint8_t>(out), std::nullopt, ec);
    CHECK(found);
    CHECK(!ec);
    CHECK(out == data);
}

TEST_CASE("fixed-buffer read size mismatch leaves named payloads intact")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("named_fixed_read_mismatch");
    std::vector<uint8_t> payload = {1, 2, 3, 4, 5};
    std::vector<uint8_t> metadata = {9, 8, 7};
    cache.write(key, payload, "payload");
    cache.write(key, metadata, "metadata");

    std::vector<uint8_t> too_large(6);
    std::error_code ec;
    bool found = cache.try_read(key, std::span<uint8_t>(too_large.data(), too_large.size()), "payload", ec);
    CHECK_FALSE(found);
    CHECK(!ec);
    CHECK(cache.contains(key, "payload"));
    CHECK(cache.contains(key, "metadata"));

    auto payload_out = cache.read(key, "payload");
    REQUIRE(payload_out.has_value());
    CHECK(*payload_out == payload);

    auto out = cache.read(key, "metadata");
    REQUIRE(out.has_value());
    CHECK(*out == metadata);
}

TEST_CASE("named sub-data payloads round-trip independently")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("named_sub_data_test");
    std::vector<uint8_t> blob = {1, 2, 3};
    std::vector<uint8_t> metadata = {4, 5};
    std::vector<uint8_t> shader = {6, 7, 8, 9};

    cache.write(key, blob);
    cache.write(key, metadata, "metadata");
    cache.write(key, shader, "shader.bin");

    CHECK(cache.contains(key));
    CHECK(cache.contains(key, "metadata"));
    CHECK(cache.contains(key, "shader.bin"));

    CHECK(cache.get_path(key).filename().string() == std::string(BlobCache::DEFAULT_SUB_DATA_NAME));
    CHECK(cache.get_path(key, "metadata").filename().string() == "metadata");
    CHECK(cache.get_path(key, "shader.bin").filename().string() == "shader.bin");

    auto metadata_size = cache.get_entry_size(key, "metadata");
    REQUIRE(metadata_size.has_value());
    CHECK(*metadata_size == metadata.size());

    auto out = cache.read(key);
    REQUIRE(out.has_value());
    CHECK(*out == blob);

    out = cache.read(key, "metadata");
    REQUIRE(out.has_value());
    CHECK(*out == metadata);

    out = cache.read(key, "shader.bin");
    REQUIRE(out.has_value());
    CHECK(*out == shader);

    auto s = cache.stats();
    CHECK(s.entry_count == 1);
    CHECK(s.total_size == blob.size() + metadata.size() + shader.size());
}

TEST_CASE("read with span")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("span_test");
    std::vector<uint8_t> data = {10, 20, 30, 40, 50};
    cache.write(key, std::span<const uint8_t>(data));

    std::vector<uint8_t> out(5);
    bool found = cache.read(key, std::span<uint8_t>(out));
    CHECK(found);
    CHECK(out == data);
}

TEST_CASE("read miss returns false")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("nonexistent_key");
    auto out = cache.read(key);
    CHECK_FALSE(out.has_value());
}

TEST_CASE("contains and get_entry_size handle named misses")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("contains_named_test");
    std::vector<uint8_t> data = {1, 2, 3};

    CHECK_FALSE(cache.contains(key));
    CHECK_FALSE(cache.contains(key, "metadata"));
    CHECK_FALSE(cache.get_entry_size(key).has_value());
    CHECK_FALSE(cache.get_entry_size(key, "metadata").has_value());

    cache.write(key, data, "metadata");

    CHECK_FALSE(cache.contains(key));
    CHECK(cache.contains(key, "metadata"));
    CHECK_FALSE(cache.get_entry_size(key).has_value());

    auto size = cache.get_entry_size(key, "metadata");
    REQUIRE(size.has_value());
    CHECK(*size == data.size());
}

TEST_CASE("stats scan reflects rewritten payload sizes")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("stats_size_update");
    std::vector<uint8_t> large(100, 0xCD);
    std::vector<uint8_t> small(25, 0xEF);
    std::vector<uint8_t> metadata(10, 0xAB);

    cache.write(key, large);
    CHECK(cache.stats().total_size == large.size());

    cache.write(key, small);
    CHECK(cache.stats().total_size == small.size());

    cache.write(key, metadata, "metadata");
    auto s = cache.stats();
    CHECK(s.entry_count == 1);
    CHECK(s.total_size == small.size() + metadata.size());
}

TEST_CASE("get_path returns the directory layout path")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("path_test");
    auto path = cache.get_path(key, "metadata");
    CHECK(!path.empty());
    CHECK(path.filename().string() == "metadata");
    CHECK(path.parent_path().filename().string().size() == 40);
    CHECK(path.parent_path().parent_path().filename().string().size() == 2);
    CHECK(path.string().find("blob_cache") != std::string::npos);
}

TEST_CASE("duplicate write replaces the same named payload")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("replace_payload_test");
    std::vector<uint8_t> first = {1, 2, 3, 4, 5};
    std::vector<uint8_t> second = {9, 8};

    cache.write(key, first, "payload");
    cache.write(key, second, "payload");

    auto out = cache.read(key, "payload");
    REQUIRE(out.has_value());
    CHECK(*out == second);
    CHECK(cache.stats().total_size == second.size());
}

TEST_CASE("read as FileStream works for named payloads")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("file_stream_test");
    std::vector<uint8_t> data = {100, 200, 50, 75};
    cache.write(key, data, "stream");

    auto stream = cache.read_as_file_stream(key, "stream");
    REQUIRE(stream != nullptr);
    CHECK(stream->is_open());
    CHECK(stream->is_readable());
    CHECK(stream->size() == data.size());

    std::vector<uint8_t> out(data.size());
    stream->read(out.data(), out.size());
    CHECK(out == data);
}

TEST_CASE("read as MemoryMappedFileStream works for named payloads")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("mmap_stream_test");
    std::vector<uint8_t> data(1024);
    std::iota(data.begin(), data.end(), uint8_t(0));
    cache.write(key, data, "mapped");

    auto stream = cache.read_as_memory_mapped_file_stream(key, "mapped");
    REQUIRE(stream != nullptr);
    CHECK(stream->size() == data.size());
}

TEST_CASE("empty payloads round-trip but are not memory mapped")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("empty_payload_test");
    std::vector<uint8_t> data;
    cache.write(key, data);

    CHECK(cache.contains(key));

    auto out = cache.read(key);
    REQUIRE(out.has_value());
    CHECK(out->empty());

    auto stream = cache.read_as_memory_mapped_file_stream(key);
    CHECK(stream == nullptr);
    CHECK(cache.contains(key));
}

TEST_CASE("streaming write works for default and named payloads")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("streaming_write_test");
    auto default_write = cache.write_stream(key);
    auto default_stream = default_write.stream();
    REQUIRE(default_stream != nullptr);

    std::vector<uint8_t> blob = {11, 22, 33, 44, 55};
    default_stream->write(blob.data(), blob.size());
    default_write.commit();

    auto named_write = cache.write_stream(key, "metadata");
    auto named_stream = named_write.stream();
    REQUIRE(named_stream != nullptr);

    std::vector<uint8_t> metadata = {1, 3, 5};
    named_stream->write(metadata.data(), metadata.size());
    named_write.commit();

    auto out = cache.read(key);
    REQUIRE(out.has_value());
    CHECK(*out == blob);

    out = cache.read(key, "metadata");
    REQUIRE(out.has_value());
    CHECK(*out == metadata);
}

TEST_CASE("uncommitted write stream cleans up temp file")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("write_stream_abort_test");
    std::filesystem::path tmp_path;
    {
        auto write = cache.write_stream(key);
        REQUIRE(write.stream() != nullptr);
        tmp_path = write.stream()->path();
        std::vector<uint8_t> data = {1, 2, 3, 4};
        write.stream()->write(data.data(), data.size());
        CHECK(std::filesystem::exists(tmp_path));
    }

    CHECK_FALSE(std::filesystem::exists(tmp_path));
    CHECK_FALSE(cache.contains(key));
}

TEST_CASE("write stream commits payloads larger than one quarter of max size")
{
    auto opts = make_test_options();
    opts.max_size = 1000;
    BlobCache cache(opts);

    auto key = make_digest("write_stream_try_commit_error_test");
    auto write = cache.write_stream(key);
    REQUIRE(write.stream() != nullptr);
    auto tmp_path = write.stream()->path();

    std::vector<uint8_t> large_payload(300, 0xCC);
    write.stream()->write(large_payload.data(), large_payload.size());

    std::error_code ec;
    CHECK(write.try_commit(ec));
    CHECK(!ec);
    CHECK_FALSE(std::filesystem::exists(tmp_path));
    CHECK(cache.contains(key));

    auto out = cache.read(key);
    REQUIRE(out.has_value());
    CHECK(*out == large_payload);
}

TEST_CASE("accessing named payload refreshes only key directory timestamp")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("timestamp_refresh_test");
    std::vector<uint8_t> data = {1, 2, 3, 4};
    cache.write(key, data, "a");
    cache.write(key, data, "b");

    auto key_dir = key_dir_for_path(cache.get_path(key, "a"));
    auto old_time = std::filesystem::file_time_type::clock::now() - std::chrono::minutes(10);
    set_entry_time(cache.get_path(key, "a"), old_time);
    set_entry_time(cache.get_path(key, "b"), old_time);
    std::filesystem::last_write_time(key_dir, old_time);
    auto payload_time = std::filesystem::last_write_time(cache.get_path(key, "b"));

    auto out = cache.read(key, "b");
    REQUIRE(out.has_value());
    CHECK(*out == data);

    auto refreshed_time = std::filesystem::last_write_time(key_dir);
    CHECK(refreshed_time > old_time);
    CHECK(std::filesystem::last_write_time(cache.get_path(key, "b")) == payload_time);
}

TEST_CASE("eviction uses key directory timestamps and removes whole key directories")
{
    auto opts = make_test_options();
    opts.max_size = 1024 * 1024;
    BlobCache cache(opts);

    auto keep_key = make_digest("evict_keep");
    auto victim_key_0 = make_digest("evict_victim_0");
    auto victim_key_1 = make_digest("evict_victim_1");

    std::vector<uint8_t> keep_a(40, 0xA0);
    std::vector<uint8_t> keep_b(40, 0xB0);
    std::vector<uint8_t> victim(80, 0xC0);

    cache.write(keep_key, keep_a, "a");
    cache.write(keep_key, keep_b, "b");
    cache.write(victim_key_0, victim);
    cache.write(victim_key_1, victim);

    auto now = std::filesystem::file_time_type::clock::now();
    set_entry_time(cache.get_path(keep_key, "a"), now - std::chrono::minutes(30));
    set_entry_time(cache.get_path(keep_key, "b"), now - std::chrono::minutes(30));
    set_entry_time(cache.get_path(victim_key_0), now - std::chrono::minutes(20));
    set_entry_time(cache.get_path(victim_key_1), now - std::chrono::minutes(10));

    auto out = cache.read(keep_key, "b");
    REQUIRE(out.has_value());

    auto evict_opts = opts;
    evict_opts.max_size = 240;
    evict_opts.eviction_target_pct = 50;
    BlobCache evict_cache(evict_opts);
    evict_cache.evict();

    CHECK(evict_cache.contains(keep_key, "a"));
    CHECK(evict_cache.contains(keep_key, "b"));
    CHECK_FALSE(evict_cache.contains(victim_key_0));
    CHECK_FALSE(evict_cache.contains(victim_key_1));
    CHECK_FALSE(std::filesystem::exists(key_dir_for_path(evict_cache.get_path(victim_key_0))));
}

TEST_CASE("eviction triggers after writes")
{
    auto opts = make_test_options();
    opts.max_size = 1000;
    opts.eviction_threshold_pct = 80;
    opts.eviction_target_pct = 60;
    BlobCache cache(opts);

    for (int i = 0; i < 10; ++i) {
        auto key = make_digest("evict_trigger_" + std::to_string(i));
        std::vector<uint8_t> data(100, static_cast<uint8_t>(i));
        cache.write(key, data);
    }

    auto s = cache.stats();
    CHECK(s.total_size <= 600);
}

TEST_CASE("stats ignores temp files and unrelated files")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("stats_ignore_real");
    std::vector<uint8_t> data = {1, 2, 3, 4};
    cache.write(key, data);

    auto temp_path = key_dir_for_path(cache.get_path(key)) / "metadata.tmp.deadbeef";
    {
        std::ofstream f(temp_path, std::ios::binary | std::ios::trunc);
        f << "temporary";
    }

    auto unrelated_path = *opts.root_dir / "index.bin";
    {
        std::ofstream f(unrelated_path, std::ios::binary | std::ios::trunc);
        f << "unrelated";
    }

    auto s = cache.stats();
    CHECK(s.entry_count == 1);
    CHECK(s.total_size == data.size());
}

TEST_CASE("clear wipes cache entries, preserves lock file, and leaves cache usable")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("clear_test");
    std::vector<uint8_t> data = {1, 2, 3};
    cache.write(key, data);
    cache.write(key, data, "metadata");
    CHECK(cache.contains(key));
    CHECK(cache.contains(key, "metadata"));

    auto lock_path = *opts.root_dir / "cache.lock";
    CHECK(std::filesystem::exists(lock_path));

    cache.clear();
    CHECK_FALSE(cache.contains(key));
    CHECK_FALSE(cache.contains(key, "metadata"));
    CHECK(std::filesystem::exists(lock_path));

    auto key2 = make_digest("clear_reuse_test");
    std::vector<uint8_t> data2 = {4, 5, 6, 7};
    cache.write(key2, data2);

    auto out = cache.read(key2);
    REQUIRE(out.has_value());
    CHECK(*out == data2);
}

TEST_CASE("size-mismatched fixed-size read leaves payload intact")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("size_mismatch_test");
    std::vector<uint8_t> data = {1, 2, 3, 4, 5};
    cache.write(key, data);

    auto path = cache.get_path(key);
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f << "x";
    }

    std::vector<uint8_t> out(5);
    std::error_code ec;
    bool found = cache.try_read(key, std::span<uint8_t>(out.data(), out.size()), std::nullopt, ec);
    CHECK_FALSE(found);
    CHECK(!ec);
    CHECK(std::filesystem::exists(path));
    CHECK(cache.contains(key));
}

TEST_CASE("old temp file cleanup on construction")
{
    auto opts = make_test_options();
    auto root = *opts.root_dir;
    auto key_dir = root / "ab" / "ab00000000000000000000000000000000000000";
    std::filesystem::create_directories(key_dir);

    auto tmp_path = key_dir / "blob.tmp.deadbeef";
    {
        std::ofstream f(tmp_path, std::ios::binary | std::ios::trunc);
        f << "temp data";
    }

    auto old_time = std::filesystem::file_time_type::clock::now() - std::chrono::minutes(2);
    std::filesystem::last_write_time(tmp_path, old_time);

    auto unrelated_path = root / "index.bin";
    {
        std::ofstream f(unrelated_path, std::ios::binary | std::ios::trunc);
        f << "unrelated";
    }

    CHECK(std::filesystem::exists(tmp_path));
    CHECK(std::filesystem::exists(unrelated_path));

    BlobCache cache(opts);
    CHECK_FALSE(std::filesystem::exists(tmp_path));
    CHECK(std::filesystem::exists(unrelated_path));
}

TEST_CASE("invalid sub-data names report errors and throwing overloads throw")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("invalid_sub_data_name_test");
    std::vector<uint8_t> data = {1, 2, 3};
    std::error_code ec;

    cache.try_write(key, std::span<const uint8_t>(data.data(), data.size()), "bad/name", ec);
    CHECK(ec == BlobCache::Error::invalid_sub_data_name);

    cache.try_write(key, std::span<const uint8_t>(data.data(), data.size()), "blob.tmp.deadbeef", ec);
    CHECK(ec == BlobCache::Error::invalid_sub_data_name);

    CHECK_THROWS_AS(cache.write_stream(key, ""), std::system_error);

    std::vector<uint8_t> out;
    bool found = cache.try_read(key, std::span<uint8_t>(out), "..", ec);
    CHECK_FALSE(found);
    CHECK(ec == BlobCache::Error::invalid_sub_data_name);

    CHECK_THROWS_AS(cache.contains(key, "bad\\name"), std::system_error);
    CHECK_THROWS_AS(cache.get_entry_size(key, "bad name"), std::system_error);
    CHECK_THROWS_AS(cache.write(key, data, "bad/name"), std::system_error);
    CHECK_THROWS_AS(cache.get_path(key, "."), std::system_error);
}

TEST_CASE("large named payloads are allowed")
{
    auto opts = make_test_options();
    opts.max_size = 1000;
    BlobCache cache(opts);

    auto key = make_digest("per_payload_limit");
    std::vector<uint8_t> first(240, 0xAA);
    std::vector<uint8_t> second(240, 0xBB);
    std::vector<uint8_t> large(300, 0xCC);

    std::error_code ec;
    cache.try_write(key, first, "first", ec);
    CHECK(!ec);
    cache.try_write(key, second, "second", ec);
    CHECK(!ec);
    cache.try_write(key, large, "large", ec);
    CHECK(!ec);

    CHECK(cache.contains(key, "first"));
    CHECK(cache.contains(key, "second"));
    CHECK(cache.contains(key, "large"));

    auto s = cache.stats();
    CHECK(s.entry_count == 1);
    CHECK(s.total_size == first.size() + second.size() + large.size());
}

TEST_CASE("thread safety parallel writes")
{
    auto opts = make_test_options();
    opts.max_size = 10 * 1024 * 1024;
    BlobCache cache(opts);

    constexpr int num_threads = 8;
    constexpr int writes_per_thread = 10;

    std::vector<std::thread> threads;
    for (int t = 0; t < num_threads; ++t) {
        threads.emplace_back(
            [&cache, t]()
            {
                for (int i = 0; i < writes_per_thread; ++i) {
                    auto key = make_digest("thread_" + std::to_string(t) + "_" + std::to_string(i));
                    std::vector<uint8_t> data(64, static_cast<uint8_t>(t * writes_per_thread + i));
                    std::error_code ec;
                    cache.try_write(key, data, std::nullopt, ec);
                }
            }
        );
    }

    for (auto& t : threads)
        t.join();

    int found_count = 0;
    for (int t = 0; t < num_threads; ++t) {
        for (int i = 0; i < writes_per_thread; ++i) {
            auto key = make_digest("thread_" + std::to_string(t) + "_" + std::to_string(i));
            if (cache.contains(key))
                found_count++;
        }
    }

    CHECK(found_count == num_threads * writes_per_thread);
}

TEST_CASE("try write/read do not throw")
{
    auto opts = make_test_options();
    BlobCache cache(opts);

    auto key = make_digest("noexcept_test");
    std::vector<uint8_t> data = {1, 2, 3};

    std::error_code ec;
    cache.try_write(key, data, "metadata", ec);
    CHECK(!ec);

    std::vector<uint8_t> out(data.size());
    bool found = cache.try_read(key, std::span<uint8_t>(out), "metadata", ec);
    CHECK(!ec);
    CHECK(found);
    CHECK(out == data);
}

TEST_SUITE_END();
