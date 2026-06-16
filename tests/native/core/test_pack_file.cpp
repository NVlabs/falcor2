// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/pack_file.h"
#include "falcor2/core/error.h"

#include <sgl/core/file_stream.h>
#include <sgl/core/memory_mapped_file.h>
#include <sgl/core/memory_stream.h>
#include <sgl/core/stream.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("PackFile");

using detail::PACK_TOC_TAG;

// Helper: write to file and return file path.
static std::filesystem::path write_test_file(PackWriter& writer, const std::string& name)
{
    auto path = falcor::testing::get_case_temp_directory() / name;
    writer.finish();
    writer.write_to_file(path);
    return path;
}

// Helper: write to memory buffer via stream, return buffer.
static std::vector<uint8_t> write_to_buffer(PackWriter& writer)
{
    writer.finish();

    auto stream = sgl::make_ref<sgl::MemoryStream>();
    writer.write_to_stream(*stream);

    std::vector<uint8_t> buffer(stream->size());
    std::memcpy(buffer.data(), stream->data(), stream->size());
    return buffer;
}

static size_t root_ptoc_payload_offset()
{
    return 2 * sizeof(detail::PackChunkHeader);
}

static detail::PackTocHeader read_toc_header(const std::vector<uint8_t>& buffer, size_t ptoc_payload_offset)
{
    detail::PackTocHeader header;
    std::memcpy(&header, buffer.data() + ptoc_payload_offset, sizeof(header));
    return header;
}

static const detail::PackTocEntry* toc_entries_at(const std::vector<uint8_t>& buffer, size_t ptoc_payload_offset)
{
    return reinterpret_cast<const detail::PackTocEntry*>(
        buffer.data() + ptoc_payload_offset + sizeof(detail::PackTocHeader)
    );
}

static const detail::PackTocEntry* root_toc_entries(const std::vector<uint8_t>& buffer)
{
    return toc_entries_at(buffer, root_ptoc_payload_offset());
}

static void write_u32(std::vector<uint8_t>& buffer, size_t offset, uint32_t value)
{
    std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

static void write_u64(std::vector<uint8_t>& buffer, size_t offset, uint64_t value)
{
    std::memcpy(buffer.data() + offset, &value, sizeof(value));
}

static uint64_t read_u64(const std::vector<uint8_t>& buffer, size_t offset)
{
    uint64_t value;
    std::memcpy(&value, buffer.data() + offset, sizeof(value));
    return value;
}

// Basic roundtrips and layout.

TEST_CASE("FourCC - make_fourcc and fourcc_to_string roundtrip")
{
    constexpr uint32_t tag = make_fourcc('T', 'E', 'S', 'T');
    CHECK(fourcc_to_string(tag) == "TEST");

    constexpr uint32_t tag2 = make_fourcc('A', 'B', 'C', 'D');
    CHECK(fourcc_to_string(tag2) == "ABCD");

    constexpr uint32_t ptoc = make_fourcc('P', 'T', 'O', 'C');
    CHECK(ptoc == PACK_TOC_TAG);
    CHECK(fourcc_to_string(ptoc) == "PTOC");
}

TEST_CASE("Empty pack - zero top-level chunks")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    PackWriter writer(ROOT, 1);

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.tag() == ROOT);
    CHECK(reader.version() == 1);
    CHECK(reader.chunk_count() == 0);

    size_t count = 0;
    for ([[maybe_unused]] auto chunk : reader.chunks())
        ++count;
    CHECK(count == 0);
}

TEST_CASE("Data chunk - roundtrip payload and version")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA, 42);
    std::vector<uint8_t> payload = {1, 2, 3, 4, 5, 6, 7, 8};
    writer.write(payload.data(), payload.size());
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.chunk_count() == 1);
    auto chunk = reader.find_chunk(DATA);
    REQUIRE(chunk.has_value());
    CHECK(chunk->tag() == DATA);
    CHECK(chunk->version() == 42);
    CHECK(chunk->size() == 8);
    CHECK(!chunk->is_container());

    auto data = chunk->data();
    CHECK(data.size() == 8);
    CHECK(std::equal(data.begin(), data.end(), payload.begin()));
}

TEST_CASE("Multiple chunks - order, versions, and find_chunk are preserved")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t AAAA = make_fourcc('A', 'A', 'A', 'A');
    constexpr uint32_t BBBB = make_fourcc('B', 'B', 'B', 'B');
    constexpr uint32_t CCCC = make_fourcc('C', 'C', 'C', 'C');

    PackWriter writer(ROOT, 100);
    uint8_t a = 'A', b = 'B', c = 'C';
    writer.begin_data_chunk(AAAA, 1);
    writer.write(&a, 1);
    writer.end_chunk();
    writer.begin_data_chunk(BBBB, 2);
    writer.write(&b, 1);
    writer.end_chunk();
    writer.begin_data_chunk(CCCC, 3);
    writer.write(&c, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.version() == 100);
    CHECK(reader.chunk_count() == 3);

    // Verify order.
    std::vector<std::pair<uint32_t, uint32_t>> tag_versions;
    for (auto chunk : reader.chunks())
        tag_versions.push_back({chunk.tag(), chunk.version()});
    CHECK(tag_versions.size() == 3);
    CHECK(tag_versions[0].first == AAAA);
    CHECK(tag_versions[0].second == 1);
    CHECK(tag_versions[1].first == BBBB);
    CHECK(tag_versions[1].second == 2);
    CHECK(tag_versions[2].first == CCCC);
    CHECK(tag_versions[2].second == 3);

    // Find by tag.
    auto found = reader.find_chunk(BBBB);
    REQUIRE(found.has_value());
    CHECK(found->version() == 2);
    auto data = found->data();
    CHECK(data.size() == 1);
    CHECK(data[0] == 'B');
}

TEST_CASE("Duplicate tags - find_chunk returns first and iteration returns all")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DUPL = make_fourcc('D', 'U', 'P', 'L');

    PackWriter writer(ROOT);
    uint8_t v1 = 1, v2 = 2;
    writer.begin_data_chunk(DUPL, 0);
    writer.write(&v1, 1);
    writer.end_chunk();
    writer.begin_data_chunk(DUPL, 0);
    writer.write(&v2, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.chunk_count() == 2);

    // find_chunk returns first match.
    auto first = reader.find_chunk(DUPL);
    REQUIRE(first.has_value());
    CHECK(first->data()[0] == 1);

    // Iterate to get both.
    std::vector<uint8_t> values;
    for (auto chunk : reader.chunks()) {
        if (chunk.tag() == DUPL) {
            values.push_back(chunk.data()[0]);
        }
    }
    CHECK(values.size() == 2);
    CHECK(values[0] == 1);
    CHECK(values[1] == 2);
}

TEST_CASE("Container chunk - direct children roundtrip")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CH_A = make_fourcc('C', 'H', '_', 'A');
    constexpr uint32_t CH_B = make_fourcc('C', 'H', '_', 'B');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    {
        writer.begin_data_chunk(CH_A);
        uint8_t a = 10;
        writer.write(&a, 1);
        writer.end_chunk();

        writer.begin_data_chunk(CH_B);
        uint8_t b = 20;
        writer.write(&b, 1);
        writer.end_chunk();
    }
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.chunk_count() == 1);
    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    CHECK(parent->is_container());
    CHECK(parent->child_count() == 2);

    auto child_a = parent->find_child(CH_A);
    REQUIRE(child_a.has_value());
    CHECK(child_a->data()[0] == 10);

    auto child_b = parent->find_child(CH_B);
    REQUIRE(child_b.has_value());
    CHECK(child_b->data()[0] == 20);
}

TEST_CASE("Container chunk - empty container roundtrip")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t NONE = make_fourcc('N', 'O', 'N', 'E');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    CHECK(parent->is_container());
    CHECK(parent->child_count() == 0);
    CHECK(!parent->find_child(NONE).has_value());

    size_t count = 0;
    for ([[maybe_unused]] auto child : parent->children())
        ++count;
    CHECK(count == 0);
}

TEST_CASE("Container chunk - deep nesting roundtrip")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t LVL1 = make_fourcc('L', 'V', 'L', '1');
    constexpr uint32_t LVL2 = make_fourcc('L', 'V', 'L', '2');
    constexpr uint32_t LEAF = make_fourcc('L', 'E', 'A', 'F');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(LVL1);
    {
        writer.begin_container_chunk(LVL2);
        {
            writer.begin_data_chunk(LEAF);
            uint32_t val = 0xDEADBEEF;
            writer.write(&val, sizeof(val));
            writer.end_chunk();
        }
        writer.end_chunk();
    }
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto lvl1 = reader.find_chunk(LVL1);
    REQUIRE(lvl1.has_value());
    CHECK(lvl1->is_container());

    auto lvl2 = lvl1->find_child(LVL2);
    REQUIRE(lvl2.has_value());
    CHECK(lvl2->is_container());

    auto leaf = lvl2->find_child(LEAF);
    REQUIRE(leaf.has_value());
    CHECK(!leaf->is_container());
    CHECK(leaf->size() == 4);
    auto data = leaf->data();
    uint32_t read_val;
    std::memcpy(&read_val, data.data(), sizeof(read_val));
    CHECK(read_val == 0xDEADBEEF);
}

TEST_CASE("Container chunks - sibling containers roundtrip")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t CTR1 = make_fourcc('C', 'T', 'R', '1');
    constexpr uint32_t CTR2 = make_fourcc('C', 'T', 'R', '2');
    constexpr uint32_t CH_A = make_fourcc('C', 'H', '_', 'A');
    constexpr uint32_t CH_B = make_fourcc('C', 'H', '_', 'B');
    constexpr uint32_t CH_C = make_fourcc('C', 'H', '_', 'C');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(CTR1);
    {
        writer.begin_data_chunk(CH_A);
        uint8_t a = 10;
        writer.write(&a, 1);
        writer.end_chunk();

        writer.begin_data_chunk(CH_B);
        uint8_t b_data[] = {20, 21, 22};
        writer.write(b_data, 3);
        writer.end_chunk();
    }
    writer.end_chunk();

    writer.begin_container_chunk(CTR2);
    {
        writer.begin_data_chunk(CH_C);
        uint32_t val = 0xCAFEBABE;
        writer.write(&val, sizeof(val));
        writer.end_chunk();
    }
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(reader.chunk_count() == 2);

    // Verify all top-level chunk offsets are aligned.
    size_t ptoc_payload_offset = root_ptoc_payload_offset();
    detail::PackTocHeader toc_header = read_toc_header(buf, ptoc_payload_offset);
    const auto* entries = toc_entries_at(buf, ptoc_payload_offset);
    for (uint32_t i = 0; i < toc_header.child_count; ++i) {
        CHECK(entries[i].offset % detail::PACK_ALIGNMENT == 0);
    }

    // Verify first container and its children.
    auto ctr1 = reader.find_chunk(CTR1);
    REQUIRE(ctr1.has_value());
    CHECK(ctr1->is_container());
    CHECK(ctr1->child_count() == 2);

    auto ch_a = ctr1->find_child(CH_A);
    REQUIRE(ch_a.has_value());
    CHECK(ch_a->data()[0] == 10);

    auto ch_b = ctr1->find_child(CH_B);
    REQUIRE(ch_b.has_value());
    CHECK(ch_b->data().size() == 3);
    CHECK(ch_b->data()[0] == 20);

    // Verify second container and its child.
    auto ctr2 = reader.find_chunk(CTR2);
    REQUIRE(ctr2.has_value());
    CHECK(ctr2->is_container());
    CHECK(ctr2->child_count() == 1);

    auto ch_c = ctr2->find_child(CH_C);
    REQUIRE(ch_c.has_value());
    uint32_t read_val;
    std::memcpy(&read_val, ch_c->data().data(), sizeof(read_val));
    CHECK(read_val == 0xCAFEBABE);
}

TEST_CASE("Pack layout - TOC entries point to aligned chunk headers")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t AAAA = make_fourcc('A', 'A', 'A', 'A');
    constexpr uint32_t BBBB = make_fourcc('B', 'B', 'B', 'B');
    constexpr uint32_t CCCC = make_fourcc('C', 'C', 'C', 'C');

    PackWriter writer(ROOT);
    // Write chunks of varying sizes to test padding.
    writer.begin_data_chunk(AAAA);
    uint8_t data1[] = {1, 2, 3}; // 3 bytes: needs 13 bytes padding.
    writer.write(data1, 3);
    writer.end_chunk();

    writer.begin_data_chunk(BBBB);
    uint8_t data2[] = {4, 5, 6, 7, 8, 9, 10}; // 7 bytes.
    writer.write(data2, 7);
    writer.end_chunk();

    writer.begin_data_chunk(CCCC);
    uint8_t data3[] = {11}; // 1 byte.
    writer.write(data3, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    // Verify chunk header offsets are aligned by reading TOC entries directly.
    size_t ptoc_payload_offset = root_ptoc_payload_offset();
    detail::PackTocHeader toc_header = read_toc_header(buf, ptoc_payload_offset);
    CHECK(toc_header.child_count == 3);
    const auto* entries = toc_entries_at(buf, ptoc_payload_offset);
    for (uint32_t i = 0; i < toc_header.child_count; ++i) {
        CHECK(entries[i].offset % detail::PACK_ALIGNMENT == 0);
        CHECK(entries[i].offset < buf.size());
        const auto* header = reinterpret_cast<const detail::PackChunkHeader*>(buf.data() + entries[i].offset);
        CHECK(header->tag == entries[i].tag);
    }

    // Also verify root header at offset 0.
    CHECK(buf.size() > 0);
    // Root is at offset 0 which is aligned.
    CHECK(buf.size() % detail::PACK_ALIGNMENT == 0);
}

TEST_CASE("Data chunk - zero-byte payload is valid")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t EMPT = make_fourcc('E', 'M', 'P', 'T');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(EMPT);
    writer.write(std::span<const uint8_t>{});
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto chunk = reader.find_chunk(EMPT);
    REQUIRE(chunk.has_value());
    CHECK(!chunk->is_container());
    CHECK(chunk->size() == 0);
    CHECK(chunk->data().empty());
}

TEST_CASE("Data chunk - large payload roundtrip")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t BIG = make_fourcc('B', 'I', 'G', ' ');

    std::vector<uint8_t> big_data(100'000);
    std::iota(big_data.begin(), big_data.end(), static_cast<uint8_t>(0));

    PackWriter writer(ROOT);
    writer.begin_data_chunk(BIG);
    writer.write(big_data);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto chunk = reader.find_chunk(BIG);
    REQUIRE(chunk.has_value());
    CHECK(chunk->size() == 100'000);
    auto data = chunk->data();
    CHECK(std::equal(data.begin(), data.end(), big_data.begin()));
}

TEST_CASE("PackReader path constructor - opens memory-mapped pack file")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT, 5);
    writer.begin_data_chunk(DATA);
    uint32_t val = 12345;
    writer.write(&val, sizeof(val));
    writer.end_chunk();

    auto path = write_test_file(writer, "mmap_test.pack");
    PackReader reader(path);

    CHECK(reader.tag() == ROOT);
    CHECK(reader.version() == 5);
    CHECK(reader.chunk_count() == 1);

    auto chunk = reader.find_chunk(DATA);
    REQUIRE(chunk.has_value());
    auto data = chunk->data();
    uint32_t read_val;
    std::memcpy(&read_val, data.data(), sizeof(read_val));
    CHECK(read_val == 12345);
}

// Reader API behavior.

TEST_CASE("Missing child lookup - find_chunk and find_child return nullopt")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', 'L', 'D');
    constexpr uint32_t NONE = make_fourcc('N', 'O', 'N', 'E');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(CHLD);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    CHECK(!reader.find_chunk(NONE).has_value());

    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    CHECK(!parent->find_child(NONE).has_value());
}

TEST_CASE("ChunkIterator equality - different ranges are not equal at same index")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', 'L', 'D');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(CHLD);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto root_range = reader.chunks();
    auto root_it = root_range.begin();
    CHECK(root_it == root_range.begin());

    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    auto child_range = parent->children();
    auto child_it = child_range.begin();

    CHECK(root_it != child_it);
}

TEST_CASE("PackReader move semantics - move-construct and move-assign")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 42;
    writer.write(&v, 1);
    writer.end_chunk();
    writer.finish();

    auto path = falcor::testing::get_case_temp_directory() / "move_test.pack";
    writer.write_to_file(path);

    // Move-construct reader.
    PackReader reader1(path);
    PackReader reader2(std::move(reader1));
    CHECK(reader2.tag() == ROOT);
    CHECK(reader2.chunk_count() == 1);

    // Move-assign reader.
    constexpr uint32_t ROOT2 = make_fourcc('R', 'T', '2', ' ');
    PackWriter writer3(ROOT2);
    writer3.finish();
    auto path2 = falcor::testing::get_case_temp_directory() / "move_test2.pack";
    writer3.write_to_file(path2);
    PackReader reader3(path2);
    reader3 = std::move(reader2);
    CHECK(reader3.tag() == ROOT);
}

TEST_CASE("PackReader - internal PTOC is hidden")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    // PTOC should not be visible in top-level chunks.
    for (auto chunk : reader.chunks()) {
        CHECK(chunk.tag() != PACK_TOC_TAG);
    }
    CHECK(!reader.find_chunk(PACK_TOC_TAG).has_value());
}

TEST_CASE("ChunkView size - data chunks report payload size and containers report zero")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(DATA);
    uint8_t payload[] = {1, 2, 3, 4, 5};
    writer.write(payload, 5);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    CHECK(parent->size() == 0); // Container returns 0.

    auto data = parent->find_child(DATA);
    REQUIRE(data.has_value());
    CHECK(data->size() == 5); // Data chunk returns payload size.
}

TEST_CASE("ChunkView invalid accessors throw for wrong chunk kind")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', 'L', 'D');
    constexpr uint32_t XXXX = make_fourcc('X', 'X', 'X', 'X');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(CHLD);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    PackReader reader(buf);

    auto parent = reader.find_chunk(PRNT);
    REQUIRE(parent.has_value());
    CHECK_THROWS(parent->data());

    auto chunk = parent->find_child(CHLD);
    REQUIRE(chunk.has_value());
    CHECK_THROWS(chunk->children());
    CHECK_THROWS(chunk->find_child(XXXX));
    CHECK_THROWS(chunk->child_count());
}

TEST_CASE("PackReader path constructor - missing file throws")
{
    auto path = falcor::testing::get_case_temp_directory() / "does_not_exist.pack";
    CHECK_THROWS(PackReader(path));
}

// Writer API behavior.

TEST_CASE("PackWriter explicit chunk modes - data chunks and container chunks are exclusive")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', '_', 'A');
    constexpr uint32_t SUB = make_fourcc('S', 'U', 'B', ' ');

    uint8_t v = 1;
    {
        PackWriter writer(ROOT);
        writer.begin_data_chunk(DATA);
        CHECK_THROWS(writer.begin_data_chunk(SUB));
        CHECK_THROWS(writer.begin_container_chunk(SUB));
    }

    {
        PackWriter writer(ROOT);
        writer.begin_container_chunk(PRNT);
        writer.begin_data_chunk(CHLD);
        writer.end_chunk();
        CHECK_THROWS(writer.write(&v, 1));
    }
}

TEST_CASE("PackWriter reserved tag - rejects PTOC")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');

    CHECK_THROWS(PackWriter(PACK_TOC_TAG));

    PackWriter writer(ROOT);
    CHECK_THROWS(writer.begin_data_chunk(PACK_TOC_TAG));
    CHECK_THROWS(writer.begin_container_chunk(PACK_TOC_TAG));
}

TEST_CASE("PackWriter state validation - invalid operations throw")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');
    uint8_t v = 1;

    {
        PackWriter writer(ROOT);
        auto path = falcor::testing::get_case_temp_directory() / "not_finished.pack";
        CHECK_THROWS(writer.write_to_file(path));
    }

    {
        PackWriter writer(ROOT);
        writer.finish();
        CHECK_THROWS(writer.begin_data_chunk(DATA));
        CHECK_THROWS(writer.begin_container_chunk(DATA));
        CHECK_THROWS(writer.write(&v, 1));
        CHECK_THROWS(writer.end_chunk());
    }

    {
        PackWriter writer(ROOT);
        writer.begin_data_chunk(DATA);
        CHECK_THROWS(writer.finish());
        writer.end_chunk();
        writer.finish();
    }

    {
        PackWriter writer(ROOT);
        CHECK_THROWS(writer.end_chunk());
        CHECK_THROWS(writer.write(&v, 1));
    }
}

TEST_CASE("PackWriter write - non-empty null data throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    CHECK_THROWS(writer.write(nullptr, 1));
    writer.end_chunk();
}

TEST_CASE("PackWriter write_to_stream - output matches write_to_file")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t AAAA = make_fourcc('A', 'A', 'A', 'A');
    constexpr uint32_t BBBB = make_fourcc('B', 'B', 'B', 'B');

    PackWriter writer(ROOT, 7);
    writer.begin_data_chunk(AAAA, 1);
    uint8_t a_data[] = {1, 2, 3, 4, 5};
    writer.write(a_data, 5);
    writer.end_chunk();
    writer.begin_data_chunk(BBBB, 2);
    uint8_t b_data[] = {10, 20};
    writer.write(b_data, 2);
    writer.end_chunk();
    writer.finish();

    // Write via stream.
    auto mem_stream = sgl::make_ref<sgl::MemoryStream>();
    writer.write_to_stream(*mem_stream);
    size_t stream_size = mem_stream->size();
    std::vector<uint8_t> stream_buf(stream_size);
    std::memcpy(stream_buf.data(), mem_stream->data(), stream_size);

    // Write via file.
    auto path = falcor::testing::get_case_temp_directory() / "stream_test.pack";
    writer.write_to_file(path);
    sgl::MemoryMappedFile mmap(path);
    REQUIRE(mmap.is_open());
    std::vector<uint8_t> file_buf(mmap.size());
    std::memcpy(file_buf.data(), mmap.data(), mmap.size());

    // Both outputs must be identical.
    REQUIRE(stream_buf.size() == file_buf.size());
    CHECK(std::equal(stream_buf.begin(), stream_buf.end(), file_buf.begin()));

    // Read back from the stream buffer and verify content.
    PackReader reader(stream_buf);
    CHECK(reader.tag() == ROOT);
    CHECK(reader.version() == 7);
    CHECK(reader.chunk_count() == 2);

    auto chunk_a = reader.find_chunk(AAAA);
    REQUIRE(chunk_a.has_value());
    CHECK(chunk_a->version() == 1);
    CHECK(chunk_a->data().size() == 5);

    auto chunk_b = reader.find_chunk(BBBB);
    REQUIRE(chunk_b.has_value());
    CHECK(chunk_b->version() == 2);
    CHECK(chunk_b->data()[0] == 10);
}

TEST_CASE("PackWriter write_to_stream - validates stream state")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');

    {
        PackWriter writer(ROOT);
        writer.finish();

        uint8_t dummy[16] = {};
        auto ro_stream = sgl::make_ref<sgl::MemoryStream>(static_cast<const void*>(dummy), sizeof(dummy));
        CHECK_THROWS(writer.write_to_stream(*ro_stream));
    }

    {
        PackWriter writer(ROOT);
        writer.finish();

        auto stream = sgl::make_ref<sgl::MemoryStream>();
        uint8_t v = 1;
        stream->write(&v, 1);
        CHECK_THROWS(writer.write_to_stream(*stream));
    }

    {
        PackWriter writer(ROOT);
        writer.finish();

        auto stream = sgl::make_ref<sgl::MemoryStream>();
        uint8_t v = 1;
        stream->write(&v, 1);
        stream->seek(0);
        CHECK_THROWS(writer.write_to_stream(*stream));
    }
}

// Corrupt input validation.

TEST_CASE("Corrupt file - truncated header throws")
{
    std::vector<uint8_t> bad_data(16, 0); // Too small for a pack chunk header (32 bytes).
    CHECK_THROWS(PackReader(bad_data));
}

TEST_CASE("Corrupt file - root PTOC tag throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');

    PackWriter writer(ROOT);
    auto buf = write_to_buffer(writer);

    write_u32(buf, offsetof(detail::PackChunkHeader, tag), PACK_TOC_TAG);

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - truncated chunk throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    buf.resize(buf.size() - 10);

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - duplicate TOC child offset throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t AAAA = make_fourcc('A', 'A', 'A', 'A');
    constexpr uint32_t BBBB = make_fourcc('B', 'B', 'B', 'B');

    PackWriter writer(ROOT);
    uint8_t a = 1;
    uint8_t b = 2;
    writer.begin_data_chunk(AAAA);
    writer.write(&a, 1);
    writer.end_chunk();
    writer.begin_data_chunk(BBBB);
    writer.write(&b, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    const auto* entries = root_toc_entries(buf);
    size_t second_offset_pos = root_ptoc_payload_offset() + sizeof(detail::PackTocHeader) + sizeof(detail::PackTocEntry)
        + offsetof(detail::PackTocEntry, offset);
    write_u64(buf, second_offset_pos, entries[0].offset);

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - unreferenced container bytes throw")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    uint8_t v = 1;
    writer.begin_data_chunk(DATA);
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);
    const uint64_t old_root_size = read_u64(buf, offsetof(detail::PackChunkHeader, size));
    const uint64_t new_root_size = old_root_size + detail::PACK_ALIGNMENT;

    buf.resize(buf.size() + detail::PACK_ALIGNMENT);
    write_u64(buf, offsetof(detail::PackChunkHeader, size), new_root_size);
    write_u64(buf, offsetof(detail::PackChunkHeader, uncompressed_size), new_root_size);

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - TOC offset out of range throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    size_t toc_offset_pos
        = root_ptoc_payload_offset() + sizeof(detail::PackTocHeader) + offsetof(detail::PackTocEntry, offset);
    uint64_t bad_offset = std::numeric_limits<uint64_t>::max();
    std::memcpy(buf.data() + toc_offset_pos, &bad_offset, sizeof(bad_offset));

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt nested container - invalid child count throws during validation")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', 'L', 'D');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(CHLD);
    uint8_t v = 42;
    writer.write(&v, 1);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    // Find the nested container's PTOC child_count and corrupt it.
    // The parent chunk is at the offset given by the root TOC.
    size_t root_payload_offset = root_ptoc_payload_offset();
    detail::PackTocHeader root_toc_header = read_toc_header(buf, root_payload_offset);
    REQUIRE(root_toc_header.child_count == 1);
    const auto* root_entries = toc_entries_at(buf, root_payload_offset);
    size_t parent_offset = root_entries[0].offset;

    // The nested PTOC child_count is at parent_offset + 2 * sizeof(chunk header).
    size_t nested_ptoc_payload_offset = parent_offset + 2 * sizeof(detail::PackChunkHeader);
    // Set child_count to a huge number that would overflow the data span.
    uint32_t bad_count = 999999;
    std::memcpy(buf.data() + nested_ptoc_payload_offset, &bad_count, sizeof(bad_count));

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - chunk payload size exceeds file boundary throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    // Find the data chunk's header and corrupt its size field to extend past file boundary.
    const auto* root_entries = root_toc_entries(buf);
    size_t chunk_offset = root_entries[0].offset;

    uint64_t bad_size = std::numeric_limits<uint64_t>::max();
    std::memcpy(buf.data() + chunk_offset + offsetof(detail::PackChunkHeader, size), &bad_size, sizeof(bad_size));

    // Should throw because payload extends past file boundary.
    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("PackReader span constructor - unaligned data throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');

    PackWriter writer(ROOT);
    auto buf = write_to_buffer(writer);

    std::vector<uint8_t> unaligned(buf.size() + 1);
    std::memcpy(unaligned.data() + 1, buf.data(), buf.size());

    CHECK_THROWS(PackReader(std::span<const uint8_t>(unaligned.data() + 1, buf.size())));
}

TEST_CASE("Corrupt file - unsupported chunk flags throw")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    const auto* root_entries = root_toc_entries(buf);
    size_t chunk_offset = root_entries[0].offset;

    uint32_t bad_flags = static_cast<uint32_t>(PackChunkFlags::compressed);
    std::memcpy(buf.data() + chunk_offset + offsetof(detail::PackChunkHeader, flags), &bad_flags, sizeof(bad_flags));

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt file - uncompressed size mismatch throws")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t DATA = make_fourcc('D', 'A', 'T', 'A');

    PackWriter writer(ROOT);
    writer.begin_data_chunk(DATA);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    const auto* root_entries = root_toc_entries(buf);
    size_t chunk_offset = root_entries[0].offset;

    uint64_t bad_uncompressed_size = 2;
    std::memcpy(
        buf.data() + chunk_offset + offsetof(detail::PackChunkHeader, uncompressed_size),
        &bad_uncompressed_size,
        sizeof(bad_uncompressed_size)
    );

    CHECK_THROWS(PackReader(buf));
}

TEST_CASE("Corrupt nested container - TOC offset out of range throws during validation")
{
    constexpr uint32_t ROOT = make_fourcc('R', 'O', 'O', 'T');
    constexpr uint32_t PRNT = make_fourcc('P', 'R', 'N', 'T');
    constexpr uint32_t CHLD = make_fourcc('C', 'H', 'L', 'D');

    PackWriter writer(ROOT);
    writer.begin_container_chunk(PRNT);
    writer.begin_data_chunk(CHLD);
    uint8_t v = 1;
    writer.write(&v, 1);
    writer.end_chunk();
    writer.end_chunk();

    auto buf = write_to_buffer(writer);

    const auto* root_entries = root_toc_entries(buf);
    size_t parent_offset = root_entries[0].offset;

    size_t nested_ptoc_payload_offset = parent_offset + 2 * sizeof(detail::PackChunkHeader);
    size_t nested_toc_offset_pos
        = nested_ptoc_payload_offset + sizeof(detail::PackTocHeader) + offsetof(detail::PackTocEntry, offset);
    uint64_t bad_offset = std::numeric_limits<uint64_t>::max();
    std::memcpy(buf.data() + nested_toc_offset_pos, &bad_offset, sizeof(bad_offset));

    CHECK_THROWS(PackReader(buf));
}

TEST_SUITE_END();
