// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/pack_file.h"
#include "falcor2/core/error.h"

#include <sgl/core/file_stream.h>
#include <sgl/core/memory_mapped_file.h>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace falcor {

using detail::PACK_ALIGNMENT;
using detail::PackChunkHeader;
using detail::PACK_TOC_TAG;
using detail::PackTocHeader;
using detail::PackTocEntry;

// -------------------------------------------------------------------------------------------------
// FourCC helpers
// -------------------------------------------------------------------------------------------------

std::string fourcc_to_string(uint32_t tag)
{
    std::string result(4, '\0');
    result[0] = static_cast<char>(tag & 0xFF);
    result[1] = static_cast<char>((tag >> 8) & 0xFF);
    result[2] = static_cast<char>((tag >> 16) & 0xFF);
    result[3] = static_cast<char>((tag >> 24) & 0xFF);
    return result;
}

// -------------------------------------------------------------------------------------------------
// Low-level helpers
// -------------------------------------------------------------------------------------------------

static size_t align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

static uint64_t checked_align_up(uint64_t value, uint64_t alignment)
{
    FALCOR_CHECK(value <= std::numeric_limits<uint64_t>::max() - (alignment - 1), "alignment overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}

static bool is_aligned(uint64_t value, size_t alignment)
{
    return value % alignment == 0;
}

static bool is_aligned(const void* ptr, size_t alignment)
{
    return reinterpret_cast<uintptr_t>(ptr) % alignment == 0;
}

static bool is_valid_range(size_t data_size, uint64_t offset, uint64_t size)
{
    if (offset > static_cast<uint64_t>(data_size))
        return false;
    return size <= static_cast<uint64_t>(data_size) - offset;
}

static bool is_range_within(uint64_t begin, uint64_t end, uint64_t offset, uint64_t size)
{
    if (offset < begin || offset > end)
        return false;
    return size <= end - offset;
}

static uint64_t range_end(uint64_t offset, uint64_t size)
{
    FALCOR_CHECK(offset <= std::numeric_limits<uint64_t>::max() - size, "range overflow");
    return offset + size;
}

static void write_padding(sgl::Stream& stream, size_t count)
{
    static constexpr uint8_t zeros[PACK_ALIGNMENT] = {};
    if (count > 0)
        stream.write(zeros, count);
}

// -------------------------------------------------------------------------------------------------
// PackWriter
// -------------------------------------------------------------------------------------------------

PackWriter::PackWriter(uint32_t tag, uint32_t version)
{
    FALCOR_CHECK(tag != PACK_TOC_TAG, "tag 'PTOC' is reserved");

    m_root.tag = tag;
    m_root.version = version;
    m_root.flags = PackChunkFlags::container;
    m_root.data_offset = 0;
    m_root.data_size = 0;
    m_root.absolute_offset = 0;
    m_root.payload_size = 0;
    m_stack.push_back(&m_root);
}

PackWriter::~PackWriter() = default;

void PackWriter::begin_data_chunk(uint32_t tag, uint32_t version)
{
    begin_chunk_internal(tag, version, PackChunkFlags::none);
}

void PackWriter::begin_container_chunk(uint32_t tag, uint32_t version)
{
    begin_chunk_internal(tag, version, PackChunkFlags::container);
}

void PackWriter::begin_chunk_internal(uint32_t tag, uint32_t version, PackChunkFlags flags)
{
    FALCOR_CHECK(!m_finalized, "writer is finalized");
    FALCOR_CHECK(tag != PACK_TOC_TAG, "tag 'PTOC' is reserved");

    ChunkNode* parent = m_stack.back();
    FALCOR_CHECK(is_set(parent->flags, PackChunkFlags::container), "cannot add children to a data chunk");

    ChunkNode child;
    child.tag = tag;
    child.version = version;
    child.flags = flags;
    child.data_offset = m_buffer.size();
    child.data_size = 0;
    child.absolute_offset = 0;
    child.payload_size = 0;
    parent->children.push_back(std::move(child));
    m_stack.push_back(&parent->children.back());
}

void PackWriter::write(const void* data, size_t size)
{
    FALCOR_CHECK(!m_finalized, "writer is finalized");
    FALCOR_CHECK(!m_stack.empty(), "no open chunk");

    ChunkNode* current = m_stack.back();
    FALCOR_CHECK(!is_set(current->flags, PackChunkFlags::container), "cannot write data to a container chunk");
    FALCOR_CHECK(current != &m_root, "cannot write data to root container");

    if (size == 0)
        return;

    FALCOR_CHECK(data != nullptr, "cannot write non-empty data from a null pointer");

    const uint8_t* bytes = static_cast<const uint8_t*>(data);
    m_buffer.insert(m_buffer.end(), bytes, bytes + size);
    current->data_size += size;
}

void PackWriter::write(std::span<const uint8_t> data)
{
    write(data.data(), data.size());
}

void PackWriter::end_chunk()
{
    FALCOR_CHECK(!m_finalized, "writer is finalized");
    FALCOR_CHECK(m_stack.size() > 1, "end_chunk() called without matching begin_data_chunk()/begin_container_chunk()");

    m_stack.pop_back();
}

void PackWriter::finish()
{
    if (m_finalized)
        return;
    FALCOR_CHECK(
        m_stack.size() == 1,
        "unbalanced begin_data_chunk()/begin_container_chunk()/end_chunk: {} chunks still open",
        m_stack.size() - 1
    );
    m_finalized = true;
}

size_t PackWriter::ptoc_payload_size(const ChunkNode& node)
{
    return sizeof(PackTocHeader) + node.children.size() * sizeof(PackTocEntry);
}

void PackWriter::compute_layout(ChunkNode& node, size_t offset)
{
    node.absolute_offset = offset;

    if (!is_set(node.flags, PackChunkFlags::container)) {
        node.payload_size = node.data_size;
        return;
    }

    // Payload starts after the node's own header.
    size_t pos = offset + sizeof(PackChunkHeader);

    // PTOC chunk header at pos, payload after that.
    size_t ptoc_total = sizeof(PackChunkHeader) + ptoc_payload_size(node);
    size_t ptoc_padded = align_up(ptoc_total, PACK_ALIGNMENT);
    pos += ptoc_padded;

    // Children.
    for (size_t i = 0; i < node.children.size(); ++i) {
        ChunkNode& child = node.children[i];
        compute_layout(child, pos);

        pos += sizeof(PackChunkHeader) + align_up(child.payload_size, PACK_ALIGNMENT);
    }

    // Cache the serialized payload size for this container.
    node.payload_size = pos - (offset + sizeof(PackChunkHeader));
}

void PackWriter::write_node(sgl::Stream& stream, const ChunkNode& node)
{
    PackChunkHeader header;
    header.tag = node.tag;
    header.version = node.version;
    header.flags = static_cast<uint32_t>(node.flags);
    header.reserved = 0;

    header.size = node.payload_size;
    header.uncompressed_size = node.payload_size;

    stream.write(&header, sizeof(header));

    if (is_set(node.flags, PackChunkFlags::container)) {
        // Write PTOC chunk.
        PackTocHeader toc_header;
        toc_header.child_count = static_cast<uint32_t>(node.children.size());
        toc_header.reserved = 0;
        size_t ptoc_payload = ptoc_payload_size(node);

        PackChunkHeader ptoc_header;
        ptoc_header.tag = PACK_TOC_TAG;
        ptoc_header.version = 0;
        ptoc_header.flags = static_cast<uint32_t>(PackChunkFlags::none);
        ptoc_header.reserved = 0;
        ptoc_header.size = ptoc_payload;
        ptoc_header.uncompressed_size = ptoc_payload;
        stream.write(&ptoc_header, sizeof(ptoc_header));

        // PTOC payload: header + entries.
        stream.write(&toc_header, sizeof(toc_header));

        for (const ChunkNode& child : node.children) {
            PackTocEntry entry;
            entry.tag = child.tag;
            entry.version = child.version;
            entry.offset = child.absolute_offset;
            stream.write(&entry, sizeof(entry));
        }

        // PTOC padding.
        size_t ptoc_total = sizeof(PackChunkHeader) + ptoc_payload;
        size_t ptoc_padded = align_up(ptoc_total, PACK_ALIGNMENT);
        write_padding(stream, ptoc_padded - ptoc_total);

        // Write children.
        for (size_t i = 0; i < node.children.size(); ++i) {
            write_node(stream, node.children[i]);
        }
    } else {
        // Data chunk: write payload from buffer.
        if (node.data_size > 0) {
            stream.write(m_buffer.data() + node.data_offset, node.data_size);
        }
    }

    // Trailing padding.
    size_t padded = align_up(node.payload_size, PACK_ALIGNMENT);
    write_padding(stream, padded - node.payload_size);
}

void PackWriter::write_to_file(const std::filesystem::path& path)
{
    FALCOR_CHECK(m_finalized, "writer must be finalized before writing");
    compute_layout(m_root, 0);
    auto stream = sgl::make_ref<sgl::FileStream>(path, sgl::FileStream::Mode::write);
    write_node(*stream, m_root);
}

void PackWriter::write_to_stream(sgl::Stream& stream)
{
    FALCOR_CHECK(m_finalized, "writer must be finalized before writing");
    FALCOR_CHECK(stream.is_writable(), "stream is not writable");
    FALCOR_CHECK(stream.tell() == 0, "stream must be positioned at offset 0 before writing");
    FALCOR_CHECK(stream.size() == 0, "stream must be empty before writing");
    compute_layout(m_root, 0);
    write_node(stream, m_root);
}

// -------------------------------------------------------------------------------------------------
// PackReader
// -------------------------------------------------------------------------------------------------

struct TocView {
    const PackChunkHeader* header;
    const PackTocEntry* entries;
    uint32_t child_count;
    uint64_t payload_offset;
};

static void validate_header_fields(const PackChunkHeader& header, uint64_t offset)
{
    constexpr uint32_t supported_flags
        = static_cast<uint32_t>(PackChunkFlags::compressed) | static_cast<uint32_t>(PackChunkFlags::container);
    FALCOR_CHECK(
        (header.flags & ~supported_flags) == 0,
        "chunk '{}' at offset {} has unsupported flags {:#x}",
        fourcc_to_string(header.tag),
        offset,
        header.flags
    );
    FALCOR_CHECK(
        !is_set(static_cast<PackChunkFlags>(header.flags), PackChunkFlags::compressed),
        "compressed chunk '{}' at offset {} is not supported",
        fourcc_to_string(header.tag),
        offset
    );
    FALCOR_CHECK(
        header.reserved == 0,
        "chunk '{}' at offset {} has non-zero reserved field",
        fourcc_to_string(header.tag),
        offset
    );
    FALCOR_CHECK(
        header.uncompressed_size == header.size,
        "chunk '{}' at offset {} has uncompressed size {} but size {}",
        fourcc_to_string(header.tag),
        offset,
        header.uncompressed_size,
        header.size
    );
}

static const PackChunkHeader* chunk_header_at(std::span<const uint8_t> data, uint64_t offset)
{
    FALCOR_CHECK(
        is_valid_range(data.size(), offset, sizeof(PackChunkHeader)),
        "chunk header at offset {} extends past data boundary (size {})",
        offset,
        data.size()
    );
    const uint8_t* ptr = data.data() + static_cast<size_t>(offset);
    FALCOR_CHECK(
        is_aligned(ptr, alignof(PackChunkHeader)),
        "chunk header at offset {} is not {}-byte aligned",
        offset,
        alignof(PackChunkHeader)
    );
    return reinterpret_cast<const PackChunkHeader*>(ptr);
}

static TocView
read_toc(std::span<const uint8_t> data, uint64_t container_offset, const PackChunkHeader& container_header)
{
    FALCOR_CHECK(
        is_set(static_cast<PackChunkFlags>(container_header.flags), PackChunkFlags::container),
        "chunk '{}' is not a container",
        fourcc_to_string(container_header.tag)
    );

    const uint64_t container_payload_offset = container_offset + sizeof(PackChunkHeader);
    const uint64_t container_payload_end = range_end(container_payload_offset, container_header.size);

    const uint64_t ptoc_offset = container_payload_offset;
    const PackChunkHeader* ptoc_header = chunk_header_at(data, ptoc_offset);
    validate_header_fields(*ptoc_header, ptoc_offset);
    FALCOR_CHECK(
        ptoc_header->tag == PACK_TOC_TAG,
        "expected PTOC at offset {}, found '{}'",
        ptoc_offset,
        fourcc_to_string(ptoc_header->tag)
    );
    FALCOR_CHECK(
        ptoc_header->version == 0,
        "PTOC at offset {} has unsupported version {}",
        ptoc_offset,
        ptoc_header->version
    );
    FALCOR_CHECK(
        ptoc_header->flags == static_cast<uint32_t>(PackChunkFlags::none),
        "PTOC at offset {} has non-zero flags",
        ptoc_offset
    );

    const uint64_t ptoc_payload_offset = ptoc_offset + sizeof(PackChunkHeader);
    FALCOR_CHECK(
        is_range_within(container_payload_offset, container_payload_end, ptoc_payload_offset, ptoc_header->size),
        "PTOC payload for container '{}' extends past container boundary",
        fourcc_to_string(container_header.tag)
    );
    FALCOR_CHECK(ptoc_header->size >= sizeof(PackTocHeader), "PTOC payload too small");

    const auto* toc_header
        = reinterpret_cast<const PackTocHeader*>(data.data() + static_cast<size_t>(ptoc_payload_offset));

    FALCOR_CHECK(toc_header->reserved == 0, "PTOC at offset {} has non-zero reserved field", ptoc_offset);

    const uint64_t expected_ptoc_size
        = sizeof(PackTocHeader) + uint64_t(toc_header->child_count) * sizeof(PackTocEntry);
    FALCOR_CHECK(
        ptoc_header->size == expected_ptoc_size,
        "PTOC size mismatch: expected {} for {} children, got {}",
        expected_ptoc_size,
        toc_header->child_count,
        ptoc_header->size
    );

    const uint64_t entries_offset = ptoc_payload_offset + sizeof(PackTocHeader);

    return TocView{
        ptoc_header,
        reinterpret_cast<const PackTocEntry*>(data.data() + static_cast<size_t>(entries_offset)),
        toc_header->child_count,
        ptoc_payload_offset,
    };
}

static const PackChunkHeader* validate_chunk(
    std::span<const uint8_t> data,
    uint64_t offset,
    uint64_t parent_payload_begin,
    uint64_t parent_payload_end,
    bool is_root
)
{
    FALCOR_CHECK(is_aligned(offset, PACK_ALIGNMENT), "chunk offset {} is not {}-byte aligned", offset, PACK_ALIGNMENT);
    FALCOR_CHECK(
        is_range_within(parent_payload_begin, parent_payload_end, offset, sizeof(PackChunkHeader)),
        "chunk at offset {} is outside its parent container",
        offset
    );

    const PackChunkHeader* header = chunk_header_at(data, offset);
    validate_header_fields(*header, offset);

    if (!is_root) {
        FALCOR_CHECK(header->tag != PACK_TOC_TAG, "PTOC cannot be exposed as a user chunk");
    }

    const uint64_t payload_offset = offset + sizeof(PackChunkHeader);
    FALCOR_CHECK(
        is_range_within(parent_payload_begin, parent_payload_end, payload_offset, header->size),
        "chunk '{}' at offset {} with size {} extends past parent container boundary",
        fourcc_to_string(header->tag),
        offset,
        header->size
    );

    if (!is_set(static_cast<PackChunkFlags>(header->flags), PackChunkFlags::container))
        return header;

    TocView toc = read_toc(data, offset, *header);

    const uint64_t ptoc_total_end = range_end(toc.payload_offset, toc.header->size);
    const uint64_t payload_end = range_end(payload_offset, header->size);
    uint64_t expected_child_offset = checked_align_up(ptoc_total_end, PACK_ALIGNMENT);

    FALCOR_CHECK(
        expected_child_offset <= payload_end,
        "PTOC for container '{}' extends past container payload after padding",
        fourcc_to_string(header->tag)
    );

    for (uint32_t i = 0; i < toc.child_count; ++i) {
        const PackTocEntry& entry = toc.entries[i];
        FALCOR_CHECK(
            entry.offset == expected_child_offset,
            "TOC entry {} for container '{}' points to offset {}, expected {}",
            i,
            fourcc_to_string(header->tag),
            entry.offset,
            expected_child_offset
        );
        const PackChunkHeader* child_header
            = validate_chunk(data, entry.offset, expected_child_offset, payload_end, false);
        FALCOR_CHECK(
            entry.tag == child_header->tag,
            "TOC entry tag '{}' does not match chunk header tag '{}'",
            fourcc_to_string(entry.tag),
            fourcc_to_string(child_header->tag)
        );
        FALCOR_CHECK(
            entry.version == child_header->version,
            "TOC entry for tag '{}' has version {} but chunk header has version {}",
            fourcc_to_string(entry.tag),
            entry.version,
            child_header->version
        );

        uint64_t child_payload_offset = range_end(entry.offset, sizeof(PackChunkHeader));
        uint64_t child_payload_size = checked_align_up(child_header->size, PACK_ALIGNMENT);
        uint64_t child_payload_end = range_end(child_payload_offset, child_header->size);
        uint64_t child_padded_end = range_end(child_payload_offset, child_payload_size);
        FALCOR_CHECK(
            child_payload_end <= payload_end && child_padded_end <= payload_end,
            "chunk '{}' at offset {} extends past parent container after padding",
            fourcc_to_string(child_header->tag),
            entry.offset
        );
        expected_child_offset = child_padded_end;
    }

    FALCOR_CHECK(
        expected_child_offset == payload_end,
        "container '{}' has unreferenced or missing bytes between offset {} and {}",
        fourcc_to_string(header->tag),
        expected_child_offset,
        payload_end
    );

    return header;
}

PackReader::PackReader(const std::filesystem::path& path)
{
    m_mmap = std::make_unique<sgl::MemoryMappedFile>(path);
    FALCOR_CHECK(m_mmap->is_open(), "failed to open file: {}", path.string());
    m_data = std::span<const uint8_t>(static_cast<const uint8_t*>(m_mmap->data()), m_mmap->size());
    validate();
}

PackReader::PackReader(std::span<const uint8_t> data)
    : m_data(data)
{
    validate();
}

PackReader::~PackReader() = default;

PackReader::PackReader(PackReader&& other) noexcept = default;
PackReader& PackReader::operator=(PackReader&& other) noexcept = default;

void PackReader::validate()
{
    const PackChunkHeader* root_header = chunk_header_at(m_data, 0);
    validate_header_fields(*root_header, 0);
    FALCOR_CHECK(root_header->tag != PACK_TOC_TAG, "root chunk tag 'PTOC' is reserved");
    FALCOR_CHECK(
        is_set(static_cast<PackChunkFlags>(root_header->flags), PackChunkFlags::container),
        "root chunk is not a container"
    );

    size_t expected_payload_size = m_data.size() - sizeof(PackChunkHeader);
    FALCOR_CHECK(
        root_header->size == expected_payload_size,
        "root chunk size {} does not match payload size {}",
        root_header->size,
        expected_payload_size
    );

    validate_chunk(m_data, 0, 0, m_data.size(), true);
}

uint32_t PackReader::tag() const
{
    return reinterpret_cast<const PackChunkHeader*>(m_data.data())->tag;
}

uint32_t PackReader::version() const
{
    return reinterpret_cast<const PackChunkHeader*>(m_data.data())->version;
}

size_t PackReader::chunk_count() const
{
    const auto* root_header = reinterpret_cast<const PackChunkHeader*>(m_data.data());
    return read_toc(m_data, 0, *root_header).child_count;
}

std::optional<PackReader::ChunkView> PackReader::find_chunk(uint32_t tag) const
{
    const auto* root_header = reinterpret_cast<const PackChunkHeader*>(m_data.data());
    TocView toc = read_toc(m_data, 0, *root_header);
    for (uint32_t i = 0; i < toc.child_count; ++i) {
        if (toc.entries[i].tag == tag) {
            const auto* header = chunk_header_at(m_data, toc.entries[i].offset);
            return ChunkView(header, m_data);
        }
    }
    return std::nullopt;
}

PackReader::ChunkRange PackReader::chunks() const
{
    const auto* root_header = reinterpret_cast<const PackChunkHeader*>(m_data.data());
    TocView toc = read_toc(m_data, 0, *root_header);
    return ChunkRange(m_data, toc.entries, toc.child_count);
}

// -------------------------------------------------------------------------------------------------
// PackReader::ChunkView
// -------------------------------------------------------------------------------------------------

PackReader::ChunkView::ChunkView(const PackChunkHeader* header, std::span<const uint8_t> data)
    : m_header(header)
    , m_data(data)
{
}

uint32_t PackReader::ChunkView::tag() const
{
    return m_header->tag;
}

uint32_t PackReader::ChunkView::version() const
{
    return m_header->version;
}

PackChunkFlags PackReader::ChunkView::flags() const
{
    return static_cast<PackChunkFlags>(m_header->flags);
}

bool PackReader::ChunkView::is_container() const
{
    return is_set(flags(), PackChunkFlags::container);
}

uint64_t PackReader::ChunkView::size() const
{
    if (is_container())
        return 0;
    return m_header->size;
}

std::span<const uint8_t> PackReader::ChunkView::data() const
{
    FALCOR_CHECK(!is_container(), "cannot read data from container chunk '{}'", fourcc_to_string(m_header->tag));
    uint64_t offset
        = static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(m_header) - m_data.data()) + sizeof(PackChunkHeader);
    FALCOR_CHECK(
        is_valid_range(m_data.size(), offset, m_header->size),
        "chunk '{}' payload at offset {} with size {} extends past data boundary (size {})",
        fourcc_to_string(m_header->tag),
        offset,
        m_header->size,
        m_data.size()
    );
    return m_data.subspan(static_cast<size_t>(offset), static_cast<size_t>(m_header->size));
}

size_t PackReader::ChunkView::child_count() const
{
    FALCOR_CHECK(is_container(), "cannot get child_count from data chunk '{}'", fourcc_to_string(m_header->tag));
    uint64_t header_offset = static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(m_header) - m_data.data());
    return read_toc(m_data, header_offset, *m_header).child_count;
}

std::optional<PackReader::ChunkView> PackReader::ChunkView::find_child(uint32_t tag) const
{
    FALCOR_CHECK(is_container(), "cannot find_child on data chunk '{}'", fourcc_to_string(m_header->tag));
    uint64_t header_offset = static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(m_header) - m_data.data());
    TocView toc = read_toc(m_data, header_offset, *m_header);
    for (uint32_t i = 0; i < toc.child_count; ++i) {
        if (toc.entries[i].tag == tag) {
            const auto* child_header = chunk_header_at(m_data, toc.entries[i].offset);
            return ChunkView(child_header, m_data);
        }
    }
    return std::nullopt;
}

PackReader::ChunkRange PackReader::ChunkView::children() const
{
    FALCOR_CHECK(is_container(), "cannot iterate children of data chunk '{}'", fourcc_to_string(m_header->tag));
    uint64_t header_offset = static_cast<uint64_t>(reinterpret_cast<const uint8_t*>(m_header) - m_data.data());
    TocView toc = read_toc(m_data, header_offset, *m_header);
    return ChunkRange(m_data, toc.entries, toc.child_count);
}

// -------------------------------------------------------------------------------------------------
// PackReader::ChunkRange
// -------------------------------------------------------------------------------------------------

PackReader::ChunkRange::ChunkRange(std::span<const uint8_t> data, const PackTocEntry* entries, uint32_t count)
    : m_data(data)
    , m_entries(entries)
    , m_count(count)
{
}

PackReader::ChunkIterator PackReader::ChunkRange::begin() const
{
    return ChunkIterator(m_data, m_entries, 0);
}

PackReader::ChunkIterator PackReader::ChunkRange::end() const
{
    return ChunkIterator(m_data, m_entries, m_count);
}

// -------------------------------------------------------------------------------------------------
// PackReader::ChunkIterator
// -------------------------------------------------------------------------------------------------

PackReader::ChunkIterator::ChunkIterator(std::span<const uint8_t> data, const PackTocEntry* entries, uint32_t index)
    : m_data(data)
    , m_entries(entries)
    , m_index(index)
{
}

PackReader::ChunkView PackReader::ChunkIterator::operator*() const
{
    const auto* header = chunk_header_at(m_data, m_entries[m_index].offset);
    return ChunkView(header, m_data);
}

PackReader::ChunkIterator& PackReader::ChunkIterator::operator++()
{
    ++m_index;
    return *this;
}

PackReader::ChunkIterator PackReader::ChunkIterator::operator++(int)
{
    ChunkIterator tmp = *this;
    ++m_index;
    return tmp;
}

bool PackReader::ChunkIterator::operator==(const ChunkIterator& other) const
{
    return m_data.data() == other.m_data.data() && m_data.size() == other.m_data.size() && m_entries == other.m_entries
        && m_index == other.m_index;
}

bool PackReader::ChunkIterator::operator!=(const ChunkIterator& other) const
{
    return !(*this == other);
}

} // namespace falcor
