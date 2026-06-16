// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"
#include "falcor2/core/macros.h"

#include <sgl/core/fwd.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace falcor {

// -------------------------------------------------------------------------------------------------
// Pack file format
// -------------------------------------------------------------------------------------------------
//
// A pack file is a tree of chunks serialized as one root container chunk. The root chunk begins at
// byte offset 0 and uses the tag/version supplied to PackWriter or expected by the caller.
//
// Scalar fields are stored as fixed-size unsigned integers. FourCC tags are packed little-endian by
// make_fourcc(), so make_fourcc('A', 'B', 'C', 'D') is stored as the byte sequence "ABCD".
//
// File layout:
//
//     PackChunkHeader root_header
//     uint8_t root_payload[root_header.size]
//
// The file ends immediately after the root payload. root_header.size must equal the total file size
// minus sizeof(PackChunkHeader).
//
// Each chunk starts with PackChunkHeader:
//
//     uint32_t tag
//     uint32_t version
//     uint32_t flags
//     uint32_t reserved
//     uint64_t size
//     uint64_t uncompressed_size
//
// The reserved field must be zero. size is the unpadded payload size for data chunks. For container
// chunks, size covers the full container payload: the internal PTOC chunk, PTOC padding, child chunk
// headers, child payloads, and child padding. uncompressed_size must match size because compression
// is reserved for a future format revision; readers reject compressed chunks today.
//
// Data chunk payload:
//
//     PackChunkHeader data_header
//     uint8_t payload[data_header.size]
//     zero padding to detail::PACK_ALIGNMENT
//
// Container chunk payload:
//
//     PackChunkHeader container_header
//     PackChunkHeader ptoc_header
//     PackTocHeader ptoc_payload_header
//     PackTocEntry children[ptoc_payload_header.child_count]
//     zero padding to detail::PACK_ALIGNMENT
//     child chunk 0
//     child chunk 1
//     ...
//
// The internal PTOC chunk always has tag "PTOC", version 0, flags PackChunkFlags::none, and a
// payload containing PackTocHeader followed by PackTocEntry records. The PTOC is an implementation
// detail and is hidden from PackReader iteration/lookups. User-visible chunks, including the root,
// may not use the "PTOC" tag.
//
// PackTocHeader:
//
//     uint32_t child_count
//     uint32_t reserved
//
// PackTocEntry:
//
//     uint32_t tag
//     uint32_t version
//     uint64_t offset
//
// Toc entry offsets are absolute byte offsets from the start of the pack file, not relative to the
// container. Entries are stored in serialized child order. Reader validation requires each entry to
// point to the next aligned child chunk, and the last child padding must end exactly at the
// container payload end. This rejects duplicate offsets, overlapping child ranges, out-of-order
// children, gaps, and hidden trailing bytes.
//
// Chunk headers are aligned to detail::PACK_ALIGNMENT bytes. Padding bytes are inserted after chunk
// payloads as needed, and writer output size is also aligned to detail::PACK_ALIGNMENT. Padding
// bytes written by PackWriter are zero.
//
// PackWriter requires callers to declare whether each non-root chunk is data or a container. Data
// chunks accept write() calls and cannot contain children. Container chunks accept child chunks and
// cannot contain raw data. Calling begin_data_chunk() followed immediately by end_chunk() creates a
// zero-byte data chunk; calling begin_container_chunk() followed immediately by end_chunk() creates
// an empty container with an internal PTOC.

// -------------------------------------------------------------------------------------------------
// FourCC helpers
// -------------------------------------------------------------------------------------------------

/// Create a little-endian FourCC tag from four characters.
constexpr uint32_t make_fourcc(char a, char b, char c, char d)
{
    return static_cast<uint32_t>(static_cast<uint8_t>(a)) | (static_cast<uint32_t>(static_cast<uint8_t>(b)) << 8)
        | (static_cast<uint32_t>(static_cast<uint8_t>(c)) << 16)
        | (static_cast<uint32_t>(static_cast<uint8_t>(d)) << 24);
}

/// Convert a FourCC tag to a four-character string.
FALCOR_API std::string fourcc_to_string(uint32_t tag);

// -------------------------------------------------------------------------------------------------
// PackChunkFlags
// -------------------------------------------------------------------------------------------------

/// Flags stored in each pack chunk header.
enum class PackChunkFlags : uint32_t {
    /// Plain data chunk.
    none = 0x0,
    /// Chunk payload contains an internal PTOC followed by child chunks.
    container = 0x2,
    /// Reserved for future use; currently rejected by PackReader.
    compressed = 0x1,
};

FALCOR_ENUM_CLASS_OPERATORS(PackChunkFlags)

// -------------------------------------------------------------------------------------------------
// Detail
// -------------------------------------------------------------------------------------------------

namespace detail {

constexpr size_t PACK_ALIGNMENT = 16;

struct PackChunkHeader {
    uint32_t tag;
    uint32_t version;
    uint32_t flags;
    uint32_t reserved;
    uint64_t size;
    uint64_t uncompressed_size;
};

static_assert(sizeof(PackChunkHeader) % PACK_ALIGNMENT == 0);

constexpr uint32_t PACK_TOC_TAG = make_fourcc('P', 'T', 'O', 'C');

struct PackTocHeader {
    uint32_t child_count;
    uint32_t reserved;
};

struct PackTocEntry {
    uint32_t tag;
    uint32_t version;
    uint64_t offset;
};

static_assert(sizeof(PackTocHeader) % alignof(PackTocEntry) == 0);
static_assert(sizeof(PackTocEntry) % PACK_ALIGNMENT == 0);

} // namespace detail

// -------------------------------------------------------------------------------------------------
// PackWriter
// -------------------------------------------------------------------------------------------------

/// Incrementally builds a pack file chunk tree and serializes it to a file or stream.
class FALCOR_API PackWriter {
public:
    /// Create a writer with the given root chunk tag and version.
    PackWriter(uint32_t tag, uint32_t version = 0);

    /// Destroy the writer and any buffered chunk data.
    ~PackWriter();

    /// Begin a data child chunk under the current open container chunk.
    void begin_data_chunk(uint32_t tag, uint32_t version = 0);

    /// Begin a container child chunk under the current open container chunk.
    void begin_container_chunk(uint32_t tag, uint32_t version = 0);

    /// Append raw payload bytes to the current chunk.
    void write(const void* data, size_t size);

    /// Append raw payload bytes to the current chunk.
    void write(std::span<const uint8_t> data);

    /// End the current chunk and return to its parent.
    void end_chunk();

    /// Finalize the chunk tree. All chunks must be closed before finalization.
    void finish();

    /// Write the finalized pack to a file.
    void write_to_file(const std::filesystem::path& path);

    /// Write the pack to an arbitrary stream. The stream must be writable, empty, and positioned at
    /// the beginning (offset 0), because PTOC entries store absolute byte offsets from the stream
    /// start.
    void write_to_stream(sgl::Stream& stream);

private:
    struct ChunkNode {
        uint32_t tag;
        uint32_t version;
        PackChunkFlags flags;
        size_t data_offset;
        size_t data_size;
        size_t absolute_offset;
        size_t payload_size; ///< Cached serialized payload size (set by compute_layout).
        std::vector<ChunkNode> children;
    };

    static size_t ptoc_payload_size(const ChunkNode& node);
    static void compute_layout(ChunkNode& node, size_t offset);
    void begin_chunk_internal(uint32_t tag, uint32_t version, PackChunkFlags flags);
    void write_node(sgl::Stream& stream, const ChunkNode& node);

    ChunkNode m_root;
    std::vector<uint8_t> m_buffer;
    std::vector<ChunkNode*> m_stack;
    bool m_finalized = false;

    FALCOR_NON_COPYABLE_AND_MOVABLE(PackWriter);
};

// -------------------------------------------------------------------------------------------------
// PackReader
// -------------------------------------------------------------------------------------------------

/// Reads and validates a pack file from disk or a borrowed memory span.
class FALCOR_API PackReader {
public:
    // Forward declarations for nested types.
    class ChunkView;
    class ChunkIterator;
    class ChunkRange;

    // ----- ChunkView -----

    /// Lightweight view of a single chunk inside a PackReader.
    class FALCOR_API ChunkView {
    public:
        /// Return the chunk FourCC tag.
        uint32_t tag() const;

        /// Return the chunk version.
        uint32_t version() const;

        /// Return the chunk flags.
        PackChunkFlags flags() const;

        /// Return true if this chunk contains child chunks.
        bool is_container() const;

        /// Return the data payload size. Container chunks report zero.
        uint64_t size() const;

        /// Return the data payload bytes. Throws if this is a container chunk.
        std::span<const uint8_t> data() const;

        /// Return the number of direct child chunks. Throws if this is a data chunk.
        size_t child_count() const;

        /// Find the first direct child with the given tag. Throws if this is a data chunk.
        std::optional<ChunkView> find_child(uint32_t tag) const;

        /// Iterate over direct child chunks. Throws if this is a data chunk.
        ChunkRange children() const;

    private:
        friend class PackReader;
        friend class ChunkRange;
        ChunkView(const detail::PackChunkHeader* header, std::span<const uint8_t> data);
        const detail::PackChunkHeader* m_header;
        std::span<const uint8_t> m_data;
    };

    // ----- ChunkIterator -----

    /// Input iterator over chunk views.
    class FALCOR_API ChunkIterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = ChunkView;
        using difference_type = std::ptrdiff_t;
        using pointer = const ChunkView*;
        using reference = ChunkView;

        /// Return the current chunk view.
        ChunkView operator*() const;

        /// Advance to the next chunk.
        ChunkIterator& operator++();

        /// Advance to the next chunk and return the previous iterator.
        ChunkIterator operator++(int);

        /// Compare iterator positions.
        bool operator==(const ChunkIterator& other) const;

        /// Compare iterator positions.
        bool operator!=(const ChunkIterator& other) const;

    private:
        friend class ChunkRange;
        ChunkIterator(std::span<const uint8_t> data, const detail::PackTocEntry* entries, uint32_t index);
        std::span<const uint8_t> m_data;
        const detail::PackTocEntry* m_entries;
        uint32_t m_index;
    };

    // ----- ChunkRange -----

    /// Iterable range of chunk views.
    class FALCOR_API ChunkRange {
    public:
        /// Return an iterator to the first chunk.
        ChunkIterator begin() const;

        /// Return the end iterator.
        ChunkIterator end() const;

    private:
        friend class PackReader;
        friend class ChunkView;
        ChunkRange(std::span<const uint8_t> data, const detail::PackTocEntry* entries, uint32_t count);
        std::span<const uint8_t> m_data;
        const detail::PackTocEntry* m_entries;
        uint32_t m_count;
    };

    // ----- PackReader methods -----

    /// Open, memory-map, and validate a pack file from disk.
    PackReader(const std::filesystem::path& path);

    /// Validate and read a pack file from a borrowed memory span. The span must outlive the reader,
    /// and its data pointer must be aligned to detail::PackChunkHeader.
    PackReader(std::span<const uint8_t> data);

    /// Destroy the reader and release any owned memory mapping.
    ~PackReader();

    /// Move construct a reader, transferring ownership of any memory mapping.
    PackReader(PackReader&& other) noexcept;

    /// Move assign a reader, transferring ownership of any memory mapping.
    PackReader& operator=(PackReader&& other) noexcept;

    /// Return the root chunk FourCC tag.
    uint32_t tag() const;

    /// Return the root chunk version.
    uint32_t version() const;

    /// Return the number of top-level chunks.
    size_t chunk_count() const;

    /// Find the first top-level chunk with the given tag.
    std::optional<ChunkView> find_chunk(uint32_t tag) const;

    /// Iterate over top-level chunks.
    ChunkRange chunks() const;

private:
    void validate();

    std::unique_ptr<sgl::MemoryMappedFile> m_mmap;
    std::span<const uint8_t> m_data;

    FALCOR_NON_COPYABLE(PackReader);
};

} // namespace falcor
