// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/error.h"
#include "falcor2/core/types.h"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <concepts>
#include <cstring>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace falcor {

// -------------------------------------------------------------------------------------------------
// Archive wire format
// -------------------------------------------------------------------------------------------------
//
// An archive is a compact record format for structured metadata inside a bounded byte span. The
// archive layer is payload-only: it does not write a top-level magic number, version, checksum, or
// byte-order marker. Outer containers such as PackFile chunks own framing and version policy.
//
// A record is a sequence of fields sorted by strictly increasing nonzero field ID:
//
//     Field field_0
//     Field field_1
//     ...
//
// Field IDs are local to one record. Nested records start new ID namespaces, so a parent record, a
// section record, and each record-list element may all use field ID 1 independently.
//
// Each field starts with a varuint header:
//
//     header = (uint64_t(field_id) << 3) | uint64_t(ArchiveEncoding)
//
// The low three bits store ArchiveEncoding, and the remaining high bits store the absolute field ID.
// Field ID zero, invalid encodings, IDs outside uint32_t, duplicate IDs, and decreasing IDs are
// rejected by readers.
//
// Primitive payload encodings:
//
//     byte:    uint8_t
//     varuint: canonical unsigned LEB128, at most 10 bytes
//     varsint: zig-zag encoded int64_t, then varuint
//     fixed4:  4 raw bytes
//     fixed8:  8 raw bytes
//     fixed12: 12 raw bytes
//     fixed16: 16 raw bytes
//     sized:   varuint byte_count, then byte_count raw bytes
//
// Raw fixed payloads are little-endian host memory copies for explicitly supported trivially
// copyable value types. The built-in raw codecs require little-endian hosts at compile time.
//
// Higher-level helpers are encoded as regular sized fields so unknown readers can skip them without
// understanding the helper:
//
//     section(id):
//         field header with ArchiveEncoding::sized
//         varuint section_byte_count
//         nested record bytes
//
//     list<T>(id):
//         field header with ArchiveEncoding::sized
//         varuint list_byte_count
//         varuint element_count
//         element payloads encoded by ArchiveCodec<T>, without per-element field headers
//
//     record_list<T>(id):
//         field header with ArchiveEncoding::sized
//         varuint list_byte_count
//         varuint element_count
//         repeated sized nested records:
//             varuint element_record_byte_count
//             element record bytes
//
// Missing fields leave caller-provided defaults unchanged. Unknown fields are skipped by their
// encoding. Requested fields are decoded into local temporaries and assigned only after the full
// payload validates. Callers must explicitly call ArchiveDecoder::finish() or
// ArchiveDeserializer::finish(); destructors do not validate trailing bytes because they must not
// throw.

// -------------------------------------------------------------------------------------------------
// ArchiveEncoding
// -------------------------------------------------------------------------------------------------

/// Payload encoding stored in the low three bits of each archive field header.
enum class ArchiveEncoding : uint8_t {
    /// A single byte payload.
    byte = 0,
    /// Canonical unsigned LEB128 payload.
    varuint = 1,
    /// Zig-zag signed integer payload encoded as varuint.
    varsint = 2,
    /// Four raw bytes.
    fixed4 = 3,
    /// Eight raw bytes.
    fixed8 = 4,
    /// Twelve raw bytes.
    fixed12 = 5,
    /// Sixteen raw bytes.
    fixed16 = 6,
    /// Varuint byte count followed by that many raw bytes.
    sized = 7,
};

// -------------------------------------------------------------------------------------------------
// Raw byte views
// -------------------------------------------------------------------------------------------------

/// Zero-copy view of a sized archive payload.
struct ArchiveBlobView {
    /// Borrowed bytes. The backing archive bytes must outlive this view.
    std::span<const uint8_t> bytes;
};

// -------------------------------------------------------------------------------------------------
// ArchiveSink
// -------------------------------------------------------------------------------------------------

/// Raw byte destination used by ArchiveEncoder.
class FALCOR_API ArchiveSink {
public:
    virtual ~ArchiveSink() = default;

    /// Append raw bytes to the sink.
    virtual void write_bytes(const uint8_t* data, size_t byte_count) = 0;
};

/// ArchiveSink implementation backed by a growing byte buffer.
class FALCOR_API BufferArchiveSink final : public ArchiveSink {
public:
    /// Reserve storage in the backing buffer.
    void reserve(size_t byte_count) { m_buffer.reserve(byte_count); }
    /// Return the number of bytes written.
    size_t size() const { return m_buffer.size(); }
    /// Return true if no bytes have been written.
    bool empty() const { return m_buffer.empty(); }
    /// Return the written bytes.
    std::span<const uint8_t> bytes() const { return m_buffer; }
    /// Return the backing byte buffer.
    const std::vector<uint8_t>& buffer() const { return m_buffer; }

    /// Append raw bytes to the backing buffer.
    void write_bytes(const uint8_t* data, size_t byte_count) override
    {
        if (byte_count == 0)
            return;
        FALCOR_CHECK(data != nullptr, "cannot write non-empty archive bytes from a null pointer");
        size_t offset = m_buffer.size();
        m_buffer.resize(offset + byte_count);
        std::memcpy(m_buffer.data() + offset, data, byte_count);
    }

private:
    std::vector<uint8_t> m_buffer;
};

/// ArchiveSink implementation that writes into caller-provided fixed storage.
class FALCOR_API SpanArchiveSink final : public ArchiveSink {
public:
    /// Create a sink over a mutable byte span.
    explicit SpanArchiveSink(std::span<uint8_t> bytes)
        : m_bytes(bytes)
    {
    }

    /// Return the current write offset.
    size_t offset() const { return m_offset; }
    /// Return the number of bytes written.
    size_t size() const { return m_offset; }
    /// Return the total writable capacity.
    size_t capacity() const { return m_bytes.size(); }
    /// Return the remaining writable capacity.
    size_t remaining() const { return m_bytes.size() - m_offset; }
    /// Return true if no bytes have been written.
    bool empty() const { return m_offset == 0; }
    /// Return the written prefix of the provided span.
    std::span<const uint8_t> bytes() const { return m_bytes.first(m_offset); }

    /// Append raw bytes. Throws if the span capacity would be exceeded.
    void write_bytes(const uint8_t* data, size_t byte_count) override
    {
        if (byte_count == 0)
            return;
        FALCOR_CHECK(data != nullptr, "cannot write non-empty archive bytes from a null pointer");
        FALCOR_CHECK(byte_count <= remaining(), "archive span sink capacity exceeded");
        std::memcpy(m_bytes.data() + m_offset, data, byte_count);
        m_offset += byte_count;
    }

private:
    std::span<uint8_t> m_bytes;
    size_t m_offset = 0;
};

// -------------------------------------------------------------------------------------------------
// ArchiveSource
// -------------------------------------------------------------------------------------------------

/// Raw byte source used by ArchiveDecoder.
class FALCOR_API ArchiveSource {
public:
    virtual ~ArchiveSource() = default;

    /// Return the current read offset.
    virtual size_t offset() const = 0;
    /// Return the number of unread bytes.
    virtual size_t remaining() const = 0;
    /// Read raw bytes and advance the cursor. The returned pointer is borrowed from the source.
    virtual const uint8_t* read_bytes(size_t byte_count) = 0;
    /// Advance the cursor without returning bytes.
    virtual void skip(size_t byte_count) = 0;
};

/// ArchiveSource implementation over a borrowed bounded byte span.
class FALCOR_API SpanArchiveSource final : public ArchiveSource {
public:
    /// Create a source over borrowed archive bytes.
    explicit SpanArchiveSource(std::span<const uint8_t> bytes)
        : m_bytes(bytes)
    {
    }

    /// Return the current read offset.
    size_t offset() const override { return m_offset; }
    /// Return the number of unread bytes.
    size_t remaining() const override { return m_bytes.size() - m_offset; }
    /// Read raw bytes and advance the cursor. Throws if the read extends past the span.
    const uint8_t* read_bytes(size_t byte_count) override
    {
        FALCOR_CHECK(byte_count <= remaining(), "archive read extends past end of bounded byte span");
        const uint8_t* result = m_bytes.data() + m_offset;
        m_offset += byte_count;
        return result;
    }
    /// Advance the cursor. Throws if the skip extends past the span.
    void skip(size_t byte_count) override
    {
        FALCOR_CHECK(byte_count <= remaining(), "archive skip extends past end of bounded byte span");
        m_offset += byte_count;
    }

private:
    std::span<const uint8_t> m_bytes;
    size_t m_offset = 0;
};

// -------------------------------------------------------------------------------------------------
// ArchiveEncoder
// -------------------------------------------------------------------------------------------------

/// Writes primitive archive wire encodings to an ArchiveSink.
class FALCOR_API ArchiveEncoder {
public:
    /// Create an encoder that writes to the given sink.
    explicit ArchiveEncoder(ArchiveSink& sink);

    /// Write one raw byte.
    void write_byte(uint8_t value);
    /// Write raw bytes without a length prefix.
    void write_bytes(const uint8_t* data, size_t byte_count);
    /// Write a canonical unsigned LEB128 integer.
    void write_varuint(uint64_t value);
    /// Write a zig-zag signed integer.
    void write_varsint(int64_t value);
    /// Write a sized payload: varuint byte count followed by raw bytes.
    void write_sized(const uint8_t* data, size_t byte_count);
    /// Write a sized payload: varuint byte count followed by raw bytes.
    void write_sized(std::span<const uint8_t> bytes) { write_sized(bytes.data(), bytes.size()); }

private:
    ArchiveSink& m_sink;
};

// -------------------------------------------------------------------------------------------------
// ArchiveDecoder
// -------------------------------------------------------------------------------------------------

/// Reads primitive archive wire encodings from an ArchiveSource.
class FALCOR_API ArchiveDecoder {
public:
    /// Create a decoder that reads from the given source.
    explicit ArchiveDecoder(ArchiveSource& source);

    /// Return the current read offset in the source.
    size_t offset() const;
    /// Return the number of unread bytes.
    size_t remaining() const;
    /// Return true if the source has no unread bytes.
    bool empty() const;

    /// Read one raw byte.
    uint8_t read_byte();
    /// Read raw bytes without a length prefix.
    const uint8_t* read_bytes(size_t byte_count);
    /// Skip raw bytes.
    void skip(size_t byte_count);
    /// Read a canonical unsigned LEB128 integer.
    uint64_t read_varuint();
    /// Read a varuint and reject values above max_value.
    uint64_t read_varuint(uint64_t max_value);
    /// Read a zig-zag signed integer.
    int64_t read_varsint();
    /// Read a varsint and reject values outside the inclusive range.
    int64_t read_varsint(int64_t min_value, int64_t max_value);
    /// Read a sized payload and return a borrowed span into the source.
    std::span<const uint8_t> read_sized();
    /// Read a sized payload with an exact byte count and return a borrowed pointer into the source.
    const uint8_t* read_sized(size_t expected_byte_count);
    /// Verify that the source has been fully consumed.
    void finish() const;

private:
    ArchiveSource& m_source;
};

// -------------------------------------------------------------------------------------------------
// ArchiveSerializer
// -------------------------------------------------------------------------------------------------

/// Writes ordered archive fields and nested records to an ArchiveEncoder.
class FALCOR_API ArchiveSerializer {
public:
    /// Create a root serializer over a caller-provided encoder.
    explicit ArchiveSerializer(ArchiveEncoder& encoder);

    ArchiveSerializer(const ArchiveSerializer&) = delete;
    ArchiveSerializer& operator=(const ArchiveSerializer&) = delete;
    /// Move construct a serializer. Moving a serializer with open child sections is rejected.
    ArchiveSerializer(ArchiveSerializer&& other);
    ArchiveSerializer& operator=(ArchiveSerializer&&) = delete;

    /// Write a field. Field IDs must be nonzero and strictly increasing within this record.
    template<typename T>
    void field(uint32_t id, const T& value);

    /// Write a counted list of codec-backed element payloads as a sized field.
    template<typename T>
    void list(uint32_t id, std::span<const T> values);
    /// Write a counted list of codec-backed element payloads as a sized field.
    template<typename T>
    void list(uint32_t id, const std::vector<T>& values)
    {
        list<T>(id, std::span<const T>(values));
    }

    /// Write a counted list of nested archive records as a sized field.
    template<typename T, typename WriteFn>
    void record_list(uint32_t id, std::span<const T> values, WriteFn&& write_element);
    /// Write a counted list of nested archive records as a sized field.
    template<typename T, typename WriteFn>
    void record_list(uint32_t id, const std::vector<T>& values, WriteFn&& write_element)
    {
        record_list<T>(id, std::span<const T>(values), std::forward<WriteFn>(write_element));
    }

    /// Create a nested record field. The returned section must be finished before the parent can continue.
    ArchiveSerializer section(uint32_t id);

    /// Finish this serializer. Section serializers flush their sized payload to the parent here.
    void finish();

private:
    ArchiveSerializer(ArchiveSerializer& parent, uint32_t id);
    ArchiveEncoder& encoder();
    void validate_next_field_id(uint32_t id) const;
    void commit_field_id(uint32_t id);

    ArchiveSerializer* m_parent = nullptr;
    uint32_t m_section_id = 0;
    std::optional<BufferArchiveSink> m_owned_sink;
    std::optional<ArchiveEncoder> m_owned_encoder;
    ArchiveEncoder* m_encoder = nullptr;
    uint32_t m_previous_field_id = 0;
    uint32_t m_open_child_sections = 0;
    bool m_finished = false;
};

// -------------------------------------------------------------------------------------------------
// ArchiveDeserializer
// -------------------------------------------------------------------------------------------------

/// Reads ordered archive fields and nested records from an ArchiveDecoder.
class FALCOR_API ArchiveDeserializer {
public:
    /// Create a root deserializer over a caller-provided decoder.
    explicit ArchiveDeserializer(ArchiveDecoder& decoder);

    ArchiveDeserializer(const ArchiveDeserializer&) = delete;
    ArchiveDeserializer& operator=(const ArchiveDeserializer&) = delete;
    /// Move construct a deserializer, transferring any owned nested source.
    ArchiveDeserializer(ArchiveDeserializer&& other);
    ArchiveDeserializer& operator=(ArchiveDeserializer&&) = delete;

    /// Read a field by ID. Returns false when missing and leaves value unchanged.
    template<typename T>
    bool field(uint32_t id, T& value);

    /// Read a counted list of codec-backed element payloads.
    template<typename T>
    bool list(uint32_t id, std::vector<T>& values);

    /// Read a counted list of nested archive records.
    template<typename T, typename ReadFn>
    bool record_list(uint32_t id, std::vector<T>& values, ReadFn&& read_element);

    /// Read a nested record field. Returns std::nullopt when missing.
    std::optional<ArchiveDeserializer> section(uint32_t id);

    /// Skip and validate all remaining fields, then verify that the record is fully consumed.
    void finish();

private:
    struct FieldHeader {
        uint32_t id;
        ArchiveEncoding encoding;
    };

    explicit ArchiveDeserializer(std::span<const uint8_t> bytes);
    ArchiveDecoder& decoder();
    FieldHeader read_field_header();
    void read_next_field_if_needed();
    void skip_pending_field();
    void validate_next_requested_field_id(uint32_t id) const;
    void commit_requested_field_id(uint32_t id);

    std::optional<SpanArchiveSource> m_owned_source;
    std::optional<ArchiveDecoder> m_owned_decoder;
    ArchiveDecoder* m_decoder = nullptr;
    uint32_t m_previous_requested_field_id = 0;
    uint32_t m_previous_encoded_field_id = 0;
    std::optional<FieldHeader> m_pending_field;
    bool m_finished = false;
};

namespace detail {

template<typename T>
inline constexpr bool dependent_false_v = false;

FALCOR_API void write_field_header(ArchiveEncoder& encoder, uint32_t id, ArchiveEncoding encoding);
FALCOR_API void skip_archive_payload(ArchiveDecoder& decoder, ArchiveEncoding encoding);

template<typename T>
concept ArchiveClass = std::same_as<std::remove_cvref_t<T>, ArchiveSerializer>
    || std::same_as<std::remove_cvref_t<T>, ArchiveDeserializer>;

template<typename Archive, typename T>
using ArchiveObject = std::conditional_t<std::same_as<std::remove_cvref_t<Archive>, ArchiveSerializer>, const T, T>;

template<typename T>
concept HasAdlWriteArchive = requires(ArchiveSerializer& serializer, const T& value) {
    { archive(serializer, value) } -> std::same_as<void>;
};

template<typename T>
concept HasAdlReadArchive = requires(ArchiveDeserializer& deserializer, T& value) {
    { archive(deserializer, value) } -> std::same_as<void>;
};

template<typename T>
concept AdlArchiveRecordType
    = std::default_initializable<T> && HasAdlWriteArchive<T> && HasAdlReadArchive<T> && !std::is_enum_v<T>;

template<typename T>
void invoke_adl_archive(ArchiveSerializer& serializer, const T& value)
{
    archive(serializer, value);
}

template<typename T>
void invoke_adl_archive(ArchiveDeserializer& deserializer, T& value)
{
    archive(deserializer, value);
}

/// Validate a counted list size before reserving destination storage.
inline size_t validate_list_element_count(uint64_t element_count, size_t remaining_byte_count, size_t max_element_count)
{
    FALCOR_CHECK(
        element_count <= static_cast<uint64_t>(remaining_byte_count),
        "archive list element count exceeds payload byte count"
    );
    FALCOR_CHECK(
        element_count <= static_cast<uint64_t>(max_element_count),
        "archive list element count exceeds destination capacity"
    );
    return static_cast<size_t>(element_count);
}

} // namespace detail

/// Define a single-body ADL archive hook for TYPE.
///
/// The generated function has fixed parameter names `archive` and `value`, and must be used in the
/// same namespace as TYPE so ADL can find it. The `value` parameter is const when writing and
/// mutable when reading.
#define FALCOR_ARCHIVE(TYPE)                                                                                           \
    template<typename Archive>                                                                                         \
        requires ::falcor::detail::ArchiveClass<Archive>                                                               \
    void archive(Archive& archive, ::falcor::detail::ArchiveObject<Archive, TYPE>& value)

/// Type-specific archive payload codec. Types may specialize this template explicitly or define
/// ADL archive hooks.
template<typename T>
struct ArchiveCodec { };

/// Generic enum archive codec. Signed enum values are encoded as signed varints
/// and unsigned enum values are encoded as unsigned varints.
template<typename T>
    requires std::is_enum_v<T>
struct ArchiveCodec<T> {
    using UnderlyingT = std::underlying_type_t<T>;

    static constexpr ArchiveEncoding encoding
        = std::is_signed_v<UnderlyingT> ? ArchiveEncoding::varsint : ArchiveEncoding::varuint;

    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        UnderlyingT underlying = static_cast<UnderlyingT>(value);
        if constexpr (std::is_signed_v<UnderlyingT>) {
            encoder.write_varsint(static_cast<int64_t>(underlying));
        } else {
            encoder.write_varuint(static_cast<uint64_t>(underlying));
        }
    }

    static T read_payload(ArchiveDecoder& decoder)
    {
        if constexpr (std::is_signed_v<UnderlyingT>) {
            int64_t value = decoder.read_varsint(
                static_cast<int64_t>(std::numeric_limits<UnderlyingT>::min()),
                static_cast<int64_t>(std::numeric_limits<UnderlyingT>::max())
            );
            return static_cast<T>(static_cast<UnderlyingT>(value));
        } else {
            uint64_t value = decoder.read_varuint(static_cast<uint64_t>(std::numeric_limits<UnderlyingT>::max()));
            return static_cast<T>(static_cast<UnderlyingT>(value));
        }
    }
};

/// Generic archive codec for record types with free-standing ADL archive hooks.
template<typename T>
    requires detail::AdlArchiveRecordType<T>
struct ArchiveCodec<T> {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::sized;

    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        BufferArchiveSink nested_sink;
        ArchiveEncoder nested_encoder(nested_sink);
        ArchiveSerializer serializer(nested_encoder);
        detail::invoke_adl_archive(serializer, value);
        serializer.finish();
        encoder.write_sized(nested_sink.bytes());
    }

    static T read_payload(ArchiveDecoder& decoder)
    {
        std::span<const uint8_t> bytes = decoder.read_sized();
        SpanArchiveSource nested_source(bytes);
        ArchiveDecoder nested_decoder(nested_source);
        ArchiveDeserializer deserializer(nested_decoder);

        T value{};
        detail::invoke_adl_archive(deserializer, value);
        deserializer.finish();
        return value;
    }
};

/// A value type with a defined archive payload codec.
template<typename T>
concept ArchiveValueType = requires(ArchiveEncoder& encoder, ArchiveDecoder& decoder, const std::decay_t<T>& value) {
    { ArchiveCodec<std::decay_t<T>>::encoding } -> std::convertible_to<ArchiveEncoding>;
    { ArchiveCodec<std::decay_t<T>>::write_payload(encoder, value) } -> std::same_as<void>;
    { ArchiveCodec<std::decay_t<T>>::read_payload(decoder) } -> std::same_as<std::decay_t<T>>;
};

template<typename T>
void ArchiveSerializer::field(uint32_t id, const T& value)
{
    if constexpr (ArchiveValueType<T>) {
        using ValueT = std::decay_t<T>;
        validate_next_field_id(id);
        ArchiveEncoder& archive_encoder = encoder();
        detail::write_field_header(archive_encoder, id, ArchiveCodec<ValueT>::encoding);
        ArchiveCodec<ValueT>::write_payload(archive_encoder, value);
        commit_field_id(id);
    } else {
        static_assert(detail::dependent_false_v<T>, "ArchiveCodec<T> is not defined for this type");
    }
}

template<typename T>
void ArchiveSerializer::list(uint32_t id, std::span<const T> values)
{
    if constexpr (ArchiveValueType<T>) {
        using ValueT = std::decay_t<T>;
        validate_next_field_id(id);

        BufferArchiveSink list_sink;
        ArchiveEncoder list_encoder(list_sink);
        list_encoder.write_varuint(static_cast<uint64_t>(values.size()));
        for (const T& value : values)
            ArchiveCodec<ValueT>::write_payload(list_encoder, value);

        ArchiveEncoder& archive_encoder = encoder();
        detail::write_field_header(archive_encoder, id, ArchiveEncoding::sized);
        archive_encoder.write_sized(list_sink.bytes());
        commit_field_id(id);
    } else {
        static_assert(detail::dependent_false_v<T>, "ArchiveCodec<T> is not defined for this type");
    }
}

template<typename T, typename WriteFn>
void ArchiveSerializer::record_list(uint32_t id, std::span<const T> values, WriteFn&& write_element)
{
    validate_next_field_id(id);

    BufferArchiveSink list_sink;
    ArchiveEncoder list_encoder(list_sink);
    list_encoder.write_varuint(static_cast<uint64_t>(values.size()));
    for (const T& value : values) {
        BufferArchiveSink element_sink;
        ArchiveEncoder element_encoder(element_sink);
        ArchiveSerializer element_serializer(element_encoder);
        write_element(element_serializer, value);
        element_serializer.finish();
        list_encoder.write_sized(element_sink.bytes());
    }

    ArchiveEncoder& archive_encoder = encoder();
    detail::write_field_header(archive_encoder, id, ArchiveEncoding::sized);
    archive_encoder.write_sized(list_sink.bytes());
    commit_field_id(id);
}

template<typename T>
bool ArchiveDeserializer::field(uint32_t id, T& value)
{
    if constexpr (ArchiveValueType<T>) {
        using ValueT = std::decay_t<T>;
        validate_next_requested_field_id(id);
        commit_requested_field_id(id);

        while (true) {
            read_next_field_if_needed();
            if (!m_pending_field.has_value())
                return false;

            if (m_pending_field->id < id) {
                skip_pending_field();
                continue;
            }

            if (m_pending_field->id > id)
                return false;

            FALCOR_CHECK(
                m_pending_field->encoding == ArchiveCodec<ValueT>::encoding,
                "archive field {} has unexpected encoding",
                id
            );

            ValueT decoded = ArchiveCodec<ValueT>::read_payload(decoder());
            m_pending_field.reset();
            value = std::move(decoded);
            return true;
        }
    } else {
        static_assert(detail::dependent_false_v<T>, "ArchiveCodec<T> is not defined for this type");
        return false;
    }
}

template<typename T>
bool ArchiveDeserializer::list(uint32_t id, std::vector<T>& values)
{
    if constexpr (ArchiveValueType<T>) {
        using ValueT = std::decay_t<T>;
        ArchiveBlobView list_bytes;
        if (!field(id, list_bytes))
            return false;

        SpanArchiveSource list_source(list_bytes.bytes);
        ArchiveDecoder list_decoder(list_source);
        uint64_t element_count = list_decoder.read_varuint();

        std::vector<T> decoded;
        size_t count = detail::validate_list_element_count(element_count, list_decoder.remaining(), decoded.max_size());
        decoded.reserve(count);
        for (size_t i = 0; i < count; ++i)
            decoded.push_back(ArchiveCodec<ValueT>::read_payload(list_decoder));

        list_decoder.finish();
        values = std::move(decoded);
        return true;
    } else {
        static_assert(detail::dependent_false_v<T>, "ArchiveCodec<T> is not defined for this type");
        return false;
    }
}

template<typename T, typename ReadFn>
bool ArchiveDeserializer::record_list(uint32_t id, std::vector<T>& values, ReadFn&& read_element)
{
    ArchiveBlobView list_bytes;
    if (!field(id, list_bytes))
        return false;

    SpanArchiveSource list_source(list_bytes.bytes);
    ArchiveDecoder list_decoder(list_source);
    uint64_t element_count = list_decoder.read_varuint();

    std::vector<T> decoded;
    size_t count = detail::validate_list_element_count(element_count, list_decoder.remaining(), decoded.max_size());
    decoded.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        std::span<const uint8_t> element_bytes = list_decoder.read_sized();
        ArchiveDeserializer element_deserializer(element_bytes);
        T value = read_element(element_deserializer);
        element_deserializer.finish();
        decoded.push_back(std::move(value));
    }

    list_decoder.finish();
    values = std::move(decoded);
    return true;
}

namespace detail {

struct ArchiveBoolCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::byte;
    static void write_payload(ArchiveEncoder& encoder, const bool& value) { encoder.write_byte(value ? 1 : 0); }
    static bool read_payload(ArchiveDecoder& decoder)
    {
        uint8_t value = decoder.read_byte();
        FALCOR_CHECK(value <= 1, "archive bool payload is not canonical");
        return value != 0;
    }
};

template<typename T>
struct ArchiveByteCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::byte;
    static void write_payload(ArchiveEncoder& encoder, const T& value) { encoder.write_byte(value); }
    static T read_payload(ArchiveDecoder& decoder) { return decoder.read_byte(); }
};

template<typename T>
struct ArchiveVaruintCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::varuint;
    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        encoder.write_varuint(static_cast<uint64_t>(value));
    }
    static T read_payload(ArchiveDecoder& decoder)
    {
        return static_cast<T>(decoder.read_varuint(static_cast<uint64_t>(std::numeric_limits<T>::max())));
    }
};

template<typename T>
struct ArchiveVarsintCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::varsint;
    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        encoder.write_varsint(static_cast<int64_t>(value));
    }
    static T read_payload(ArchiveDecoder& decoder)
    {
        return static_cast<T>(decoder.read_varsint(
            static_cast<int64_t>(std::numeric_limits<T>::min()),
            static_cast<int64_t>(std::numeric_limits<T>::max())
        ));
    }
};

struct ArchiveStringCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::sized;
    static void write_payload(ArchiveEncoder& encoder, const std::string& value)
    {
        encoder.write_sized(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    }
    static std::string read_payload(ArchiveDecoder& decoder)
    {
        std::span<const uint8_t> bytes = decoder.read_sized();
        return std::string(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
};

struct ArchiveBlobViewCodec {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::sized;
    static void write_payload(ArchiveEncoder& encoder, const ArchiveBlobView& value)
    {
        encoder.write_sized(value.bytes);
    }
    static ArchiveBlobView read_payload(ArchiveDecoder& decoder) { return ArchiveBlobView{decoder.read_sized()}; }
};

template<typename T, ArchiveEncoding Encoding, size_t ByteCount>
struct ArchiveRawFixedCodec {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) == ByteCount);
    static_assert(std::endian::native == std::endian::little, "archive raw payloads require little-endian hosts");
    static constexpr ArchiveEncoding encoding = Encoding;
    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        encoder.write_bytes(reinterpret_cast<const uint8_t*>(&value), ByteCount);
    }
    static T read_payload(ArchiveDecoder& decoder)
    {
        const uint8_t* bytes = decoder.read_bytes(ByteCount);
        T value;
        std::memcpy(&value, bytes, ByteCount);
        return value;
    }
};

template<typename T, size_t ByteCount>
struct ArchiveRawSizedCodec {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(sizeof(T) == ByteCount);
    static_assert(std::endian::native == std::endian::little, "archive raw payloads require little-endian hosts");
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::sized;
    static void write_payload(ArchiveEncoder& encoder, const T& value)
    {
        encoder.write_sized(reinterpret_cast<const uint8_t*>(&value), ByteCount);
    }
    static T read_payload(ArchiveDecoder& decoder)
    {
        const uint8_t* bytes = decoder.read_sized(ByteCount);
        T value;
        std::memcpy(&value, bytes, ByteCount);
        return value;
    }
};

struct ArchivePathCodec {
    static constexpr ArchiveEncoding encoding = ArchiveStringCodec::encoding;
    static void write_payload(ArchiveEncoder& encoder, const std::filesystem::path& value)
    {
        ArchiveStringCodec::write_payload(encoder, value.string());
    }
    static std::filesystem::path read_payload(ArchiveDecoder& decoder)
    {
        return std::filesystem::path(ArchiveStringCodec::read_payload(decoder));
    }
};

} // namespace detail

#define FALCOR_DEFINE_ARCHIVE_CODEC(TYPE, ...)                                                                         \
    template<>                                                                                                         \
    struct ArchiveCodec<TYPE> : __VA_ARGS__ { }

// Built-in scalar codecs.
FALCOR_DEFINE_ARCHIVE_CODEC(bool, detail::ArchiveBoolCodec);
FALCOR_DEFINE_ARCHIVE_CODEC(uint8_t, detail::ArchiveByteCodec<uint8_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint16_t, detail::ArchiveVaruintCodec<uint16_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint32_t, detail::ArchiveVaruintCodec<uint32_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint64_t, detail::ArchiveVaruintCodec<uint64_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(int8_t, detail::ArchiveVarsintCodec<int8_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(int16_t, detail::ArchiveVarsintCodec<int16_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(int32_t, detail::ArchiveVarsintCodec<int32_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(int64_t, detail::ArchiveVarsintCodec<int64_t>);
FALCOR_DEFINE_ARCHIVE_CODEC(float, detail::ArchiveRawFixedCodec<float, ArchiveEncoding::fixed4, 4>);
FALCOR_DEFINE_ARCHIVE_CODEC(double, detail::ArchiveRawFixedCodec<double, ArchiveEncoding::fixed8, 8>);
FALCOR_DEFINE_ARCHIVE_CODEC(std::string, detail::ArchiveStringCodec);
FALCOR_DEFINE_ARCHIVE_CODEC(ArchiveBlobView, detail::ArchiveBlobViewCodec);
FALCOR_DEFINE_ARCHIVE_CODEC(std::filesystem::path, detail::ArchivePathCodec);

// Built-in Falcor value codecs.
FALCOR_DEFINE_ARCHIVE_CODEC(int2, detail::ArchiveRawFixedCodec<int2, ArchiveEncoding::fixed8, 8>);
FALCOR_DEFINE_ARCHIVE_CODEC(int3, detail::ArchiveRawFixedCodec<int3, ArchiveEncoding::fixed12, 12>);
FALCOR_DEFINE_ARCHIVE_CODEC(int4, detail::ArchiveRawFixedCodec<int4, ArchiveEncoding::fixed16, 16>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint2, detail::ArchiveRawFixedCodec<uint2, ArchiveEncoding::fixed8, 8>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint3, detail::ArchiveRawFixedCodec<uint3, ArchiveEncoding::fixed12, 12>);
FALCOR_DEFINE_ARCHIVE_CODEC(uint4, detail::ArchiveRawFixedCodec<uint4, ArchiveEncoding::fixed16, 16>);
FALCOR_DEFINE_ARCHIVE_CODEC(float2, detail::ArchiveRawFixedCodec<float2, ArchiveEncoding::fixed8, 8>);
FALCOR_DEFINE_ARCHIVE_CODEC(float3, detail::ArchiveRawFixedCodec<float3, ArchiveEncoding::fixed12, 12>);
FALCOR_DEFINE_ARCHIVE_CODEC(float4, detail::ArchiveRawFixedCodec<float4, ArchiveEncoding::fixed16, 16>);
FALCOR_DEFINE_ARCHIVE_CODEC(float3x3, detail::ArchiveRawSizedCodec<float3x3, 36>);
FALCOR_DEFINE_ARCHIVE_CODEC(float4x4, detail::ArchiveRawSizedCodec<float4x4, 64>);
FALCOR_DEFINE_ARCHIVE_CODEC(quatf, detail::ArchiveRawFixedCodec<quatf, ArchiveEncoding::fixed16, 16>);

#undef FALCOR_DEFINE_ARCHIVE_CODEC

} // namespace falcor
