// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/archive.h"

#include <limits>

namespace falcor {

namespace {

constexpr uint8_t FIELD_ENCODING_BITS = 3;
constexpr uint64_t FIELD_ENCODING_MASK = (uint64_t(1) << FIELD_ENCODING_BITS) - 1;
constexpr uint8_t VARUINT_CONTINUATION_BIT = 0x80;
constexpr uint8_t VARUINT_VALUE_MASK = 0x7f;
constexpr size_t MAX_VARUINT_BYTES = 10;

uint64_t zig_zag_encode(int64_t value)
{
    uint64_t unsigned_value = static_cast<uint64_t>(value);
    uint64_t sign = uint64_t(0) - (unsigned_value >> 63);
    return (unsigned_value << 1) ^ sign;
}

int64_t zig_zag_decode(uint64_t value)
{
    uint64_t decoded = (value >> 1) ^ (uint64_t(0) - (value & 1));
    return static_cast<int64_t>(decoded);
}

bool is_valid_encoding_bits(uint64_t encoding_bits)
{
    return encoding_bits <= static_cast<uint64_t>(ArchiveEncoding::sized);
}

} // namespace

// -------------------------------------------------------------------------------------------------
// ArchiveEncoder
// -------------------------------------------------------------------------------------------------

ArchiveEncoder::ArchiveEncoder(ArchiveSink& sink)
    : m_sink(sink)
{
}

void ArchiveEncoder::write_byte(uint8_t value)
{
    m_sink.write_bytes(&value, 1);
}

void ArchiveEncoder::write_bytes(const uint8_t* data, size_t byte_count)
{
    m_sink.write_bytes(data, byte_count);
}

void ArchiveEncoder::write_varuint(uint64_t value)
{
    uint8_t bytes[MAX_VARUINT_BYTES];
    size_t byte_count = 0;
    while (value >= VARUINT_CONTINUATION_BIT) {
        bytes[byte_count++] = static_cast<uint8_t>((value & VARUINT_VALUE_MASK) | VARUINT_CONTINUATION_BIT);
        value >>= 7;
    }
    bytes[byte_count++] = static_cast<uint8_t>(value);
    write_bytes(bytes, byte_count);
}

void ArchiveEncoder::write_varsint(int64_t value)
{
    write_varuint(zig_zag_encode(value));
}

void ArchiveEncoder::write_sized(const uint8_t* data, size_t byte_count)
{
    write_varuint(byte_count);
    write_bytes(data, byte_count);
}

// -------------------------------------------------------------------------------------------------
// ArchiveDecoder
// -------------------------------------------------------------------------------------------------

ArchiveDecoder::ArchiveDecoder(ArchiveSource& source)
    : m_source(source)
{
}

size_t ArchiveDecoder::offset() const
{
    return m_source.offset();
}

size_t ArchiveDecoder::remaining() const
{
    return m_source.remaining();
}

bool ArchiveDecoder::empty() const
{
    return m_source.remaining() == 0;
}

uint8_t ArchiveDecoder::read_byte()
{
    return *m_source.read_bytes(1);
}

const uint8_t* ArchiveDecoder::read_bytes(size_t byte_count)
{
    return m_source.read_bytes(byte_count);
}

void ArchiveDecoder::skip(size_t byte_count)
{
    m_source.skip(byte_count);
}

uint64_t ArchiveDecoder::read_varuint()
{
    uint64_t value = 0;
    for (size_t shift = 0; shift < 64; shift += 7) {
        uint8_t byte = read_byte();
        uint64_t payload = byte & VARUINT_VALUE_MASK;

        if (shift == 63)
            FALCOR_CHECK(payload <= 1, "archive varuint overflows uint64_t");

        value |= payload << shift;

        if ((byte & VARUINT_CONTINUATION_BIT) == 0) {
            FALCOR_CHECK(shift == 0 || payload != 0, "archive varuint is not canonical");
            return value;
        }
    }

    FALCOR_THROW("archive varuint exceeds maximum encoded length");
}

uint64_t ArchiveDecoder::read_varuint(uint64_t max_value)
{
    uint64_t value = read_varuint();
    FALCOR_CHECK(value <= max_value, "archive unsigned integer out of range");
    return value;
}

int64_t ArchiveDecoder::read_varsint()
{
    return zig_zag_decode(read_varuint());
}

int64_t ArchiveDecoder::read_varsint(int64_t min_value, int64_t max_value)
{
    int64_t value = read_varsint();
    FALCOR_CHECK(value >= min_value && value <= max_value, "archive signed integer out of range");
    return value;
}

std::span<const uint8_t> ArchiveDecoder::read_sized()
{
    uint64_t byte_count = read_varuint();
    FALCOR_CHECK(byte_count <= static_cast<uint64_t>(remaining()), "archive sized payload extends past end of record");
    return {read_bytes(static_cast<size_t>(byte_count)), static_cast<size_t>(byte_count)};
}

const uint8_t* ArchiveDecoder::read_sized(size_t expected_byte_count)
{
    uint64_t byte_count = read_varuint();
    FALCOR_CHECK(
        byte_count == static_cast<uint64_t>(expected_byte_count),
        "archive sized payload has unexpected byte count"
    );
    return read_bytes(expected_byte_count);
}

void ArchiveDecoder::finish() const
{
    FALCOR_CHECK(empty(), "archive decoder did not consume the full byte span");
}

// -------------------------------------------------------------------------------------------------
// ArchiveSerializer
// -------------------------------------------------------------------------------------------------

ArchiveSerializer::ArchiveSerializer(ArchiveEncoder& encoder)
    : m_encoder(&encoder)
{
}

ArchiveSerializer::ArchiveSerializer(ArchiveSerializer& parent, uint32_t id)
{
    parent.validate_next_field_id(id);

    m_owned_sink.emplace();
    m_owned_encoder.emplace(*m_owned_sink);
    m_encoder = &*m_owned_encoder;
    m_parent = &parent;
    m_section_id = id;

    parent.commit_field_id(id);
    ++parent.m_open_child_sections;
}

ArchiveSerializer::ArchiveSerializer(ArchiveSerializer&& other)
    : m_parent(other.m_parent)
    , m_section_id(other.m_section_id)
    , m_owned_sink(std::move(other.m_owned_sink))
    , m_previous_field_id(other.m_previous_field_id)
    , m_open_child_sections(other.m_open_child_sections)
    , m_finished(other.m_finished)
{
    FALCOR_CHECK(other.m_open_child_sections == 0, "cannot move archive serializer with open child sections");
    if (m_owned_sink.has_value()) {
        m_owned_encoder.emplace(*m_owned_sink);
        m_encoder = &*m_owned_encoder;
    } else {
        m_encoder = other.m_encoder;
    }
    other.m_parent = nullptr;
    other.m_encoder = nullptr;
    other.m_open_child_sections = 0;
    other.m_finished = true;
}

ArchiveSerializer ArchiveSerializer::section(uint32_t id)
{
    FALCOR_CHECK(!m_finished, "archive serializer is already finished");
    return ArchiveSerializer(*this, id);
}

void ArchiveSerializer::finish()
{
    FALCOR_CHECK(!m_finished, "archive serializer is already finished");
    FALCOR_CHECK(m_open_child_sections == 0, "archive serializer has unfinished child sections");
    if (m_parent != nullptr) {
        FALCOR_CHECK(m_owned_sink.has_value(), "archive section serializer has no owned sink");
        FALCOR_CHECK(m_parent->m_open_child_sections > 0, "archive parent serializer has no open child sections");
        detail::write_field_header(m_parent->encoder(), m_section_id, ArchiveEncoding::sized);
        m_parent->encoder().write_sized(m_owned_sink->bytes());
        --m_parent->m_open_child_sections;
    }
    m_finished = true;
}

ArchiveEncoder& ArchiveSerializer::encoder()
{
    FALCOR_CHECK(m_encoder != nullptr, "archive serializer has no encoder");
    return *m_encoder;
}

void ArchiveSerializer::validate_next_field_id(uint32_t id) const
{
    FALCOR_CHECK(!m_finished, "archive serializer is already finished");
    FALCOR_CHECK(m_open_child_sections == 0, "archive serializer has unfinished child sections");
    FALCOR_CHECK(id != 0, "archive field id 0 is invalid");
    FALCOR_CHECK(id > m_previous_field_id, "archive fields must be written in strictly increasing id order");
}

void ArchiveSerializer::commit_field_id(uint32_t id)
{
    m_previous_field_id = id;
}

// -------------------------------------------------------------------------------------------------
// ArchiveDeserializer
// -------------------------------------------------------------------------------------------------

ArchiveDeserializer::ArchiveDeserializer(ArchiveDecoder& decoder)
    : m_decoder(&decoder)
{
}

ArchiveDeserializer::ArchiveDeserializer(std::span<const uint8_t> bytes)
{
    m_owned_source.emplace(bytes);
    m_owned_decoder.emplace(*m_owned_source);
    m_decoder = &*m_owned_decoder;
}

ArchiveDeserializer::ArchiveDeserializer(ArchiveDeserializer&& other)
    : m_owned_source(std::move(other.m_owned_source))
    , m_previous_requested_field_id(other.m_previous_requested_field_id)
    , m_previous_encoded_field_id(other.m_previous_encoded_field_id)
    , m_pending_field(std::move(other.m_pending_field))
    , m_finished(other.m_finished)
{
    if (m_owned_source.has_value()) {
        m_owned_decoder.emplace(*m_owned_source);
        m_decoder = &*m_owned_decoder;
    } else {
        m_decoder = other.m_decoder;
    }
    other.m_decoder = nullptr;
    other.m_finished = true;
}

std::optional<ArchiveDeserializer> ArchiveDeserializer::section(uint32_t id)
{
    ArchiveBlobView section_bytes;
    if (!field(id, section_bytes))
        return std::nullopt;
    return std::optional<ArchiveDeserializer>(ArchiveDeserializer(section_bytes.bytes));
}

ArchiveDecoder& ArchiveDeserializer::decoder()
{
    FALCOR_CHECK(m_decoder != nullptr, "archive deserializer has no decoder");
    return *m_decoder;
}

ArchiveDeserializer::FieldHeader ArchiveDeserializer::read_field_header()
{
    uint64_t encoded_header = decoder().read_varuint();
    uint64_t encoding_bits = encoded_header & FIELD_ENCODING_MASK;
    uint64_t field_id = encoded_header >> FIELD_ENCODING_BITS;

    FALCOR_CHECK(is_valid_encoding_bits(encoding_bits), "archive field has invalid encoding");
    FALCOR_CHECK(field_id != 0, "archive field id 0 is invalid");
    FALCOR_CHECK(field_id <= std::numeric_limits<uint32_t>::max(), "archive field id exceeds uint32_t range");
    FALCOR_CHECK(
        field_id > m_previous_encoded_field_id,
        "archive encoded fields must be in strictly increasing id order"
    );

    m_previous_encoded_field_id = static_cast<uint32_t>(field_id);
    return FieldHeader{static_cast<uint32_t>(field_id), static_cast<ArchiveEncoding>(encoding_bits)};
}

void ArchiveDeserializer::read_next_field_if_needed()
{
    if (m_pending_field.has_value() || decoder().empty())
        return;
    m_pending_field = read_field_header();
}

void ArchiveDeserializer::skip_pending_field()
{
    FALCOR_CHECK(m_pending_field.has_value(), "archive deserializer has no pending field to skip");
    detail::skip_archive_payload(decoder(), m_pending_field->encoding);
    m_pending_field.reset();
}

void ArchiveDeserializer::validate_next_requested_field_id(uint32_t id) const
{
    FALCOR_CHECK(!m_finished, "archive deserializer is already finished");
    FALCOR_CHECK(id != 0, "archive field id 0 is invalid");
    FALCOR_CHECK(
        id > m_previous_requested_field_id,
        "archive fields must be requested in strictly increasing id order"
    );
}

void ArchiveDeserializer::commit_requested_field_id(uint32_t id)
{
    m_previous_requested_field_id = id;
}

void ArchiveDeserializer::finish()
{
    FALCOR_CHECK(!m_finished, "archive deserializer is already finished");
    while (true) {
        read_next_field_if_needed();
        if (!m_pending_field.has_value())
            break;
        skip_pending_field();
    }
    decoder().finish();
    m_finished = true;
}

// -------------------------------------------------------------------------------------------------
// Detail
// -------------------------------------------------------------------------------------------------

namespace detail {

void write_field_header(ArchiveEncoder& encoder, uint32_t id, ArchiveEncoding encoding)
{
    FALCOR_CHECK(id != 0, "archive field id 0 is invalid");
    uint8_t encoding_bits = static_cast<uint8_t>(encoding);
    FALCOR_CHECK(is_valid_encoding_bits(encoding_bits), "archive field has invalid encoding");
    uint64_t encoded_header = (uint64_t(id) << FIELD_ENCODING_BITS) | encoding_bits;
    encoder.write_varuint(encoded_header);
}

void skip_archive_payload(ArchiveDecoder& decoder, ArchiveEncoding encoding)
{
    switch (encoding) {
    case ArchiveEncoding::byte:
        decoder.skip(1);
        break;
    case ArchiveEncoding::varuint:
        decoder.read_varuint();
        break;
    case ArchiveEncoding::varsint:
        decoder.read_varsint();
        break;
    case ArchiveEncoding::fixed4:
        decoder.skip(4);
        break;
    case ArchiveEncoding::fixed8:
        decoder.skip(8);
        break;
    case ArchiveEncoding::fixed12:
        decoder.skip(12);
        break;
    case ArchiveEncoding::fixed16:
        decoder.skip(16);
        break;
    case ArchiveEncoding::sized: {
        uint64_t byte_count = decoder.read_varuint();
        FALCOR_CHECK(
            byte_count <= static_cast<uint64_t>(decoder.remaining()),
            "archive sized payload extends past end of record"
        );
        decoder.skip(static_cast<size_t>(byte_count));
        break;
    }
    default:
        FALCOR_THROW("archive field has invalid encoding");
    }
}

} // namespace detail

} // namespace falcor
