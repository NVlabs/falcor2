// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/archive.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("Archive");

struct ArchiveTestStruct {
    uint32_t id = 0;
    std::string name;

    bool operator==(const ArchiveTestStruct& other) const { return id == other.id && name == other.name; }
};

enum class ArchiveSignedEnum : int32_t {
    negative = -7,
    positive = 9,
};

enum class ArchiveUnsignedEnum : uint8_t {
    high = 200,
};

enum class ArchiveWideUnsignedEnum : uint64_t {
    high = 0x8000000000000000ull,
    max = 0xffffffffffffffffull,
};

namespace archive_adl_tests {

struct MacroFoo {
    int32_t i = 0;

    bool operator==(const MacroFoo& other) const { return i == other.i; }
};

FALCOR_ARCHIVE(MacroFoo)
{
    archive.field(1, value.i);
}

struct MacroBar {
    int32_t i = 0;
    MacroFoo foo;
    std::string name;

    bool operator==(const MacroBar& other) const { return i == other.i && foo == other.foo && name == other.name; }
};

FALCOR_ARCHIVE(MacroBar)
{
    archive.field(1, value.i);
    archive.field(2, value.foo);
    archive.field(3, value.name);
}

struct DefaultedRecord {
    int32_t required = 3;
    int32_t optional = 17;

    bool operator==(const DefaultedRecord& other) const
    {
        return required == other.required && optional == other.optional;
    }
};

FALCOR_ARCHIVE(DefaultedRecord)
{
    archive.field(1, value.required);
    archive.field(2, value.optional);
}

struct SeparateHook {
    int32_t i = 0;
    bool loaded = false;
};

void archive(falcor::ArchiveSerializer& archive, const SeparateHook& value)
{
    archive.field(1, value.i);
}

void archive(falcor::ArchiveDeserializer& archive, SeparateHook& value)
{
    value.loaded = archive.field(1, value.i);
}

struct MemberOnly {
    int32_t i = 0;

    void archive(falcor::ArchiveSerializer& archive) const { archive.field(1, i); }
    void archive(falcor::ArchiveDeserializer& archive) { archive.field(1, i); }
};

} // namespace archive_adl_tests

static_assert(ArchiveValueType<ArchiveSignedEnum>);
static_assert(ArchiveValueType<ArchiveUnsignedEnum>);
static_assert(ArchiveValueType<ArchiveWideUnsignedEnum>);
static_assert(ArchiveCodec<ArchiveSignedEnum>::encoding == ArchiveEncoding::varsint);
static_assert(ArchiveCodec<ArchiveUnsignedEnum>::encoding == ArchiveEncoding::varuint);
static_assert(ArchiveCodec<ArchiveWideUnsignedEnum>::encoding == ArchiveEncoding::varuint);
static_assert(ArchiveValueType<archive_adl_tests::MacroFoo>);
static_assert(ArchiveValueType<archive_adl_tests::MacroBar>);
static_assert(ArchiveValueType<archive_adl_tests::DefaultedRecord>);
static_assert(ArchiveValueType<archive_adl_tests::SeparateHook>);
static_assert(!ArchiveValueType<archive_adl_tests::MemberOnly>);

template<>
struct falcor::ArchiveCodec<ArchiveTestStruct> {
    static constexpr ArchiveEncoding encoding = ArchiveEncoding::sized;

    static void write_payload(ArchiveEncoder& encoder, const ArchiveTestStruct& value)
    {
        BufferArchiveSink nested_sink;
        ArchiveEncoder nested_encoder(nested_sink);
        ArchiveSerializer serializer(nested_encoder);
        serializer.field(1, value.id);
        serializer.field(2, value.name);
        encoder.write_sized(nested_sink.bytes());
    }

    static ArchiveTestStruct read_payload(ArchiveDecoder& decoder)
    {
        std::span<const uint8_t> bytes = decoder.read_sized();
        SpanArchiveSource nested_source(bytes);
        ArchiveDecoder nested_decoder(nested_source);
        ArchiveDeserializer deserializer(nested_decoder);

        ArchiveTestStruct value;
        deserializer.field(1, value.id);
        deserializer.field(2, value.name);
        deserializer.finish();
        return value;
    }
};

static std::vector<uint8_t> make_bytes(std::initializer_list<uint8_t> values)
{
    return std::vector<uint8_t>(values);
}

static std::vector<uint8_t> to_vector(std::span<const uint8_t> bytes)
{
    return std::vector<uint8_t>(bytes.begin(), bytes.end());
}

static std::vector<uint8_t> write_varuint_bytes(uint64_t value)
{
    BufferArchiveSink sink;
    ArchiveEncoder encoder(sink);
    encoder.write_varuint(value);
    return sink.buffer();
}

static uint64_t read_varuint_bytes(std::span<const uint8_t> bytes)
{
    SpanArchiveSource source(bytes);
    ArchiveDecoder decoder(source);
    uint64_t value = decoder.read_varuint();
    decoder.finish();
    return value;
}

template<typename T>
static std::vector<uint8_t> write_value(uint32_t id, const T& value)
{
    BufferArchiveSink sink;
    ArchiveEncoder encoder(sink);
    ArchiveSerializer serializer(encoder);
    serializer.field(id, value);
    return sink.buffer();
}

template<typename T>
static T read_value(std::span<const uint8_t> bytes, uint32_t id, T initial = {})
{
    SpanArchiveSource source(bytes);
    ArchiveDecoder decoder(source);
    ArchiveDeserializer deserializer(decoder);
    T value = std::move(initial);
    CHECK(deserializer.field(id, value));
    deserializer.finish();
    return value;
}

static std::vector<uint8_t> write_unknowns_and_values()
{
    BufferArchiveSink sink;
    ArchiveEncoder encoder(sink);
    ArchiveSerializer serializer(encoder);
    serializer.field(1, uint8_t(0xaa));
    serializer.field(2, uint32_t(55));
    serializer.field(3, uint32_t(99));
    serializer.field(5, float3{1.0f, 2.0f, 3.0f});
    serializer.field(6, float4{4.0f, 5.0f, 6.0f, 7.0f});
    serializer.field(7, std::string("kept"));
    serializer.field(10, int32_t(-42));
    serializer.field(11, 1.25f);
    serializer.field(12, 2.5);
    serializer.field(20, ArchiveBlobView{std::span<const uint8_t>()});
    return sink.buffer();
}

static void check_varuint_throws(std::vector<uint8_t> bytes)
{
    SpanArchiveSource source(bytes);
    ArchiveDecoder decoder(source);
    CHECK_THROWS(decoder.read_varuint());
}

TEST_CASE("raw source sink and encoder decoder cursors")
{
    BufferArchiveSink byte_sink;
    uint8_t x = 9;
    byte_sink.write_bytes(&x, 1);
    CHECK(byte_sink.buffer() == make_bytes({x}));

    uint8_t fixed_storage[5] = {};
    SpanArchiveSink span_sink(fixed_storage);
    span_sink.write_bytes(&x, 1);
    CHECK(to_vector(span_sink.bytes()) == make_bytes({x}));

    ArchiveEncoder fixed_encoder(span_sink);
    uint8_t bytes[] = {1, 2, 3};
    fixed_encoder.write_sized(bytes);
    CHECK(to_vector(span_sink.bytes()) == make_bytes({x, 3, 1, 2, 3}));
    CHECK_THROWS(fixed_encoder.write_byte(4));

    BufferArchiveSink sink;
    ArchiveEncoder encoder(sink);
    encoder.write_sized(bytes);
    CHECK(sink.buffer() == make_bytes({3, 1, 2, 3}));

    SpanArchiveSource source(sink.bytes());
    ArchiveDecoder decoder(source);
    std::span<const uint8_t> read = decoder.read_sized();
    CHECK(read.size() == 3);
    CHECK(std::equal(read.begin(), read.end(), bytes));
    decoder.finish();

    std::vector<uint8_t> truncated = {4, 1, 2, 3};
    SpanArchiveSource truncated_source(truncated);
    ArchiveDecoder truncated_decoder(truncated_source);
    CHECK_THROWS(truncated_decoder.read_sized());

    BufferArchiveSink nested_sink;
    ArchiveEncoder nested_encoder(nested_sink);
    uint8_t nested_value = 7;
    nested_encoder.write_sized(&nested_value, 1);
    CHECK(nested_sink.buffer() == make_bytes({1, 7}));

    std::vector<uint8_t> cursor_bytes = make_bytes({1, 2, 3});
    SpanArchiveSource cursor_source(cursor_bytes);
    CHECK(cursor_source.offset() == 0);
    CHECK(cursor_source.remaining() == 3);
    CHECK(*cursor_source.read_bytes(1) == 1);
    CHECK(cursor_source.offset() == 1);
    cursor_source.skip(2);
    CHECK(cursor_source.remaining() == 0);

    std::vector<uint8_t> unfinished_bytes = make_bytes({1});
    SpanArchiveSource unfinished(unfinished_bytes);
    ArchiveDecoder unfinished_decoder(unfinished);
    CHECK_THROWS(unfinished_decoder.finish());
}

TEST_CASE("raw varuint golden bytes")
{
    struct Case {
        uint64_t value;
        std::vector<uint8_t> bytes;
    };

    std::vector<Case> cases = {
        {0, {0x00}},
        {1, {0x01}},
        {127, {0x7f}},
        {128, {0x80, 0x01}},
        {16383, {0xff, 0x7f}},
        {16384, {0x80, 0x80, 0x01}},
        {std::numeric_limits<uint64_t>::max(), {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}},
    };

    for (const Case& c : cases) {
        CHECK(write_varuint_bytes(c.value) == c.bytes);
        CHECK(read_varuint_bytes(c.bytes) == c.value);
    }
}

TEST_CASE("raw varsint golden bytes")
{
    struct Case {
        int64_t value;
        std::vector<uint8_t> bytes;
    };

    std::vector<Case> cases = {
        {0, {0x00}},
        {-1, {0x01}},
        {1, {0x02}},
        {-2, {0x03}},
        {63, {0x7e}},
        {-64, {0x7f}},
        {64, {0x80, 0x01}},
        {std::numeric_limits<int64_t>::max(), {0xfe, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}},
        {std::numeric_limits<int64_t>::min(), {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01}},
    };

    for (const Case& c : cases) {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        encoder.write_varsint(c.value);
        CHECK(sink.buffer() == c.bytes);

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        CHECK(decoder.read_varsint() == c.value);
        decoder.finish();
    }
}

TEST_CASE("bounded primitive reads")
{
    {
        std::vector<uint8_t> bytes = write_varuint_bytes(127);
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        CHECK(decoder.read_varuint(127) == 127);
        decoder.finish();
    }

    {
        std::vector<uint8_t> bytes = write_varuint_bytes(128);
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        CHECK_THROWS(decoder.read_varuint(127));
    }

    {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        encoder.write_varsint(-128);
        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        CHECK(decoder.read_varsint(-128, 127) == -128);
        decoder.finish();
    }

    {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        encoder.write_varsint(-129);
        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        CHECK_THROWS(decoder.read_varsint(-128, 127));
    }
}

TEST_CASE("field header golden bytes")
{
    CHECK(write_value(1, uint8_t(7)) == make_bytes({0x08, 0x07}));
    CHECK(write_value(1, uint32_t(300)) == make_bytes({0x09, 0xac, 0x02}));
    CHECK(write_value(2, int32_t(-3)) == make_bytes({0x12, 0x05}));
    CHECK(write_value(15, 1.0f) == make_bytes({0x7b, 0x00, 0x00, 0x80, 0x3f}));
    CHECK(write_value(16, 1.0) == make_bytes({0x84, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f}));
    CHECK(write_value(17, std::string("abc")) == make_bytes({0x8f, 0x01, 0x03, 'a', 'b', 'c'}));
    CHECK(
        write_value(1, float3{1.0f, 2.0f, 3.0f})
        == make_bytes({0x0d, 0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x40, 0x40})
    );
    CHECK(
        write_value(1, uint4{1, 2, 3, 4})
        == make_bytes(
            {0x0e, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00}
        )
    );
}

TEST_CASE("primitive round trips")
{
    CHECK(read_value(write_value(1, true), 1, false) == true);
    CHECK(read_value(write_value(1, false), 1, true) == false);
    CHECK(read_value(write_value(2, uint8_t(255)), 2, uint8_t(0)) == uint8_t(255));
    CHECK(read_value(write_value(3, uint16_t(65535)), 3, uint16_t(0)) == uint16_t(65535));
    CHECK(read_value(write_value(4, uint32_t(0xfedcba98)), 4, uint32_t(0)) == uint32_t(0xfedcba98));
    CHECK(
        read_value(write_value(5, std::numeric_limits<uint64_t>::max()), 5, uint64_t(0))
        == std::numeric_limits<uint64_t>::max()
    );
    CHECK(read_value(write_value(6, int8_t(-128)), 6, int8_t(0)) == int8_t(-128));
    CHECK(read_value(write_value(7, int16_t(-12345)), 7, int16_t(0)) == int16_t(-12345));
    CHECK(read_value(write_value(8, int32_t(-123456789)), 8, int32_t(0)) == int32_t(-123456789));
    CHECK(
        read_value(write_value(9, std::numeric_limits<int64_t>::min()), 9, int64_t(0))
        == std::numeric_limits<int64_t>::min()
    );

    CHECK(read_value(write_value(10, 3.5f), 10, 0.0f) == 3.5f);
    CHECK(read_value(write_value(11, -0.25), 11, 0.0) == -0.25);
    CHECK(read_value(write_value(12, std::string("hello archive")), 12, std::string()) == "hello archive");
}

TEST_CASE("enum round trips")
{
    CHECK(
        read_value(write_value(1, ArchiveSignedEnum::negative), 1, ArchiveSignedEnum::positive)
        == ArchiveSignedEnum::negative
    );
    CHECK(read_value(write_value(2, ArchiveUnsignedEnum::high), 2, ArchiveUnsignedEnum()) == ArchiveUnsignedEnum::high);
    CHECK(
        read_value(write_value(3, ArchiveWideUnsignedEnum::high), 3, ArchiveWideUnsignedEnum())
        == ArchiveWideUnsignedEnum::high
    );
    CHECK(
        read_value(write_value(4, ArchiveWideUnsignedEnum::max), 4, ArchiveWideUnsignedEnum())
        == ArchiveWideUnsignedEnum::max
    );
    CHECK(write_value(2, ArchiveUnsignedEnum::high) == make_bytes({0x11, 0xc8, 0x01}));

    std::vector<uint8_t> out_of_range = write_value(1, uint16_t(300));
    SpanArchiveSource source(out_of_range);
    ArchiveDecoder decoder(source);
    ArchiveDeserializer deserializer(decoder);
    ArchiveUnsignedEnum value = ArchiveUnsignedEnum::high;
    CHECK_THROWS(deserializer.field(1, value));
    CHECK(value == ArchiveUnsignedEnum::high);
}

TEST_CASE("blob view is zero copy")
{
    std::vector<uint8_t> payload = {1, 2, 3, 4};
    std::vector<uint8_t> bytes = write_value(1, ArchiveBlobView{payload});

    SpanArchiveSource source(bytes);
    ArchiveDecoder decoder(source);
    ArchiveDeserializer deserializer(decoder);
    ArchiveBlobView view;
    CHECK(deserializer.field(1, view));
    deserializer.finish();

    REQUIRE(view.bytes.size() == payload.size());
    CHECK(std::equal(view.bytes.begin(), view.bytes.end(), payload.begin()));
    CHECK(view.bytes.data() >= bytes.data());
    CHECK(view.bytes.data() < bytes.data() + bytes.size());
}

TEST_CASE("nested records round trip")
{
    ArchiveTestStruct value{42, "nested"};
    CHECK(read_value(write_value(5, value), 5, ArchiveTestStruct{}) == value);
}

TEST_CASE("ADL record archive hooks")
{
    using namespace archive_adl_tests;

    MacroFoo foo{42};
    CHECK(read_value(write_value(1, foo), 1, MacroFoo{}) == foo);

    MacroBar bar{7, {9}, "bar"};
    CHECK(read_value(write_value(1, bar), 1, MacroBar{}) == bar);

    SeparateHook separate{123, false};
    SeparateHook decoded_separate = read_value(write_value(1, separate), 1, SeparateHook{});
    CHECK(decoded_separate.i == 123);
    CHECK(decoded_separate.loaded);

    {
        BufferArchiveSink nested_sink;
        ArchiveEncoder nested_encoder(nested_sink);
        ArchiveSerializer nested_serializer(nested_encoder);
        nested_serializer.field(1, int32_t(99));
        nested_serializer.field(4, std::string("unknown"));
        nested_serializer.finish();

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.field(1, ArchiveBlobView{nested_sink.bytes()});
        serializer.finish();

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        DefaultedRecord decoded;
        CHECK(deserializer.field(1, decoded));
        deserializer.finish();

        CHECK(decoded.required == 99);
        CHECK(decoded.optional == 17);
    }

    {
        std::vector<MacroFoo> values = {{1}, {2}, {3}};

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.list(1, values);
        serializer.finish();

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<MacroFoo> decoded;
        CHECK(deserializer.list(1, decoded));
        deserializer.finish();

        CHECK(decoded == values);
    }
}

TEST_CASE("scoped sections")
{
    {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        ArchiveSerializer section = serializer.section(1);
        CHECK_THROWS(serializer.field(2, uint8_t(1)));
        CHECK_THROWS(serializer.section(2));
        CHECK_THROWS(serializer.finish());
        section.field(1, uint8_t(7));
        section.finish();
        CHECK_THROWS(section.finish());
        CHECK(sink.buffer() == make_bytes({0x0f, 0x02, 0x08, 0x07}));
    }

    BufferArchiveSink sink;
    ArchiveEncoder encoder(sink);
    ArchiveSerializer serializer(encoder);
    serializer.field(1, uint32_t(11));
    ArchiveSerializer written_section = serializer.section(2);
    written_section.field(1, std::string("inside"));
    written_section.field(4, uint32_t(99));
    written_section.finish();
    serializer.field(4, uint32_t(22));

    SpanArchiveSource source(sink.bytes());
    ArchiveDecoder decoder(source);
    ArchiveDeserializer deserializer(decoder);
    uint32_t before = 0;
    uint32_t after = 0;
    std::string section_text = "default";
    uint32_t section_number = 0;

    CHECK(deserializer.field(1, before));
    std::optional<ArchiveDeserializer> section = deserializer.section(2);
    REQUIRE(section.has_value());
    CHECK(section->field(1, section_text));
    CHECK(section->field(4, section_number));
    section->finish();
    CHECK(deserializer.field(4, after));
    deserializer.finish();

    CHECK(before == 11);
    CHECK(after == 22);
    CHECK(section_text == "inside");
    CHECK(section_number == 99);

    SpanArchiveSource missing_source(sink.bytes());
    ArchiveDecoder missing_decoder(missing_source);
    ArchiveDeserializer missing_deserializer(missing_decoder);
    CHECK(missing_deserializer.field(1, before));
    CHECK(!missing_deserializer.section(3).has_value());
    CHECK(missing_deserializer.field(4, after));
    missing_deserializer.finish();
}

TEST_CASE("counted lists")
{
    {
        std::vector<uint32_t> values = {1, 300};

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.list(1, values);
        CHECK(sink.buffer() == make_bytes({0x0f, 0x04, 0x02, 0x01, 0xac, 0x02}));

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<uint32_t> decoded;
        CHECK(deserializer.list(1, decoded));
        deserializer.finish();
        CHECK(decoded == values);
    }

    {
        std::vector<uint32_t> values = {11, 22};
        std::vector<uint32_t> empty;

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.field(1, uint32_t(7));
        serializer.list(3, values);
        serializer.list(4, empty);

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint32_t skipped = 0;
        std::vector<uint32_t> missing = {99};
        std::vector<uint32_t> decoded;
        std::vector<uint32_t> decoded_empty = {1};

        CHECK(deserializer.field(1, skipped));
        CHECK(!deserializer.list(2, missing));
        CHECK(missing == std::vector<uint32_t>{99});
        CHECK(deserializer.list(3, decoded));
        CHECK(deserializer.list(4, decoded_empty));
        deserializer.finish();

        CHECK(decoded == values);
        CHECK(decoded_empty.empty());
    }

    {
        std::vector<ArchiveTestStruct> values = {{1, "one"}, {2, "two"}};

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.record_list(
            1,
            values,
            [](ArchiveSerializer& element, const ArchiveTestStruct& value)
            {
                element.field(1, value.id);
                element.field(2, value.name);
            }
        );

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<ArchiveTestStruct> decoded;
        CHECK(deserializer.record_list(
            1,
            decoded,
            [](ArchiveDeserializer& element)
            {
                ArchiveTestStruct value;
                element.field(1, value.id);
                element.field(2, value.name);
                return value;
            }
        ));
        deserializer.finish();
        CHECK(decoded == values);
    }
}

TEST_CASE("math and path codecs")
{
    CHECK(read_value(write_value(1, int2{-1, 2}), 1, int2{}) == int2{-1, 2});
    CHECK(read_value(write_value(1, int3{-1, 2, -3}), 1, int3{}) == int3{-1, 2, -3});
    CHECK(read_value(write_value(1, int4{-1, 2, -3, 4}), 1, int4{}) == int4{-1, 2, -3, 4});

    CHECK(read_value(write_value(1, uint2{1, 2}), 1, uint2{}) == uint2{1, 2});
    CHECK(read_value(write_value(1, uint3{1, 2, 3}), 1, uint3{}) == uint3{1, 2, 3});
    CHECK(read_value(write_value(1, uint4{1, 2, 3, 4}), 1, uint4{}) == uint4{1, 2, 3, 4});

    CHECK(read_value(write_value(1, float2{1.5f, -2.5f}), 1, float2{}) == float2{1.5f, -2.5f});
    CHECK(read_value(write_value(1, float3{1.5f, -2.5f, 3.25f}), 1, float3{}) == float3{1.5f, -2.5f, 3.25f});
    CHECK(
        read_value(write_value(1, float4{1.5f, -2.5f, 3.25f, -4.75f}), 1, float4{})
        == float4{1.5f, -2.5f, 3.25f, -4.75f}
    );

    float3x3 matrix3{{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    float4x4 matrix4{{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}};
    CHECK(read_value(write_value(1, matrix3), 1, float3x3{}) == matrix3);
    CHECK(read_value(write_value(1, matrix4), 1, float4x4{}) == matrix4);

    quatf rotation(0.25f, -0.5f, 0.75f, 1.0f);
    CHECK(read_value(write_value(1, rotation), 1, quatf{}) == rotation);

    std::filesystem::path path("assets/textures/albedo.png");
    CHECK(read_value(write_value(1, path), 1, std::filesystem::path()) == path);
}

TEST_CASE("missing and unknown fields")
{
    {
        std::vector<uint8_t> bytes = write_unknowns_and_values();

        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint32_t number = 1234;
        std::string text = "default";

        CHECK(deserializer.field(3, number));
        CHECK(number == 99);
        CHECK(!deserializer.field(4, number));
        CHECK(number == 99);
        CHECK(deserializer.field(7, text));
        CHECK(text == "kept");
        deserializer.finish();
    }

    {
        std::vector<uint8_t> bytes = write_unknowns_and_values();
        bytes.push_back(0x8d); // Truncated header for a trailing malformed unknown field.

        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint32_t number = 0;
        CHECK(deserializer.field(3, number));
        CHECK_THROWS(deserializer.finish());
    }
}

TEST_CASE("malformed varuints")
{
    check_varuint_throws(make_bytes({0x80}));
    check_varuint_throws(make_bytes({0x80, 0x00}));
    check_varuint_throws(make_bytes({0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x02}));
    check_varuint_throws(make_bytes({0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x00}));
}

TEST_CASE("malformed field ordering and encodings")
{
    CHECK_THROWS(write_value(0, uint8_t(1)));

    {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.field(1, uint8_t(1));
        CHECK_THROWS(serializer.field(1, uint8_t(2)));
        CHECK_THROWS(serializer.field(0, uint8_t(2)));
    }

    {
        std::vector<uint8_t> bytes = {0x00, 0x01};
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint8_t value = 0;
        CHECK_THROWS(deserializer.field(1, value));
    }

    {
        std::vector<uint8_t> bytes = {0x08, 0x01, 0x08, 0x02};
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint8_t value = 0;
        CHECK(deserializer.field(1, value));
        CHECK_THROWS(deserializer.finish());
    }

    {
        std::vector<uint8_t> bytes = {0x10, 0x01, 0x08, 0x02};
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint8_t value = 0;
        CHECK(deserializer.field(2, value));
        CHECK_THROWS(deserializer.finish());
    }

    {
        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        encoder.write_varuint((uint64_t(std::numeric_limits<uint32_t>::max()) + 1) << 3);
        encoder.write_byte(0);

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        CHECK_THROWS(deserializer.finish());
    }

    {
        std::vector<uint8_t> bytes = write_value(2, uint8_t(1));
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint8_t value = 0;
        CHECK(deserializer.field(2, value));
        CHECK_THROWS(deserializer.field(1, value));
    }
}

TEST_CASE("malformed payloads")
{
    {
        std::vector<uint8_t> bytes = {0x0b, 0x00, 0x00, 0x80}; // Truncated fixed4 field.
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        float value = 5.0f;
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == 5.0f);
    }

    {
        std::vector<uint8_t> bytes = {0x0f, 0x04, 1, 2}; // Sized length exceeds remaining bytes.
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::string value = "unchanged";
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == "unchanged");
    }

    {
        std::vector<uint8_t> bytes = {0x08, 0x02};
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        bool value = true;
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == true);
    }

    {
        std::vector<uint8_t> bytes = write_value(1, uint32_t(std::numeric_limits<uint16_t>::max()) + 1);
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint16_t value = 77;
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == 77);
    }

    {
        std::vector<uint8_t> bytes = write_value(1, uint8_t(7));
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        bool value = false;
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == false);
    }

    {
        std::vector<uint8_t> bytes = {
            0x0c, // Field 1, fixed8.
            0x00,
            0x00,
            0x80,
            0x3f, // Only 1.0f; the second float is truncated.
        };
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        float2 value{9.0f, 9.0f};
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == float2{9.0f, 9.0f});
    }

    {
        std::vector<uint8_t> bytes = {
            0x0f,
            0x08, // Field 1, sized payload with too few bytes for float3x3.
            0x00,
            0x00,
            0x80,
            0x3f,
            0x00,
            0x00,
            0x00,
            0x40,
        };
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        float3x3 value{{1, 0, 0, 0, 1, 0, 0, 0, 1}};
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == float3x3{{1, 0, 0, 0, 1, 0, 0, 0, 1}});
    }
}

TEST_CASE("malformed list payloads")
{
    {
        std::vector<uint8_t> bytes = {
            0x0f,
            0x01, // Field 1, sized payload with only a count.
            0x02, // Count says two elements, but no element payload remains.
        };
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<uint32_t> value = {7};
        CHECK_THROWS(deserializer.list(1, value));
        CHECK(value == std::vector<uint32_t>{7});
    }

    {
        std::vector<uint8_t> bytes = {
            0x0f,
            0x03, // Field 1, sized payload with two declared elements.
            0x02,
            0x80,
            0x01, // First element consumes all remaining bytes.
        };
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<uint32_t> value = {7};
        CHECK_THROWS(deserializer.list(1, value));
        CHECK(value == std::vector<uint32_t>{7});
    }

    {
        std::vector<uint8_t> bytes = {
            0x0f,
            0x05, // Field 1, sized list payload.
            0x01, // One nested record element.
            0x03, // Nested record byte count.
            0x09,
            0x01,
            0x80, // Trailing corrupt varuint inside the nested record.
        };
        SpanArchiveSource source(bytes);
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        std::vector<ArchiveTestStruct> initial = {{9, "initial"}};
        std::vector<ArchiveTestStruct> value = initial;
        CHECK_THROWS(deserializer.record_list(
            1,
            value,
            [](ArchiveDeserializer& element)
            {
                ArchiveTestStruct decoded;
                element.field(1, decoded.id);
                return decoded;
            }
        ));
        CHECK(value == initial);
    }
}

TEST_CASE("nested malformed payloads")
{
    {
        ArchiveTestStruct initial{9, "initial"};
        ArchiveTestStruct value = initial;

        BufferArchiveSink nested_sink;
        ArchiveEncoder nested_encoder(nested_sink);
        ArchiveSerializer nested_serializer(nested_encoder);
        nested_serializer.field(1, uint32_t(123));
        nested_encoder.write_byte(0x80); // Trailing corrupt varuint inside nested record.

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.field(1, ArchiveBlobView{nested_sink.bytes()});

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        CHECK_THROWS(deserializer.field(1, value));
        CHECK(value == initial);
    }

    {
        BufferArchiveSink nested_sink;
        ArchiveEncoder nested_encoder(nested_sink);
        ArchiveSerializer nested_serializer(nested_encoder);
        nested_serializer.field(1, uint32_t(123));
        nested_encoder.write_byte(0x80); // Trailing corrupt varuint inside the section.

        BufferArchiveSink sink;
        ArchiveEncoder encoder(sink);
        ArchiveSerializer serializer(encoder);
        serializer.field(1, ArchiveBlobView{nested_sink.bytes()});

        SpanArchiveSource source(sink.bytes());
        ArchiveDecoder decoder(source);
        ArchiveDeserializer deserializer(decoder);
        uint32_t value = 0;
        std::optional<ArchiveDeserializer> section = deserializer.section(1);
        REQUIRE(section.has_value());
        CHECK(section->field(1, value));
        CHECK_THROWS(section->finish());
    }
}

TEST_SUITE_END();
