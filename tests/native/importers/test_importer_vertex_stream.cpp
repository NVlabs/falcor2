// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"

#include <type_traits>
#include <utility>
#include <algorithm>
#include <iterator>

using namespace falcor;

static_assert(std::forward_iterator<ImporterVertexStream<float>::Iterator>);
static_assert(std::forward_iterator<ImporterVertexStream<const float>::Iterator>);
static_assert(!std::contiguous_iterator<ImporterVertexStream<float>::Iterator>);
static_assert(
    std::is_same_v<std::iter_reference_t<decltype(std::declval<const ImporterVertexStream<float>&>().begin())>, float&>
);
static_assert(std::is_same_v<std::iter_reference_t<ImporterVertexStream<const float>::Iterator>, const float&>);

TEST_CASE("ImporterVertexStream.iteration")
{
    ImporterVertexBuffer buffer{.stride = 2 * sizeof(float)};
    buffer.resize_vertices(3);
    const auto padding = buffer.get_stream<float>(0, 3);
    const auto values = buffer.get_stream<float>(sizeof(float), 3);
    std::fill(padding.begin(), padding.end(), -1.f);
    std::fill(values.begin(), values.end(), 2.f);
    for (float& value : values)
        value += 1.f;
    for (size_t i = 0; i < values.size; ++i) {
        CHECK_EQ(values[i], 3.f);
        CHECK_EQ(padding[i], -1.f);
    }

    // The temporary stream wrapper does not own iterator state.
    auto it = std::as_const(buffer).get_stream<float>(sizeof(float), 3).begin();
    const auto end = std::as_const(buffer).get_stream<float>(sizeof(float), 3).end();
    CHECK_EQ(std::distance(it, end), 3);
    const auto copy = it;
    CHECK_EQ(*it++, 3.f);
    CHECK(copy != it);
    CHECK_EQ(*copy, 3.f);

    // Resizing invalidates views; reacquire before accessing the grown buffer.
    buffer.resize_vertices(4);
    CHECK_EQ(buffer.get_stream<float>(sizeof(float), 4)[2], 3.f);
    CHECK_EQ(buffer.get_stream<float>(sizeof(float), 4)[3], 0.f);

    float tight[] = {1.f, 2.f};
    const ImporterVertexStream<float> contiguous{reinterpret_cast<uint8_t*>(tight), sizeof(float), 2};
    std::fill(contiguous.begin(), contiguous.end(), 4.f);
    CHECK_EQ(tight[0], 4.f);
    CHECK_EQ(tight[1], 4.f);

    const ImporterVertexStream<float> empty{};
    CHECK(empty.begin() == empty.end());
    std::fill(empty.begin(), empty.end(), 1.f);
    CHECK(ImporterVertexStream<float>::Iterator{} == ImporterVertexStream<float>::Iterator{});
}

TEST_CASE("ImporterCurve buffers preserve samples when resized")
{
    ImporterCurve curve;
    CHECK_EQ(curve.sample_count(), 1);
    CHECK_EQ(curve.position_stream().size, 0);
    curve.resize_vertices(2, 2);
    curve.position_stream()[0] = float3(1.f, 2.f, 3.f);
    curve.radius_stream()[0] = 0.25f;
    curve.position_stream(1)[1] = float3(4.f, 5.f, 6.f);
    curve.radius_stream(1)[1] = 0.5f;

    curve.resize_vertices(3, 3);
    const auto positions = std::as_const(curve).position_stream(1);
    const auto radii = std::as_const(curve).radius_stream(1);
    static_assert(std::is_same_v<decltype(positions[0]), const float3&>);
    static_assert(std::is_same_v<decltype(radii[0]), const float&>);
    CHECK_EQ(positions.size, 3);
    CHECK_EQ(positions.stride, sizeof(float3));
    CHECK_EQ(radii.stride, sizeof(float));
    CHECK_EQ(positions[1], float3(4.f, 5.f, 6.f));
    CHECK_EQ(radii[1], 0.5f);
    CHECK_EQ(positions[2], float3(0.f));
    CHECK_EQ(radii[2], 0.f);
    CHECK_EQ(curve.position_stream(2)[0], float3(0.f));
    CHECK_EQ(curve.radius_stream(2)[0], 0.f);

    curve.resize_vertices(1, 1);
    CHECK_EQ(curve.position_stream()[0], float3(1.f, 2.f, 3.f));
    CHECK_EQ(curve.radius_stream()[0], 0.25f);
    CHECK_THROWS(curve.position_stream(1));
    CHECK_THROWS(curve.radius_stream(1));
    curve.resize_vertices(2, 2);
    CHECK_EQ(curve.position_stream(1)[1], float3(0.f));
    CHECK_EQ(curve.radius_stream(1)[1], 0.f);
    CHECK_EQ(curve.position_stream()[1], float3(0.f));
    CHECK_EQ(curve.radius_stream()[1], 0.f);

    curve.resize_vertices(0);
    CHECK_EQ(curve.vertex_count(), 0);
    CHECK_EQ(curve.radius_stream().size, 0);
    CHECK_THROWS(curve.resize_vertices(1, 0));
}

static_assert(std::is_same_v<decltype(ImporterVertexStream<const float>::data), const uint8_t*>);
static_assert(std::is_same_v<decltype(std::declval<ImporterVertexStream<const float>&>().first()), const float*>);
static_assert(std::is_same_v<decltype(std::declval<ImporterVertexStream<const float>&>()[0]), const float&>);
static_assert(!std::is_assignable_v<decltype(std::declval<ImporterVertexStream<const float>&>()[0]), float>);
static_assert(std::is_same_v<decltype(std::declval<const ImporterVertexStream<const float>&>().first()), const float*>);
static_assert(std::is_same_v<decltype(std::declval<const ImporterVertexStream<const float>&>()[0]), const float&>);

static_assert(std::is_same_v<decltype(ImporterVertexStream<float>::data), uint8_t*>);
static_assert(std::is_same_v<decltype(std::declval<ImporterVertexStream<float>&>().first()), float*>);
static_assert(std::is_assignable_v<decltype(std::declval<ImporterVertexStream<float>&>()[0]), float>);
static_assert(std::is_same_v<decltype(std::declval<const ImporterVertexStream<float>&>().first()), float*>);
static_assert(std::is_same_v<decltype(std::declval<const ImporterVertexStream<float>&>()[0]), float&>);
static_assert(std::is_assignable_v<decltype(std::declval<const ImporterVertexStream<float>&>()[0]), float>);

static_assert(std::is_convertible_v<ImporterVertexStream<float>, ImporterVertexStream<const float>>);
static_assert(std::is_convertible_v<const ImporterVertexStream<float>&, ImporterVertexStream<const float>>);
static_assert(!std::is_convertible_v<ImporterVertexStream<const float>, ImporterVertexStream<float>>);
static_assert(!std::is_convertible_v<ImporterVertexStream<int>, ImporterVertexStream<const unsigned int>>);
static_assert(std::is_aggregate_v<ImporterVertexStream<float>>);
static_assert(std::is_aggregate_v<ImporterVertexStream<const float>>);

TEST_CASE("ImporterVertexStream.const_conversion")
{
    float values[] = {1.f, -1.f, 2.f, -2.f};
    const ImporterVertexStream<float> writable{reinterpret_cast<uint8_t*>(values), 2 * sizeof(float), 2};
    ImporterVertexStream<const float> readonly = writable;
    CHECK(readonly.data == writable.data);
    CHECK(readonly.stride == writable.stride);
    CHECK(readonly.size == writable.size);
    CHECK(readonly.first() == values);
    CHECK(readonly[1] == 2.f);
    writable[1] = 3.f;
    CHECK(readonly[1] == 3.f);
    CHECK_THROWS(readonly[2]);

    constexpr ImporterVertexStream<float> missing{};
    constexpr ImporterVertexStream<const float> readonly_missing = missing;
    static_assert(readonly_missing.data == nullptr && readonly_missing.stride == 0 && readonly_missing.size == 0);
    CHECK_FALSE(readonly_missing.valid());
    CHECK_THROWS(readonly_missing.first());
    CHECK_THROWS(readonly_missing[0]);
}

TEST_CASE("ImporterVertexStream.const_wrapper")
{
    float values[] = {1.f, -1.f, 2.f, -2.f};
    const ImporterVertexStream<float> stream{reinterpret_cast<uint8_t*>(values), 2 * sizeof(float), 2};
    CHECK(stream.valid());
    CHECK(stream.first() == values);
    *stream.first() = 3.f;
    stream[1] = 4.f;
    CHECK(values[0] == 3.f);
    CHECK(values[1] == -1.f);
    CHECK(values[2] == 4.f);
    CHECK(values[3] == -2.f);
    CHECK_THROWS(stream[2]);

    const ImporterVertexStream<float> missing{nullptr, 0, 0};
    CHECK_FALSE(missing.valid());
    CHECK_THROWS(missing.first());
    CHECK_THROWS(missing[0]);
}

TEST_CASE("ImporterVertexStream.const_elements")
{
    const float values[] = {1.f, -1.f, 2.f, -2.f};
    ImporterVertexStream<const float> stream{reinterpret_cast<const uint8_t*>(values), 2 * sizeof(float), 2};
    CHECK(stream.valid());
    CHECK(stream.first() == values);
    CHECK(stream[0] == 1.f);
    CHECK(stream[1] == 2.f);
    CHECK(std::as_const(stream).first() == values);
    CHECK(std::as_const(stream)[1] == 2.f);
    CHECK_THROWS(stream[2]);
    CHECK_THROWS(std::as_const(stream)[2]);

    ImporterMesh mesh;
    mesh.ensure_attributes({{ImporterSemantic::position}});
    mesh.allocate_vertices(2, true);
    mesh.position_stream()[1] = float3(1.f, 2.f, 3.f);
    auto positions = std::as_const(mesh).position_stream();
    static_assert(std::is_same_v<decltype(positions), ImporterVertexStream<const float3>>);
    CHECK(positions.size == mesh.vertex_count());
    CHECK(positions.stride == mesh.position_stream().stride);
    CHECK(positions.first() == mesh.position_stream().first());
    CHECK(positions[1].y == 2.f);

    auto missing = std::as_const(mesh).normal_stream();
    CHECK_FALSE(missing.valid());
    CHECK(missing.size == 0);
    CHECK(missing.stride == 0);
    CHECK_THROWS(missing.first());
    CHECK_THROWS(std::as_const(missing).first());
    CHECK_THROWS(missing[0]);
}
