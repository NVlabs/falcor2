// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/geometry/tangent_transfer.h"
#include "falcor2/importers/importer_types.h"
#include <array>
#include <cmath>

using namespace falcor;

TEST_SUITE_BEGIN("MeshMotion");

TEST_CASE_NIGHTLY("ImporterMesh - degenerate static mesh normals use a finite fallback")
{
    ImporterMesh mesh;
    mesh.ensure_attributes({{ImporterSemantic::position}});
    mesh.allocate_vertices(3, true);
    mesh.subgeometries.emplace_back().indices = {uint3(0, 1, 2)};
    mesh.add_normals_from_faces();
    for (size_t i = 0; i < mesh.vertex_count(); ++i)
        CHECK(mesh.normal_stream()[i] == float3(0, 0, 1));
}

TEST_CASE_NIGHTLY("mesh motion tangent transfer rotates and handles degenerate geometry")
{
    const std::array<uint3, 1> triangles{uint3(0, 1, 2)};
    const std::array<float3, 3> positions{float3(0), float3(1, 0, 0), float3(0, 1, 0)};
    const std::array<float3, 3> normals{float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1)};
    const std::array<float3, 3> tangents{float3(1, 0, 0), float3(1, 0, 0), float3(1, 0, 0)};
    TangentTransfer transfer(triangles, positions, normals, tangents);
    std::array<float3, 3> output;
    const std::array<float3, 3> rotated{float3(0), float3(0, 1, 0), float3(-1, 0, 0)};
    transfer.evaluate(rotated, normals, output);
    for (float3 tangent : output)
        CHECK(length(tangent - float3(0, 1, 0)) < 1e-6f);
    const std::array<float3, 3> collapsed{float3(0), float3(0), float3(0)};
    transfer.evaluate(collapsed, normals, output);
    for (float3 tangent : output)
        CHECK(length(tangent - float3(1, 0, 0)) < 1e-6f);
    transfer.evaluate(positions, normals, output);
    for (float3 tangent : output)
        CHECK(length(tangent - float3(1, 0, 0)) < 1e-6f);
}

TEST_CASE_NIGHTLY("mesh motion tangent transfer preserves an unchanged non-axis frame")
{
    const std::array<uint3, 1> triangles{uint3(0, 1, 2)};
    const std::array<float3, 3> positions{float3(0), float3(1.f, 0.25f, 0), float3(-0.25f, 1.f, 0)};
    const std::array<float3, 3> normals{float3(0, 0, 1), float3(0, 0, 1), float3(0, 0, 1)};
    const float3 reference_tangent = normalize(float3(1.f, 2.f, 0));
    const std::array<float3, 3> tangents{reference_tangent, reference_tangent, reference_tangent};
    TangentTransfer transfer(triangles, positions, normals, tangents);

    std::array<float3, 3> output;
    transfer.evaluate(positions, normals, output);
    for (size_t vertex = 0; vertex < output.size(); ++vertex) {
        CAPTURE(vertex);
        for (int component = 0; component < 3; ++component) {
            CAPTURE(component);
            CHECK(std::isfinite(output[vertex][component]));
            CHECK_LE(std::abs(output[vertex][component] - reference_tangent[component]), 1e-5f);
        }
        CHECK_LE(std::abs(length(output[vertex]) - 1.f), 1e-5f);
        CHECK_LE(std::abs(dot(output[vertex], normals[vertex])), 1e-5f);
    }
}

TEST_SUITE_END();
