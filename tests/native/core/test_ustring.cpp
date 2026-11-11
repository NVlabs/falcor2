// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/ustring.h"

#include <doctest/doctest.h>

#include <array>
#include <barrier>
#include <string>
#include <thread>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("ustring");

TEST_CASE("ustring preserves values and shares storage")
{
    CHECK(ustring().empty());
    CHECK(ustring().data() == ustring("").data());
    CHECK(ustring(nullptr).data() == ustring("").data());
    CHECK(ustring().hash() == std::hash<std::string_view>()({}));
    const std::string text("ustring embedded\0null", 21);
    const ustring first(text);
    const ustring second(text);
    CHECK(first.string() == text);
    CHECK(first.data() == second.data());
    CHECK(first.c_str()[text.size()] == '\0');
}

TEST_CASE("ustring allocations remain intact across page boundaries")
{
    std::vector<std::string> values;
    std::vector<ustring> interned;
    // Varied lengths force allocations to encounter partially filled 64 KiB pages.
    for (size_t i = 0; i < 80; ++i) {
        values.push_back("ustring page boundary " + std::to_string(i) + std::string(3000 + i * 97 % 2048, 'x'));
        interned.emplace_back(values.back());
    }
    for (size_t i = 0; i < values.size(); ++i) {
        CHECK(interned[i].string() == values[i]);
        CHECK(interned[i].c_str()[values[i].size()] == '\0');
        CHECK(interned[i].data() == ustring(values[i]).data());
    }
}

TEST_CASE("ustring supports strings at and beyond page size")
{
    constexpr size_t page_size = 64 * 1024;
    std::vector<std::string> values;
    std::vector<ustring> interned;
    for (size_t size : {page_size - 1, page_size, page_size + 1, 3 * page_size + 17}) {
        values.emplace_back(size, 'L');
        interned.emplace_back(values.back());
        // Interleave ordinary allocations with oversized ones.
        values.push_back("ustring after large allocation " + std::to_string(size));
        interned.emplace_back(values.back());
    }
    for (size_t i = 0; i < values.size(); ++i) {
        CHECK(interned[i].string() == values[i]);
        CHECK(interned[i].c_str()[values[i].size()] == '\0');
        CHECK(interned[i].data() == ustring(values[i]).data());
    }
}

TEST_CASE("ustring concurrent insertion returns shared storage")
{
    constexpr size_t thread_count = 8;
    constexpr size_t value_count = 4096;
    std::vector<std::string> values;
    for (size_t i = 0; i < value_count; ++i)
        values.push_back("ustring concurrent growth " + std::to_string(i));
    std::barrier start(thread_count);
    std::array<std::vector<ustring>, thread_count> results;
    std::array<bool, thread_count> stable;
    stable.fill(true);
    std::vector<std::jthread> threads;
    for (size_t i = 0; i < thread_count; ++i) {
        threads.emplace_back(
            [&, i]()
            {
                results[i].reserve(value_count);
                start.arrive_and_wait();
                for (size_t j = 0; j < value_count; ++j) {
                    // Different insertion orders overlap reads and competing insertions during growth.
                    const size_t index = (j + i * (value_count / thread_count)) % value_count;
                    results[i].emplace_back(values[index]);
                    const ustring& first = results[i].front();
                    const std::string& first_value = values[i * (value_count / thread_count)];
                    if (first != std::string_view(first_value) || first.data() != ustring(first_value).data())
                        stable[i] = false;
                }
            }
        );
    }
    threads.clear(); // Join before inspecting results.
    for (size_t i = 0; i < thread_count; ++i) {
        CHECK(stable[i]);
        REQUIRE(results[i].size() == value_count);
        for (size_t j = 0; j < value_count; ++j) {
            const size_t index = (j + i * (value_count / thread_count)) % value_count;
            CHECK(results[i][j].data() == results[0][index].data());
        }
    }
    for (size_t j = 0; j < value_count; ++j) {
        CHECK(results[0][j].string() == values[j]);
        CHECK(results[0][j].hash() == std::hash<std::string_view>()(values[j]));
        CHECK(results[0][j].c_str()[values[j].size()] == '\0');
    }
}

TEST_SUITE_END();
