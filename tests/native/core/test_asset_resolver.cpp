// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/asset_resolver.h"

#include <filesystem>
#include <fstream>
#include <string>
#include <type_traits>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("AssetResolver");

static_assert(!std::is_copy_constructible_v<AssetResolver>);
static_assert(!std::is_copy_assignable_v<AssetResolver>);
static_assert(!std::is_move_constructible_v<AssetResolver>);
static_assert(!std::is_move_assignable_v<AssetResolver>);

namespace {

class CurrentPathGuard {
public:
    explicit CurrentPathGuard(const std::filesystem::path& path)
        : m_original_path(std::filesystem::current_path())
    {
        std::filesystem::current_path(path);
    }

    ~CurrentPathGuard() { std::filesystem::current_path(m_original_path); }

private:
    std::filesystem::path m_original_path;
};

void write_file(const std::filesystem::path& path, std::string_view contents = "test")
{
    std::filesystem::create_directories(path.parent_path());
    std::ofstream stream(path);
    stream << contents;
}

std::vector<std::filesystem::path> paths(std::span<const std::filesystem::path> values)
{
    return {values.begin(), values.end()};
}

} // namespace

TEST_CASE("push returns a normalized higher-priority resolver without mutating its source")
{
    const auto root = testing::get_case_temp_directory();
    CurrentPathGuard guard(root);

    AssetResolver resolver;
    auto first = resolver.push("first");
    const std::vector<std::filesystem::path> next_paths = {"second", "third"};
    auto next = first->push(next_paths);

    CHECK(resolver.get_search_paths().empty());
    CHECK(paths(first->get_search_paths()) == std::vector<std::filesystem::path>{root / "first"});
    CHECK(
        paths(next->get_search_paths())
        == std::vector<std::filesystem::path>{root / "second", root / "third", root / "first"}
    );
}

TEST_CASE("constructors create normalized resolvers without an empty intermediate")
{
    const auto root = testing::get_case_temp_directory();
    CurrentPathGuard guard(root);

    AssetResolver split(" first ; second ");
    AssetResolver literal_path(std::filesystem::path("literal;path"));
    const std::vector<std::filesystem::path> literal_paths = {"third", "literal;name"};
    AssetResolver literal{std::span<const std::filesystem::path>(literal_paths)};

    CHECK(paths(split.get_search_paths()) == std::vector<std::filesystem::path>{root / "first", root / "second"});
    CHECK(paths(literal_path.get_search_paths()) == std::vector<std::filesystem::path>{root / "literal;path"});
    CHECK(
        paths(literal.get_search_paths()) == std::vector<std::filesystem::path>{root / "third", root / "literal;name"}
    );
}

TEST_CASE("semicolon paths are trimmed while batch elements remain literal")
{
    const auto root = testing::get_case_temp_directory();
    CurrentPathGuard guard(root);

    AssetResolver resolver;
    auto split = resolver.push(" first ; ; second ");
    auto literal_path = resolver.push(std::filesystem::path("literal;path"));
    CHECK(paths(split->get_search_paths()) == std::vector<std::filesystem::path>{root / "first", root / "second"});
    CHECK(paths(literal_path->get_search_paths()) == std::vector<std::filesystem::path>{root / "literal;path"});

    const std::vector<std::filesystem::path> literal = {"literal;name"};
    auto unsplit = resolver.push(literal);
    CHECK(paths(unsplit->get_search_paths()) == std::vector<std::filesystem::path>{root / "literal;name"});
}

TEST_CASE("duplicates are removed and repushing promotes a root")
{
    const auto root = testing::get_case_temp_directory();
    CurrentPathGuard guard(root);

    AssetResolver resolver;
    const std::vector<std::filesystem::path> initial_paths = {"first", "second", "first"};
    auto initial = resolver.push(initial_paths);
    auto promoted = initial->push("second");

    CHECK(paths(initial->get_search_paths()) == std::vector<std::filesystem::path>{root / "first", root / "second"});
    CHECK(paths(promoted->get_search_paths()) == std::vector<std::filesystem::path>{root / "second", root / "first"});
}

TEST_CASE("explicit roots beat the working directory and absolute roots remain stable")
{
    const auto root = testing::get_case_temp_directory();
    const auto working = root / "working";
    const auto search = root / "search";
    const auto other = root / "other";
    write_file(working / "shared.txt", "working");
    write_file(search / "shared.txt", "search");
    std::filesystem::create_directories(other);

    CurrentPathGuard guard(working);
    AssetResolver resolver;
    auto configured = resolver.push(search);
    CHECK(configured->resolve_path("shared.txt") == std::filesystem::canonical(search / "shared.txt"));
    CHECK(resolver.resolve_path("shared.txt") == std::filesystem::canonical(working / "shared.txt"));

    std::filesystem::current_path(other);
    CHECK(configured->get_search_paths()[0] == search.lexically_normal());
    CHECK(configured->resolve_path("shared.txt") == std::filesystem::canonical(search / "shared.txt"));
    CHECK(configured->resolve_path("missing.txt").empty());

    try {
        configured->resolve_path("missing.txt", true);
        FAIL("Expected required missing path to throw");
    } catch (const std::exception& error) {
        const std::string message = error.what();
        CHECK(message.find("required asset path") != std::string::npos);
        CHECK(message.find("missing.txt") != std::string::npos);
        CHECK(message.find("AssetResolver(search_paths=[") != std::string::npos);
        CHECK(message.find(search.string()) != std::string::npos);
    }
}

TEST_CASE("string representation owns resolver search diagnostics")
{
    const auto root = testing::get_case_temp_directory();
    const auto first = root / "first";
    const auto second = root / "second";
    CurrentPathGuard guard(root);

    const std::vector<std::filesystem::path> roots = {first, second};
    AssetResolver resolver{std::span<const std::filesystem::path>(roots)};
    const std::string description = resolver.to_string();

    CHECK(description.find("AssetResolver(search_paths=[") != std::string::npos);
    CHECK(description.find(first.string()) != std::string::npos);
    CHECK(description.find(second.string()) != std::string::npos);
    CHECK(description.find("working_directory_fallback=\"" + root.string() + "\"") != std::string::npos);
}

TEST_CASE("patterns are sorted within and limited to the first matching root")
{
    const auto root = testing::get_case_temp_directory();
    const auto high = root / "high";
    const auto low = root / "low";
    write_file(high / "tiles" / "z_1002.exr");
    write_file(high / "tiles" / "a_1001.exr");
    write_file(low / "tiles" / "0_1000.exr");

    AssetResolver resolver;
    const std::vector<std::filesystem::path> roots = {high, low};
    auto configured = resolver.push(roots);

    CHECK(
        configured->resolve_path_pattern("tiles", ".*_[0-9]+\\.exr", false)
        == std::vector<std::filesystem::path>{high / "tiles" / "a_1001.exr", high / "tiles" / "z_1002.exr"}
    );
    CHECK(
        configured->resolve_path_pattern("tiles", ".*_[0-9]+\\.exr", true)
        == std::vector<std::filesystem::path>{high / "tiles" / "a_1001.exr"}
    );
}

TEST_SUITE_END();
