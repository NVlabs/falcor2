// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/platform.h"

#include <string>
#include <thread>
#include <chrono>

using namespace falcor;

TEST_SUITE_BEGIN("platform");

// Test types for demangle_type_name
enum class TestEnumClass { A, B };

enum TestEnum { X, Y };

class TestClass { };

struct TestStruct { };

// Helper to check that a string does not contain any of the given substrings
static bool contains_any(const std::string& str, std::initializer_list<std::string_view> substrings)
{
    for (auto substr : substrings)
        if (str.find(substr) != std::string::npos)
            return true;
    return false;
}

TEST_CASE("demangle_type_name")
{
    SUBCASE("enum class")
    {
        std::string name = platform::demangle_type_name(typeid(TestEnumClass));
        CHECK(name == "TestEnumClass");
    }

    SUBCASE("enum")
    {
        std::string name = platform::demangle_type_name(typeid(TestEnum));
        CHECK(name == "TestEnum");
    }

    SUBCASE("class")
    {
        std::string name = platform::demangle_type_name(typeid(TestClass));
        CHECK(name == "TestClass");
    }

    SUBCASE("struct")
    {
        std::string name = platform::demangle_type_name(typeid(TestStruct));
        CHECK(name == "TestStruct");
    }

    SUBCASE("int")
    {
        std::string name = platform::demangle_type_name(typeid(int));
        CHECK(name == "int");
    }

    SUBCASE("double")
    {
        std::string name = platform::demangle_type_name(typeid(double));
        CHECK(name == "double");
    }

    SUBCASE("pointer type")
    {
        std::string name = platform::demangle_type_name(typeid(int*));
        CHECK(name.find("int") != std::string::npos);
        CHECK(name.find("*") != std::string::npos);
        CHECK_FALSE(contains_any(name, {"__ptr64"}));
    }

    SUBCASE("function pointer type")
    {
        using FuncPtr = void (*)();
        std::string name = platform::demangle_type_name(typeid(FuncPtr));
        CHECK(name.find("void") != std::string::npos);
        CHECK(name.find("(*)") != std::string::npos);
        CHECK(contains_any(name, {"()", "(void)"}));
        CHECK_FALSE(contains_any(name, {"__cdecl", "__stdcall", "__fastcall", "__thiscall", "__vectorcall"}));
    }

    SUBCASE("std::string")
    {
        std::string name = platform::demangle_type_name(typeid(std::string));
        CHECK(!name.empty());
    }
}

TEST_CASE("app_data_directory")
{
    std::filesystem::path dir = platform::app_data_directory();
    CHECK(!dir.empty());
    CHECK(std::filesystem::exists(dir));
    CHECK(std::filesystem::is_directory(dir));
}

TEST_CASE("FileLock basic lock and unlock")
{
    auto lock_file = falcor::testing::get_case_temp_directory() / "test.lock";
    platform::FileLock lock(lock_file);
    CHECK(!lock.is_locked());
    CHECK(lock.try_lock());
    CHECK(lock.is_locked());
    lock.unlock();
    CHECK(!lock.is_locked());
}

TEST_CASE("FileLock destructor releases lock")
{
    auto lock_file = falcor::testing::get_case_temp_directory() / "test.lock";
    {
        platform::FileLock lock(lock_file);
        CHECK(lock.try_lock());
        CHECK(lock.is_locked());
    }
    // After destruction, we should be able to lock again
    platform::FileLock lock2(lock_file);
    CHECK(lock2.try_lock());
    lock2.unlock();
}

TEST_CASE("FileLock double lock from same instance returns true")
{
    auto lock_file = falcor::testing::get_case_temp_directory() / "test.lock";
    platform::FileLock lock(lock_file);
    CHECK(lock.try_lock());
    // Second call should return true immediately (already locked)
    CHECK(lock.try_lock());
    lock.unlock();
}

TEST_CASE("FileLock contention from another thread times out")
{
    auto lock_file = falcor::testing::get_case_temp_directory() / "test.lock";
    platform::FileLock lock1(lock_file);
    REQUIRE(lock1.try_lock());

    // Try to lock from another thread - should time out
    bool second_lock_result = true;
    std::thread t(
        [&]()
        {
            platform::FileLock lock2(lock_file);
            second_lock_result = lock2.try_lock(std::chrono::milliseconds(50));
        }
    );
    t.join();
    CHECK_FALSE(second_lock_result);

    lock1.unlock();
}

TEST_CASE("FileLock can be acquired after release from another thread")
{
    auto lock_file = falcor::testing::get_case_temp_directory() / "test.lock";
    platform::FileLock lock1(lock_file);
    REQUIRE(lock1.try_lock());
    lock1.unlock();

    // Another thread should now be able to acquire the lock
    bool second_lock_result = false;
    std::thread t(
        [&]()
        {
            platform::FileLock lock2(lock_file);
            second_lock_result = lock2.try_lock(std::chrono::milliseconds(1000));
            lock2.unlock();
        }
    );
    t.join();
    CHECK(second_lock_result);
}

TEST_SUITE_END();
