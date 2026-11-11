// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/core/config.h>

// If crashpad is enabled, we disable doctest's SEH and signal handlers.
// This allows crashpad to capture crashes and dump diagnostic information.
#if SGL_HAS_CRASHPAD
#define DOCTEST_CONFIG_NO_WINDOWS_SEH
#define DOCTEST_CONFIG_NO_POSIX_SIGNALS
#endif

#include <array>
#include <doctest/doctest.h>
#include <filesystem>
#include <optional>
#include <sgl/device/device.h>
#include <string>
#include <vector>

namespace falcor::testing {

struct Options {
    bool nightly{false};
    bool module_cache_enabled{false};
    bool shader_cache_enabled{false};
    std::filesystem::path module_and_shader_cache_dir = ".shader-cache/tests";
};

inline Options& options()
{
    static Options opts;
    return opts;
}

// Reserved doctest description metadata. Selection preserves test names and suites.
inline constexpr const char* NIGHTLY_TEST_DESCRIPTION = "falcor2 nightly";

bool is_nightly_test(const doctest::TestCaseData& test);

/// Construct the doctest subcase only when nightly work is enabled, so an excluded
/// branch is never entered, scheduled, or reported as a successful subcase.
class NightlySubcase {
public:
    NightlySubcase(const doctest::String& name, const char* file, int line)
    {
        if (options().nightly)
            m_subcase.emplace(name, file, line);
    }

    explicit operator bool() const { return m_subcase && static_cast<bool>(*m_subcase); }

private:
    std::optional<doctest::detail::Subcase> m_subcase;
};

/// Get name of running test suite (note: defined in sgl_tests.cpp).
std::string get_current_test_suite_name();

/// Get name of running test case (note: defined in sgl_tests.cpp).
std::string get_current_test_case_name();

/// Get global temp directory for tests.
std::filesystem::path get_test_temp_directory();

/// Get temp directory for current test suite.
std::filesystem::path get_suite_temp_directory();

/// Get temp directory for current test case.
std::filesystem::path get_case_temp_directory();

const std::filesystem::path& project_directory();

/// Start the crashpad handler.
/// Return true on success, false on failure or if crashpad is not supported.
bool start_crashpad_handler();

void static_init();
void static_shutdown();

/// GPU APIs exercised by the native tests, in execution order.
/// API-independent checks can use the first entry to run only once.
#if SGL_WINDOWS
inline constexpr std::array DEFAULT_DEVICE_TYPES{sgl::DeviceType::d3d12, sgl::DeviceType::vulkan};
#elif SGL_LINUX
inline constexpr std::array DEFAULT_DEVICE_TYPES{sgl::DeviceType::vulkan};
#elif SGL_MACOS
inline constexpr std::array DEFAULT_DEVICE_TYPES{sgl::DeviceType::metal};
#endif

struct GpuTestContext {
    sgl::Device* device;
};

void run_gpu_test(void (*func)(GpuTestContext&));

/// Make a device descriptor for a testing device.
/// This sets up the device descriptor with default settings for testing, including
/// enabling debug layers and setting up the slang include paths and cache paths.
sgl::DeviceDesc make_test_device_desc(sgl::DeviceType device_type);

void release_cached_devices();

} // namespace falcor::testing


#define DOCTEST_TEST_CASE_GPU(f, name)                                                                                 \
    static void f(::falcor::testing::GpuTestContext& ctx);                                                             \
    TEST_CASE(name)                                                                                                    \
    {                                                                                                                  \
        ::falcor::testing::run_gpu_test(f);                                                                            \
    }                                                                                                                  \
    static void f(::falcor::testing::GpuTestContext& ctx)


#define TEST_CASE_GPU(name) DOCTEST_TEST_CASE_GPU(DOCTEST_ANONYMOUS(gpu_test), name)

#define TEST_CASE_NIGHTLY(name) TEST_CASE(name* doctest::description(::falcor::testing::NIGHTLY_TEST_DESCRIPTION))
#define TEST_CASE_GPU_NIGHTLY(name)                                                                                    \
    TEST_CASE_GPU(name* doctest::description(::falcor::testing::NIGHTLY_TEST_DESCRIPTION))

#define SUBCASE_NIGHTLY(name)                                                                                          \
    if (const ::falcor::testing::NightlySubcase DOCTEST_ANONYMOUS(nightly_subcase){name, __FILE__, __LINE__})
