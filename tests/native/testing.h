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

#include <doctest/doctest.h>
#include <filesystem>
#include <sgl/device/device.h>
#include <string>

namespace falcor::testing {

struct Options {
    bool module_cache_enabled{false};
    bool shader_cache_enabled{false};
    std::filesystem::path module_and_shader_cache_dir = ".shader-cache/tests";
};

inline Options& options()
{
    static Options opts;
    return opts;
}

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
