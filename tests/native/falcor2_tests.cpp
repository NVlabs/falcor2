// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define DOCTEST_CONFIG_IMPLEMENT

#include "testing.h"
#include "falcor2/falcor2.h"
#include "sgl/sgl.h"
#include "sgl/device/device.h"
#include "sgl/device/agility_sdk.h"
#include "sgl/core/object.h"
#include "sgl/core/logger.h"
#include "sgl/core/error.h"

#include <cstdio>

SGL_EXPORT_AGILITY_SDK

using namespace sgl;

namespace falcor::testing {

// Helpers to get current test suite and case name.
// See https://github.com/doctest/doctest/issues/345.
// Has to be defined in the same file as DOCTEST_CONFIG_IMPLEMENT
std::string get_current_test_suite_name()
{
    return doctest::detail::g_cs->currentTest->m_test_suite;
}
std::string get_current_test_case_name()
{
    return doctest::detail::g_cs->currentTest->m_name;
}

// Keep the doctest registry dependency beside the existing context helpers in
// this implementation TU. Removing reserved nightly metadata before Context::run
// also applies to list/count queries and does not override user filters or skips.
static void select_tests()
{
    auto& tests = doctest::detail::getRegisteredTests();
    size_t excluded_nightly = 0;
    size_t excluded_device = 0;
    for (auto it = tests.begin(); it != tests.end();) {
        if (!options().nightly && is_nightly_test(*it)) {
            it = tests.erase(it);
            ++excluded_nightly;
        } else if (!is_test_device_selected(*it)) {
            it = tests.erase(it);
            ++excluded_device;
        } else {
            ++it;
        }
    }
    std::printf("[falcor2] Excluded %zu nightly test cases; use -nightly for the full tier.\n", excluded_nightly);
    if (options().device_type) {
        std::printf(
            "[falcor2] Selected device type %s; excluded %zu backend-specific test cases.\n",
            enum_to_string(*options().device_type).c_str(),
            excluded_device
        );
    }
}

} // namespace falcor::testing

int main(int argc, char** argv)
{
    // Parse extra command line options.
    {
        auto& options = falcor::testing::options();
        options.nightly = doctest::parseFlag(argc, argv, "nightly");
        if (doctest::parseFlag(argc, argv, "module-cache"))
            options.module_cache_enabled = true;
        if (doctest::parseFlag(argc, argv, "no-module-cache"))
            options.module_cache_enabled = false;

        if (doctest::parseFlag(argc, argv, "shader-cache"))
            options.shader_cache_enabled = true;
        if (doctest::parseFlag(argc, argv, "no-shader-cache"))
            options.shader_cache_enabled = false;

        doctest::String cache_dir;
        if (doctest::parseOption(argc, argv, "module-and-shader-cache-dir=", &cache_dir))
            options.module_and_shader_cache_dir = cache_dir.c_str();

        doctest::String device_type;
        if (doctest::parseOption(argc, argv, "device-type=", &device_type)
            || doctest::parseFlag(argc, argv, "device-type") || doctest::parseFlag(argc, argv, "device-type=")) {
            options.device_type = falcor::testing::parse_test_device_type(device_type.c_str());
            if (!options.device_type) {
                std::fprintf(stderr, "Invalid -device-type=%s. Available graphics APIs:", device_type.c_str());
                for (DeviceType available : falcor::testing::DEFAULT_DEVICE_TYPES)
                    std::fprintf(stderr, " %s", enum_to_string(available).c_str());
                std::fprintf(stderr, "\n");
                return 2;
            }
        }
    }

#if SGL_HAS_CRASHPAD
    if (!falcor::testing::start_crashpad_handler()) {
        return 1;
    }
#endif

    static_init();
    falcor::static_init();

    Logger::get().remove_all_outputs();
    Logger::get().add_debug_console_output();
    Logger::get().add_file_output("falcor2_tests.log");

    /// Do not break debugger on exceptions when running tests.
    set_exception_diagnostics(ExceptionDiagnosticFlags::none);

    int result = 1;
    {
        falcor::testing::static_init();

        falcor::testing::select_tests();
        doctest::Context context(argc, argv);

        // Select specific test suite to run
        // context.setOption("-ts", "formats");
        // Report successful tests
        // context.setOption("success", true);

        result = context.run();

        if (doctest::parseFlag(argc, argv, "help") || doctest::parseFlag(argc, argv, "h")
            || doctest::parseFlag(argc, argv, "?")) {
            std::printf(
                "\nAdditional options:\n"
                " -nightly                            Include nightly cases and subcases\n"
                " -device-type=<name>                 Select one platform graphics API (default: all)\n"
                " -module-cache                       Enable the persistent module cache\n"
                " -no-module-cache                    Disable the persistent module cache\n"
                " -shader-cache                       Enable the persistent shader cache\n"
                " -no-shader-cache                    Disable the persistent shader cache\n"
                " -module-and-shader-cache-dir=<path> Set the persistent module and shader cache root\n"
            );
        }

        falcor::testing::static_shutdown();
    }

    Device::close_all_devices();

#if SGL_ENABLE_OBJECT_TRACKING
    Logger::get().add_console_output();
    Object::report_live_objects();
#endif

    falcor::static_shutdown();
    static_shutdown();

    return result;
}
