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

} // namespace falcor::testing

int main(int argc, char** argv)
{
    // Parse extra command line options.
    {
        auto& options = falcor::testing::options();
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
