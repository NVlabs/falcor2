// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include <sgl/core/macros.h>
#include <sgl/core/platform.h>
#include <sgl/utils/crashpad.h>

#include <sgl/device/device.h>

#include <map>
#include <ctime>
#include <fstream>
#include <string>

using namespace sgl;

namespace falcor::testing {

// Cached devices.
static std::map<DeviceType, ref<Device>> g_cached_devices;

// Temp directory to create files for teting in.
static std::filesystem::path g_test_temp_directory;

// Calculates a files sytem compatible date string formatted YYYY-MM-DD-hh-mm-ss.
static std::string build_current_date_string()
{
    time_t now;
    time(&now);
    struct tm tm;
#if SGL_WINDOWS
    localtime_s(&tm, &now);
#else
    localtime_r(&now, &tm);
#endif
    char result[128];
    std::strftime(result, sizeof(result), "%Y-%m-%d-%H-%M-%S", &tm);
    return result;
}

std::filesystem::path get_test_temp_directory()
{
    if (g_test_temp_directory == "") {
        std::string datetime_str = build_current_date_string();
        g_test_temp_directory = std::filesystem::current_path() / ".test_temp" / datetime_str;
        std::filesystem::create_directories(g_test_temp_directory);
    }
    return g_test_temp_directory;
}

std::filesystem::path get_suite_temp_directory()
{
    auto path = get_test_temp_directory() / get_current_test_suite_name();
    std::filesystem::create_directories(path);
    return path;
}

std::filesystem::path get_case_temp_directory()
{
    auto path = get_test_temp_directory() / get_current_test_suite_name() / get_current_test_case_name();
    std::filesystem::create_directories(path);
    return path;
}

const std::filesystem::path& project_directory()
{
    static std::filesystem::path path = []()
    {
        std::filesystem::path path_ = std::filesystem::path{FALCOR_PROJECT_DIR}.lexically_normal();
        return path_;
    }();
    return path;
};

sgl::DeviceDesc make_test_device_desc(sgl::DeviceType device_type)
{
    DeviceDesc desc{
        .type = device_type,
        .enable_debug_layers = true,
        .compiler_options = {
            .include_paths = {
                // The slang code, so we can import things from falcor2
                project_directory() / "slang",
                // tests, so we can import test-specific .slang files
                project_directory() / "tests" / "native",
                // slangpy module
                project_directory() / "external" / "slangpy" / "slangpy" / "slang",
            },
        }
    };

    std::filesystem::path cache_root = options().module_and_shader_cache_dir / "native";
    if (cache_root.is_relative())
        cache_root = falcor::testing::project_directory() / cache_root;
    if (options().module_cache_enabled)
        desc.module_cache_path = cache_root / "module";
    if (options().shader_cache_enabled)
        desc.shader_cache_path = cache_root / "shader";

    return desc;
}

static std::filesystem::path g_crashpad_database;
static bool g_crashpad_enabled = false;
static std::string g_current_test_case;

void notify_crashpad_current_test(const std::string& name)
{
    if (!g_crashpad_enabled)
        return;

    try {
        std::filesystem::create_directories(g_crashpad_database);
        std::ofstream file(g_crashpad_database / (std::to_string(platform::current_process_id()) + ".txt"));
        file << name << "\n";
    } catch (...) {
        // Crash diagnostics must not affect the test result.
    }
}

class CrashpadListener final : public doctest::IReporter {
public:
    explicit CrashpadListener(const doctest::ContextOptions&) { }

    void report_query(const doctest::QueryData&) override { }
    void test_run_start() override { }
    void test_run_end(const doctest::TestRunStats&) override { }

    void test_case_start(const doctest::TestCaseData& in) override
    {
        g_current_test_case = in.m_name;
        if (in.m_test_suite && in.m_test_suite[0] != '\0')
            g_current_test_case = std::string(in.m_test_suite) + "::" + g_current_test_case;
        notify_crashpad_current_test(g_current_test_case);
    }

    void test_case_reenter(const doctest::TestCaseData& in) override { test_case_start(in); }
    void test_case_end(const doctest::CurrentTestCaseStats&) override { }
    void test_case_exception(const doctest::TestCaseException&) override { }

    void subcase_start(const doctest::SubcaseSignature& in) override
    {
        notify_crashpad_current_test(g_current_test_case + "/" + in.m_name.c_str());
    }

    void subcase_end() override { notify_crashpad_current_test(g_current_test_case); }
    void log_assert(const doctest::AssertData&) override { }
    void log_message(const doctest::MessageData&) override { }
    void test_case_skipped(const doctest::TestCaseData&) override { }
};

REGISTER_LISTENER("crashpad", 100, CrashpadListener);

bool start_crashpad_handler()
{
    if (!crashpad::is_supported())
        return false;

    try {
        g_crashpad_database = project_directory() / ".crashpad" / "native";
        std::filesystem::create_directories(g_crashpad_database);

        crashpad::start_handler(
            {},
            g_crashpad_database,
            std::map<std::string, std::string>{{"project", "falcor"}, {"test_kind", "native"}}
        );
        g_crashpad_enabled = true;
        printf("Crashpad handler started.\n");
        return true;
    } catch (const std::exception& e) {
        printf("Failed to start Crashpad handler (%s)\n", e.what());
    } catch (...) {
        printf("Failed to start Crashpad handler.\n");
    }
    return false;
}

void static_init() { }

void static_shutdown()
{
    // Clean up temp files
    std::filesystem::remove_all(g_test_temp_directory);

    g_cached_devices.clear();
}

void run_gpu_test(void (*func)(GpuTestContext&))
{
#if SGL_WINDOWS
    std::vector<DeviceType> device_types{DeviceType::d3d12, DeviceType::vulkan};
#elif SGL_LINUX
    std::vector<DeviceType> device_types{DeviceType::vulkan};
#elif SGL_MACOS
    std::vector<DeviceType> device_types{DeviceType::metal};
#endif

    bool use_cached_device = true;

    for (DeviceType device_type : device_types) {
        SUBCASE(enum_to_string(device_type).c_str())
        {
            ref<Device> device;
            if (use_cached_device) {
                auto it = g_cached_devices.find(device_type);
                if (it != g_cached_devices.end())
                    device = it->second;
            }

            if (!device) {
                DeviceDesc desc = make_test_device_desc(device_type);
                device = Device::create(desc);
                if (use_cached_device)
                    g_cached_devices[device_type] = device;
            }

            GpuTestContext ctx{.device = device};
            func(ctx);

            if (!use_cached_device)
                device->close();
        }
    }
}

} // namespace falcor::testing
