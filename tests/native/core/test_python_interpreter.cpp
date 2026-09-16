// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/python_interpreter.h"

#include <fstream>
#include <thread>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("PythonInterpreter");

TEST_CASE("create context and execute simple code")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    ctx.execute_string("x = 1 + 2");
}

TEST_CASE("context state persists across calls")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    ctx.execute_string("value = 42");
    // Variable should be accessible in the same context.
    ctx.execute_string("assert value == 42");
}

TEST_CASE("contexts are isolated from each other")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx1 = py.create_context();
    PythonContext ctx2 = py.create_context();
    ctx1.execute_string("isolated_var = 'hello'");
    // ctx2 should not see ctx1's variable.
    CHECK_THROWS_AS(ctx2.execute_string("assert isolated_var"), PythonException);
}

TEST_CASE("syntax error throws PythonException")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    CHECK_THROWS_AS(ctx.execute_string("def broken("), PythonException);
}

TEST_CASE("import error throws PythonException with message")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    try {
        ctx.execute_string("import no_such_module_xyz_123");
        FAIL("Expected PythonException");
    } catch (const PythonException& e) {
        std::string msg = e.what();
        CHECK(msg.find("ModuleNotFoundError") != std::string::npos);
    }
}

TEST_CASE("runtime error throws PythonException with traceback")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    try {
        ctx.execute_string("1 / 0");
        FAIL("Expected PythonException");
    } catch (const PythonException& e) {
        std::string msg = e.what();
        CHECK(msg.find("ZeroDivisionError") != std::string::npos);
    }
}

TEST_CASE("context remains usable after error")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    CHECK_THROWS_AS(ctx.execute_string("1 / 0"), PythonException);
    // Context should still work after an error.
    ctx.execute_string("result = 'ok'");
    ctx.execute_string("assert result == 'ok'");
}

TEST_CASE("execute_file runs a script")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();

    auto temp_dir = falcor::testing::get_case_temp_directory();
    auto script_path = temp_dir / "test_script.py";

    {
        std::ofstream f(script_path);
        f << "script_var = 'from_file'\n";
    }

    ctx.execute_file(script_path);
    ctx.execute_string("assert script_var == 'from_file'");

    // Check that __file__ uses forward slashes (Python convention).
    ctx.execute_string("assert '\\\\' not in __file__");
}

TEST_CASE("execute_file controls persistent module identity")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();

    auto temp_dir = falcor::testing::get_case_temp_directory();
    auto script_path = temp_dir / "module_identity.py";

    {
        std::ofstream f(script_path);
        f << "guard_count = globals().get('guard_count', 0)\n";
        f << "if __name__ == '__main__':\n";
        f << "    guard_count += 1\n";
        f << "observed_name = __name__\n";
    }

    ctx.execute_file(script_path);
    ctx.execute_string("assert observed_name == '__main__'");
    ctx.execute_string("assert guard_count == 1");

    ctx.execute_file(script_path, "__falcor2_scene__");
    ctx.execute_string("assert observed_name == '__falcor2_scene__'");
    ctx.execute_string("assert __name__ == '__falcor2_scene__'");
    ctx.execute_string("assert guard_count == 1");

    ctx.execute_file(script_path);
    ctx.execute_string("assert observed_name == '__main__'");
    ctx.execute_string("assert __name__ == '__main__'");
    ctx.execute_string("assert guard_count == 2");
}

TEST_CASE("execute_file search paths are temporary and restored after errors")
{
    PythonContext ctx = PythonInterpreter::get().create_context();
    const std::filesystem::path temp_dir = testing::get_case_temp_directory();
    const std::filesystem::path helper_path = temp_dir / "pyscene_python_path_helper.py";
    const std::filesystem::path script_path = temp_dir / "uses_helper.py";
    const std::filesystem::path failing_script_path = temp_dir / "uses_helper_then_fails.py";

    {
        std::ofstream helper(helper_path);
        helper << "VALUE = 17\n";
    }
    {
        std::ofstream script(script_path);
        script << "from pyscene_python_path_helper import VALUE\n";
        script << "assert VALUE == 17\n";
    }
    {
        std::ofstream script(failing_script_path);
        script << "from pyscene_python_path_helper import VALUE\n";
        script << "raise RuntimeError(f'failure after importing {VALUE}')\n";
    }

    ctx.execute_string("import sys\nsaved_sys_path = sys.path\nsaved_sys_path_values = list(sys.path)");
    const std::vector<std::filesystem::path> search_paths{temp_dir};
    ctx.execute_file(script_path, "__search_path_test__", search_paths);
    ctx.execute_string("assert sys.path is saved_sys_path\nassert sys.path == saved_sys_path_values");

    CHECK_THROWS_AS(
        ctx.execute_file(failing_script_path, "__search_path_failure_test__", search_paths),
        PythonException
    );
    ctx.execute_string("assert sys.path is saved_sys_path\nassert sys.path == saved_sys_path_values");
}

#if defined(_WIN32)
TEST_CASE("execute_file supports non-ASCII Windows paths")
{
    PythonContext ctx = PythonInterpreter::get().create_context();
    std::wstring directory_name = L"python_path_";
    directory_name.push_back(static_cast<wchar_t>(0x00e9));
    const std::filesystem::path unicode_dir = testing::get_case_temp_directory() / directory_name;
    std::filesystem::create_directories(unicode_dir);

    const std::filesystem::path helper_path = unicode_dir / "pyscene_unicode_path_helper.py";
    const std::filesystem::path script_path = unicode_dir / "uses_unicode_path_helper.py";
    {
        std::ofstream helper(helper_path);
        helper << "VALUE = 23\n";
    }
    {
        std::ofstream script(script_path);
        script << "from pyscene_unicode_path_helper import VALUE\n";
        script << "assert VALUE == 23\n";
        script << "assert '\\\\' not in __file__\n";
    }

    const std::vector<std::filesystem::path> search_paths{unicode_dir};
    ctx.execute_file(script_path, "__unicode_path_test__", search_paths);
}
#endif

TEST_CASE("execute_file with non-existent path throws")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    CHECK_THROWS_AS(ctx.execute_file("no_such_file_xyz.py"), PythonException);
}

TEST_CASE("move constructor transfers ownership")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    ctx.execute_string("moved_var = 99");

    PythonContext ctx2 = std::move(ctx);
    // ctx2 should have the state.
    ctx2.execute_string("assert moved_var == 99");
}

TEST_CASE("move assignment transfers ownership")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx1 = py.create_context();
    ctx1.execute_string("a_var = 1");

    PythonContext ctx2 = py.create_context();
    ctx2 = std::move(ctx1);
    // ctx2 should now have ctx1's state.
    ctx2.execute_string("assert a_var == 1");
}

TEST_CASE("context has __name__ set to __main__")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();
    ctx.execute_string("assert __name__ == '__main__'");
}

TEST_CASE("execute_file syntax error includes filename")
{
    PythonInterpreter& py = PythonInterpreter::get();
    PythonContext ctx = py.create_context();

    auto temp_dir = falcor::testing::get_case_temp_directory();
    auto script_path = temp_dir / "bad_syntax.py";

    {
        std::ofstream f(script_path);
        f << "def broken(\n";
    }

    try {
        ctx.execute_file(script_path);
        FAIL("Expected PythonException");
    } catch (const PythonException& e) {
        std::string msg = e.what();
        CHECK(msg.find("bad_syntax.py") != std::string::npos);
        CHECK(msg.find("SyntaxError") != std::string::npos);
    }
}

TEST_CASE("separate contexts work from multiple threads")
{
    PythonInterpreter& py = PythonInterpreter::get();

    auto work = [&](int id)
    {
        PythonContext ctx = py.create_context();
        ctx.execute_string(fmt::format("thread_id = {}", id));
        ctx.execute_string(fmt::format("assert thread_id == {}", id));
    };

    std::thread t1(work, 1);
    std::thread t2(work, 2);
    t1.join();
    t2.join();
}

TEST_SUITE_END();
