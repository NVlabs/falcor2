// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/python_interpreter.h"

#include <fstream>
#include <thread>

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
