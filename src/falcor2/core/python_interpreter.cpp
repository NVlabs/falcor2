// SPDX-License-Identifier: Apache-2.0

#include "python_interpreter.h"

#include "falcor2/core/error.h"

#include <fstream>

// On Windows, Python.h in debug mode tries to link against python3XX_d.lib
// which is not available in standard Python distributions.
#if defined(_DEBUG) && defined(_MSC_VER)
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif

namespace falcor {

/// RAII wrapper for PyGILState_Ensure/PyGILState_Release.
/// Safe to use when the GIL is already held (e.g. called from Python).
class GILGuard {
public:
    GILGuard() noexcept { m_state = PyGILState_Ensure(); }
    ~GILGuard() { PyGILState_Release(m_state); }
    GILGuard(const GILGuard&) = delete;
    GILGuard& operator=(const GILGuard&) = delete;

private:
    PyGILState_STATE m_state;
};

// ----------------------------------------------------------------------------
// PythonInterpreter
// ----------------------------------------------------------------------------

PythonInterpreter::PythonInterpreter()
{
    initialize();
}

PythonInterpreter::~PythonInterpreter()
{
    // When we started the interpreter ourselves (embedded use case),
    // finalize it for a clean shutdown. In the extension case
    // (m_owns_python == false), Python is owned by the host process.
    //
    // WARNING: This runs during static destruction. Other static/global
    // destructors (e.g. pybind11 type internals, extension module state)
    // may still reference the interpreter. Py_FinalizeEx() is best-effort
    // per CPython docs. If shutdown crashes occur, consider removing this
    // call -- many embedders intentionally skip finalization.
    // if (m_owns_python) {
    //     PyGILState_Ensure();
    //     Py_FinalizeEx();
    // }
}

PythonInterpreter& PythonInterpreter::get()
{
    static PythonInterpreter instance;
    return instance;
}

PythonContext PythonInterpreter::create_context()
{
    FALCOR_ASSERT(Py_IsInitialized());
    return PythonContext();
}

void PythonInterpreter::initialize()
{
    if (Py_IsInitialized())
        return;

    PyConfig config;
    PyConfig_InitPythonConfig(&config);

#ifdef FALCOR_PYTHON_HOME
    // Set the Python home to the prefix discovered at build time so the
    // embedded interpreter can locate the standard library even when the
    // host executable lives outside the Python installation tree.
    // The PYTHONHOME environment variable still takes precedence because
    // config.use_environment defaults to 1.
    std::wstring home = L"" FALCOR_PYTHON_HOME;
    PyConfig_SetString(&config, &config.home, home.c_str());
#endif

    PyStatus status = Py_InitializeFromConfig(&config);
    PyConfig_Clear(&config);

    if (PyStatus_Exception(status)) {
        throw PythonException(
            fmt::format("Failed to initialize Python: {}", status.err_msg ? status.err_msg : "unknown error")
        );
    }

    m_owns_python = true;

    // Release the GIL so that any thread (including main) can acquire it
    // on demand via GILGuard / PyGILState_Ensure.
    PyEval_SaveThread();
}

/// @pre Caller must hold the GIL.
std::string PythonInterpreter::capture_python_error()
{
    if (!PyErr_Occurred())
        return {};

    std::string message;

#if PY_VERSION_HEX >= 0x030C0000
    // Python 3.12+: use the modern non-deprecated API.
    PyObject* exc = PyErr_GetRaisedException();
    if (!exc)
        return {};

    PyObject* tb_module = PyImport_ImportModule("traceback");
    if (tb_module) {
        PyObject* format_func = PyObject_GetAttrString(tb_module, "format_exception");
        if (format_func && PyCallable_Check(format_func)) {
            PyObject* args = PyTuple_Pack(1, exc);
            if (args) {
                PyObject* formatted_list = PyObject_CallObject(format_func, args);
                if (formatted_list) {
                    PyObject* separator = PyUnicode_FromString("");
                    if (separator) {
                        PyObject* joined = PyUnicode_Join(separator, formatted_list);
                        if (joined) {
                            const char* utf8 = PyUnicode_AsUTF8(joined);
                            if (utf8)
                                message = utf8;
                            Py_DECREF(joined);
                        }
                        Py_DECREF(separator);
                    }
                    Py_DECREF(formatted_list);
                }
                Py_DECREF(args);
            }
        }
        Py_XDECREF(format_func);
        Py_DECREF(tb_module);
    }

    // Fallback: just str() the exception value.
    if (message.empty()) {
        PyObject* str_obj = PyObject_Str(exc);
        if (str_obj) {
            const char* utf8 = PyUnicode_AsUTF8(str_obj);
            if (utf8)
                message = utf8;
            Py_DECREF(str_obj);
        }
    }

    Py_DECREF(exc);
#else
    // Python < 3.12: use the legacy Fetch/Normalize API.
    PyObject* type = nullptr;
    PyObject* value = nullptr;
    PyObject* traceback = nullptr;
    PyErr_Fetch(&type, &value, &traceback);
    PyErr_NormalizeException(&type, &value, &traceback);

    // Try to get the full traceback as a string.
    PyObject* tb_module = PyImport_ImportModule("traceback");
    if (tb_module) {
        PyObject* format_func = PyObject_GetAttrString(tb_module, "format_exception");
        if (format_func && PyCallable_Check(format_func)) {
            PyObject* args
                = PyTuple_Pack(3, type ? type : Py_None, value ? value : Py_None, traceback ? traceback : Py_None);
            if (args) {
                PyObject* formatted_list = PyObject_CallObject(format_func, args);
                if (formatted_list) {
                    PyObject* separator = PyUnicode_FromString("");
                    if (separator) {
                        PyObject* joined = PyUnicode_Join(separator, formatted_list);
                        if (joined) {
                            const char* utf8 = PyUnicode_AsUTF8(joined);
                            if (utf8)
                                message = utf8;
                            Py_DECREF(joined);
                        }
                        Py_DECREF(separator);
                    }
                    Py_DECREF(formatted_list);
                }
                Py_DECREF(args);
            }
        }
        Py_XDECREF(format_func);
        Py_DECREF(tb_module);
    }

    // Fallback: just str() the exception value.
    if (message.empty() && value) {
        PyObject* str_obj = PyObject_Str(value);
        if (str_obj) {
            const char* utf8 = PyUnicode_AsUTF8(str_obj);
            if (utf8)
                message = utf8;
            Py_DECREF(str_obj);
        }
    }

    Py_XDECREF(type);
    Py_XDECREF(value);
    Py_XDECREF(traceback);
#endif

    if (message.empty())
        message = "Unknown Python error";

    // Clear any secondary error that occurred during formatting.
    PyErr_Clear();

    return message;
}

// ----------------------------------------------------------------------------
// PythonContext
// ----------------------------------------------------------------------------

PythonContext::PythonContext()
{
    GILGuard gil;

    PyObject* globals = PyDict_New();
    if (!globals) {
        std::string msg = PythonInterpreter::capture_python_error();
        throw PythonException(msg.empty() ? "Failed to create globals dict" : msg);
    }

    PyObject* builtins = PyEval_GetBuiltins(); // borrowed ref
    PyDict_SetItemString(globals, "__builtins__", builtins);

    // Set __name__ so code that checks `if __name__ == '__main__'` or
    // uses relative imports has a well-defined module name.
    PyObject* name = PyUnicode_FromString("__main__");
    if (name) {
        PyDict_SetItemString(globals, "__name__", name);
        Py_DECREF(name);
    }

    m_globals = globals;
}

PythonContext::~PythonContext()
{
    if (m_globals) {
        GILGuard gil;
        Py_DECREF(static_cast<PyObject*>(m_globals));
    }
}

PythonContext::PythonContext(PythonContext&& other) noexcept
    : m_globals(other.m_globals)
{
    other.m_globals = nullptr;
}

PythonContext& PythonContext::operator=(PythonContext&& other) noexcept
{
    if (this != &other) {
        if (m_globals) {
            GILGuard gil;
            Py_DECREF(static_cast<PyObject*>(m_globals));
        }
        m_globals = other.m_globals;
        other.m_globals = nullptr;
    }
    return *this;
}

void PythonContext::execute_string(const std::string& code)
{
    FALCOR_ASSERT(Py_IsInitialized());
    FALCOR_ASSERT(m_globals);

    GILGuard gil;

    PyObject* globals = static_cast<PyObject*>(m_globals);
    PyObject* ret = PyRun_String(code.c_str(), Py_file_input, globals, globals);

    if (ret) {
        Py_DECREF(ret);
    } else {
        std::string msg = PythonInterpreter::capture_python_error();
        throw PythonException(msg);
    }
}

void PythonContext::execute_file(const std::filesystem::path& path)
{
    FALCOR_ASSERT(Py_IsInitialized());
    FALCOR_ASSERT(m_globals);

    std::string path_str = path.string();

    // Read the file with C++ streams to avoid CRT mismatch issues on Windows
    // debug builds (our code links debug CRT, Python links release CRT).
    std::error_code ec;
    auto file_size = std::filesystem::file_size(path, ec);
    if (ec)
        throw PythonException(fmt::format("Cannot open file: {}", path_str));
    std::string code(file_size, '\0');

    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        throw PythonException(fmt::format("Cannot open file: {}", path_str));
    }
    ifs.read(code.data(), static_cast<std::streamsize>(file_size));
    code.resize(static_cast<size_t>(ifs.gcount()));

    GILGuard gil;

    PyObject* globals = static_cast<PyObject*>(m_globals);

    // Set __file__ with forward slashes for Python convention.
    std::string generic_path_str = path.generic_string();
    PyObject* py_path = PyUnicode_FromString(generic_path_str.c_str());
    if (py_path) {
        PyDict_SetItemString(globals, "__file__", py_path);
        Py_DECREF(py_path);
    }

    // Compile with the filename for meaningful tracebacks.
    PyObject* code_obj = Py_CompileString(code.c_str(), path_str.c_str(), Py_file_input);
    if (!code_obj) {
        std::string msg = PythonInterpreter::capture_python_error();
        throw PythonException(msg);
    }

    PyObject* ret = PyEval_EvalCode(code_obj, globals, globals);
    Py_DECREF(code_obj);

    if (ret) {
        Py_DECREF(ret);
    } else {
        std::string msg = PythonInterpreter::capture_python_error();
        throw PythonException(msg);
    }
}

} // namespace falcor
