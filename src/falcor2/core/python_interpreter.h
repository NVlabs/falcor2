// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"

#include <filesystem>
#include <stdexcept>
#include <string>

namespace falcor {

/// Exception thrown when Python code execution fails.
/// Contains the full Python stack trace in the message.
class FALCOR_API PythonException : public std::runtime_error {
public:
    explicit PythonException(const std::string& message)
        : std::runtime_error(message)
    {
    }
};

class PythonInterpreter;

/// An isolated Python execution context with its own globals dict.
///
/// Created via PythonInterpreter::create_context(). Each context has
/// a fresh globals dict seeded only with __builtins__, providing full
/// isolation between contexts. State accumulates within a context
/// across calls (e.g. imports and variables persist).
///
/// @note All public methods automatically acquire/release the GIL,
///       making them safe to call from any thread, including from
///       Python (where the GIL is already held).
///
/// @note Each PythonContext instance should only be used from a single
///       thread at a time. While individual operations are GIL-safe,
///       concurrent calls on the same context may interleave and
///       cause logical races on shared state in the globals dict.
///
/// @warning PythonContext objects must not outlive program main().
///          Storing them in static/global variables risks destruction
///          after the Python interpreter state becomes invalid.
///
/// Usage:
/// @code
/// auto ctx = PythonInterpreter::get().create_context();
/// ctx.execute_string("x = 42");
/// ctx.execute_string("print(x)");  // works, same context
/// @endcode
class FALCOR_API PythonContext {
public:
    ~PythonContext();

    PythonContext(PythonContext&& other) noexcept;
    PythonContext& operator=(PythonContext&& other) noexcept;

    /// Execute a string of Python code in this context.
    /// @param code Python source code to execute.
    /// @throws PythonException on failure with the full Python traceback.
    void execute_string(const std::string& code);

    /// Execute a Python file in this context.
    /// @param path Path to the .py file to execute.
    /// @throws PythonException on failure with the full Python traceback.
    void execute_file(const std::filesystem::path& path);

private:
    PythonContext();

    /// Opaque pointer to the PyObject* globals dict.
    void* m_globals{nullptr};

    FALCOR_NON_COPYABLE(PythonContext);

    friend class PythonInterpreter;
};

/// Singleton interface for the process-global Python interpreter.
///
/// CPython is inherently process-global, so this class exposes it as
/// a singleton. The interpreter is lazily initialized on the first
/// call to get().
///
/// Usage:
/// @code
/// auto& py = PythonInterpreter::get();
///
/// auto ctx = py.create_context();
/// try {
///     ctx.execute_string("print('hello')");
/// } catch (const PythonException& e) {
///     log_error("{}", e.what());
/// }
/// @endcode
class FALCOR_API PythonInterpreter {
public:
    /// Get the singleton instance, lazily initializing the interpreter.
    /// @throws PythonException if initialization fails.
    static PythonInterpreter& get();

    /// Create a new isolated execution context.
    /// The context has a fresh globals dict seeded only with __builtins__.
    PythonContext create_context();

private:
    PythonInterpreter();
    ~PythonInterpreter();

    /// Initialize the Python interpreter.
    /// @throws PythonException if initialization fails.
    void initialize();

    /// Capture the current Python exception as a string and clear it.
    /// Returns an empty string if no exception is pending.
    /// @pre Caller must hold the GIL.
    static std::string capture_python_error();

    /// Whether this instance initialized the interpreter.
    bool m_owns_python{false};

    FALCOR_NON_COPYABLE_AND_MOVABLE(PythonInterpreter);

    friend class PythonContext;
};

} // namespace falcor
