// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"
#include "falcor2/core/signal.h"

namespace falcor {

namespace detail {

/// Helper to extract Args... from Signal<void(Args...)> for binding.
template<typename SignalType>
struct BindSignal;

template<typename... Args>
struct BindSignal<Signal<void(Args...)>> {
    template<typename... Extra>
    static nb::class_<Signal<void(Args...)>> bind(nb::handle scope, const char* name, Extra&&... extra)
    {
        using SignalType = Signal<void(Args...)>;

        return nb::class_<SignalType>(scope, name, std::forward<Extra>(extra)...)
            .def(nb::init<>())
            .def(
                "connect",
                [](SignalType& self, std::function<void(Args...)> callback, nb::object owner) -> SignalConnection
                {
                    SignalConnection conn = self.connect(
                        [callback](Args... args)
                        {
                            nb::gil_scoped_acquire guard;
                            callback(args...);
                        }
                    );

                    if (!owner.is_none()) {
                        // Clone the connection for the weak-ref finalizer so both
                        // the caller's handle and the finalizer can independently
                        // disconnect. Cast to Python object so the weak-ref
                        // callback holds a reference to the Python wrapper.
                        nb::object py_conn = nb::cast(conn.share());
                        nb::module_::import_("weakref").attr("finalize")(owner, py_conn.attr("disconnect"));
                    }

                    return conn;
                },
                "callback"_a,
                "owner"_a = nb::none(),
                "Connect a callable slot. Returns a SignalConnection handle.\n\n"
                "Args:\n"
                "    callback: The callable to invoke when the signal is emitted.\n"
                "    owner: Optional owner object. When provided, the connection is\n"
                "        automatically disconnected when the owner is garbage collected.\n"
                "        Note: passing a bound method of owner as callback creates a\n"
                "        reference cycle that prevents collection. Use a lambda instead."
            )
            .def(
                "emit",
                &SignalType::emit,
                "Emit the signal, calling all connected slots.\n\n"
                "Disconnect during emit suppresses later callbacks in the same emission.\n"
                "New connections created during emit are deferred until the next emission."
            )
            .def("disconnect_all", &SignalType::disconnect_all, "Disconnect all slots.")
            .def_prop_ro("empty", &SignalType::empty, "Returns true if no slots are connected.")
            .def_prop_ro("size", &SignalType::size, "Returns the number of connected slots.");
    }
};

} // namespace detail

/// Bind a Signal type to Python.
/// Creates a nanobind class with connect, emit, empty, and size.
/// The connect method wraps the Python callable with GIL acquisition
/// so callbacks are safe to invoke from any context.
///
/// Usage:
///   using MySignal = Signal<void(int)>;
///   bind_signal<MySignal>(m, "MySignal");
template<typename SignalType, typename... Extra>
nb::class_<SignalType> bind_signal(nb::handle scope, const char* name, Extra&&... extra)
{
    return detail::BindSignal<SignalType>::bind(scope, name, std::forward<Extra>(extra)...);
}

} // namespace falcor
