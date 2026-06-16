// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"
#include "signal.h"

#include "falcor2/core/signal.h"

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(core_signal)
{
    using namespace falcor;

    nb::class_<SignalConnection>(m, "SignalConnection", D(SignalConnection))
        .def("disconnect", &SignalConnection::disconnect, D(SignalConnection, disconnect))
        .def_prop_ro("connected", &SignalConnection::connected, D(SignalConnection, connected))
        .def("share", &SignalConnection::share, D(SignalConnection, share));

    nb::class_<SignalConnectionGroup>(m, "SignalConnectionGroup", D(SignalConnectionGroup))
        .def(nb::init<>(), D(SignalConnectionGroup, SignalConnectionGroup))
        .def(
            "add",
            [](SignalConnectionGroup& self, SignalConnection& connection)
            {
                self.add(std::move(connection));
            },
            "connection"_a,
            D(SignalConnectionGroup, add)
        )
        .def("disconnect_all", &SignalConnectionGroup::disconnect_all, D(SignalConnectionGroup, disconnect_all))
        .def_prop_ro("empty", &SignalConnectionGroup::empty, D(SignalConnectionGroup, empty))
        .def_prop_ro("size", &SignalConnectionGroup::size, D(SignalConnectionGroup, size));

    // Bind Signal<void(nb::object)> as the primary Python-facing signal type.
    bind_signal<Signal<void(nb::object)>>(m, "Signal", "Signal passing a single Python object argument.");
}
