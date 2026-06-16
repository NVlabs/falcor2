// SPDX-License-Identifier: Apache-2.0

// Portions of this file are derived from Mitsuba 3.
// Copyright (c) Mitsuba contributors.
// Licensed under the BSD 3-Clause License.
// See LICENSES/mitsuba3.txt for the full license text.

#pragma once

#include "falcor2/core/any.h"
#include "nanobind.h"

namespace falcor {

/// Create an Any object that stores a Python object.
Any any_wrap(nb::handle obj);

/// If the Any holds a Python object (created via any_wrap), return its handle.
/// Otherwise return an invalid handle.
nb::handle any_try_unwrap(const Any& a);

/// Convert a type-erased C++ value (type_info + void*) back to a Python object
/// using nanobind's internal type registry. This is a last-resort fallback for
/// values that were NOT created via any_wrap but whose C++ type has nanobind bindings.
/// Returns an invalid handle if the type is not registered.
nb::handle nb_type_to_python(const std::type_info& type, void* data);

} // namespace falcor
