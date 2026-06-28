// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#define SLANGPY_SKIP_PY_DOC_INCLUDE
#include <slangpy_ext/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/typing.h>
#include <nanobind/stl/unordered_map.h>

#include <cstddef>
#include <initializer_list>
#include <memory>

#include "py_doc.h"

namespace falcor {

template<typename T, typename FillFunc>
nb::ndarray<nb::numpy, T> make_numpy_copy(size_t value_count, std::initializer_list<size_t> shape, FillFunc fill)
{
    auto data = std::make_unique<T[]>(value_count);
    fill(data.get());

    T* raw_data = data.release();
    nb::capsule owner(
        raw_data,
        [](void* p) noexcept
        {
            delete[] static_cast<T*>(p);
        }
    );
    return nb::ndarray<nb::numpy, T>(raw_data, shape, owner);
}

} // namespace falcor

#define FALCOR_DICT_TO_DESC_BEGIN(type) SGL_DICT_TO_DESC_BEGIN(type)
#define FALCOR_DICT_TO_DESC_FIELD(name, type) SGL_DICT_TO_DESC_FIELD(name, type)
#define FALCOR_DICT_TO_DESC_FIELD_LIST(name, type) SGL_DICT_TO_DESC_FIELD_LIST(name, type)
#define FALCOR_DICT_TO_DESC_FIELD_CUSTOM(name, code) SGL_DICT_TO_DESC_FIELD_CUSTOM(name, code)
#define FALCOR_DICT_TO_DESC_END() SGL_DICT_TO_DESC_END()

#define FALCOR_PY_DECLARE(name) extern void falcor2_python_export_##name(nb::module_& m)
#define FALCOR_PY_EXPORT(name) void falcor2_python_export_##name([[maybe_unused]] ::nb::module_& m)
#define FALCOR_PY_IMPORT(name) falcor2_python_export_##name(m)

#undef D
#define D(...) DOC(falcor, __VA_ARGS__)

// Helper macros to simplify defining properties

#define DEF_RO(cls, prop, ...) def_ro(#prop, &cls::prop, D(cls, prop), ##__VA_ARGS__)
#define DEF_RW(cls, prop, ...) def_rw(#prop, &cls::prop, D(cls, prop), ##__VA_ARGS__)
#define DEF_PROP_RO(cls, prop, ...) def_prop_ro(#prop, &cls::prop, D(cls, prop), ##__VA_ARGS__)
#define DEF_PROP_RW(cls, prop, ...) def_prop_rw(#prop, &cls::prop, &cls::set_##prop, D(cls, prop), ##__VA_ARGS__)

#define DEF_RO_2(parent, cls, prop, ...) def_ro(#prop, &parent::cls::prop, D(parent, cls, prop), ##__VA_ARGS__)
#define DEF_RW_2(parent, cls, prop, ...) def_rw(#prop, &parent::cls::prop, D(parent, cls, prop), ##__VA_ARGS__)
#define DEF_PROP_RO_2(parent, cls, prop, ...)                                                                          \
    def_prop_ro(#prop, &parent::cls::prop, D(parent, cls, prop), ##__VA_ARGS__)
#define DEF_PROP_RW_2(parent, cls, prop, ...)                                                                          \
    def_prop_rw(#prop, &parent::cls::prop, &cls::set_##prop, D(parent, cls, prop), ##__VA_ARGS__)
