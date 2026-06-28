// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "nanobind.h"
#include "falcor2/utils/idictionary.h"
#include "falcor2/core/error.h"

namespace falcor {

class NBDictionary : public IDictionary {
public:
    NBDictionary() { m_dict_stack.push_back(nb::dict()); }

    template<typename T>
        requires requires(T v, IDictionary& d) { v.to_dictionary(d); }
    static nb::dict get_uniforms(T& self)
    {
        NBDictionary dict;
        self.to_dictionary(dict);
        return dict.nbdict();
    }

    template<typename T>
        requires requires(T v, IDictionary& d) { v.to_globals_dictionary(d); }
    static nb::dict get_global_uniforms(T& self)
    {
        NBDictionary dict;
        self.to_globals_dictionary(dict);
        return dict.nbdict();
    }

    nb::dict& nbdict()
    {
        FALCOR_CHECK_EQ(m_dict_stack.size(), 1);
        return m_dict_stack.back();
    }

    void push(std::string_view nested_name) override
    {
        std::string name(nested_name);

        nb::dict& top = m_dict_stack.back();
        nb::dict child_dict;
        if (top.contains(name.c_str())) {
            nb::object child = top[name.c_str()];
            nb::try_cast<nb::dict>(child, child_dict, false);
        }

        m_dict_stack.push_back(child_dict);
        m_dict_names.push_back(std::move(name));
    }

    void pop() override
    {
        FALCOR_CHECK_GE(m_dict_stack.size(), 2);
        FALCOR_CHECK_GE(m_dict_names.size(), 1);

        nb::dict& nested = m_dict_stack[m_dict_stack.size() - 1];
        nb::dict& main = m_dict_stack[m_dict_stack.size() - 2];
        const std::string& name = m_dict_names.back();

        main[name.c_str()] = nested;

        m_dict_stack.pop_back();
        m_dict_names.pop_back();
    }

    template<typename T>
    void set_value(std::string_view name, const T& value)
    {
        nb::dict& dict = m_dict_stack.back();
        dict[std::string(name).c_str()] = value;
    }

    template<typename T>
    void set_span(std::string_view name, std::span<const T> value)
    {
        nb::list list;
        for (size_t i = 0; i < value.size(); ++i)
            list.append(value[i]);

        nb::dict& dict = m_dict_stack.back();
        dict[std::string(name).c_str()] = list;
    }

#define X(type_)                                                                                                       \
    void set(std::string_view name, const type_& value) override { set_value(name, value); }                           \
    void set(std::string_view name, std::span<const type_> value) override { set_span(name, value); }
    FALCOR_DICTIONARY_TYPES
#undef X

private:
    std::vector<nb::dict> m_dict_stack;
    std::vector<std::string> m_dict_names;
};

} // namespace falcor
