// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "core/reflection/python_class_reflection.h"

#include "falcor2/ui/property_editor.h"

#include <imgui.h>

#include <string>
#include <string_view>

namespace nb = nanobind;
using namespace nb::literals;

namespace falcor::ui {

namespace {

bool render_python_properties(nb::object instance, PropertyEditorContext& context)
{
    reflection::PythonClassReflection& class_reflection = reflection::PythonClassReflection::find_or_create(instance);
    class_reflection.ensure_up_to_date();

    nb::handle instance_handle = instance;
    return properties_editor(
        class_reflection.descriptors(),
        &instance_handle,
        &class_reflection,
        class_reflection.generation(),
        context
    );
}

/// Stateful bridge from Python reflected objects to the native property editor.
class PythonPropertyEditor {
public:
    bool render(std::string_view label, nb::object instance)
    {
        bool changed = false;
        const std::string label_string(label);

        ImGui::PushID(instance.ptr());
        if (ImGui::CollapsingHeader(label_string.c_str(), ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Indent();
            changed = render_python_properties(instance, m_context);
            ImGui::Unindent();
        }
        ImGui::PopID();

        return changed;
    }

    PropertyEditorContext& context() { return m_context; }

private:
    PropertyEditorContext m_context;
};

} // namespace

} // namespace falcor::ui

FALCOR_PY_EXPORT(ui_property_editor)
{
    nb::module_ ui = nb::module_::import_("falcor2.ui");

    nb::class_<falcor::ui::PropertyEditorContext>(ui, "PropertyEditorContext")
        .def(nb::init<>())
        .def_rw("show_advanced", &falcor::ui::PropertyEditorContext::show_advanced)
        .def_rw("show_read_only", &falcor::ui::PropertyEditorContext::show_read_only);

    ui.def(
        "properties_editor",
        [](nb::object instance, falcor::ui::PropertyEditorContext& context)
        {
            ImGui::PushID(instance.ptr());
            bool changed = falcor::ui::render_python_properties(instance, context);
            ImGui::PopID();
            return changed;
        },
        "instance"_a,
        "context"_a,
        "Render a Python @reflected object using an explicit property editor context."
    );

    nb::class_<falcor::ui::PythonPropertyEditor>(ui, "PropertyEditor")
        .def(nb::init<>())
        .def_prop_ro(
            "context",
            &falcor::ui::PythonPropertyEditor::context,
            nb::rv_policy::reference_internal,
            "Property editor context controlling visibility and layout caching."
        )
        .def(
            "render",
            &falcor::ui::PythonPropertyEditor::render,
            "label"_a,
            "instance"_a,
            "Render a Python @reflected object using Falcor2's native property editor."
        );
}
