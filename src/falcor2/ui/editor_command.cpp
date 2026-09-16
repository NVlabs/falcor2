// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/ui/editor_command.h"

#include "falcor2/core/error.h"
#include "falcor2/core/format.h"

#include <algorithm>
#include <utility>

namespace falcor::ui {

bool CommandShortcut::matches(const sgl::KeyboardEvent& event) const
{
    return event.is_key_press() && event.key == key && event.mods == modifiers;
}

void EditorCommandRegistry::register_command(EditorCommand command)
{
    FALCOR_CHECK(!command.id.empty(), "Editor command ID cannot be empty.");
    FALCOR_CHECK(!find(command.id), "Editor command '{}' is already registered.", command.id);
    FALCOR_CHECK(command.execute, "Editor command '{}' has no execute callback.", command.id);
    m_commands.push_back(std::move(command));
}

const EditorCommand* EditorCommandRegistry::find(const CommandId& id) const
{
    auto it = std::find_if(
        m_commands.begin(),
        m_commands.end(),
        [&id](const EditorCommand& command)
        {
            return command.id == id;
        }
    );
    return it == m_commands.end() ? nullptr : &*it;
}

bool EditorCommandRegistry::execute(const CommandId& id) const
{
    const EditorCommand& command = require(id);
    if (command.can_execute && !command.can_execute())
        return false;
    command.execute();
    return true;
}

bool EditorCommandRegistry::can_execute(const CommandId& id) const
{
    const EditorCommand& command = require(id);
    return !command.can_execute || command.can_execute();
}

bool EditorCommandRegistry::is_checked(const CommandId& id) const
{
    const EditorCommand& command = require(id);
    return command.is_checked && command.is_checked();
}

bool EditorCommandRegistry::dispatch_shortcut(const sgl::KeyboardEvent& event) const
{
    for (const EditorCommand& command : m_commands) {
        if (command.shortcut && command.shortcut->matches(event) && (!command.can_execute || command.can_execute())) {
            command.execute();
            return true;
        }
    }
    return false;
}

const EditorCommand& EditorCommandRegistry::require(const CommandId& id) const
{
    const EditorCommand* command = find(id);
    FALCOR_CHECK(command, "Unknown editor command '{}'.", id);
    return *command;
}

} // namespace falcor::ui
