// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/macros.h"
#include "falcor2/core/ustring.h"

#include <sgl/core/input.h>

#include <deque>
#include <functional>
#include <optional>
#include <string>

namespace falcor::ui {

using CommandId = ustring;

/// Keyboard shortcut associated with an editor command.
struct FALCOR_API CommandShortcut {
    sgl::KeyCode key{sgl::KeyCode::unknown};
    sgl::KeyModifierFlags modifiers{sgl::KeyModifierFlags::none};
    std::string display_name;

    bool matches(const sgl::KeyboardEvent& event) const;
};

/// Persistent definition of a user-invokable editor command.
struct FALCOR_API EditorCommand {
    CommandId id;
    std::string label;
    std::optional<CommandShortcut> shortcut;
    std::function<void()> execute;
    std::function<bool()> can_execute;
    std::function<bool()> is_checked;
};

/// Registry and dispatcher for editor commands.
class FALCOR_API EditorCommandRegistry {
public:
    void register_command(EditorCommand command);

    const EditorCommand* find(const CommandId& id) const;

    bool execute(const CommandId& id) const;
    bool can_execute(const CommandId& id) const;
    bool is_checked(const CommandId& id) const;

    /// Execute the first enabled command matching @p event.
    /// Returns true if a command was executed.
    bool dispatch_shortcut(const sgl::KeyboardEvent& event) const;

private:
    const EditorCommand& require(const CommandId& id) const;

    // Keep pointers returned by find() stable when commands are registered.
    std::deque<EditorCommand> m_commands;
};

} // namespace falcor::ui
