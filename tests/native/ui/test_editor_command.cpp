// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/ui/editor_command.h"

using namespace falcor;
using namespace falcor::ui;

TEST_SUITE_BEGIN("EditorCommand");

TEST_CASE("editor commands execute and expose state")
{
    static const CommandId COMMAND_ID{"test.execute"};

    int execution_count = 0;
    bool enabled = true;
    bool checked = false;

    EditorCommandRegistry commands;
    commands.register_command({
        .id = COMMAND_ID,
        .label = "Execute",
        .execute =
            [&execution_count]
        {
            ++execution_count;
        },
        .can_execute =
            [&enabled]
        {
            return enabled;
        },
        .is_checked =
            [&checked]
        {
            return checked;
        },
    });

    REQUIRE(commands.find(COMMAND_ID));
    CHECK(commands.find(COMMAND_ID)->label == "Execute");
    CHECK(commands.can_execute(COMMAND_ID));
    CHECK_FALSE(commands.is_checked(COMMAND_ID));
    CHECK(commands.execute(COMMAND_ID));
    CHECK(execution_count == 1);

    enabled = false;
    checked = true;
    CHECK_FALSE(commands.can_execute(COMMAND_ID));
    CHECK(commands.is_checked(COMMAND_ID));
    CHECK_FALSE(commands.execute(COMMAND_ID));
    CHECK(execution_count == 1);
}

TEST_CASE("editor commands dispatch exact keyboard shortcuts")
{
    static const CommandId PLAIN_COMMAND_ID{"test.shortcut.plain"};
    static const CommandId CTRL_COMMAND_ID{"test.shortcut.ctrl"};

    int plain_count = 0;
    int ctrl_count = 0;
    bool plain_enabled = true;
    EditorCommandRegistry commands;
    commands.register_command({
        .id = PLAIN_COMMAND_ID,
        .label = "Plain",
        .shortcut = CommandShortcut{sgl::KeyCode::q, sgl::KeyModifierFlags::none, "Q"},
        .execute =
            [&plain_count]
        {
            ++plain_count;
        },
        .can_execute =
            [&plain_enabled]
        {
            return plain_enabled;
        },
    });
    commands.register_command({
        .id = CTRL_COMMAND_ID,
        .label = "Ctrl",
        .shortcut = CommandShortcut{sgl::KeyCode::q, sgl::KeyModifierFlags::ctrl, "Ctrl+Q"},
        .execute = [&ctrl_count]
        {
            ++ctrl_count;
        },
    });

    CHECK(commands.dispatch_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::q}));
    CHECK(plain_count == 1);
    CHECK(ctrl_count == 0);

    CHECK(
        commands.dispatch_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::q, 0, sgl::KeyModifierFlags::ctrl})
    );
    CHECK(plain_count == 1);
    CHECK(ctrl_count == 1);

    CHECK_FALSE(commands.dispatch_shortcut(
        {sgl::KeyboardEventType::key_press, sgl::KeyCode::q, 0, sgl::KeyModifierFlags::shift}
    ));
    CHECK_FALSE(commands.dispatch_shortcut({sgl::KeyboardEventType::key_release, sgl::KeyCode::q}));

    plain_enabled = false;
    CHECK_FALSE(commands.dispatch_shortcut({sgl::KeyboardEventType::key_press, sgl::KeyCode::q}));
    CHECK(plain_count == 1);
}

TEST_CASE("editor command IDs must be unique")
{
    static const CommandId COMMAND_ID{"test.duplicate"};
    EditorCommandRegistry commands;
    commands.register_command({
        .id = COMMAND_ID,
        .label = "First",
        .execute = []
        {
        },
    });
    CHECK_THROWS(commands.register_command({
        .id = COMMAND_ID,
        .label = "Second",
        .execute = []
        {
        },
    }));
}

TEST_SUITE_END();
