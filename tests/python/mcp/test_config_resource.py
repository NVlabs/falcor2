# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from importlib import resources

from falcor2.mcp.offline import python_info


def test_config_resource_loads():
    config_text = resources.files("falcor2.mcp").joinpath("config.json").read_text(encoding="utf-8")
    config = json.loads(config_text)

    assert config["name"] == "falcor2"
    endpoint_names = {endpoint["name"] for endpoint in config["endpoints"]}
    assert endpoint_names == {
        "falcor2.bridge.ping",
        "falcor2.app.status",
        "falcor2.app.screenshot",
        "falcor2.offline.python_info",
        "falcor2.offline.render_scene",
        "falcor2.echo",
    }

    endpoints = {endpoint["name"]: endpoint for endpoint in config["endpoints"]}
    screenshot_description = endpoints["falcor2.app.screenshot"]["description"]
    render_description = endpoints["falcor2.offline.render_scene"]["description"]
    assert "MANDATORY PRESENTATION REQUIREMENT" in screenshot_description
    assert "clickable Markdown link using markdown_path" in screenshot_description
    assert (
        "inline Markdown image using ![Falcor2 screenshot](markdown_path)" in screenshot_description
    )
    assert "final response" in screenshot_description
    assert "MANDATORY PRESENTATION REQUIREMENT" in render_description
    assert "clickable Markdown link using images[].markdown_path" in render_description
    assert "inline Markdown image using ![<camera_name>](<markdown_path>)" in render_description
    assert "final response" in render_description


def test_python_info_offline_endpoint_returns_context():
    result = python_info({"hello": "world"}, {"workspace_root": "workspace"})

    assert "python" in result
    assert result["params"] == {"hello": "world"}
    assert result["context"] == {"workspace_root": "workspace"}
