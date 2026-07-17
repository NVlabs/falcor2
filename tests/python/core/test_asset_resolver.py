# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

import falcor2 as f2


def test_constructor_dispatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    single = f2.AssetResolver(" first ; second ")
    pathlike = f2.AssetResolver(Path("literal;path"))
    batch = f2.AssetResolver([Path("third"), Path("literal;name")])

    assert single.search_paths == [tmp_path / "first", tmp_path / "second"]
    assert pathlike.search_paths == [tmp_path / "literal;path"]
    assert batch.search_paths == [tmp_path / "third", tmp_path / "literal;name"]


def test_push_dispatch_and_immutability(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    resolver = f2.AssetResolver()

    single = resolver.push("single")
    pathlike = resolver.push(Path("literal;path"))
    batch = resolver.push([Path("first"), Path("second")])

    assert resolver.search_paths == []
    assert single is not resolver
    assert single.search_paths == [tmp_path / "single"]
    assert pathlike.search_paths == [tmp_path / "literal;path"]
    assert batch.search_paths == [tmp_path / "first", tmp_path / "second"]


def test_semicolon_paths_and_literal_batch_elements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    resolver = f2.AssetResolver()

    split = resolver.push(" first ; ; second ")
    literal = resolver.push([Path("literal;name")])

    assert split.search_paths == [tmp_path / "first", tmp_path / "second"]
    assert literal.search_paths == [tmp_path / "literal;name"]


def test_resolution_priority_and_missing_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    working = tmp_path / "working"
    search = tmp_path / "search"
    working.mkdir()
    search.mkdir()
    (working / "shared.txt").write_text("working", encoding="utf-8")
    (search / "shared.txt").write_text("search", encoding="utf-8")
    monkeypatch.chdir(working)

    resolver = f2.AssetResolver()
    configured = resolver.push(search)

    assert resolver.resolve_path("shared.txt") == (working / "shared.txt").resolve()
    assert configured.resolve_path("shared.txt") == (search / "shared.txt").resolve()
    assert configured.resolve_path("missing.txt") == Path()

    with pytest.raises(RuntimeError) as exc_info:
        configured.resolve_path("missing.txt", required=True)
    message = str(exc_info.value)
    assert "required asset path" in message
    assert "missing.txt" in message
    assert "AssetResolver(search_paths=[" in message
    assert str(search) in message


def test_pattern_results_are_deterministic_and_do_not_mix_roots(tmp_path: Path) -> None:
    high = tmp_path / "high" / "tiles"
    low = tmp_path / "low" / "tiles"
    high.mkdir(parents=True)
    low.mkdir(parents=True)
    (high / "z_1002.exr").touch()
    (high / "a_1001.exr").touch()
    (low / "0_1000.exr").touch()

    resolver = f2.AssetResolver().push([high.parent, low.parent])

    assert resolver.resolve_path_pattern("tiles", r".*_[0-9]+\.exr", False) == [
        high / "a_1001.exr",
        high / "z_1002.exr",
    ]
    assert resolver.resolve_path_pattern("tiles", r".*_[0-9]+\.exr", True) == [high / "a_1001.exr"]
