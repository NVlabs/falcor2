# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dispatch Falcor2 setup and optional host integration tools.

Keep this module independent of native Falcor2 imports so setup works before a
build or GPU configuration. Source scripts invoke this file directly; the installed
f2 launcher loads it without executing falcor2/__init__.py. Hydra commands use
the tools in an existing Falcor2 checkout.

Setup resolves configuration, preflights managed-file conflicts, then applies the
changes and records hashes for future refreshes. Instruction entrypoints belong
to the project: only their marked Falcor2 sections are managed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Source checkouts keep canonical resources at the root; wheels package them here.
SKILLS = (
    _SOURCE_ROOT / "skills"
    if (_SOURCE_ROOT / "pyproject.toml").is_file()
    else Path(__file__).parent / "skills"
)
DESTINATIONS = {"codex": ".agents/skills", "claude": ".claude/skills"}
MANIFEST = ".falcor2/skills.json"

INSTRUCTIONS = ".falcor2/instructions.md"
ENTRYPOINTS = {"codex": "AGENTS.md", "claude": "CLAUDE.md"}
BEGIN = b"<!-- BEGIN FALCOR2 INSTRUCTIONS -->"
END = b"<!-- END FALCOR2 INSTRUCTIONS -->"


def _hash(content: bytes) -> str:
    """Fingerprint exact installed bytes to detect local edits on refresh."""
    return hashlib.sha256(content).hexdigest()


def _block(agent: str, newline: bytes) -> bytes:
    """Render the agent's guidance reference using the entrypoint's line endings."""
    reference = (
        "Before working with Falcor2, read [.falcor2/instructions.md](.falcor2/instructions.md)."
        if agent == "codex"
        else "@.falcor2/instructions.md"
    )
    return newline.join([BEGIN, b"", b"## Falcor2", b"", reference.encode(), b"", END])


def plan_blocks(
    contents: dict[str, bytes | None],
    previous: dict[str, str],
    agents: list[str],
) -> tuple[dict[str, bytes], dict[str, str]]:
    """Plan entrypoint updates without writing, preserving bytes outside markers.

    contents maps entrypoint names to existing bytes, or None for absent files.
    previous contains hashes of formerly owned sections; agents selects the new
    sections to keep. Return whole-file replacements and new section hashes.
    Raise ValueError for malformed, locally edited, or conflicting unowned blocks.
    """
    selected = {ENTRYPOINTS[agent]: agent for agent in agents}
    writes = {}
    hashes = {}
    for name, content in contents.items():
        current = content if content is not None else b""
        newline = b"\r\n" if b"\r\n" in current else b"\n"
        desired = _block(selected[name], newline) if name in selected else None
        start, end = current.find(BEGIN), current.find(END)
        if start != -1 or end != -1:
            if (
                current.count(BEGIN) != 1
                or current.count(END) != 1
                or end < start
                or (start > 0 and current[start - 1 : start] != b"\n")
            ):
                raise ValueError(f"Malformed Falcor2 section in {name}; nothing changed.")
            end += len(END)
            if current[end : end + 1] not in (b"", b"\r", b"\n"):
                raise ValueError(f"Malformed Falcor2 section in {name}; nothing changed.")
            existing = current[start:end]
            # An identical desired block can be adopted even without prior ownership.
            if _hash(existing) != previous.get(name) and existing != desired:
                recovery = (
                    "restore the marked section or move the file aside"
                    if name in previous
                    else "remove the unowned marked section"
                )
                raise ValueError(
                    f"Modified or unowned Falcor2 section in {name}; nothing changed. "
                    f"Preserve your edits, then {recovery} and retry."
                )
            replacement = current[:start] + (desired or b"") + current[end:]
        elif name in previous and content is not None:
            raise ValueError(
                f"Missing managed Falcor2 section in {name}; nothing changed. "
                "Restore the marked section or move the file aside before retrying."
            )
        elif desired is not None:
            separator = newline if current and not current.endswith(b"\n") else b""
            replacement = current + separator + desired + newline
        else:
            continue
        if desired is not None:
            hashes[name] = _hash(desired)
        if replacement != current:
            writes[name] = replacement
    return writes, hashes


def _git(directory: Path, *args: str) -> str:
    """Query rev-parse, returning an empty string when Git context is unavailable."""
    try:
        result = subprocess.run(
            ["git", "-C", str(directory), "rev-parse", *args],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _is_source(path: Path) -> bool:
    """Recognize a source checkout without requiring generated installation files."""
    return all(
        (path / name).exists()
        for name in ("falcor2/__init__.py", "src/falcor2", "pyproject.toml", "skills/PLANS.md")
    )


def resolve_context(project: Path | None, source: Path | None) -> tuple[Path, Path | None]:
    """Return the destination project and optional validated Falcor2 source root.

    An explicit project wins. Otherwise use the current Git root, redirecting a
    Falcor2 submodule to its immediate superproject, or fall back to cwd. Source
    discovery uses this supplying checkout unless the caller explicitly overrides it.
    """
    supplying = Path(__file__).resolve().parents[2]
    if source is not None:
        source = source.resolve()
        if not _is_source(source):
            raise ValueError(f"Not a Falcor2 source checkout: {source}")
    elif _is_source(supplying):
        source = supplying
    if project is not None:
        return project.resolve(), source
    root = _git(Path.cwd(), "--show-toplevel")
    destination = Path(root).resolve() if root else Path.cwd().resolve()
    if _is_source(destination):
        parent = _git(destination, "--show-superproject-working-tree")
        if parent:
            destination = Path(parent).resolve()
    return destination, source


def detect_agents(project: Path) -> list[str]:
    """Detect supported agents from configuration directories or PATH, without launching them."""
    home = Path.home()
    codex_home = Path(os.environ.get("CODEX_HOME", str(home / ".codex")))
    claude_home = Path(os.environ.get("CLAUDE_CONFIG_DIR", str(home / ".claude")))
    candidates = {
        "codex": (project / ".agents", project / ".codex", codex_home, home / ".agents"),
        "claude": (project / ".claude", claude_home),
    }
    return [
        name
        for name, paths in candidates.items()
        if any(path.is_dir() for path in paths) or shutil.which(name)
    ]


def _safe_path(project: Path, relative: str) -> Path:
    """Resolve a managed relative path beneath an absolute, resolved project root.

    Reject traversal, symlinks, and Windows reparse points (including junctions)
    before reading or writing. Manifest paths are treated as untrusted input.
    """
    parts = PurePosixPath(relative).parts
    if (
        not parts
        or "\\" in relative
        or ":" in relative
        or PurePosixPath(relative).is_absolute()
        or ".." in parts
    ):
        raise ValueError(f"Unsafe installation path: {relative}")
    target = project.joinpath(*parts)
    current = target
    while current != project:
        try:
            attributes = getattr(current.lstat(), "st_file_attributes", 0)
        except FileNotFoundError:
            attributes = 0
        if current.is_symlink() or attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            raise ValueError(f"Linked installation path: {current}")
        current = current.parent
    if not target.resolve().is_relative_to(project):
        raise ValueError(f"Installation path escapes project: {relative}")
    return target


def _managed_path(relative: str) -> bool:
    """Check the whole-file ownership allowlist; _safe_path checks containment."""
    return isinstance(relative, str) and (
        relative in (INSTRUCTIONS, ".agents/PLANS.md")
        or any(relative.startswith(prefix + "/") for prefix in DESTINATIONS.values())
    )


def _read_manifest(path: Path) -> dict:
    """Load and validate ownership/settings, or return empty ownership on first use.

    Schema 1 predates saved instruction choices. Preserve absent fields so setup
    can distinguish an unanswered choice from an explicitly disabled option.
    """
    if not path.exists():
        return {"files": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") not in (1, 2)
        or not isinstance(data.get("files"), dict)
    ):
        raise ValueError(f"Invalid skills manifest: {path}")
    for name, digest in data["files"].items():
        if not _managed_path(name) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid managed file: {name}")
    agents = data.get("agents", [])
    if (
        not isinstance(agents, list)
        or not all(isinstance(agent, str) and agent in DESTINATIONS for agent in agents)
        or len(set(agents)) != len(agents)
    ):
        raise ValueError(f"Invalid agent choices in {path}")
    if "instructions" in data and not isinstance(data["instructions"], bool):
        raise ValueError(f"Invalid instruction choice in {path}")
    if "profile" in data and data["profile"] not in ("user", "developer"):
        raise ValueError(f"Invalid profile in {path}")
    if data.get("source") is not None and not isinstance(data["source"], str):
        raise ValueError(f"Invalid source location in {path}")
    blocks = data.get("instruction_blocks", {})
    if not isinstance(blocks, dict) or any(
        name not in ENTRYPOINTS.values() or not isinstance(digest, str) or len(digest) != 64
        for name, digest in blocks.items()
    ):
        raise ValueError(f"Invalid instruction section metadata in {path}")
    return data


def read_settings(project: Path) -> dict:
    """Read prior setup choices without importing Falcor2 or writing files."""
    return _read_manifest(_safe_path(project, MANIFEST))


def _instruction_guide(agents: list[str], profile: str, source: Path | None) -> bytes:
    """Render shared guidance with installed locations and developer source context."""
    guide = (SKILLS / "instructions.md").read_text(encoding="utf-8")
    guide += "\n## Installed skill locations\n\n"
    guide += "".join(f"- {agent}: `{DESTINATIONS[agent]}`\n" for agent in agents)
    if profile == "developer":
        assert source is not None
        guide += (
            f"\n## Developing Falcor2\n\nFalcor2 source checkout: `{source.as_posix()}`.\n"
            "Read AGENTS.md in that checkout before changing Falcor2 itself. Follow\n"
            "its build/test policy; do not apply repository-specific commands to the\n"
            "consuming application. For planned changes read the installed\n"
            ".agents/PLANS.md in this project and .plans/GUIDELINES.md in that checkout.\n"
            "Developer skills provide plan review, code-review,\n"
            "and explicitly requested design-document maintenance workflows.\n"
        )
    return guide.encode("utf-8")


def _atomic_write(path: Path, data: bytes) -> None:
    """Replace one file through a sibling temporary file, cleaning up on failure.

    Each replacement is atomic; the complete multi-file installation is not.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".f2-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _desired_files(agents: list[str], profile: str, source: Path | None) -> dict[str, bytes]:
    """Build the desired whole-file contents, keyed by project-relative POSIX paths.

    Developer installations include user skills and require a source checkout.
    Generated context anchors repository policies there even when the installed
    skills live in a consuming application. No destination files are modified.
    """
    desired = {}
    if profile == "developer":
        assert source is not None
        context = (
            "<!-- Generated by f2 setup from skills/PLANS.md. -->\n\n"
            f"Falcor2 source root: `{source.as_posix()}`\n\n"
        )
        desired[".agents/PLANS.md"] = context.encode() + (SKILLS / "PLANS.md").read_bytes()
    groups = ["user", "developer"] if profile == "developer" else ["user"]
    for group in groups:
        for skill in sorted((SKILLS / group).iterdir()):
            if not (skill / "SKILL.md").is_file():
                continue
            for file in sorted(skill.rglob("*")):
                if not file.is_file() or "__pycache__" in file.parts or file.suffix == ".pyc":
                    continue
                content = file.read_bytes()
                for agent in agents:
                    desired[
                        f"{DESTINATIONS[agent]}/{skill.name}/{file.relative_to(skill).as_posix()}"
                    ] = content
            if group == "developer":
                assert source is not None
                context = (
                    "# Installed Falcor2 source context\n\n"
                    f"Falcor2 source root: `{source.as_posix()}`\n\n"
                    "Resolve repository paths in this skill against this source root, not the\n"
                    "consuming application's root. Run repository commands from this checkout.\n"
                    "Read its AGENTS.md and referenced policy files. If the checkout has moved\n"
                    "or is unavailable, rerun f2 setup --source PATH --profile developer.\n"
                    "For planning, read ../../../.agents/PLANS.md relative to this file.\n"
                    "This shared template for Codex and Claude applies to\n"
                    "the Falcor2 source root above.\n"
                ).encode()
                for agent in agents:
                    desired[f"{DESTINATIONS[agent]}/{skill.name}/source-context.md"] = context
    return desired


def install(
    project: Path | None = None,
    agent: str = "auto",
    profile: str = "auto",
    source: Path | None = None,
    dry_run: bool = False,
    instructions: bool = False,
    preserve_instructions: bool = False,
) -> int:
    """Install or preview resolved choices without prompting.

    Preflight all owned files and instruction blocks before mutation. Replace or
    remove only unchanged managed content; preserve unrelated files and raise
    ValueError on conflicts. Dry runs perform the same checks without writing.
    Return zero on success, including when automatic detection finds no agents.
    Filesystem failures propagate to main for CLI diagnostics.
    preserve_instructions leaves shared guidance and entrypoint blocks untouched,
    retaining their recorded ownership so a later explicit update detects edits.
    """
    project, source = resolve_context(project, source)
    if profile == "auto":
        profile = "developer" if project == source else "user"
    if profile == "developer" and source is None:
        raise ValueError("Developer skills require a Falcor2 checkout; pass --source PATH.")
    agents = (
        detect_agents(project)
        if agent == "auto"
        else list(DESTINATIONS) if agent == "both" else [agent]
    )
    if not agents:
        print(
            "No supported AI tools detected. Run f2 setup --agent codex, --agent claude, or --agent both."
        )
        return 0
    manifest_path = _safe_path(project, MANIFEST)
    if any(parent.exists() and not parent.is_dir() for parent in manifest_path.parents):
        raise ValueError(f"Manifest parent is not a directory: {manifest_path.parent}")
    previous = _read_manifest(manifest_path)
    desired = _desired_files(agents, profile, source)
    if preserve_instructions:
        instructions = previous.get("instructions", False)
    elif instructions:
        desired[INSTRUCTIONS] = _instruction_guide(agents, profile, source)
    previous_blocks = previous.get("instruction_blocks", {})
    block_names = (
        set()
        if preserve_instructions
        else set(previous_blocks) | ({ENTRYPOINTS[a] for a in agents} if instructions else set())
    )
    contents = {}
    for name in sorted(block_names):
        path = _safe_path(project, name)
        contents[name] = path.read_bytes() if path.exists() else None
    block_writes, block_hashes = plan_blocks(
        contents, previous_blocks, agents if instructions else []
    )
    if preserve_instructions:
        block_hashes = previous_blocks
    old = previous["files"]
    conflicts = []
    writes = {}
    deletes = []
    # Include formerly managed paths so upgrades and profile changes retire old files.
    for name in sorted(old.keys() | desired.keys()):
        if preserve_instructions and name == INSTRUCTIONS:
            continue
        path = _safe_path(project, name)
        # A file in an ancestor position also prevents a successful installation.
        if any(parent.exists() and not parent.is_dir() for parent in path.parents):
            conflicts.append(name)
            continue
        if path.exists() and not path.is_file():
            conflicts.append(name)
            continue
        current = path.read_bytes() if path.exists() else None
        expected = desired.get(name)
        if current == expected:
            # Identical content is safe to adopt; two absent values need no cleanup.
            continue
        if current is not None and (name not in old or _hash(current) != old[name]):
            conflicts.append(name)
        elif expected is not None:
            writes[name] = expected
        elif current is not None:
            deletes.append(name)
    if conflicts:
        raise ValueError(
            "Conflicting files (nothing changed):\n  "
            + "\n  ".join(conflicts)
            + "\nPreserve your edits elsewhere or restore the installed copies, then retry."
        )
    try:
        supplying_version = version("falcor2")
    except PackageNotFoundError:
        supplying_version = "source"
    metadata = {
        "schema": 2,
        "version": supplying_version,
        "agents": agents,
        "profile": profile,
        "source": str(source) if source else None,
        "instructions": instructions,
        "instruction_blocks": block_hashes,
        "files": {name: _hash(data) for name, data in sorted(desired.items())},
    }
    if preserve_instructions and INSTRUCTIONS in old:
        metadata["files"][INSTRUCTIONS] = old[INSTRUCTIONS]
    print(
        f"{'Would install' if dry_run else 'Installing'} {profile} skills for {', '.join(agents)} in {project}"
    )
    print(
        f"{len(writes) + len(block_writes)} files to write, {len(deletes)} obsolete files to remove."
    )
    if dry_run:
        return 0
    # No writes occur until both whole-file and instruction-block preflights pass.
    for name, content in (writes | block_writes).items():
        _atomic_write(_safe_path(project, name), content)
    for name in deletes:
        path = _safe_path(project, name)
        path.unlink()
        parent = path.parent
        # Remove only empty directories; unrelated skill contents prevent pruning.
        while parent != project and parent not in [
            project / prefix for prefix in DESTINATIONS.values()
        ]:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent
    # Record ownership last, after all content changes have succeeded.
    _atomic_write(manifest_path, (json.dumps(metadata, indent=2) + "\n").encode())
    print("Skills installed. Start a new AI session if the skills are not yet listed.")
    return 0


def _choose(prompt: str, choices: tuple[str, ...], default: str) -> str:
    """Prompt until a valid choice is entered; blank input selects the default."""
    while True:
        answer = input(f"{prompt} ({'/'.join(choices)}) [{default}]: ").strip().lower()
        if choices == ("yes", "no"):
            answer = {"y": "yes", "n": "no"}.get(answer, answer)
        if not answer:
            return default
        if answer in choices:
            return answer
        print(f"Choose {', '.join(choices)}.")


def _setup(args: argparse.Namespace) -> int:
    """Combine explicit flags, saved choices, and first-run answers, then install.

    Explicit flags win over saved settings. Forced interaction revisits only
    choices without flags; non-interactive mode uses saved choices or defaults.
    No configuration is persisted until installation succeeds.
    """
    # Source setup defaults apply only where the user did not provide an override.
    if args.skills_only:
        if args.interactive is None:
            args.interactive = False
        if args.agent is None:
            args.agent = "both"
    preserve_instructions = args.skills_only and args.instructions is None
    project, source = resolve_context(args.project, args.source)
    saved = read_settings(project)
    profile = args.profile or saved.get("profile", "auto")
    if args.source is None and saved.get("source") and profile == "developer":
        _, source = resolve_context(project, Path(saved["source"]))
    previous_agents = saved.get("agents", [])
    saved_agent = "both" if len(previous_agents) == 2 else next(iter(previous_agents), "auto")
    agent = args.agent or saved_agent
    instructions = args.instructions
    ask_agent = args.agent is None and (args.interactive or "agents" not in saved)
    ask_instructions = (
        not preserve_instructions
        and instructions is None
        and (args.interactive or "instructions" not in saved)
    )
    # None means normal first-run prompting; explicit True also allows piped answers.
    if args.interactive is not False and (ask_agent or ask_instructions):
        if args.interactive is None and not sys.stdin.isatty():
            raise ValueError(
                "First-run setup needs a terminal for unanswered choices. Use --non-interactive "
                "for defaults, or supply --agent and --instructions/--no-instructions. "
                "Use --interactive to answer through redirected input."
            )
        print(f"Configure Falcor2 AI support in {project}")
        if ask_agent:
            detected = detect_agents(project)
            default = (
                saved_agent if previous_agents else (detected[0] if len(detected) == 1 else "both")
            )
            print(f"Detected agents: {', '.join(detected) if detected else 'none'}")
            agent = _choose(
                "Which agents should this project support?", ("codex", "claude", "both"), default
            )
        if ask_instructions:
            print(
                "Instruction integration adds a Falcor2 section to the selected agents' "
                "AGENTS.md/CLAUDE.md files and installs .falcor2/instructions.md."
            )
            default = "yes" if saved.get("instructions", True) else "no"
            instructions = (
                _choose("Create or update those instruction references?", ("yes", "no"), default)
                == "yes"
            )
    if instructions is None:
        instructions = saved.get("instructions", False)
    return install(
        project,
        agent,
        profile,
        source,
        args.dry_run,
        instructions=instructions,
        preserve_instructions=preserve_instructions,
    )


def _hydra(argv: list[str]) -> int:
    """Delegate to an existing checkout, which owns the Hydra build/install tools."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source", type=Path)
    options, arguments = parser.parse_known_args(argv)
    try:
        project, source = resolve_context(None, options.source)
        if source is None:
            saved = read_settings(project).get("source")
            candidate = Path(saved) if saved else Path(_git(Path.cwd(), "--show-toplevel") or ".")
            if _is_source(candidate):
                source = candidate.resolve()
        if source is None or not (source / "tools/hydra.py").is_file():
            raise ValueError("Hydra tools require a Falcor2 checkout; use --source PATH.")
        return subprocess.call([sys.executable, str(source / "tools/hydra.py"), *arguments])
    except (OSError, ValueError) as error:
        print(f"f2 hydra: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Route commands, leaving Hydra arguments to the checkout tool's parser."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "hydra":
        return _hydra(arguments[1:])
    parser = argparse.ArgumentParser(prog="f2", description="Falcor2 project tools")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("hydra", help="Install and verify prebuilt Hydra plugins")
    setup = commands.add_parser(
        "setup",
        help="Configure project-local AI skills and instructions",
        description="Ask for agent and instruction choices on first use; reuse saved choices on later runs.",
    )
    setup.add_argument(
        "--project", type=Path, help="Destination project (default: Git root or cwd)"
    )
    setup.add_argument(
        "--agent",
        choices=("auto", "codex", "claude", "both"),
        help="Override the saved agent choice; auto detects installed tools",
    )
    setup.add_argument(
        "--profile", choices=("auto", "user", "developer"), help="Override the saved skill profile"
    )
    interaction = setup.add_mutually_exclusive_group()
    interaction.add_argument(
        "--interactive",
        dest="interactive",
        action="store_true",
        help="Ask again for choices not supplied as flags",
    )
    interaction.add_argument(
        "--non-interactive",
        dest="interactive",
        action="store_false",
        help="Never prompt; reuse saved choices or use detection and no instructions",
    )
    setup.set_defaults(interactive=None)
    setup.add_argument(
        "--instructions",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable shared Falcor2 guidance and managed AGENTS.md/CLAUDE.md references",
    )
    setup.add_argument("--source", type=Path, help="Falcor2 checkout for developer instructions")
    setup.add_argument("--dry-run", action="store_true", help="Check and report without writing")
    setup.add_argument(
        "--skills-only",
        action="store_true",
        help="Default to both agents without prompts; preserve instruction files unless explicitly overridden",
    )
    setup.add_argument(
        "--no-skills", action="store_true", help="Skip AI setup (used by source setup scripts)"
    )
    args = parser.parse_args(arguments)
    if args.no_skills:
        print("Skipping AI skill installation (--no-skills).")
        return 0
    try:
        return _setup(args)
    except EOFError:
        print("\nSetup cancelled; nothing changed.", file=sys.stderr)
        return 130
    except KeyboardInterrupt:
        print(
            "\nSetup interrupted. If installation had started, rerun setup to finish.",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError) as error:
        print(f"f2 setup: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
