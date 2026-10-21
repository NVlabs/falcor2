# AI skills

Falcor2 includes project-local skills for Codex and Claude Code. After installing
Falcor2 into your project's Python environment, run:

```sh
f2 setup
# Equivalent, with an explicitly selected interpreter:
python -m runcli setup
```

In a source checkout, the command is also available as
`python falcor2/tools/cli.py setup`. Canonical skills live in the repository's
`skills/` folder and are packaged under `falcor2.tools.skills`. The small installed
launcher bypasses the native imports in `falcor2/__init__.py`.

On the first run, setup asks which agents to support (Codex, Claude Code, or both)
and whether to create or update instruction references. Detected tools provide
the suggested agent choice. Successful choices are saved in `.falcor2/skills.json`;
subsequent runs refresh the same configuration without asking again.

The command does not import native Falcor2 libraries or require a GPU. It copies
the supplying version's skills into `.agents/skills` and/or `.claude/skills`.
Start a new AI session if discovery has not refreshed.

## Interactive and unattended setup

```sh
f2 setup --interactive                      # Revisit saved choices
f2 setup --agent both --instructions        # Explicit choices, no questions
f2 setup --agent codex --no-instructions    # Skills only
f2 setup --non-interactive                 # Saved settings, otherwise defaults
f2 setup --non-interactive --dry-run       # Preview without questions or writes
```

Choice flags override saved values and suppress their corresponding questions.
`--interactive` asks again for choices not supplied as flags. `--non-interactive`
never prompts: it reuses saved values, otherwise detects agents and leaves
instruction integration off. `--agent auto` explicitly reruns tool detection from
project/user configuration and PATH; if none are found, setup prints instructions
without writing anything.

Without a terminal, unanswered first-run choices produce an actionable error.
Use `--non-interactive` or supply both agent and instruction choices. Explicit
`--interactive` also permits answering through redirected input. Cancelling during
the questions or using `--dry-run` does not save choices. Existing manifests from
before instruction integration retain their agent choice and ask the new
instruction question once when run interactively.

`--project PATH` selects the exact destination. Otherwise setup uses the current
Git root or current directory; a Falcor2 submodule redirects to its immediate
superproject. A direct supplying checkout receives user and developer skills;
other projects receive user skills on first setup. The selected profile and source
are also saved; `--profile auto` recomputes the profile from the current context.
`--profile developer --source PATH` enables
developer skills against a specific Falcor2 checkout. Developer skills always
apply repository policies and commands to that checkout, not the application.

Source `setup.bat` and `setup.sh` run this installer after successful environment
setup with `--skills-only`: non-interactive installation for both Codex and Claude,
leaving AGENTS.md, CLAUDE.md, and `.falcor2/instructions.md` untouched even on reruns.
Explicit `--agent` or interaction flags override these defaults; `--instructions`
or `--no-instructions` explicitly opts into updating or removing instruction
integration. Standalone `f2 setup` retains its first-run questions.
The scripts accept the same options plus `--no-skills`. Run them from the source
checkout with the intended Python environment activated; use `--project` for
another destination. They use `python` and `pip` from that environment on PATH.
`--dry-run` previews skill installation only; it does
not turn dependency installation/builds in the source setup scripts into a dry run.

The only user skill, `falcor2-health-check`, verifies package installation, native
imports, and GPU device creation, with driver diagnostics. It makes no environment
changes. Refresh removes unchanged managed copies of the former getting-started,
render-scene, and create-app skills; locally edited copies produce conflicts. Developer
skills cover plan review, code review, and comparative research. Use `comparative-research`
to investigate a question across named systems, implementations, or papers with
delegated researchers, saved evidence, and a synthesis that accounts for missing
coverage. The research workflow is domain-independent and uses available tools;
it does not require an external research service. Reviewer definitions ship
alongside the code-review skill. Developer setup also copies
`skills/PLANS.md` to `.agents/PLANS.md` for both agents, including Claude-only
installations. This shared template covers preparation and execution and records
the Falcor2 source location. Read `.plans/GUIDELINES.md` when the checkout has it.
The template has the same refresh and local-edit protection as skills. Refresh
removes unchanged managed copies of the retired `prepare-executable-plan` and
`execute-reviewed-plan` skills; edited copies produce conflicts. Unmanaged copies
are left alone.

## Refresh and conflicts

When enabled, instruction integration installs `.falcor2/instructions.md` with
general Falcor2 guidance and links to relevant skills. It adds a delimited Falcor2
section to `AGENTS.md` (a read instruction) and/or `CLAUDE.md` (an `@` import),
creating entrypoint files when needed. It preserves all existing bytes outside
the managed section, including project-specific instructions and line endings.
Keep custom instructions outside the `BEGIN/END FALCOR2 INSTRUCTIONS` markers.

`--no-instructions` removes the shared guide and owned references; switching agents
removes the deselected agent's owned reference. Entrypoint files are retained,
including their project-owned content. Modified, missing, or malformed managed
sections produce conflicts before any writes. Preserve edits elsewhere and restore
the managed section before retrying. An unowned section can be removed after saving
its contents; a missing owned section can be recovered by moving the entrypoint
aside, rerunning setup, then restoring project content around the new section.

Rerun setup after upgrading Falcor2 or editing canonical skill sources under
`skills`. Use `--dry-run` to check changes without writing. The
`.falcor2/skills.json` manifest records installed hashes, version, profile, tools,
and source path, plus instruction integration and section hashes. Selecting a different profile/tool set removes obsolete managed
files, but never unrelated files. Keep this manifest with the installed copies.

Locally modified or conflicting files cause a nonzero exit before any writes.
Preserve your changes elsewhere, then restore the installed copy or remove the
conflicting file and rerun. An interrupted install can be retried: copies already
matching the requested version are accepted. Do not hand-edit manifest hashes.
Linked destination directories are rejected; choose an ordinary project path.

Generated skills and the shared guide are ignored in the Falcor2 repository. Consuming projects decide
whether to commit copies and the manifest or add local ignore rules. Setup only
edits the owned sections of AGENTS.md/CLAUDE.md when integration is selected and
never edits the consuming project's ignore files. If a project
moves, rerun with the new source location before using developer skills.

## Installer development

Run `python -m pytest tests/cli -n 4 -p no:cacheprovider` for the installer and
setup-wrapper tests. These tests live outside the native Python test suite so
they can run without importing Falcor2 or requiring a GPU. Wrapper tests use
stub dependency installers; they do not install packages or invoke CMake.
