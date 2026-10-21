---
name: falcor2-code-review
description: Multi-agent Falcor2 code review using predefined specialist sub-agents. Use when the user explicitly asks for sub-agents, parallel review, specialist reviewers, or a deeper delegated code review of local diffs, branches, PRs, MRs, or specific Falcor2 files.
---

# Falcor2 Code Review

## Source checkout

Read [source-context.md](source-context.md) first when installed. It identifies the
Falcor2 checkout for all repository paths and commands below. When reading the
canonical packaged source directly, use its enclosing Falcor2 checkout instead.
Supporting references in this skill are relative to this SKILL.md.


Use this skill to run a coordinated multi-agent review. The parent agent is the review captain: it defines the review scope, dispatches specialist reviewers, deduplicates findings, and presents a concise final review.

## Predefined Reviewers

Use these bundled reviewer definitions, relative to this skill:

- `references/falcor2-correctness-reviewer.md`
- `references/falcor2-rendering-reviewer.md`
- `references/falcor2-api-bindings-reviewer.md`
- `references/falcor2-test-reviewer.md`
- `references/falcor2-performance-reviewer.md`
- `references/falcor2-maintainability-reviewer.md`
- `references/falcor2-build-ci-reviewer.md`

Use available Codex agent or Claude Code subagent tools. Prefer matching specialist types when available; otherwise supply the bundled reviewer prompt to a generic read-only agent. Run all seven roles in bounded batches when concurrency is limited. Collect results before releasing agents using available lifecycle operations. If delegation is unavailable, report the limitation instead of claiming a multi-agent review.

## Workflow

1. Establish review scope by querying user what to compare against, defaulting to origin/main:
   - Most commonly, comparing to origin/main
   - local working tree diff,
   - branch comparison to origin/main
   - PR/MR diff,
   - explicit files,
   - or an explicit patch supplied by the user.

2. Inspect enough local context to create a useful reviewer packet:
   - use `docs/contributing/coding-style.md` in the Falcor2 checkout identified above; read SCOPE.1 and route
     only the relevant rule sections to each reviewer,
   - changed file list,
   - relevant diff,
   - nearby code patterns,
   - tests touched or missing,
   - build files affected.

3. Run all predefined reviewers in parallel, giving them the same scope and context. Each reviewer should return findings in the specified format.

4. Collate results:
   - Deduplicate overlapping findings.
   - Keep only concrete, defensible issues.
   - Drop vague style preferences unless they indicate real risk.
   - Require applicable rule IDs for convention findings. Preserve the responsible reviewers' audit results,
     supporting evidence, permitted exceptions, and unverified checks; do not expand into unrelated cleanup.
   - Resolve conflicting reviewer opinions by checking the code locally.
   - Sort by severity.

5. Report in code-review style:
   - Findings first.
   - Then open questions.
   - Then test gaps and verification notes.
   - Summary last and brief.

## Reviewer Prompt Template

```text
Use the predefined reviewer in <agent-definition-path>.

Review target:
<scope and diff/files>

Context:
<project-specific notes, relevant conventions, nearby files, or test/build notes>

Rules:
- Read only. Do not edit files.
- Return only actionable findings.
- Include file and line/function where possible.
- Explain impact and suggested fix.
- Say "No findings" if no actionable issue is found.
- Include the reviewer definition's required audit results even when there are no findings.
```

## Final Output

Use this shape:

```markdown
**Findings**
- [severity] file:line - Issue, impact, and suggested fix.

**Open Questions**
- Question or assumption, if any.

**Test Gaps**
- Missing or recommended tests.

**Verification**
- Commands run, commands not run, and why.
- API and maintainability audit results, with scope, evidence, exceptions, and unverified checks.
```

If there are no findings, say that clearly and still mention any residual test or verification gaps.
