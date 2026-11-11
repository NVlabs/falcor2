(falcor2-design-documents)=
# Writing and maintaining design documents

Falcor2 design documents are concise explanations of the architecture that exists now. They give humans and coding
agents enough context to find ownership boundaries, follow runtime flow, and understand enduring constraints without
reading implementation diaries. This guide is the policy authority for pages under `docs/design/` and for the
manual process that distills useful knowledge from `.plans/`.

Apply [Writing technical documentation for humans and agents](documentation-writing.md) for general writing
quality. This design-document policy defines the architecture-specific page contract, evidence requirements,
length guidance, and review lifecycle.

## Purpose and boundaries

A design page answers one architectural question. It describes responsibilities, relationships, control or data
flow, invariants, and rationale that still explains the current implementation. It should help a reader decide which
source area to inspect next and which assumptions must remain true while changing the system.

Design pages are explanation, not tutorials, API reference, roadmaps, or project history. Omit milestone chronology,
commit and merge-request history, transient test counts, exhaustive method lists, implementation diaries, and old
alternatives that no longer shape the code. Future work belongs in an active or backlog plan. The
[Diataxis framework](https://www.diataxis.fr/) provides the broader distinction between explanation, tutorials,
how-to guides, and reference material.

## Page contract

Every substantive page uses CommonMark-first MyST Markdown and contains these elements in order:

1. YAML front matter containing `status`, `audience`, `last_verified`, and `verified_against`. Use an ISO date
   for `last_verified` and the full Git commit hash for `verified_against`.
2. A unique stable MyST target followed by a descriptive level-one heading. Name the target
   `falcor2-design-<topic>` so links do not depend on title wording.
3. A 60-120 word opening summary that names the system and its role. The summary must make sense when retrieved
   without the title or surrounding page.
4. `Purpose and scope`, defining the question answered and the details intentionally left to other pages.
5. `Architecture`, describing ownership, responsibilities, and relationships.
6. `Runtime flow`, or a more precise equivalent for the page's control or data flow.
7. `Invariants and rationale`, recording constraints and enduring reasons without chronology.
8. `Source landmarks`, naming relevant implementation, binding, test, and example paths without line numbers.
9. `Related design documents`, linking only pages that exist. A planned filename may appear as inline code when
   the relationship is important, but it must not create a broken link.

A minimal opening has this shape:

```markdown
---
status: draft
audience: Falcor2 contributors and coding agents
last_verified: 2026-09-08
verified_against: "0123456789abcdef0123456789abcdef01234567"
---

(falcor2-design-example)=
# Descriptive title
```

Every new page begins as `draft`. Do not mark a page `current` in the same review cycle that first introduces or
materially rewrites it.

## Evidence and verification

Use evidence in this order:

1. Current implementation and observable runtime behavior.
2. Current bindings, tests, and examples.
3. Completed plan outcomes and accepted decisions, as evidence for enduring rationale.
4. Active and backlog plans, as future intent only.

Open defining files, trace important callers and ownership, and find tests or examples that exercise the behavior.
Search results and filenames are leads, not proof. A completed plan can explain why code took its present shape, but
verify every current-state claim against implementation or behavioral evidence. Never present an active, refining,
or backlog plan as behavior that already exists.

If evidence conflicts, describe what the current implementation does and report the disagreement during review. If
the current behavior cannot be established confidently, keep the page `draft` and state the uncertainty in the
review handoff rather than guessing. Source landmarks must exist at the verification commit, but should omit line
numbers because normal edits make them brittle.

## Scope, length, and writing style

Use the general writing guide's sections on [reader outcomes](documentation-writing.md#define-the-reader-and-outcome),
[explanation](documentation-writing.md#explanation), [organization](documentation-writing.md#organize-a-navigable-page),
[precise prose](documentation-writing.md#write-precise-prose),
[retrieval context](documentation-writing.md#support-search-and-agent-retrieval),
[authority](documentation-writing.md#establish-authority-and-freshness), and
[reader-task review](documentation-writing.md#review-documentation-through-reader-tasks).
Consult its guidance on procedures, examples, API contracts, or visual presentation when the page includes that
material. The requirements below specialize that guidance for architectural explanations.

Prefer one coherent architectural question over a source-directory tour. Target 800-1,600 words for a normal page.
Review pages below 400 words for missing substance or merger, and pages above 2,000 words for splitting. Normally
split a page above 2,500 words. Keep most sections below roughly 500 words so a retrieved section remains useful on
its own. These ranges are Falcor2 review defaults, not universal empirical limits and not automatic validation
errors.

Split by responsibility or architectural question, never by arbitrary line count. Use explicit subjects such as
`The Scene update pass` instead of contextless phrases such as `this component`. Define project-specific terms
on first use. Prefer short paragraphs and small lists. Add a compact ASCII diagram only when it materially clarifies
flow or ownership.

Oversized retrieved chunks can carry irrelevant material, while undersized chunks can lose the context needed to
interpret a claim. Microsoft's [RAG chunking guidance](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase)
recommends respecting document structure, Anthropic's
[contextual retrieval guidance](https://www.anthropic.com/engineering/contextual-retrieval) explains why retrieved
material benefits from local context, and the
[Lost in the Middle paper](https://arxiv.org/abs/2307.03172) demonstrates limits of relying on undifferentiated long
context. Falcor2 therefore treats coherent sections as the primary boundary and word counts as review signals.

## Markdown and Sphinx compatibility

Use portable Markdown wherever it is sufficient: hash headings, ordinary lists, fenced code blocks, tables, and
relative Markdown links. Use MyST syntax only where it preserves a real documentation capability:

- Put `(falcor2-design-<topic>)=` immediately before each design page's level-one heading to create a stable target.
- Use a hidden `{toctree}` directive only in documentation indexes. Keep ordinary Markdown links beside it so the
  indexes remain navigable in repository browsers that do not render MyST directives.
- Prefer standard links such as `[Scene model](../design/scene-model.md)` over Sphinx roles.

Sphinx supports `.md` sources through MyST-Parser, which extends CommonMark and registers Markdown documents with
Sphinx. The documentation build enables the `myst_parser` extension, as described by the
[Sphinx Markdown documentation](https://www.sphinx-doc.org/en/master/usage/markdown.html). MyST's
[cross-reference documentation](https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html)
documents explicit targets and recommends portable Markdown link syntax where possible.

Use ASCII source text and keep local links relative and targets unique. Follow [Building documentation](building-documentation.md)
to preview the site. Changes to the configuration, theme, dependencies, build jobs, API generation, or `llms.txt`
remain separate tooling tasks rather than automatic requirements of design page authoring.

## Review lifecycle

The design index tracks three states. `planned` means the catalog names an architectural question but no page
exists. `draft` means a page exists and has been verified by its author but awaits human acceptance. `current`
means a human approved it as the durable account at the recorded commit.

One review cycle changes at most one substantive design page, plus `docs/design/index.md` and necessary related
links. The author reports the verification commit, source and behavioral evidence sampled, uncertainties, word
count, and any plan artifacts that might later be retired. Human approval changes the page from `draft` to
`current` only after relevant changes since its verification commit have been checked. Approval does not delete
plans.

An audit that materially repairs a current page returns it to `draft` for review. Ordinary implementation work has
no automatic design-document update requirement. Auditing, distilling, and retiring design knowledge are explicit
maintenance tasks.

## Plan distillation and retirement

The `.plans/` tree holds temporary execution plans, backlog, investigations, and historical evidence. Distillation
turns useful knowledge from completed work into one concise design page. It uses plans as leads, verifies current
claims against code and behavior, creates or updates one `draft` page, and reports retained rationale, discarded
history, unresolved work, and possible retirement candidates. Distillation deletes nothing.

Retirement is a separate destructive pass. It runs only after a human explicitly approves the replacement page and
names the exact plan scope to consider. A plan artifact is deletable only when all of the following are true:

- Its implementation is complete.
- Useful current knowledge is captured in the approved page or intentionally rejected as obsolete.
- No active, refining, backlog, or otherwise unresolved work depends on it.
- Repository-wide search proves that live backlinks can be repaired.
- The final review diff pairs the accepted replacement page with the exact removed files.

Never delete active, refining, or backlog material. Preserve references needed by unresolved work. A mixed workstream
keeps its unresolved files and a short, accurate navigational `README.md`. A completed README-only workstream can
be removed entirely only after the same checks pass. Git history provides traceability; do not add tombstones or a
migration ledger.

## Maintenance skill

The repository-local `$falcor2-maintain-design-docs` skill applies this guide in Audit, Distill, or Retire mode. It is
explicit-only and must never become a scheduled job or automatic plan-closeout hook. The skill reads this policy
and the relevant sections of the general writing guide, then checks architectural accuracy and reader usefulness
separately. OpenAI's
[skill documentation](https://learn.chatgpt.com/docs/build-skills) describes the repository skill layout and
explicit invocation metadata.

## Validation checklist

Before handing off a page for review, verify its unique target, YAML metadata, opening summary, required sections,
word count, ASCII source, local links, and every source-landmark path. Confirm that each hidden toctree contains only
files that exist and that planned filenames remain unlinked. Run pre-commit on all changed files. MyST and Sphinx
validation follows [Building documentation](building-documentation.md#build-and-preview); inspect the rendered page
and report build warnings or unavailable checks. A successful HTML build does not verify examples or architectural
claims.

Apply the writing guide's [reader-task review](documentation-writing.md#review-documentation-through-reader-tasks):
check whether readers can identify the page's question, explain ownership and runtime flow, interpret a substantive
section independently, and find the relevant source. Report usability findings separately from factual errors,
with the affected passage, reader consequence, applicable principle, and a focused correction. Keep optional
editorial suggestions distinct from required fixes. A harmless style preference does not make a claim stale.

Before retirement, repeat page verification at the current commit, inspect dirty files, search the whole repository
for inbound references, and review the literal deletion set. Stop without deleting anything if approval, evidence,
scope, or link safety is incomplete.
