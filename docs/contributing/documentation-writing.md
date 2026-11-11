(falcor2-documentation-writing)=
# Writing technical documentation for humans and agents

Falcor2 documentation should help a reader find an answer, understand the conditions under which it is true,
and use it correctly. This guide is for contributors writing tutorials, task guides, reference material, and
technical explanations. It covers planning a page, writing its content, and assessing whether another person
or coding agent can use it without the author's help.

Start with the reader's task. Include the detail needed to complete that task or understand the system, and
organize deeper material so readers can find it when needed. A short page can omit essential context; a long
page can be useful when its sections answer distinct, well-connected questions.

## Find the guidance you need

- [Define the reader and outcome](#define-the-reader-and-outcome) before choosing content.
- [Choose the document type](#choose-the-document-type) for a tutorial, procedure, reference, or explanation.
- [Organize a navigable page](#organize-a-navigable-page) and [write precise prose](#write-precise-prose).
- [Write reproducible procedures](#write-reproducible-procedures), [useful examples](#write-examples-that-teach-a-specific-point),
  [API contracts](#document-api-contracts-and-boundaries), and [troubleshooting guidance](#help-readers-diagnose-failures).
- [Support search and agent retrieval](#support-search-and-agent-retrieval) and
  [choose accessible presentation](#choose-accessible-presentation).
- [Establish authority and freshness](#establish-authority-and-freshness), then
  [review the result](#review-documentation-through-reader-tasks).

The [coding-style guide](coding-style.md) governs source comments, naming, and binding
documentation. [Writing and maintaining design documents](design-documents.md) governs
architecture pages and their review lifecycle. [Building documentation](building-documentation.md)
describes the documentation toolchain. Use their specialized requirements with this general writing guidance.

## Define the reader and outcome

Before drafting, complete this sentence: "A reader who knows ___ can use this page to ___." Name the knowledge
that matters: Python familiarity, rendering terminology, a working checkout, or experience with a particular
backend. A job title alone rarely explains what the reader already knows.

List the questions the page must answer. For a task, identify the starting state and observable end state. For
an explanation, identify the decision or reasoning the reader should be able to make afterward. Exclude material
that serves a different goal and provide a link when it is a likely next step.

Open with enough information for a reader to decide whether to continue. State the subject, purpose, and
significant prerequisites. Put the recommended approach or central conclusion near the start. Avoid a long
history before explaining what the component does. These principles follow
[Google's guidance on documents](https://developers.google.com/tech-writing/one/documents).

For example, replace an opening such as "This page covers documentation functionality" with:

> Build Falcor2's HTML documentation locally to preview changes to guides and API pages. Use an existing
> development environment with the documentation dependencies installed and current Release bindings.

This opening identifies a concrete task and prevents readers from assuming that the documentation command
also configures the native build. Put less common prerequisites in a short section immediately afterward.

## Choose the document type

[Diataxis](https://diataxis.fr/) distinguishes four needs. Choose the main need for each page; link to other
forms when the reader needs a different kind of help.

| Reader's need | Form | Main obligation |
| --- | --- | --- |
| Learn an unfamiliar activity | Tutorial | Lead a beginner through a successful learning experience. |
| Complete a known task | How-to guide | Give a usable route from a stated starting state to a result. |
| Look up exact behavior | Reference | Describe the relevant interface consistently and precisely. |
| Understand relationships and reasons | Explanation | Build a mental model that supports reasoning. |

### Tutorials

Choose one useful result and a dependable path to it. Supply or identify the inputs, state the environment,
and introduce concepts when the learner encounters them. Each meaningful stage should produce something the
learner can recognize. Explain what to notice in the result so the learner connects an action to its effect.
Give the learner an early visible result when feasible; postpone optional setup that does not contribute to
the first learning goal.

Keep early choices limited. A beginner should not have to choose among several backends before learning the
operation the tutorial teaches. Link optional variations and deeper explanations after the main path. End by
identifying what the learner accomplished and a sensible next exercise.

An outline is: outcome, prerequisites, prepared inputs, guided steps with checkpoints, result, next exercise.
Review the tutorial from its stated starting state rather than from the author's already-configured machine.
See [Diataxis tutorials](https://diataxis.fr/tutorials/).

### How-to guides

Use a task title such as "Preview documentation changes locally." State when the procedure applies and what
the reader needs before starting. Assume the relevant background knowledge, but do not assume hidden files,
environment variables, or configuration.

Include decisions that affect task completion. For alternatives, give the condition for choosing each route
before its commands. For example, distinguish an incremental rebuild from a rebuild that discards cached
documentation state. Keep optional work visibly optional.

An outline is: goal, applicability, prerequisites, ordered actions, verification, likely failures, related
tasks. Complex tasks may need branches or judgment; they need not be forced into one universal recipe.
See [Diataxis how-to guides](https://diataxis.fr/how-to-guides/).

### Reference

Organize reference around the interface readers look up: a type, operation, configuration field, file format,
or command. Use a consistent structure across comparable entries so readers can find the same information
in the same place. State defaults, allowed values, constraints, and error behavior where those facts matter.

A concise usage example can clarify a reference entry. Keep the entry focused on exact behavior and link to
a tutorial for a guided learning experience. An outline is: purpose, signature or syntax, inputs, outputs,
constraints, errors, example, related entries. Omit fields that add no useful information.
See [Diataxis reference](https://diataxis.fr/reference/).

### Explanation

Answer a specific conceptual question, such as how ownership crosses the Python/native boundary. Identify
the relevant parts, their responsibilities, and their relationships before tracing a detailed flow.
Explain the invariants and reasons that help a contributor predict the effects of a change.

Discuss alternatives when they explain a lasting tradeoff. Keep chronology and abandoned experiments in
research or planning records unless they are essential to the current rationale. A useful explanation lets
a reader reason about a case the page does not explicitly walk through.

For Falcor2 architecture pages, use the existing design-document page contract rather than inventing a second
template. See [Diataxis explanation](https://diataxis.fr/explanation/).

### One subject, four different pages

The documentation build could support all four forms without repeating the same page four times:

- A tutorial, "Preview your first documentation edit," would guide a newcomer through changing a sentence,
  building the site, and finding the changed text.
- A how-to, "Rebuild documentation after changing a native API," would guide an experienced contributor through
  the relevant generation and build dependencies.
- A reference, "Documentation build inputs and outputs," would describe the configuration, generated artifacts,
  and relevant options in a consistent lookup structure.
- An explanation, "How native API help reaches the documentation site," would explain the relationships between
  declarations, bindings, extraction, and page generation.

These are possible page topics, not a requirement to create all four. Add a page when a real reader need warrants it.

## Organize a navigable page

Arrange information in the order the reader needs it: orientation before detail, prerequisites before action,
definitions before unfamiliar use, and conditions before affected instructions. Keep related claims together.
An exception several pages away is easy to miss even if both statements are individually correct.

Use a heading hierarchy that exposes the argument or procedure. Prefer "Verify the generated HTML" to
"Additional information." Use task verbs for actions and subject names for concepts. Read the headings alone:
they should show what the page offers and how its parts fit together. Follow
[Google's heading guidance](https://developers.google.com/style/headings) without renaming existing public
targets merely to make a stylistic improvement.

Split a page when its sections serve different readers, require independent maintenance, or answer separate
questions that readers commonly look up. Keep a sequence together when splitting it would hide prerequisites
or force readers to jump between pages to complete one task. Length is a review signal, not proof of quality.

Give links informative labels and explain why the destination is useful. "See the binding documentation rules
for native help extraction" is more useful than "see here." Supply a short local definition or essential
condition when following a link would interrupt understanding; link to the full authoritative treatment.
This balances discoverability with the reading cost described in
[Google's cross-reference guidance](https://developers.google.com/style/cross-references).

## Write precise prose

Use concrete subjects and verbs: name the object that owns a resource, the function that raises an error, or
the caller that must synchronize. Prefer direct instructions such as "Run the command from the repository
root." Passive voice is useful when the actor is irrelevant, but should not obscure responsibility.

Keep each paragraph centered on one idea. Start with the point, develop its explanation, and include an example
when it resolves likely confusion. Split a sentence when it combines an action, several conditions, and a
different consequence. Preserve logical relationships when splitting: readers still need to know which
condition controls which action. See Google's guidance on
[clear sentences](https://developers.google.com/tech-writing/one/clear-sentences),
[short sentences](https://developers.google.com/tech-writing/one/short-sentences), and
[paragraphs](https://developers.google.com/tech-writing/one/paragraphs).

Use one term for one concept. Introduce unfamiliar abbreviations and retain exact API names in code formatting.
Avoid switching between "buffer," "array," and "tensor" unless the distinction is intentional and explained.
Repeat a noun when "it" or "this" could refer to multiple things. See
[Google's guidance on words](https://developers.google.com/tech-writing/one/words).

Preserve useful mathematical notation. Define symbols, spaces, units, and conventions at the point where they
matter; do not rename standard domain notation just to make every identifier read like English. For example,
a direction's coordinate space and normalization requirement can be more important than expanding its name.
Follow Falcor2's existing naming conventions for code identifiers, including uppercase type-name abbreviations.

Distinguish requirements from choices. Use "must" for a genuine constraint, "should" for a recommendation with
possible exceptions, and "can" for capability. Identify the condition under which a recommendation changes.
Avoid "simply," "obviously," or "just" when they conceal setup or make a reader's difficulty sound unreasonable.

**Illustrative rewrite, using a hypothetical interface:**

> Weak: After passing the data to the renderer, it should be kept around until it finishes.
>
> Better: The renderer borrows the input buffer. The caller must keep that buffer alive until the submitted
> operation completes.

The second version identifies the owner, resource, and lifetime boundary. A real API page must establish
those facts from its implementation and contract before making the claim.

## Write reproducible procedures

Describe the procedure's starting state explicitly. Include the working directory, shell, environment,
required data, and relevant platform or backend. Link substantial setup instructions. State any small
additional prerequisite locally so the command remains understandable when copied or found by search.

Number dependent actions. Give each step a clear purpose and keep the command near that purpose. State a
branch condition before its commands. After a consequential step, identify the file, visible result, or
exit status that shows success. Put expected output in a separate block so it cannot be mistaken for input.
See [Google's procedure guidance](https://developers.google.com/style/procedures).

Command examples should name their shell. A PowerShell continuation and a Bash continuation are different
syntax; a generic "Windows" label is insufficient. Prefer single-line commands when practical, omit prompt
characters from copyable blocks, and distinguish executable examples from syntax summaries. Explain each
placeholder and quote substituted paths appropriately for the stated shell. Do not put notation such as
`[optional-argument]` in a block presented as a command to execute. See Google's guidance on
[command-line syntax](https://developers.google.com/style/code-syntax) and
[placeholders](https://developers.google.com/style/placeholders).

For actions that change persistent state, explain the relevant effect and scope before the command. Include
cleanup or rollback when it is part of the reader's task. Avoid broad reset or deletion commands as an
unexplained first troubleshooting step.

### Example of a complete command instruction

The following example uses the documented Falcor2 HTML build command; its purpose here is to show the context
that should accompany a command.

From the repository root in PowerShell, use the existing repository-local environment after completing the
[documentation prerequisites](building-documentation.md#prerequisites):

```powershell
.\.conda\python.exe -m sphinx -M html docs docs/_build
```

The command writes the HTML site to `docs/_build/html`. A successful build exits with status zero; open
`docs/_build/html/index.html` to check the generated navigation and edited page. Review warnings as well:
the normal command does not fail solely because it reports a warning. The documentation build does not
execute the examples shown in pages.

This is a materially stronger instruction than "Run Sphinx and check it worked": it states the environment,
input and output locations, result check, and limits of that check. The current build guide remains the
authority for commands and setup.

## Write examples that teach a specific point

Choose examples by the behavior they explain. Start with the smallest complete case that demonstrates the
intended use. Add a second case for a meaningful boundary or common failure when it teaches something the
successful case cannot. Avoid an exhaustive matrix that obscures the central operation.

Say whether a block is a complete program, a fragment that assumes earlier setup, pseudocode, or expected
output. Complete examples need their imports, inputs, and required initialization. A fragment should name
the objects or state it assumes. Mark omissions with comments rather than plausible-looking code that cannot
run. Use project formatting conventions. See
[Google's code-sample guidance](https://developers.google.com/style/code-samples).

Explain the result as well as the invocation. For image-producing examples, state the property the reader
should observe. For nondeterministic output, describe a stable property or supported tolerance rather than
printing an arbitrary exact value as the acceptance criterion. Use portable sample paths and make required
assets discoverable.

Keep a durable runnable example in its normal source location when possible, and link to it from the narrative.
If a page repeats a small excerpt, keep the excerpt focused and check it when its source changes. Document
special configuration beside the example so an alternate backend cannot silently appear to be the default.

Do not invent Falcor2 APIs to make an example look complete. If a conceptual example is clearer with invented
names, label the entire example as hypothetical and keep it separate from copyable project instructions.

## Document API contracts and boundaries

Reference should answer what callers need to know to use an operation correctly. Inspect the interface and
its important callers, then select the applicable questions below. These are prompts for useful content,
not a requirement to attach a comment or every field to every declaration.

| Contract area | Questions to resolve |
| --- | --- |
| Inputs | What values, shapes, units, coordinate spaces, and combinations are valid? |
| Defaults | What happens when an argument is omitted? Does the exposed language use the same default? |
| Results | What is returned or mutated? Is the result a copy, view, handle, or borrowed object? |
| Lifetime | Who owns the resource? What keeps it alive? Which operation invalidates it? |
| Execution | Is work immediate or deferred? What synchronization or calling order is required? |
| Failures | What fails, how is failure reported, and what state remains after failure? |
| Availability | Which backend, feature, build option, or version limits the behavior? |

For rendering and numerical interfaces, define the convention as well as the unit: local or world space,
normalized or unnormalized direction, probability mass or density, and the measure associated with a density.
Only include distinctions applicable to the documented operation. A name or type often cannot express these
contracts fully.

**Hypothetical contract example:** "Returns pixels" leaves important choices unresolved. A useful contract
might instead say: "Returns a borrowed view of linear RGBA values in an H-by-W-by-4 array. The view remains
valid until the next resize. Copy the array if its contents must survive a resize." This is an example of
contract precision, not a claim about a Falcor2 function.

Keep shared native contracts in their authoritative source and language-specific adapter behavior at the
adapter. Follow the [binding documentation rules](coding-style.md#binding-documentation)
for extraction, reflection, fallback help, and validation. In particular, empty generated help or Python
exposure alone does not require a new native comment. Temporary `D_NA(...)` applies when generation tools are
unavailable after checking their installation and configuration; input errors, parse failures, and stale
output need correction. A category without extraction support follows the direct-text rules. The coding-style
guide defines the exact distinctions and checks.

## Help readers diagnose failures

Organize troubleshooting by observable symptoms, including the exact diagnostic text when it is stable and
useful for search. State the affected context, then guide the reader through checks that distinguish likely
causes. Explain what each result means. A list of possible causes without a decision path leaves the reader
to perform the diagnosis alone.

Use this structure for a substantial entry:

1. Symptom and affected environment.
2. A diagnostic check and the evidence to inspect.
3. Likely cause for each relevant result.
4. A targeted remedy and its consequences.
5. A verification step and information to collect if the failure continues.

Prefer observations before disruptive repairs. "Reinstall everything" does not explain what failed or why
the action should help. Preserve useful diagnostics before a reset, and keep a workaround's limitations
beside the workaround. The [Good Docs troubleshooting guide](https://www.thegooddocsproject.dev/template/troubleshooting)
provides a useful model for symptom-based guidance.

For example, if generated documentation appears stale, first establish that the build used the intended
checkout and that the reader opened the output just generated. Then consider cached documentation state
and the freshness of generated API artifacts. A clean documentation rebuild cannot establish that native
bindings reflect a source change. Link the appropriate build procedure once the diagnostic evidence identifies
that problem.

## Support search and agent retrieval

An agent may find a page through a filename search, an exact identifier match, a link, or a retrieved excerpt.
Human readers also arrive through search and deep links. Preserve enough context at those entry points to
identify the subject and applicability.

Use exact configuration keys, type names, command names, and diagnostics alongside readable explanations.
Introduce a synonym when readers are likely to search for it, then use consistent terminology. Write a short
description beside an index link so the reader can choose the destination without opening every page.

Keep a section's subject, relevant backend or version, and significant exception within that section. Prefer
"The HTML documentation build does not execute examples" to "This does not run them." Keep a table's units
and column meanings attached to the table, and a code block's prerequisites in its immediate introduction.
If a necessary detail lives elsewhere, name that dependency and link to the exact section.

This recommendation draws on [Anthropic's contextual retrieval work](https://www.anthropic.com/engineering/contextual-retrieval)
and [Microsoft's chunking guidance](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/rag/rag-chunking-phase).
Both concern retrieval systems. Applying their lessons to authored sections is a project recommendation;
their results do not establish a universal page length or guarantee that a particular agent will load a link.

Use a concise project entry point to route readers to detailed guidance. Repeat a short critical condition
when it is needed locally, while linking the complete canonical contract. Avoid making every page carry the
entire repository policy. [Anthropic's context engineering guidance](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
supports discovering relevant detail as needed while retaining sufficient context for the task.

Maintain one underlying account of the system across human and agent views. A generated discovery index or
Markdown export can expose that content through another route. Evaluate such tooling against actual consumers;
its existence does not establish that an agent finds the right page or follows the instructions correctly.

## Choose accessible presentation

Use prose for an argument, numbered lists for dependent steps, bullets for parallel items, and tables for
comparisons with consistent fields. Give tables descriptive column headers and keep cells short enough to
scan. Move long prose out when it prevents useful comparison, and split complex tables when needed. A
procedure that should be read sequentially usually works better as steps.
See [Google's table guidance](https://developers.google.com/style/tables).

Use a diagram when it clarifies ownership, state transitions, or flow more efficiently than prose. Name the
nodes, explain what arrows mean, and state the important relationship in adjacent text. A screenshot needs
context about what to inspect; text and command examples should remain selectable. Explain an image's purpose
in alternative text, and do not rely on color or screen position alone to communicate a distinction. See
[Google's accessibility guidance](https://developers.google.com/style/accessibility).

Use portable Markdown and the project's supported MyST features. Keep local links relative and stable targets
unique. Provide ordinary navigable links alongside index directives that repository browsers do not render.
Check both source readability and rendered presentation, especially heading structure, wide tables, code
overflow, and figures. Follow the design-page formatting rules when writing in `docs/design/`.
For new interactive or visual content, check keyboard access and enlarged text. Read the page without its
images to check whether essential meaning survives. Screen-reader checks are useful when changing heading,
table, or navigation structure; a clean Markdown source alone does not establish accessible rendering.

## Establish authority and freshness

Make it clear whether a statement describes implemented behavior, a required contract, a recommendation, or
future intent. A proposed optimization in a plan must not appear in a current API page as available behavior.
If implementation and the intended contract disagree, report both with their evidence instead of silently
turning a defect into policy.

Use the appropriate authoritative home:

| Information | Falcor2 home |
| --- | --- |
| Agent workflow and execution requirements | `AGENTS.md` and applicable skills |
| Source conventions and binding documentation policy | `docs/contributing/coding-style.md` |
| Current architecture and lasting rationale | `docs/design/`, under its review policy |
| Documentation setup and build behavior | `docs/contributing/building-documentation.md` |
| Execution plans and their investigations and history | `.plans/`, under its planning guidelines |

For factual claims, use current implementation, bindings, tests, and observed behavior as evidence. Cite
external standards or algorithms when they supply a convention the reader needs to check. Preserve version
or configuration qualifiers where they affect applicability. Link source landmarks by path and symbol;
use a revision-pinned link when historical evidence is the point.

Verification metadata records what was checked; it is not a promise that later changes cannot invalidate a
page. Apply the existing metadata and approval lifecycle to design pages. Outside that lifecycle, record
version-sensitive conditions where useful without adding a ceremonial timestamp to every paragraph.

When a requested documentation change moves or renames content, check inbound links and repair affected
navigation. Keep unresolved questions visible in the review handoff. Follow the repository's explicitly
requested scope for architecture maintenance and plan retirement.

## Review documentation through reader tasks

A mechanically valid document can still fail its reader. Review the edited page in these passes, scaling
the work to the change:

| Pass | Practical check | Evidence of a useful result |
| --- | --- | --- |
| Purpose | Read only the title, opening, and headings. | The reader can identify the question and relevant section. |
| Discovery | Start from the index or search for an expected API name. | The right page can be found and distinguished from outdated material. |
| Local context | Read a substantive section without the preceding section. | Its subject, conditions, and important limitations remain clear. |
| Technical accuracy | Check decisive claims against source, contracts, and examples. | Behavior and intent are distinguished, with uncertainty recorded. |
| Procedure | Follow the stated starting conditions and steps. | The documented result is produced and recognizable. |
| Presentation | Inspect source and rendered output. | Links, tables, code, headings, and figures remain usable. |

For a tutorial, try the beginner path without unstated setup. For reference, locate a default and a boundary
condition. For explanation, ask a reader to describe ownership or predict the effect of a change. For
troubleshooting, check whether the proposed observations distinguish the named causes.
When feasible, observe an unfamiliar reader using a tutorial and record where they hesitate, make an
unexpected choice, or cannot recognize success. Those observations expose assumptions that an author may miss.

For agent usability, a bounded read-only exercise can ask an agent to locate the authoritative page and
answer a realistic question with supporting sections. Judge whether the answer identifies the applicable
conditions and source, not merely whether the agent sounds confident. Treat observed failures as specific
improvement opportunities; one successful run does not establish universal reliability.

Use existing format, link, generation, and build checks appropriate to the changed material. Follow
[Building documentation](building-documentation.md) for Sphinx validation and the
coding-style guide for affected generated API help. Report whether examples were executed or only inspected.
Building HTML does not verify the behavior of its embedded examples.

This combines technical review with the versioned-source and automated-check approach described by
[Write the Docs](https://www.writethedocs.org/guide/docs-as-code/). It does not replace the project's workflow
or create new approval gates.

### Final author checklist

- The title and opening identify a reader need and outcome.
- The structure fits the document type and exposes a useful path through the material.
- Prerequisites, conditions, defaults, and meaningful exceptions appear where readers need them.
- Examples are labeled accurately and have understandable results.
- API descriptions explain non-obvious contracts without demanding comments for self-evident declarations.
- Readers can find the page and interpret its important sections independently.
- Claims have appropriate evidence; proposals and uncertainties are identifiable.
- Links and rendered content work, and the handoff accurately states the validation performed.
