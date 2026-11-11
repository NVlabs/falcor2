---
name: falcor2-review-plan
description: Review a plan using sub agents and provide feedback on how to improve it.
---

# Falcor2 Review Plan

## Source checkout

Read [source-context.md](source-context.md) first when installed. It identifies the
Falcor2 checkout for all repository paths and commands below. When reading the
canonical packaged source directly, use its enclosing Falcor2 checkout instead.
Supporting references in this skill are relative to this SKILL.md.

Read the installed `.agents/PLANS.md` identified by source-context.md, or
`skills/PLANS.md` when working directly from the canonical checkout. It defines
plan preparation and execution; this skill provides the independent review pass.

This skill allows you to review a plan using sub agents and provide feedback on how to improve it. You can use this skill to get feedback on your plans and make them better.

Run 5 sub agents to review the plan and provide feedback on it. The agents will have the following roles:
- correctness reviewer: This agent will review the plan for correctness and identify any errors or issues.
- completeness reviewer: This agent will review the plan for completeness and identify any missing steps or information
- clarity reviewer: This agent will review the plan for clarity and identify any confusing or unclear parts.
- efficiency reviewer: This agent will review the plan for efficiency and identify any unnecessary steps or inefficient processes.
- performance reviewer: This agent will review the plan for performance and identify potential cpu or gpu performance issues

For code plans, route relevant rules from the checkout's `docs/contributing/coding-style.md` to reviewers.
Correctness/completeness cover binding documentation and validation (BIND); clarity/completeness cover naming
(NAME). Apply SCOPE.1, cite rule IDs in findings, and avoid unrelated cleanup or documentation-coverage work.

## Agent Lifecycle

Use available Codex agent or Claude Code subagent tools. Run all five roles in
bounded batches if concurrency is limited. Collect each result before releasing
the agent with an available lifecycle operation; do not assume close_agent exists.
If delegation is unavailable, report that the required multi-agent review could
not be performed instead of claiming it.

Once the agents have reviewed the plan, they will provide feedback on how to improve it. The feedback will be based on their findings and will include specific suggestions for improvement.

Once complete, release reviewer agents when supported, then present the feedback in a clear and concise manner, highlighting the key areas for improvement and providing actionable steps to enhance the plan and ask if the user would like to implement the feedback.
