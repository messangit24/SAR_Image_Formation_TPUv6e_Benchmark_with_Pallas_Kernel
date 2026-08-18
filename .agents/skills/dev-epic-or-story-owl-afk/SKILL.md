---
name: dev-epic-or-story-owl-afk
description: Autonomous Epic & Story orchestrator powered by owl (DeepCoder and bmad-code-review subagents). Executes single stories or entire multi-story Epics end-to-end in an isolated worktree, starting with an interactive architectural alignment gate (/grill-me), implementing via TDD + Red-Team Review, and opening verified Pull Requests with full Buganizer/TaskFlow traceability.
---

> **CONFIDENTIALITY NOTICE**: Google-Internal Use Only. This skill specification and its orchestration routines are proprietary and cannot be shared externally or committed to public external repositories.

# Autonomous Epic & Story Orchestrator (`dev-epic-or-story-owl-afk`)

Use this skill when you have completed a planning session (`bmad-prd`, `bmad-architecture`, `bmad-create-epics-and-stories`) and want to autonomously develop:
1. **A Single Story** (`story-X.Y` marked `execution_mode: "AFK"`).
2. **An Entire Multi-Story Epic** (`Epic X` with all constituent stories marked `execution_mode: "AFK"`).

`dev-epic-or-story-owl-afk` automates the exact developer loop: **`/grill-me` alignment $\to$ `/owl LGTM` (DeepCoder TDD) $\to$ `/owl bmad adv review` (DeepInvestigator Red-Team) $\to$ Refinement Loop $\to$ PR**.

---

## 🛑 MANDATORY ORCHESTRATOR INVARIANTS

1. **ZERO DIRECT CODE MODIFICATION BY ORCHESTRATOR**:
   - The Orchestrator agent **MUST NEVER** call `write_to_file` or `replace_file_content` to implement story logic.
   - All code generation, TDD tests, and refactoring **MUST** be performed by invoking `DeepCoder` via `invoke_subagent`.
2. **MANDATORY ADVERSARIAL RED-TEAM SUBAGENT**:
   - Every story **MUST** undergo adversarial review via `DeepInvestigator` / `bmad-code-review` via `invoke_subagent` before committing.
3. **STOP AND WAIT FOR SUBAGENTS**:
   - After invoking each subagent, the orchestrator must stop calling tools and wait for the subagent's response.

---

## 🎯 Dual Execution Modes

### Mode A: Single Story Execution (`story-X.Y`)
* **Trigger**: Invoked with a single story file (e.g. `docs/bmad/stories/story-19.1-detector.md`).
* **Workflow**:
  1. **Pre-flight**: Verify prerequisite `depends_on` stories are `COMPLETED` and `execution_mode` is `AFK`.
  2. **Socratic Alignment (`/grill-me`)**: Synthesize Story Implementation Plan (Deep Module public contract, hidden complexity, test seams) and interview developer via `/grill-me`. Wait for approval (`Proceed` / `LGTM`).
  3. **Worktree Scaffolding**: Create isolated worktree `../<repo>-<story_slug>` on branch `feat/<story_slug>` with isolated `.env`.
  4. **The `/owl` Dev & Review Loop**:
     - **`/owl LGTM`**: Dispatch `DeepCoder` for TDD implementation.
     - **`/owl bmad adv review`**: Dispatch `DeepInvestigator` for adversarial review.
     - **Refine & Repeat**: Iterate until review is 100% clean.
  5. **PR Submission**: Open Pull Request `feat(<story_slug>): <title>` linking `b/<story_bug_id>`.

---

### Mode B: Full Epic Execution (`Epic X`)
* **Trigger**: Invoked with an Epic number (e.g. `Epic 19` or `docs/bmad/planning/epics.md#epic-19`).
* **Workflow**:
  1. **Intake & DAG Analysis**: Discover all `docs/bmad/stories/story-X.*.md`, confirm all are `AFK`, and compute Topological Execution Waves (Wave 1: roots $\to$ Wave 2: dependents).
  2. **Comprehensive Epic Alignment Gate (`/grill-me`)**: Synthesize Epic Master Plan across all stories and conduct Socratic interview (`/grill-me`). Wait for approval.
  3. **Dedicated Epic Worktree**: Scaffolds `../<repo>-epic-X` on branch `feat/epic-X` with dynamic `.env`.
  4. **Topological Wave Execution**: For each story in each wave, execute the **`/owl` Dev & Review Loop**.
  5. **Consolidated Epic PR**: Open single consolidated PR `feat(epic-X): <Epic Title>` referencing parent Epic `b/<epic_bug_id>` and all child tickets.

---

## 🛠️ Step-by-Step Execution Guide

### Phase 1: Intake & Mode Routing
1. Determine target mode:
   - If a specific story path is provided $\to$ **Mode A (Single Story)**.
   - If an Epic number or title is provided $\to$ **Mode B (Full Epic)**.
2. Verify all target stories have `execution_mode: "AFK"`.

### Phase 2: Socratic Alignment Gate (`/grill-me`)
1. Synthesize Implementation Plan:
   - Deep Module interface contracts (public API functions, hidden plumbing).
   - Component boundaries, database models, and migration scripts.
   - Test strategy (acceptance criteria assertion against public interface).
2. Conduct interactive Socratic interview using `/grill-me` with the `ask_question` tool.
3. **🛑 Mandatory Gate**: Prompt the developer for explicit approval:
   > 💡 **Alignment Gate**: Please review the implementation plan above. Once you confirm (**`Proceed`** or **`LGTM`**), I will scaffold the isolated worktree and autonomously execute the `/owl LGTM` and `/owl bmad adv review` loop.
4. **WAIT** for the user to respond with `Proceed` or `LGTM` before proceeding to Phase 3.

### Phase 3: Worktree Scaffolding
* For Single Story:
  ```bash
  python3 scripts/dev_story.py --story <path> --setup-worktree
  ```
* For Full Epic:
  ```bash
  python3 scripts/dev_story.py --epic <epic_id> --setup-epic-worktree
  ```

---

## 🚀 Phase 4: The Autonomous `/owl` Implementation & Review Loop

For each target story (in topological dependency order):

### Step 4.1: Mark Story In-Progress
Update YAML frontmatter in `docs/bmad/stories/story-X.Y.md` to `stage: "IN_PROGRESS"`.

### Step 4.2: Execute `/owl LGTM` (DeepCoder Subagent Delegation)
The orchestrator executes the `owl` coding delegation routine. **DO NOT write code directly in the orchestrator session**.

Call `invoke_subagent`:
```python
invoke_subagent(
    Subagents=[{
        "TypeName": "self",
        "Role": "DeepCoder",
        "Workspace": "inherit",
        "Prompt": """Task: Implement Story docs/bmad/stories/story-X.Y.md based on the approved plan: LGTM

Additional Context:
Worktree Location: <path_to_worktree>
Story Content:
<verbatim_story_acceptance_criteria_and_deep_module_specs>

Instructions:
1. Follow strict Test-Driven Development (TDD):
   - Red: Write failing unit/integration tests in tests/ asserting acceptance criteria against the public interface. Run `uv run pytest -q` to confirm failure.
   - Green: Implement minimal production code in src/ until tests pass.
   - Refactor: Enforce strict typing, clean encapsulation, and lazy imports (write-python/write-typescript standards).
2. Run verification gates in the worktree:
   - `uv run ruff check src tests`
   - `uv run pytest -q`
3. Return a completion report summarizing implemented files and verification results."""
    }],
    toolAction="Dispatching owl DeepCoder",
    toolSummary="Execute /owl LGTM story implementation"
)
```
**🛑 STOP AND WAIT**: Wait for the `DeepCoder` subagent to return its completion report.

### Step 4.3: Execute `/owl bmad adv review` (Red-Team Review Subagent)
Once `DeepCoder` completes, the orchestrator triggers the adversarial red-team review routine.

Call `invoke_subagent`:
```python
invoke_subagent(
    Subagents=[{
        "TypeName": "self",
        "Role": "DeepInvestigator",
        "Workspace": "inherit",
        "Prompt": """Task: Conduct an adversarial red-team code review of the git diff for story-X.Y: bmad adv review

Additional Context:
Worktree Location: <path_to_worktree>
Story File: docs/bmad/stories/story-X.Y.md

Instructions:
1. Run `git diff origin/main` in the worktree.
2. Cynically evaluate the diff against:
   - Acceptance Criteria: Did the implementation satisfy ALL acceptance criteria?
   - Deep Module Encapsulation: Is the public interface simple and typed? Is internal complexity hidden?
   - Edge Cases & Memory Leaks: Are there unhandled boundary conditions, race conditions, or resource leaks?
   - Language Standards: Strict typing, lazy imports, no dead code.
3. Return a verdict:
   - `VERDICT: PASS` (if diff is clean, robust, and meets all criteria)
   - `VERDICT: ACTION REQUIRED` (with precise file:line feedback and required fixes)."""
    }],
    toolAction="Dispatching owl DeepInvestigator",
    toolSummary="Execute /owl bmad adv review"
)
```
**🛑 STOP AND WAIT**: Wait for the `DeepInvestigator` subagent's review report.

### Step 4.4: Refinement Loop (Repeat As Needed)
- If the reviewer returns **`VERDICT: ACTION REQUIRED`**:
  1. Send the review feedback back to the `DeepCoder` subagent via `send_message`:
     `Task: Resolve the following adversarial review findings: <findings>`
  2. Wait for `DeepCoder` to apply fixes and verify green.
  3. Re-run `/owl bmad adv review` via `send_message` to the review subagent.
  4. Repeat until the review report returns **`VERDICT: PASS`**.

### Step 4.5: Final Verification Gate & Atomic Git Commit
Once review is 100% clean:
1. Run test suite:
   ```bash
   uv run ruff check src tests && uv run pytest -q
   ```
2. Commit story in worktree:
   ```bash
   git add -A
   git commit -m "feat(story-X.Y): <Story Title> (closes b/<buganizer_id>)"
   ```
3. Update frontmatter in `story-X.Y.md` to `stage: "COMPLETED"`.

---

## 🏁 Phase 5: Pull Request Submission & Traceability

1. Run full verification suite across the repository:
   ```bash
   uv run ruff check src tests && uv run pytest -q
   ```
2. Push branch to origin (`git push origin feat/<slug>`).
3. Open Pull Request via `ggh pr create`:
   - Title: `feat(<slug>): <Title>`
   - Body: Summary of completed vertical slice stories, verification proofs, and links to parent Epic `b/<epic_bug_id>` and child `b/<story_bug_id>` tickets.
