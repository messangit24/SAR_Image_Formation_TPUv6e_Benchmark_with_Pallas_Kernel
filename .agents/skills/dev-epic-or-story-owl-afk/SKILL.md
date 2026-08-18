---
name: dev-epic-or-story-owl-afk
description: Autonomous Epic & Story orchestrator powered by owl (DeepCoder and bmad-code-review subagents). Executes single stories or entire multi-story Epics end-to-end in an isolated worktree, starting with an interactive architectural alignment gate (/grill-me), implementing via TDD + Red-Team Review, and opening verified Pull Requests with full Buganizer/TaskFlow traceability.
---

> **CONFIDENTIALITY NOTICE**: Google-Internal Use Only. This skill specification and its orchestration routines are proprietary and cannot be shared externally or committed to public external repositories.

# Autonomous Epic & Story Orchestrator (`dev-epic-or-story-owl-afk`)

Use this skill when you have completed a planning session (`bmad-prd`, `bmad-architecture`, `bmad-create-epics-and-stories`) and want to autonomously develop:
1. **A Single Story** (`story-X.Y` marked `execution_mode: "AFK"`).
2. **An Entire Multi-Story Epic** (`Epic X` with all constituent stories marked `execution_mode: "AFK"`).

`dev-epic-or-story-owl-afk` automates the entire development lifecycle from initial Socratic alignment (`/grill-me`) to a verified, red-team reviewed Pull Request.

---

## 🎯 Dual Execution Modes

### Mode A: Single Story Execution (`story-X.Y`)
* **Trigger**: Invoked with a single story file (e.g. `docs/bmad/stories/story-19.1-detector.md`).
* **Flow**:
  1. **Pre-flight**: Verify prerequisite `depends_on` stories are `COMPLETED` and `execution_mode` is `AFK`.
  2. **Socratic Alignment (`/grill-me`)**: Synthesize Story Implementation Plan (Deep Module public contract, hidden complexity, test seams) and interview developer via `/grill-me`. Wait for approval (`Proceed` / `LGTM`).
  3. **Worktree Scaffolding**: Create isolated worktree `../<repo>-<story_slug>` on branch `feat/<story_slug>` with isolated `.env` (dynamic DB and port offsets).
  4. **TDD Implementation via `DeepCoder`**: Red-Green-Refactor cycle pulling in `write-python` / `write-typescript`.
  5. **Verification & Adversarial Review**: Run `ruff` + `pytest`, then dispatch `bmad-code-review` subagent to inspect diff.
  6. **Commit & PR Submission**: Create atomic commit and open PR `feat(<story_slug>): <title>` linking `b/<story_bug_id>`.

---

### Mode B: Full Epic Execution (`Epic X`)
* **Trigger**: Invoked with an Epic number (e.g. `Epic 19` or `docs/bmad/planning/epics.md#epic-19`).
* **Flow**:
  1. **Intake & DAG Analysis**: Discover all `docs/bmad/stories/story-X.*.md`, confirm all are `AFK`, and compute Topological Execution Waves (Wave 1: roots $\to$ Wave 2: dependents).
  2. **🛑 Comprehensive Epic Alignment Gate (`/grill-me`)**: Synthesize Epic Master Plan across all stories and conduct Socratic interview (`/grill-me`). Wait for approval.
  3. **Dedicated Epic Worktree**: Scaffolds `../<repo>-epic-X` on branch `feat/epic-X` with dynamic `.env`.
  4. **Topological Wave Execution**:
     - For each story in each wave:
       - Spawn `DeepCoder` for TDD Red-Green-Refactor.
       - Run automated verification gates (`ruff`, `pytest`).
       - Spawn `bmad-code-review` subagent for adversarial diff review.
       - Create atomic commit `feat(story-X.Y): <title> (closes b/<id>)`.
       - Mark frontmatter `stage: "COMPLETED"`.
  5. **Consolidated Epic PR**: Open single consolidated PR `feat(epic-X): <Epic Title>` referencing parent Epic `b/<epic_bug_id>` and all child tickets.

---

## 🛠️ Step-by-Step Execution Guide

### Phase 1: Intake & Mode Routing
1. Inspect target input:
   - If a specific story path is provided $\to$ **Mode A (Single Story)**.
   - If an Epic number or title is provided $\to$ **Mode B (Full Epic)**.
2. Verify all target stories have `execution_mode: "AFK"`. If any story requires `HUMAN_MONITORED`, notify the developer before proceeding.

### Phase 2: Socratic Alignment Gate (`/grill-me`)
1. Synthesize Implementation Plan:
   - Deep Module interface contracts (public API functions, hidden plumbing).
   - Component boundaries, database models, and migration scripts.
   - Test strategy (acceptance criteria assertion against public interface).
2. Conduct interactive Socratic interview using `/grill-me` with the `ask_question` tool.
3. **🛑 Mandatory Gate**: Prompt the developer for explicit approval:
   > 💡 **Alignment Gate**: Please review the implementation plan above. Once you confirm (**`Proceed`** or **`LGTM`**), I will scaffold the isolated worktree and autonomously execute via `DeepCoder` and `bmad-code-review`.

### Phase 3: Worktree Scaffolding
* For Single Story:
  ```bash
  python3 scripts/dev_story.py --story <path> --setup-worktree
  ```
* For Full Epic:
  ```bash
  python3 scripts/dev_story.py --epic <epic_id> --setup-epic-worktree
  ```

### Phase 4: Autonomous TDD & Adversarial Red-Team Review
For each story:
1. **Mark In-Progress**: Set frontmatter `stage: "IN_PROGRESS"`.
2. **`DeepCoder` Subagent**: Dispatch `invoke_subagent` with `TypeName='DeepCoder'`, `Workspace='inherit'`, and verbatim acceptance criteria.
3. **Verification Gates**: Run `uv run ruff check src tests` and `uv run pytest -q` in the worktree.
4. **`bmad-code-review` Subagent**: Dispatch `invoke_subagent` with `TypeName='bmad-code-review'` to red-team diff against deep module encapsulation, edge cases, and memory leaks.
5. **Atomic Commit**: Commit changes with ticket closure tag.
6. **Mark Completed**: Set frontmatter `stage: "COMPLETED"`.

### Phase 5: Pull Request & Buganizer Traceability
1. Push branch to origin (`feat/<story_slug>` or `feat/epic-X`).
2. Open Pull Request via `ggh pr create` with rich markdown summary and Buganizer links.
