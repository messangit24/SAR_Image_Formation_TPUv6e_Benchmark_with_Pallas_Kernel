---
name: dev-epic-owl
description: Experimental autonomous Epic orchestrator powered by owl (DeepCoder and bmad-code-review subagents). Executes an entire multi-story Epic end-to-end in an isolated Epic worktree, starting with an interactive architectural alignment gate (/grill-me), implementing stories in topological wave order via TDD + Red-Team Review, and opening a single consolidated Pull Request with full Buganizer/TaskFlow traceability.
---

> **CONFIDENTIALITY NOTICE**: Google-Internal Use Only. This skill specification and its orchestration routines are proprietary and cannot be shared externally or committed to public external repositories.

# Autonomous Epic Orchestrator (`dev-epic-owl`)

Use this experimental skill when you have completed a thorough planning session (`bmad-prd`, `bmad-architecture`, `bmad-create-epics-and-stories`) resulting in a set of detailed, well-specified stories for an Epic that are all marked `execution_mode: "AFK"`.

`dev-epic-owl` automates the entire Epic lifecycle from initial Socratic alignment to a single, verified, consolidated Pull Request.

---

## 🔄 The 5-Phase Epic Lifecycle

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. INTAKE & DEPENDENCY DAG ANALYSIS                                         │
│ • Reads `epics.md` and discovers all child stories `story-X.*.md`.          │
│ • Verifies all stories are `AFK` and builds topological execution waves.    │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. 🛑 COMPREHENSIVE EPIC-LEVEL ALIGNMENT GATE (`/grill-me`)                 │
│ • Synthesizes Master Epic Architecture & Implementation Plan.               │
│ • Socratic interview aligns on schemas, APIs, deep modules & edge cases.    │
│ • WAITS for explicit developer confirmation before modifying code.          │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. DEDICATED EPIC WORKTREE SCAFFOLDING (`../<repo>-epic-X`)                 │
│ • Scaffolds `feat/epic-X` worktree with zero-copy symlinks (.venv, etc.).   │
│ • Generates isolated `.env` (dynamic database & port offsets).              │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. AUTONOMOUS WAVE EXECUTION (TDD + DeepCoder + Adversarial Review)         │
│ • Executes stories in topological dependency order (Wave 1 ➔ Wave 2 ➔ ...). │
│ • For each story:                                                           │
│   1. Spawn `DeepCoder` for TDD Red-Green-Refactor implementation.           │
│   2. Run verification gates (`ruff`, `pytest`).                             │
│   3. Spawn `bmad-code-review` subagent for adversarial diff inspection.    │
│   4. Create atomic Git commit `feat(story-X.Y): <title> (closes b/<id>)`.   │
│   5. Update in-repo frontmatter (`stage: "COMPLETED"`).                     │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 5. CONSOLIDATED PULL REQUEST & TRACKER TRACEABILITY                         │
│ • Runs full repository test suite.                                          │
│ • Opens single consolidated PR (`feat/epic-X: <Epic Title>`) via `ggh`.     │
│ • References parent Epic `b/<epic_bug_id>` and all child `b/<story_id>`s.  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Step-by-Step Execution Guide

### Phase 1: Intake & Dependency DAG Analysis
1. Parse `docs/bmad/planning/epics.md` for target Epic number `X`.
2. Discover all story files matching `docs/bmad/stories/story-X.*.md`.
3. Extract `depends_on` from frontmatter and build the Topological Wave Graph:
   - **Wave 1**: Root stories (`depends_on: []` / Walking Skeleton `story-X.1`).
   - **Wave 2**: Stories depending only on Wave 1 stories.
   - **Wave N**: Final integration, UI, and end-to-end verification stories.
4. Verify that all stories in the Epic are marked `execution_mode: "AFK"`. If any story requires `HUMAN_MONITORED`, highlight it to the user.

### Phase 2: Socratic Alignment Gate (`/grill-me`)
1. Synthesize an **Epic Master Implementation Plan** covering:
   - Shared data models, database migrations, and schema changes.
   - Deep Module interface contracts (public APIs, hidden complexities).
   - Component boundaries and seams.
   - Verification suite strategy (unit, integration, and E2E tests).
2. Conduct an interactive Socratic interview using `/grill-me`:
   - Resolve any underspecified requirement or architectural ambiguity across all stories up front.
3. **🛑 Mandatory Gate**: Prompt the developer for approval:
   > 💡 **Epic Alignment Gate**: Please review the Master Epic Plan above. Once you confirm (**`Proceed`** or **`LGTM`**), I will scaffold the isolated Epic worktree and autonomously execute all stories in topological wave order.

### Phase 3: Dedicated Epic Worktree Scaffolding
1. Create isolated Epic worktree at `../<repo>-epic-X` on branch `feat/epic-X`:
   ```bash
   git worktree add ../<repo>-epic-X -b feat/epic-X origin/main
   ```
2. Symlink heavy dependencies (`.venv`, `web/node_modules`, `data`).
3. Generate isolated `.env` with:
   - `DATABASE_URL=postgresql://localhost:5432/<repo>_epic_X`
   - `PORT=8000 + (X * 10)`
   - `SIDECAR_PORT=8100 + (X * 10)`
   - `WORKTREE_NAME=<repo>-epic-X`

### Phase 4: Autonomous Wave Execution
For each Wave in the topological graph, and for each story in that wave:

1. **Mark Story In-Progress**:
   - Update frontmatter in `story-X.Y.md` to `stage: "IN_PROGRESS"`.
2. **Execute TDD via `DeepCoder` Subagent**:
   - Dispatch `invoke_subagent` with `TypeName='DeepCoder'`, `Workspace='inherit'`, and prompt formatted with the verbatim story acceptance criteria and deep module rules (`write-python` / `write-typescript`).
   - Red-Green-Refactor: Author failing tests first $\to$ Implement minimal logic $\to$ Clean and type-check.
3. **Run Verification Gates**:
   - Execute `uv run ruff check src tests` and `uv run pytest -q` in the Epic worktree.
4. **Adversarial Red-Team Review Subagent**:
   - Dispatch `invoke_subagent` with `TypeName='bmad-code-review'` to cynically inspect the story's diff against deep module encapsulation, edge cases, and memory leaks. Address any actionable findings.
5. **Atomic Story Git Commit**:
   - Stage and commit changes:
     ```bash
     git add -A
     git commit -m "feat(story-X.Y): <Story Title> (closes b/<buganizer_id>)"
     ```
6. **Mark Story Completed**:
   - Update frontmatter in `story-X.Y.md` to `stage: "COMPLETED"`.

### Phase 5: Consolidated Pull Request & Tracker Traceability
1. Run the entire verification suite across the whole Epic:
   ```bash
   uv run ruff check src tests && uv run pytest -q
   ```
2. Push branch `feat/epic-X` to origin:
   ```bash
   git push origin feat/epic-X
   ```
3. Open consolidated Pull Request via `ggh pr create`:
   - Title: `feat(epic-X): <Epic Title>`
   - Body: Summary of all completed vertical slice stories, architecture updates, test coverage proofs, and links to parent Epic `b/<epic_bug_id>` and child `b/<story_bug_id>` tickets.
4. Update parent Epic ticket in Buganizer with link to the opened PR.

---

## ⚡ Execution Invariants

1. **All Stories Must Be AFK**: Only execute Epics where all constituent stories are fully detailed with clear acceptance criteria and marked `execution_mode: "AFK"`.
2. **One Consolidated PR**: The entire Epic is delivered as a single cohesive Pull Request containing clean, atomic commits for each story.
3. **No Direct Production Changes**: All work occurs in the isolated Epic worktree (`../<repo>-epic-X`).
4. **Adversarial Review per Story**: Every story must pass both automated test verification and a subagent adversarial review before being committed to the Epic branch.
