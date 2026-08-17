# AI-SDLC & BMAD Workflow Guide

> **Last Updated:** 2026-08-17  
> **Status:** Active Reference

A complete guide to feature brainstorming, architecture planning, story creation, mid-project evolution, and automated TaskFlow synchronization using the **Lean 9-Skill AI-SDLC Suite**.

---

## 📁 Standard File & Folder Layout

```
docs/bmad/
├── README.md                      # Index of all active planning documents
├── WORKFLOW.md                    # This lifecycle & evolution guide
├── planning/
│   ├── prd.md                     # Product Requirements Document (FRs, NFRs, user journeys)
│   ├── architecture.md            # Technical Architecture Spine (contracts, DB, schemas, APIs)
│   └── epics.md                   # Epic breakdown and story-to-requirement mapping
└── stories/
    ├── sprint-status.yaml         # Sprint state machine (backlog, in-progress, review, done)
    ├── story-1.1-starter.md       # Sharded, independently testable story specifications
    └── story-1.2-api.md
```

---

## 🧰 The Curated 9-Skill Suite

| # | Skill Name | Primary Output File / Action | Purpose |
| :- | :--- | :--- | :--- |
| **1** | **`bmad-brainstorming`** | `docs/bmad/planning/research/` | Explores requirements, problem space, algorithms, and trade-offs. |
| **2** | **`bmad-prd`** | `docs/bmad/planning/prd.md` | Grounds ideas into formal Functional & Non-Functional Requirements. |
| **3** | **`bmad-architecture`** | `docs/bmad/planning/architecture.md` | Defines technical architecture spine, DB schemas, API contracts, and invariants. |
| **4** | **`bmad-create-epics-and-stories`** | `planning/epics.md`<br/>`stories/story-*.md` | Breaks architecture into Epics and sharded vertical slice stories. |
| **5** | **`bmad-create-story`** | `stories/story-X.Y-*.md` | Creates a single new vertical slice story mid-project. |
| **6** | **`update-taskflow`** | **Buganizer & TaskFlow** | **The mandatory concluding step**: Mints Epics, mints Stories, links parent hierarchy, and stamps IDs into frontmatter. |
| **7** | **`dev-story`** | Isolated Git Worktree | Pre-flight dependency check, isolated worktree scaffolding, plan approval gate, verification gates, and PR submission. |
| **8** | **`bmad-code-review`** | PR Review Findings | Standalone adversarial red-team code review on `git diff`. |
| **9** | **`merge-sync`** | Main branch & Cloud Trackers | Pulls `main`, marks stories `- [x]`, transitions Buganizer/TaskFlow tickets to `FIXED`, and deletes merged worktrees. |

---

## 🔄 Mid-Project Evolution (Adding & Modifying Stories)

| Scenario | Curated Skill | Action | Concluding Step |
| :--- | :--- | :--- | :--- |
| **Add 1 New Story** | `bmad-create-story` | Creates `docs/bmad/stories/story-X.Y-*.md` with `epic: 'Epic X'` | Run `update-taskflow` (mints ticket, links to parent Epic) |
| **Add a Brand New Epic** | `bmad-create-epics-and-stories` | Adds `## Epic X: <Title>` in `epics.md` and creates story files | Run `update-taskflow` (mints Epic, mints child stories, links hierarchy) |
| **Refine Acceptance Criteria** | Edit Story Markdown | Modifies checklist or technical context in `docs/bmad/stories/` | Run `update-taskflow` (updates Buganizer comments) |

---

## 🎯 Story Specification Standards & Vertical Slices

When drafting stories with `bmad-create-story` or `bmad-create-epics-and-stories`:

1. **Explicit Story Dependencies (`depends_on`)**:
   - Upstream prerequisites declared as a typed list: `depends_on: ["story-1.1"]` (or `depends_on: []` if unblocked).
   - Pre-Flight Gate: Prerequisites must be `COMPLETED` before picking up a story.

2. **Operational Execution Mode (`execution_mode`)**:
   - `execution_mode: "AFK"` (Autonomous): Self-contained scope, deterministic contracts, unit-test coverage, low architectural blast radius. Once plan is approved, agent implements and verifies unattended.
   - `execution_mode: "HUMAN_MONITORED"` (Critical / Supervised): Requires developer oversight for architectural refactoring, destructive DB migrations, billable API spend (e.g. Gemini Vertex VLM calls), security boundaries, or complex UI alterations. Requires `human_monitoring_rationale`.

3. **Vertical Slice Architecture Principle (Tracer Bullets vs. Horizontal Layering)**:
   - **Anti-Pattern (Horizontal Layering)**: Never decompose stories by architectural tier (Story 1 = DB only, Story 2 = Backend only, Story 3 = UI only).
   - **Mandatory Vertical Slices**:
     - **Story 1 (Walking Skeleton / Tracer Bullet)**: Builds the minimal end-to-end runnable/testable path through the entire stack (Input $\to$ Core Logic $\to$ Persistence/Output $\to$ E2E Integration Test).
     - **Subsequent Stories (Progressive Enrichment)**: Add incremental depth, algorithmic tiers, and UI views.
     - **Continuous Verifiability**: Every single story contains an executable end-to-end integration test (`pytest tests/test_e2e_*.py`) proving the complete pipeline functions.

---

## 🛠️ The 7-Step Story Development Loop

1. **Pre-Flight & Scaffold Worktree**: `python3 scripts/dev_story.py --story docs/bmad/stories/<story>.md --setup-worktree` (validates `depends_on` and notes `execution_mode`).
2. **Propose Plan & Wait for Approval**: Interactive alignment between human and agent.
3. **Worktree Implementation (TDD & `/owl` Mode)**:
   - For self-contained stories without external repo dependencies, trigger `/owl` for autonomous deep reasoning and checkpointing.
   - Follow **Test-Driven Development (TDD)**: 🔴 **Red** (Author failing test asserting ACs) $\to$ 🟢 **Green** (Implement minimal code to pass) $\to$ 🔵 **Refactor** (Harden and type-check).
4. **Verification Gates**: Run automated linting (`uv run ruff check src tests scripts`) and test suites (`uv run pytest -q`).
5. **Adversarial Red-Team Review**: Scan diff for edge cases, leaks, and regressions (`bmad-code-review`).
6. **PR Submission**: Push branch, open Issue and PR via `ggh`, and update Buganizer ticket.
7. **Post-Merge**: Merge PR, run `python3 scripts/merge_sync.py` to finalize tickets to FIXED and clean worktrees.

