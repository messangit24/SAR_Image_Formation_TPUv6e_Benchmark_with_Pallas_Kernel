# AI-Native Software Development Life Cycle (AI-SDLC) Workflow Guide

> **Last Updated:** 2026-08-17  
> **Status:** Active Reference

A comprehensive operating guide to the **AI-Native Software Development Life Cycle (AI-SDLC)**. Covers the core philosophy, the 5-stage conceptual pipeline, standing up greenfield repositories (0 ➔ 1), adding features to existing codebases (1 ➔ N), Socratic alignment via `/grill-me`, and automating multi-tracker synchronization.

---

## 🏛️ Core Philosophy: Shifting Intentionality Left

### The Code Generation Inversion
In traditional human engineering, writing code is the primary bottleneck. Developers often jump straight into an IDE, writing classes and database tables before defining strict interface boundaries, leading to coupled architectures, unreviewable pull requests, and horizontal layers where nothing works end-to-end until the final week.

In the **AI-Native Era**, code generation is instantaneous ($0 marginal cost, 0 latency). The primary failure mode is generating thousands of lines of plausible-looking code that solves the wrong problem, violates architectural invariants, cannot be verified incrementally, or breaks existing systems.

### 🎤 Socratic Alignment via `/grill-me`
The `/grill-me` skill is the interactive alignment engine of the AI-SDLC. Rather than allowing the agent to assume default requirements or rush into implementation plans, `/grill-me` conducts a multi-turn Socratic interview at two critical milestones:
1. **During Brainstorming & Research**: Probing edge cases, non-functional latency/throughput constraints, failure recovery, and scope boundaries.
2. **Before / During Implementation Plan Creation**: Stress-testing file modifications, schema migrations, backward compatibility, and test coverage before touching code.

### The 5-Stage Conceptual Progression
Whether creating a brand-new service or adding a minor feature, all work progresses through 5 disciplined stages:

```
💡 1. IDEA / SPARK
       ⬇
🧠 2. BRAINSTORM & SOCRATIC ALIGNMENT (`bmad-brainstorming` + `/grill-me`)
       ⬇
📄 3. PRD, ARCHITECTURE & EPICS (`bmad-prd`, `bmad-architecture`, `bmad-create-epics-and-stories`)
       ⬇
🎯 4. TRACKER SYNC & TASKFLOW (`update-taskflow` ➔ Buganizer & TaskFlow)
       ⬇
🧪 5. DEV IMPLEMENTATION, PLANNING & TDD (`/grill-me` ➔ `dev-story` ➔ `/owl` ➔ TDD)
       ⬇
🛡️ 6. ADVERSARIAL REVIEW, CI & HUMAN MERGE (`bmad-code-review`, PR checks, merge-sync)
```

---

## 🚀 Journey A: Standing Up a Greenfield Repository (0 ➔ 1)

When bootstrapping a brand-new project, platform, or service:

1. **Clone Starter Template**:
   ```bash
   git clone git@depot.code.corp.goog:personal/ai-sdlc-starter.git my-project
   cd my-project && chmod +x scripts/*.py
   ```
2. **Brainstorm & Socratic Interview (`bmad-brainstorming` + `/grill-me`)**:
   - Explore target user personas, technical trade-offs, algorithmic models, and boundary risks.
   - Run `/grill-me` to challenge assumptions and clarify underspecified requirements.
3. **Formalize PRD & Architecture (`bmad-prd`, `bmad-architecture`)**:
   - Write `docs/bmad/planning/prd.md` to lock in Functional Requirements (FRs) and Non-Functional Requirements (NFRs).
   - Write `docs/bmad/planning/architecture.md` to define database schemas, interface contracts, and system invariants.
4. **Decompose into Epics & Vertical Slice Stories (`bmad-create-epics-and-stories`)**:
   - Break architecture into Epics.
   - **Mandatory Walking Skeleton (Story 1.1)**: Build a minimal, runnable end-to-end path (Input $\to$ Core Logic $\to$ Storage/Output $\to$ E2E Integration Test). Declares `depends_on: []` and `execution_mode: "AFK"`.
5. **Synchronize Cloud Trackers (`update-taskflow`)**:
   - Run `python3 scripts/sync_trackers.py --update-all` to mint parent Epics, child stories, and link hierarchy in Buganizer and TaskFlow.
6. **Implement Walking Skeleton via TDD (`dev-story`, `/owl`)**:
   - Scaffold worktree for Story 1.1, author integration test first, implement minimal logic, verify green, and open PR #1 to establish a runnable baseline.

---

## 🔄 Journey B: Adding a Feature to an Existing Codebase (1 ➔ N)

When adding functionality, refactoring an engine, or executing mid-project evolution:

1. **Contextual Brainstorming & Seam Analysis (`bmad-brainstorming` + `/grill-me`)**:
   - Analyze existing `prd.md` and `architecture.md`. Use `/grill-me` to probe how the new feature integrates without breaking existing invariants.
2. **Story Specification & Vertical Slice Definition (`bmad-create-epics-and-stories`)**:
   - Create `docs/bmad/stories/story-X.Y-<name>.md` declaring:
     - `depends_on: ["story-X.Y"]` (upstream prerequisites).
     - `execution_mode: "AFK"` (autonomous) or `"HUMAN_MONITORED"` (critical oversight).
     - Vertical slice scope with a verifiable end-to-end integration test.
3. **Synchronize Trackers (`update-taskflow`)**:
   - Run `update-taskflow` to mint Buganizer tickets and stamp IDs into story frontmatter.
4. **Scaffold Isolated Worktree (`dev-story`)**:
   - Run `python3 scripts/dev_story.py --story <path> --setup-worktree`.
   - Validates that all `depends_on` prerequisite stories are `COMPLETED`.
   - Symlinks heavy dependencies (`.venv`, `node_modules`, `.env`, `data`) with zero file duplication.
5. **Plan Alignment & Socratic Grilling (`/grill-me` Gate)**:
   - Agent formulates step-by-step implementation plan. Run `/grill-me` to stress-test the proposed files, schema alterations, and verification tests. Code modification **never** begins until the developer approves.
6. **Test-Driven Implementation (`/owl` Mode)**:
   - For self-contained stories, leverage `/owl` for deep autonomous reasoning.
   - Follow strict TDD: 🔴 **Red** (failing test first) $\to$ 🟢 **Green** (minimal code) $\to$ 🔵 **Refactor** (clean and type-check).
7. **Verification Gates & Adversarial Red-Team Review**:
   - `uv run ruff check src tests scripts` && `uv run pytest -q`.
   - Run `bmad-code-review` to inspect the `git diff` for edge cases, memory leaks, and boundary gaps.
8. **PR Submission, Human Merge & Automated Post-Merge Sync**:
   - Open GitHub PR via `ggh` and update Buganizer ticket with progress.
   - Developer performs final review and manually clicks **Merge**.
   - Running `git checkout main && git pull origin main` triggers the post-merge hook (`merge_sync.py`), which marks the story completed, transitions the Buganizer ticket to `FIXED`, and deletes the temporary worktree.

---

## 🧰 The Curated 9-Skill AI-SDLC Suite

| # | Skill Name | Primary Output File / Action | Purpose |
| :- | :--- | :--- | :--- |
| **1** | **`bmad-brainstorming`** | `docs/bmad/planning/research/` | Explores requirements, problem space, algorithms, and trade-offs (amplified by `/grill-me`). |
| **2** | **`bmad-prd`** | `docs/bmad/planning/prd.md` | Grounds ideas into formal Functional & Non-Functional Requirements. |
| **3** | **`bmad-architecture`** | `docs/bmad/planning/architecture.md` | Defines technical architecture spine, DB schemas, API contracts, and invariants. |
| **4** | **`bmad-create-epics-and-stories`** | `planning/epics.md`<br/>`stories/story-*.md` | Breaks architecture into Epics and sharded vertical slice stories. |
| **5** | **`bmad-create-epics-and-stories`** | `stories/story-X.Y-*.md` | Creates a single new vertical slice story mid-project. |
| **6** | **`update-taskflow`** | **Buganizer & TaskFlow** | **The mandatory concluding step**: Mints Epics, mints Stories, links parent hierarchy, and stamps IDs into frontmatter. |
| **7** | **`dev-story`** | Isolated Git Worktree | Pre-flight dependency check, isolated worktree scaffolding, plan approval gate, verification gates, and PR submission. |
| **8** | **`bmad-code-review`** | PR Review Findings | Standalone adversarial red-team code review on `git diff`. |
| **9** | **`merge-sync`** | Main branch & Cloud Trackers | Pulls `main`, marks stories `- [x]`, transitions Buganizer/TaskFlow tickets to `FIXED`, and deletes merged worktrees. |

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

## 📺 Recommended Video References & Walkthroughs

For visual walkthroughs of the architecture and live hands-on story development:

1. **[BMAD Architecture & Multi-Agent Design Walkthrough](https://www.youtube.com/watch?v=v4F1gFy-hqg)**:
   - Deep conceptual breakdown of the BMAD framework, multi-agent context boundaries, and structured artifact generation.
2. **[Live AI-Native Story Development in Action](https://www.youtube.com/watch?v=-QFHIoCo-Ko)**:
   - Real-time demonstration of an engineer and AI agent picking up a story, scaffolding an isolated worktree, writing failing tests first (TDD), executing with `/owl`, and merging.

---

## ⚡ User Slash Command Prompting Protocol

Because slash commands (like `/owl`, `/grill-me`, `/goal`) are user-facing shortcuts executed by the human in the chat UI, AI agents **cannot invoke them autonomously**. Instead, agents follow a proactive prompting protocol:

| Lifecycle Stage | Agent Action | Recommended User Prompt |
| :--- | :--- | :--- |
| **Brainstorming / PRD** | Explores scope, trade-offs | *"💡 Type `/grill-me` to stress-test these assumptions in an interactive interview."* |
| **Step 2 (Plan Approval)** | Drafts step-by-step plan | *"💡 Type `/grill-me` to stress-test this implementation plan, or reply `Proceed`/`LGTM` to start."* |
| **Step 3 (Implementation)** | Worktree scaffolded | *"💡 For deep autonomous execution with TDD checkpoints, type `/owl` (or `/goal`)."* |
| **Step 5 (Adversarial Review)** | Tests passing | Spawns `bmad-code-review` subagent or prompts: *"💡 Running adversarial red-team review."* |
