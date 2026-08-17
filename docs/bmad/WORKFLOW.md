# AI-Native Software Development Life Cycle (AI-SDLC) Workflow Guide

> **Last Updated:** 2026-08-17  
> **Status:** Active Reference

A comprehensive operating guide to the **AI-Native Software Development Life Cycle (AI-SDLC)**. Covers the core philosophy, the 5-stage conceptual pipeline, standing up greenfield repositories (0 ➔ 1), adding features to existing codebases (1 ➔ N), Socratic alignment via `/grill-me`, and automating multi-tracker synchronization.

---

## 🏛️ Core Philosophy: Shifting Intentionality Left

### The Code Generation Inversion
In traditional human engineering, writing code is the primary bottleneck. Developers often jump straight into an IDE, writing classes and database tables before defining strict interface boundaries, leading to coupled architectures, unreviewable pull requests, and horizontal layers where nothing works end-to-end until the final week.

In the **AI-Native Era**, code generation is instantaneous ($0 marginal cost, 0 latency). The primary failure mode is generating thousands of lines of plausible-looking code that solves the wrong problem, violates architectural invariants, cannot be verified incrementally, or breaks existing systems.

### 📺 Conceptual Inspiration & Recommended Walkthroughs
1. **[High-Level AI-Native Software Development Concepts](https://www.youtube.com/watch?v=v4F1gFy-hqg)**:
   - Primary conceptual inspiration for this framework—explores how engineering shifts when LLMs write code, why problem framing and boundary definitions become the primary developer tasks, and how multi-agent structures prevent cognitive saturation.
2. **[Live AI-Native Story Development in Action](https://www.youtube.com/watch?v=-QFHIoCo-Ko)**:
   - Practical demonstration of an engineer using an AI agent to pick up a story, work within an isolated worktree, apply test-driven development (TDD), and complete a verified PR.

---

## 🚀 Journey A: Standing Up a Greenfield Repository (0 ➔ 1)

When bootstrapping a brand-new project, platform, or service:

1. **Clone Starter Template**:
   ```bash
   git clone git@depot.code.corp.goog:personal/ai-sdlc-starter.git my-project
   cd my-project && chmod +x scripts/*.py
   ```
2. **Brainstorm & Socratic Interview (`bmad-brainstorming`)**:
   - Explore target user personas, technical trade-offs, algorithmic models, and boundary risks.
   - *💡 User Prompt*: Type `/grill-me` if you want to stress-test assumptions and clarify underspecified requirements through an interactive interview.
3. **Formalize PRD & Architecture (`bmad-prd`, `bmad-architecture`)**:
   - Write `docs/bmad/planning/prd.md` to lock in Functional Requirements (FRs) and Non-Functional Requirements (NFRs).
   - Write `docs/bmad/planning/architecture.md` to define database schemas, interface contracts, and system invariants.
4. **Decompose into Epics & Vertical Slice Stories (`bmad-create-epics-and-stories`)**:
   - Break architecture into Epics.
   - **Mandatory Walking Skeleton (Story 1.1)**: Build a minimal, runnable end-to-end path (Input $\to$ Core Logic $\to$ Storage/Output $\to$ E2E Integration Test). Declares `depends_on: []` and `execution_mode: "AFK"`.
5. **Human Confirmation & Cloud Tracker Sync (`update-taskflow`)**:
   - Present the breakdown to the developer. **Wait for explicit confirmation (`Proceed` / `LGTM`)** before syncing.
   - Run `python3 scripts/sync_trackers.py --update-all` to mint parent Epics, child stories, and link hierarchy in Buganizer and TaskFlow.
   - **🛑 Action**: Close the planning conversation session.
6. **Implement Walking Skeleton via TDD (`dev-story`)**:
   - Open a fresh dev session for Story 1.1. Scaffold the worktree, author the integration test first, implement minimal logic, verify green, and open PR #1 to establish a runnable baseline.

---

## 🔄 Journey B: Adding a Feature to an Existing Codebase (1 ➔ N)

When adding functionality, refactoring an engine, or executing mid-project evolution:

1. **Contextual Brainstorming & Seam Analysis (`bmad-brainstorming`)**:
   - Analyze existing `prd.md` and `architecture.md`. Use `/grill-me` to probe how the new feature integrates without breaking existing invariants.
2. **Story Specification & Deep Module Definition (`bmad-create-epics-and-stories`)**:
   - Decomposes the requirement into either a single story or a brand-new multi-story Epic:
     - **Deep Module Seam**: Small, clean public interface hiding internal complexity.
     - **Explicit Dependencies**: `depends_on: ["story-X.Y"]` (upstream prerequisites).
     - **Execution Mode**: `execution_mode: "AFK"` (autonomous) or `"HUMAN_MONITORED"` (critical oversight).
     - **Interface-Centric Acceptance Criteria**: Verifiable end-to-end integration test asserting public behavior.

3. **Human Confirmation & Tracker Sync (`update-taskflow`)**:
   - Present proposed stories/epics to the developer for review.
   - **Wait for explicit user confirmation (`Proceed` / `LGTM`)**, then run `update-taskflow` to mint Buganizer tickets and stamp IDs into story frontmatter.
   - **🛑 Action**: Close the planning session.

4. **Scaffold Isolated Worktree (`dev-story`)**:
   - Open a fresh dev session. Run `python3 scripts/dev_story.py --story <path> --setup-worktree`.
   - Validates that all `depends_on` prerequisite stories are `COMPLETED`.
   - Symlinks heavy dependencies (`.venv`, `node_modules`, `.env`, `data`) with zero file duplication.
5. **Plan Alignment & Socratic Grilling**:
   - Agent formulates step-by-step implementation plan. Developer can run `/grill-me` to stress-test the proposed files, schema alterations, and verification tests. Code modification **never** begins until the developer approves.
6. **Test-Driven Implementation (`dev-story` + TDD)**:
   - Pull in `write-python` or `write-typescript` standards.
   - Follow strict TDD: 🔴 **Red** (failing test first) $\to$ 🟢 **Green** (minimal code) $\to$ 🔵 **Refactor** (clean and type-check).
   - *💡 User Prompt*: For self-contained stories, the developer can type `/owl` (or `/goal`) for deep autonomous execution.
7. **Verification Gates & Adversarial Red-Team Review**:
   - `uv run ruff check src tests scripts` && `uv run pytest -q`.
   - The agent spawns `bmad-code-review` subagent to cynically inspect the `git diff` for edge cases, memory leaks, and boundary gaps.
8. **PR Submission, Human Merge & Automated Post-Merge Sync**:
   - Open GitHub PR via `ggh` and update Buganizer ticket with progress.
   - GitHub Actions AI Reviewer (`gemini-pr-review.yml`) checks the PR diff.
   - Developer performs final review and manually clicks **Merge**.
   - Running `git checkout main && git pull origin main` triggers the post-merge hook (`merge_sync.py`), which marks the story completed, transitions the Buganizer ticket to `FIXED`, and deletes the temporary worktree.
   - **🛑 Action**: Close the dev session.

---

## 🧰 The Curated 10-Skill AI-SDLC Suite

| # | Skill Name | Primary Output File / Action | Purpose |
| :- | :--- | :--- | :--- |
| **1** | **`bmad-brainstorming`** | `docs/bmad/planning/research/` | Explores requirements, problem space, algorithms, and trade-offs (optionally grilled with `/grill-me`). |
| **2** | **`bmad-prd`** | `docs/bmad/planning/prd.md` | Grounds ideas into formal Functional & Non-Functional Requirements. |
| **3** | **`bmad-architecture`** | `docs/bmad/planning/architecture.md` | Defines technical architecture spine, DB schemas, API contracts, and invariants. |
| **4** | **`bmad-create-epics-and-stories`** | `planning/epics.md`<br/>`stories/story-*.md` | Unified engine: decomposes architecture into Epics, new subsystems, and sharded vertical slice stories. |
| **5** | **`update-taskflow`** | **Buganizer & TaskFlow** | **The mandatory concluding step**: Mints Epics, mints Stories, links parent hierarchy, and stamps IDs into frontmatter. |
| **6** | **`dev-story`** | Isolated Git Worktree | Pre-flight dependency check, isolated worktree scaffolding, plan approval gate, verification gates, and PR submission. |
| **7** | **`bmad-code-review`** | PR Review Findings | Standalone adversarial red-team code review on `git diff` (spawned as subagent). |
| **8** | **`write-python`** | Python Source Files | Modern Python typing, lazy imports, hermetic testing, and security invariants. |
| **9** | **`write-typescript`** | TypeScript / React Files | Strict TypeScript, Zod boundary validation, and React patterns. |
| **10** | **`merge-sync`** | Main branch & Cloud Trackers | Pulls `main`, marks stories `- [x]`, transitions Buganizer/TaskFlow tickets to `FIXED`, and deletes merged worktrees. |

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

## 🌐 Worktree Resource & Machinery Isolation Patterns

When scaling parallel agent workflows across multiple Git worktrees:

1. **Database Namespacing**: `scripts/dev_story.py` writes a unique `DATABASE_URL=postgresql://localhost:5432/<repo>_<story>` into the worktree `.env`, preventing schema migration races.
2. **Port Offsets**: Web server and sidecar ports are dynamically offset (`PORT=8000+offset`) to prevent `EADDRINUSE` port collisions.
3. **Hardware FileLocks**: Exclusive hardware accelerators (TPUs/GPUs) use file-based mutexes (`filelock`) in test fixtures to queue access cleanly without crashing.
4. **12-Factor Compliance**: Application source code in `src/` only reads standard environment variables, ensuring zero worktree or dev-specific code leaks into production.
