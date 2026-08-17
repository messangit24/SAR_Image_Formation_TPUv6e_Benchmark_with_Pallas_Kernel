---
name: bmad-create-epics-and-stories
description: Unified Epic & Story creation engine. Automatically determines whether requirements represent a full greenfield decomposition, a brand-new Epic with multiple stories, or a single story addition. Decomposes architecture into vertical slice stories with deep module interfaces, depends_on, execution_mode (AFK vs HUMAN_MONITORED), and updates TaskFlow.
---

# Universal Epic & Story Creation Engine (`bmad-create-epics-and-stories`)

Use this skill immediately after brainstorming (`bmad-brainstorming` + `/grill-me`) and architectural planning (`bmad-prd`, `bmad-architecture`) to generate, shard, and structure Epics and user stories.

---

## 🎯 Universal Scope Resolution (Auto-Routing)

Upon activation, analyze user intent, brainstorm notes, and existing `docs/bmad/planning/` files to select the appropriate mode:

### Mode 1: Greenfield Project Breakdown (0 ➔ 1)
* **Trigger**: Initial project setup or when `epics.md` is empty/new.
* **Actions**:
  1. Read `docs/bmad/planning/prd.md` and `docs/bmad/planning/architecture.md`.
  2. Structure `docs/bmad/planning/epics.md` with Epics 1..N.
  3. Create **Story 1.1 as the Walking Skeleton / Tracer Bullet** (`depends_on: []`, `execution_mode: "AFK"`, minimal end-to-end runnable baseline with integration test).
  4. Shard all subsequent stories into `docs/bmad/stories/story-X.Y-*.md`.

### Mode 2: Major Subsystem / New Epic Addition (1 ➔ N)
* **Trigger**: A new multi-feature initiative, major pipeline, or architectural expansion (e.g. *Orbweaver Epic 19*).
* **Actions**:
  1. Determine next Epic number `X` in `docs/bmad/planning/epics.md`.
  2. Append `## Epic X: <Title>` and requirement mappings to `epics.md`.
  3. Create **Story X.1 as the Subsystem Walking Skeleton** (`depends_on: []`, `execution_mode: "AFK"`).
  4. Create progressive enrichment stories (`story-X.2`, `story-X.3`, ...) with explicit `depends_on: ["story-X.1"]`.

### Mode 3: Single Story Addition
* **Trigger**: An isolated feature, endpoint, or bug fix belonging to an existing Epic.
* **Actions**:
  1. Locate parent Epic in `epics.md` and assign next story number `X.Y`.
  2. Create `docs/bmad/stories/story-X.Y-<name>.md`.
  3. Declare upstream `depends_on` and `execution_mode`.

---

## 🏛️ Deep Module Planning Invariant

Every story MUST plan out a **deep module**:
1. **Simple Public Interface**: Design a small, clean, typed public API (e.g. `Worker.process(task: Task) -> Result`).
2. **Encapsulated Complexity**: Explicitly state what complexity is hidden behind the boundary (e.g. caching, retry mechanisms, socket polling, serialization, disk operations).
3. **Interface-Centric Testing**: Acceptance criteria must assert behavior strictly against the public interface, avoiding brittle tests coupled to internal private plumbing.

---

## 📋 Mandatory Story Specification Standards

Every generated story file in `docs/bmad/stories/` MUST adhere to this exact structure:

```markdown
---
title: "Story X.Y: <Title>"
stage: "BACKLOG"
epic: "Epic X (<Epic Title>)"
depends_on: ["story-X.1"]  # list of prerequisite story slugs, or [] if ready immediately
execution_mode: "AFK"      # "AFK" (autonomous) or "HUMAN_MONITORED" (critical oversight)
---
# Story X.Y: <Title>

## Status: backlog
## Epic: Epic X (<Epic Title>)
## Execution Mode: AFK (Autonomous) | HUMAN_MONITORED (Supervised)
## Dependencies: Upstream stories (`depends_on`)

## Vertical Slice Summary
Explain the end-to-end slice cutting across the stack:
Input Data $\to$ Deep Core Logic $\to$ Persistence/Output $\to$ E2E Integration Test.

## Deep Module Interface & Encapsulation
- **Public Interface Contract**: Define the clean, typed function/class signature exposed to callers.
- **Hidden Complexity**: Detail the internal state, I/O, error recovery, or orchestration hidden behind this interface.

## Architecture & Seams
Mermaid diagram and technical contracts touched by this slice.

## Acceptance Criteria
- [ ] **AC 1 — Public Interface & Core Logic**: Implement the deep module satisfying the typed public contract.
- [ ] **AC 2 — Contracts & Schema**: Type-checked input/output serialization (Pydantic / Zod).
- [ ] **AC 3 — Interface-Centric Test**: Author `tests/test_e2e_*.py` asserting slice behavior via the public interface.
- [ ] **AC 4 — Verification Gate**: `uv run ruff check` and `uv run pytest -q` pass 100% green.
```

---

## 🔄 Concluding Step: Cloud Tracker Synchronization

Immediately after writing story files:
1. Update `docs/bmad/stories/sprint-status.yaml` with the new story entries.
2. Run the tracker synchronization command:
   ```bash
   python3 scripts/sync_trackers.py --update-all
   ```
3. Verify that all Epics and child story tickets are minted in Buganizer and TaskFlow with parent-child links established and IDs stamped into frontmatter.
