# GoldThread AI-SDLC & BMAD Workflow Guide

A complete guide to feature brainstorming, architecture planning, story creation, mid-project evolution, and automated TaskFlow synchronization for developers and AI agents working on **GoldThread / TPUbench (AGO TO2)**.

---

## 📁 Standard File & Folder Layout

```
docs/bmad/
├── README.md                      # Index of all active planning documents
├── WORKFLOW.md                    # This lifecycle & evolution guide
├── planning/
│   ├── prd.md                     # Product Requirements Document (SAR, MODIS, GKE, TPU)
│   ├── architecture.md            # Technical Architecture Spine (contracts, DB, schemas, APIs)
│   └── epics.md                   # Epic breakdown and story-to-requirement mapping
└── stories/
    ├── sprint-status.yaml         # Sprint state machine (backlog, in-progress, review, done)
    ├── story-modis-1.1-*.md       # Sharded, independently testable story specifications
    └── story-infra-1.1-*.md
```

---

## 🧠 Phase 1: The Brainstorming Progression (Skills ➔ Output Files)

| Step | BMAD Skill | Primary Output File | Purpose |
| :--- | :--- | :--- | :--- |
| **1. Brainstorm** | `bmad-brainstorming` / `bmad-forge-idea` | `docs/bmad/planning/research/` | Explore requirements, SAR/MODIS algorithms, JAX TPU optimizations, and edge cases. |
| **2. PRD** | `bmad-prd` | `docs/bmad/planning/prd.md` | Ground into formal Functional (FR) & Non-Functional (NFR) requirements. |
| **3. Architecture** | `bmad-architecture` | `docs/bmad/planning/architecture.md` | Define JAX TPU kernels, GKE worker orchestration, Planck inversion, and invariants. |
| **4. Epics & Stories** | `bmad-create-epics-and-stories` | `docs/bmad/planning/epics.md`<br/>`docs/bmad/stories/story-*.md` | Break architecture into Epics and sharded, independently testable story files. |
| **5. Sprint Plan** | `bmad-sprint-planning` | `docs/bmad/stories/sprint-status.yaml` | Initialize sprint status and dependency order across stories. |
| **6. Sync** | **`update-taskflow`** | **Google TaskFlow & Buganizer** | **The mandatory concluding step**: Mints Epics, mints Stories, links hierarchy in TaskFlow Workspace 6001488. |

---

## 🔄 Phase 2: Mid-Project Evolution (Adding & Modifying Epics & Stories)

| Scenario | BMAD Skill | Action | Concluding Step |
| :--- | :--- | :--- | :--- |
| **Add 1 New Story** | `bmad-create-story` | Creates `docs/bmad/stories/story-X.Y-*.md` with `epic: 'Epic X'` | Run `update-taskflow` (mints ticket, links to parent Epic) |
| **Add a Brand New Epic** | `bmad-create-epics-and-stories` | Adds `## Epic X: <Title>` in `epics.md` and creates story files | Run `update-taskflow` (mints Epic, mints child stories, links hierarchy) |
| **Refine Acceptance Criteria** | Edit Story Markdown | Modifies checklist or technical context in `docs/bmad/stories/` | Run `update-taskflow` (updates Buganizer comments) |
| **Mid-Sprint Course Correction** | `bmad-correct-course` | Re-aligns PRD/Arch, archives obsolete stories, creates replacements | Run `update-taskflow` (syncs ticket statuses) |
| **Document Sharding** | `bmad-shard-doc` | Splits large RFC or design doc into executable stories | Run `update-taskflow` |

---

## 🛠️ Phase 3: The 7-Step Story Development Loop

1. **Scaffold Worktree**: `python3 scripts/dev_story.py --story docs/bmad/stories/<story>.md --setup-worktree`
2. **Propose Plan & Wait for Approval**: Interactive alignment between human and agent.
3. **Worktree Implementation**: Code in `../goldthread-<story>` with zero dirty `main`.
4. **Verification Gates**: `uv run ruff check src tests scripts` && `uv run pytest -q`.
5. **Adversarial Red-Team Review**: Scan diff for numerical stability, float precision, and edge cases.
6. **PR Submission**: Push branch, open Issue and PR via `ggh`, and update Buganizer ticket.
7. **Post-Merge**: Merge PR, run `python3 scripts/merge_sync.py` to finalize tickets and clean worktrees.
