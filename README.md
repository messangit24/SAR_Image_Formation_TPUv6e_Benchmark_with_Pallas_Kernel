# AI-Native Software Development Life Cycle (AI-SDLC) Starter Pack

> **Last Updated:** 2026-08-18  
> **Status:** Active Reference

A turnkey starter repository and operating template for teams building high-velocity software with autonomous AI agents (Gemini, Claude, subagents) using the **AI-Native Software Development Life Cycle (AI-SDLC)**.

---

## 🚀 What's Inside

- **`AGENTS.md`**: The repository operating manual with the canonical 7-step AI-Native Dev Loop, TDD mandate, `/owl` mode, `/grill-me` alignment, Worktree Resource Isolation patterns, and the 3-Agent Session Freshness rule.
- **`docs/bmad/`**: Full BMAD planning suite:
  - `WORKFLOW.md`: Complete Greenfield (0 ➔ 1) and Evolutionary (1 ➔ N) lifecycle guide.
  - `planning/prd.md`: Canonical Product Requirements Document.
  - `planning/architecture.md`: System architecture spine and invariants.
  - `planning/epics.md`: Epic breakdowns and story mapping.
  - `stories/`: Active sprint status and gold-standard vertical slice user stories.
- **`scripts/`**:
  - `dev_story.py`: Worktree scaffolder with dynamic resource allocation (isolated `DATABASE_URL` and `PORT` offsets).
  - `sync_trackers.py`: Idempotent TaskFlow and Buganizer parent-child ticket synchronizer with persistent `BuganizerClient`.
  - `merge_sync.py`: Post-merge sync tool that auto-reassigns tickets, finalizes them to `FIXED`, and deletes merged worktrees.
- **`.agents/skills/`**: The curated 14-skill suite (`brainstorming`, `prd`, `architecture`, `create-epics-and-stories`, `update-taskflow`, `dev-story`, `code-review`, `write-python`, `write-typescript`, `merge-sync`, `owl`, `dev-epic-or-story-owl-afk`, `grill-me`, `browser`).




- **`.github/`**: Standalone AI Code Reviewer GitHub Action (`.github/workflows/gemini-pr-review.yml`) and multi-provider review engine (`.github/pr-reviewer/`).

---

## 🔒 Confidentiality & Proprietary Skills Notice

> ⚠️ **CONFIDENTIALITY NOTICE**: The **`owl`** skill (`.agents/skills/owl/`) and its orchestration routines are **Google-Internal and Confidential**. They contain proprietary multi-agent inference-scaling pipelines and **CANNOT be shared externally, open-sourced, or published to public repositories**.

---

## 🔑 GitHub AI Reviewer Setup (Google AI Studio Key)

1. Generate a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey).
2. In your repository on GitHub / Depot, navigate to **Settings ➔ Secrets and variables ➔ Actions**.
3. Add a repository secret named **`GEMINI_API_KEY`** and paste your API key.
4. Pull Requests will automatically receive inline AI code reviews and automated verifications!
