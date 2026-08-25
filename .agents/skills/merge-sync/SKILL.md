---
name: merge-sync
description: Merges the active worktree branch's open GitHub/Depot PR (treating manual execution as explicit human approval without disturbing concurrent workers), pulls latest main, marks the story completed and Buganizer ticket FIXED, cleans up the worktree, and updates TaskFlow.
---

# Post-Merge Synchronization & Auto-Merge Skill (`merge-sync`)

Use this skill whenever:
1. You are ready to merge the Pull Request for the **active story/worktree** and finalize the lifecycle.
2. The user says "merge and sync", "merge PR", "sync main", "post merge", "/merge-sync", or "clean merged worktrees".
3. **Explicit Human Approval**: Running this skill manually signals explicit human approval to merge the open Pull Request for the active worktree branch.

---

## 🛑 Concurrency Protection Invariant

`merge-sync` defaults to merging **ONLY the Pull Request matching the active worktree branch** (or specified `--branch` / `--story`). It **NEVER** merges unrelated open PRs belonging to other concurrent workers operating in parallel worktrees.

---

## 🚀 Execution Instructions

Run the synchronization tool from the active worktree or repository root:

```bash
# Auto-detects and merges the PR for the current active worktree branch:
python3 scripts/merge_sync.py

# Or explicitly specify a target branch / story:
python3 scripts/merge_sync.py --branch feat/story-21.6
python3 scripts/merge_sync.py --story docs/bmad/stories/story-21.6-web-mercator-pyramid.md

# Operator override to merge ALL open PRs across the repository:
python3 scripts/merge_sync.py --all
```

### What It Performs Automatically:
1. **Merges Active Branch Pull Request**: Matches open PRs against the current worktree branch (or specified `--branch` / `--story`), executing `ggh pr merge` (squash merge with auto-delete remote branch). Open PRs from other workers remain untouched.
2. **Pulls Latest `main`**: Switches to `main` in the primary repository and pulls the newest commits from `origin/main`.
3. **Finalizes Buganizer / TaskFlow Tickets**:
   - Transitions the completed story's ticket to `FIXED` (auto-reassigning to active LDAP if unassigned).
   - Marks the completed story `- [x]` in `docs/bmad/stories/`.
4. **Cleans Merged Worktrees**: Automatically deletes the temporary git worktree (`../<repo>-<slug>`) and branch for the merged story.
5. **Idempotent Frontmatter Sync**: Runs `scripts/sync_trackers.py` to ensure local repository files and TaskFlow stay in 100% synchronization.
