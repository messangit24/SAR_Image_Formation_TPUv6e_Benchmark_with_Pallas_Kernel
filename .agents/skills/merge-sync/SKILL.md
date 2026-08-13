---
name: merge-sync
description: Synchronizes local main branch, TaskFlow, and Buganizer after a GitHub PR is merged. Pulls latest main, triggers post-merge hooks to mark stories completed and tickets FIXED, cleans up temporary git worktrees, and updates frontmatter.
---

# Post-Merge Synchronization Skill (`merge-sync`)

Use this skill whenever:
1. You just merged a Pull Request on GitHub or Depot.
2. The user says "sync main", "post merge", "/merge-sync", or "clean merged worktrees".
3. You want to ensure the primary workspace reflects the latest merged state and all TaskFlow tickets are finalized.

---

## 🚀 Execution Instructions

Run the synchronization tool from the repository root:

```bash
python3 scripts/merge_sync.py
```

### What It Performs Automatically:
1. **Pulls Latest `main`**: Switches to the `main` branch and pulls the newest commits from `origin/main`.
2. **Executes Post-Merge Hook**: Triggers `.git/hooks/post-merge` logic:
   - Marks completed story tasks `- [x]` in `docs/bmad/stories/`.
   - Transitions Buganizer / TaskFlow tickets to `FIXED`.
3. **Cleans Merged Worktrees**: Automatically deletes temporary git worktrees (`../orbweaver-<slug>`) whose branches have been merged into `main`.
4. **Idempotent Frontmatter Sync**: Runs `scripts/sync_trackers.py` to ensure local repository files and TaskFlow stay in 100% synchronization.
