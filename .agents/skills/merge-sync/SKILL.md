---
name: merge-sync
description: Merges open GitHub/Depot PRs (treating manual execution as explicit human approval), pulls latest main, marks stories completed and Buganizer tickets FIXED, cleans up temporary git worktrees, and updates TaskFlow.
---

# Post-Merge Synchronization & Auto-Merge Skill (`merge-sync`)

Use this skill whenever:
1. You are ready to merge open Pull Request(s) and finalize the story/epic lifecycle.
2. The user says "merge and sync", "merge PR", "sync main", "post merge", "/merge-sync", or "clean merged worktrees".
3. **Explicit Human Approval**: Running this skill manually signals explicit human approval to merge the open Pull Request(s).

---

## 🚀 Execution Instructions

Run the synchronization tool from the repository root:

```bash
python3 scripts/merge_sync.py
```

### What It Performs Automatically:
1. **Merges Open Pull Requests**: Takes manual execution as explicit human approval, running `ggh pr merge` (squash merge with auto-delete remote branch).
2. **Pulls Latest `main`**: Switches to `main` branch and pulls the newest commits from `origin/main`.
3. **Finalizes Buganizer / TaskFlow Tickets**:
   - Transitions completed story and Epic tickets to `FIXED` (auto-reassigning to active LDAP if previously unassigned).
   - Marks completed stories `- [x]` in `docs/bmad/stories/`.
4. **Cleans Merged Worktrees**: Automatically deletes temporary git worktrees (`../<repo>-<slug>`) whose branches have been merged into `main`.
5. **Idempotent Frontmatter Sync**: Runs `scripts/sync_trackers.py` to ensure local repository files and TaskFlow stay in 100% synchronization.
