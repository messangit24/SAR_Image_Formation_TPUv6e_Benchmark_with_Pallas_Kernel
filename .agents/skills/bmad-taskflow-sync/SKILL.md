---
name: bmad-taskflow-sync
description: Synchronizes newly brainstormed BMAD epics and stories with Google TaskFlow and Buganizer. Mints parent Epic tickets, embeds full Markdown story descriptions and acceptance criteria, links stories to parent Epics, and stamps IDs into in-repo story frontmatter.
---

# BMAD TaskFlow & Buganizer Synchronization Skill

This skill synchronizes in-repo BMAD epics and stories (`docs/bmad/planning/epics.md` and `docs/bmad/stories/`) with **Google TaskFlow** and **Buganizer**.

---

## 🎯 What This Skill Does

1. **Epic & Story Hierarchy**:
   - Creates parent **Epic tickets** (`PROJECT` / `GOAL` type) in TaskFlow/Buganizer under Component `1326899` with Hotlist `8635284`.
   - Links each Story ticket to its parent Epic (`buganizer_add_issue_parent`), creating the proper tree structure on the TaskFlow Roadmap & Backlog.

2. **Rich Story Content**:
   - Embeds the **full Markdown specification** into Buganizer comments (Story Description, Architecture Context, and Acceptance Criteria checklists), alongside the repository file link.

3. **Frontmatter Stamping**:
   - Writes `taskflow_goal_id`, `buganizer_id`, `epic_bug_id`, and `stage` back into each story's YAML frontmatter.

---

## 🔄 How to Run

```bash
# Preview or sync existing items:
python3 scripts/sync_trackers.py

# Create all missing Epics and Stories, link parents, and stamp frontmatter:
python3 scripts/sync_trackers.py --create-missing
```
