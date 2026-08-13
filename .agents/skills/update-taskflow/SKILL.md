---
name: update-taskflow
description: Synchronizes all BMAD epics and stories with Google TaskFlow and Buganizer. Mints parent Epic tickets, links child stories to parent Epics, embeds full Markdown story descriptions with acceptance criteria and in-repo paths, and stamps IDs into story frontmatter. Handles initial creation and mid-project additions.
---

# Update TaskFlow Skill (`update-taskflow`)

Use this skill whenever:
1. **Initial Brainstorming Completes**: You finish brainstorming or creating stories in `docs/bmad/stories/`.
2. **Adding New Stories to Existing Epics**: You create a new story file (e.g. `docs/bmad/stories/story-14.5-*.md`).
3. **Adding a New Epic**: You add a new `## Epic X: <Title>` section in `docs/bmad/planning/epics.md` and create its stories.
4. **Refining Existing Stories**: You update acceptance criteria, technical context, or edge case checklists in an existing story file.
5. **Course Corrections**: You run `bmad-correct-course` and need changes reflected in TaskFlow.

---

## 🚀 Execution Instructions

Run the synchronization command from the repository root:

```bash
python3 scripts/sync_trackers.py --update-all
```

### What It Performs Automatically:
1. **Dynamic Epic Discovery**: Scans `docs/bmad/planning/epics.md`. If a new Epic is found without a Buganizer ticket, it mints a parent `PROJECT` ticket in Component `1326899` with Hotlist `8635284` (TaskFlow Workspace `5949795`).
2. **Dynamic Story Creation**: Scans `docs/bmad/stories/`. For any new story lacking an ID, it mints a `TASK` ticket.
3. **Parent-Child Linking**: Links each story to its parent Epic (`buganizer_add_issue_parent`), ensuring accurate hierarchy in TaskFlow Backlog & Roadmap.
4. **Embeds Rich Markdown Specifications**: Posts the complete User Story description, Technical Context, Acceptance Criteria checklist, and in-repo file path into Buganizer comments.
5. **Idempotent Frontmatter Stamping**: Writes `epic_bug_id`, `buganizer_id`, and `taskflow_goal_id` into the Markdown story files without duplicating existing tickets.
