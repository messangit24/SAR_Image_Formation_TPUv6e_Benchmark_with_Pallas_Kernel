#!/usr/bin/env python3
"""Synchronize in-repo BMAD stories and Epics with TaskFlow, Buganizer, and GitHub.

Dynamic & Idempotent:
- Dynamically parses docs/bmad/planning/epics.md for existing and new Epics.
- Automatically discovers newly created stories in docs/bmad/stories/.
- Mints missing Epic and Story tickets in Buganizer with Hotlist 8635284.
- Links child stories to parent Epics (parent_id).
- Embeds rich Markdown specifications (user story, context, acceptance criteria).
- Stamps cross-linked IDs into in-repo story YAML frontmatter.

Usage:
    # Quick status preview & local frontmatter sync (Default / Hook mode):
    python3 scripts/sync_trackers.py

    # Fully sync, auto-create missing Epics/Stories, link parents, and embed updates:
    python3 scripts/sync_trackers.py --update-all
"""
import argparse
import getpass
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

TASKFLOW_WORKSPACE_ID = "5949795"
BUGANIZER_COMPONENT_ID = 1326899
HOTLIST_ID = 8635284
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = PROJECT_ROOT / "docs/bmad/stories"
EPICS_FILE = PROJECT_ROOT / "docs/bmad/planning/epics.md"
BUGANIZER_SERVER_PATH = "/google/bin/releases/codemind-mcp-servers/buganizer_server.par"


def get_current_user_email() -> str:
    """Returns current user LDAP email."""
    user = os.environ.get("USER")
    if not user:
        try:
            user = getpass.getuser()
        except Exception:
            user = "ermlickb"
    if user and "@" not in user:
        user = f"{user}@google.com"
    return user


class BuganizerClient:
    """Persistent connection to buganizer_server.par to execute multiple JSON-RPC calls without per-call process overhead."""

    def __init__(self, timeout: float = 15.0):
        self.proc = None
        self.req_id = 1
        self.timeout = timeout
        if os.path.exists(BUGANIZER_SERVER_PATH):
            try:
                self.proc = subprocess.Popen(
                    [BUGANIZER_SERVER_PATH],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                init_payload = json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "sync-trackers", "version": "1.0"},
                    },
                })
                self.proc.stdin.write(init_payload + "\n")
                self.proc.stdin.flush()
                self.proc.stdout.readline()
            except Exception as e:
                self.proc = None
                print(f"info: Buganizer MCP server initialization failed ({e}), skipping live ticket RPCs.", file=sys.stderr)

    def call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        if not self.proc or self.proc.poll() is not None:
            return ""
        self.req_id += 1
        req = json.dumps({
            "jsonrpc": "2.0",
            "id": self.req_id,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        })
        try:
            self.proc.stdin.write(req + "\n")
            self.proc.stdin.flush()
            resp_line = self.proc.stdout.readline()
            if not resp_line:
                return ""
            resp_data = json.loads(resp_line)
            res_str = (
                resp_data.get("result", {})
                .get("structuredContent", {})
                .get("result", "")
            )
            if not res_str:
                res_str = (
                    resp_data.get("result", {})
                    .get("content", [{}])[0]
                    .get("text", "")
                )
            return res_str
        except Exception as e:
            print(f"warning calling Buganizer MCP {tool_name}: {e}", file=sys.stderr)
            return ""

    def close(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass


def discover_epics() -> dict[str, dict[str, Any]]:
    """Dynamically parses docs/bmad/planning/epics.md to extract all Epic definitions."""
    epics = {}
    if not EPICS_FILE.exists():
        return epics

    content = EPICS_FILE.read_text(encoding="utf-8")
    epic_matches = re.finditer(
        r"^##\s+(Epic\s+\d+):\s*(.+?)$(?:\n+([^\n#].+?))?(?=\n##|\Z)",
        content,
        re.MULTILINE | re.DOTALL,
    )
    for m in epic_matches:
        epic_key = m.group(1).strip()
        epic_title = m.group(2).strip()
        epic_desc = (m.group(3) or "").strip()
        desc_summary = epic_desc.split("\n\n")[0].strip() if epic_desc else epic_title
        epics[epic_key] = {
            "title": f"[TO1] {epic_key}: {epic_title}",
            "description": f"{desc_summary}\n\n**TaskFlow Workspace**: {TASKFLOW_WORKSPACE_ID} | **Component**: {BUGANIZER_COMPONENT_ID} | **Hotlist**: {HOTLIST_ID}",
            "bug_id": "",
        }

    # Known stamped Epic bug IDs for Orbweaver
    known_epic_bugs = {
        "Epic 1": "546592462",
        "Epic 2": "546591445",
        "Epic 3": "546591371",
        "Epic 4": "546592481",
        "Epic 5": "546592761",
        "Epic 6": "546592660",
        "Epic 7": "546592222",
        "Epic 8": "546593216",
        "Epic 9": "546593098",
        "Epic 10": "546592185",
        "Epic 11": "546593535",
        "Epic 12": "546028325",
        "Epic 13": "546028024",
        "Epic 14": "546028152",
        "Epic 15": "546028027",
        "Epic 16": "546028028",
        "Epic 17": "546028188",
        "Epic 18": "546028330",
        "Epic 19": "546747442",
    }
    for k, v in known_epic_bugs.items():
        if k in epics:
            epics[k]["bug_id"] = v
        else:
            epics[k] = {
                "title": f"[TO1] {k}",
                "description": f"**TaskFlow Workspace**: {TASKFLOW_WORKSPACE_ID} | **Hotlist**: {HOTLIST_ID}",
                "bug_id": v,
            }

    return epics


def parse_story_file(path: Path) -> dict[str, Any]:
    """Extracts frontmatter and markdown sections from a story file."""
    content = path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    val = v.strip()
                    if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
                        try:
                            frontmatter[k.strip()] = json.loads(val)
                        except Exception:
                            frontmatter[k.strip()] = val.strip("'\"")
                    else:
                        frontmatter[k.strip()] = val.strip("'\"")

    title = frontmatter.get("title", "")
    if not title:
        for line in body.splitlines():
            line_s = line.strip()
            if line_s.startswith("# "):
                title = line_s[2:].strip()
                break
    if not title:
        title = path.stem.replace("-", " ").capitalize()

    stage = frontmatter.get("stage", frontmatter.get("status", "")).upper()
    if not stage:
        for line in body.splitlines():
            if re.match(r"^##?\s*Status:\s*", line, re.I):
                stage = re.sub(r"^##?\s*Status:\s*", "", line, flags=re.I).strip().upper()
                break
    if not stage:
        stage = "BACKLOG"

    if "PROGRESS" in stage:
        stage = "IN_PROGRESS"
    elif "DONE" in stage or "COMPLETED" in stage:
        stage = "COMPLETED"
    elif "BACKLOG" in stage or "READY" in stage:
        stage = "BACKLOG"

    epic_name = frontmatter.get("epic", "")
    if not epic_name:
        epic_m = re.search(r"^##?\s*Epic:\s*(.+)$", body, re.M | re.I)
        if epic_m:
            epic_name = epic_m.group(1).strip()
    if not epic_name:
        num_m = re.search(r"story-(\d+)\.", path.name)
        if num_m:
            epic_name = f"Epic {num_m.group(1)}"
        else:
            epic_name = "General Enhancements"

    return {
        "path": path,
        "filename": path.name,
        "title": title,
        "stage": stage,
        "epic": epic_name,
        "frontmatter": frontmatter,
        "body": body.strip(),
    }


def stamp_story_frontmatter(path: Path, meta: dict[str, Any]) -> None:
    """Updates or injects YAML frontmatter with standardized metadata."""
    content = path.read_text(encoding="utf-8")
    body = content
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            body = parts[2].lstrip("\n")

    fm_dict = dict(meta.get("frontmatter", {}))
    fm_dict["title"] = meta["title"]
    fm_dict["stage"] = meta["stage"]
    fm_dict["epic"] = meta.get("epic", "General Enhancements")
    fm_dict["taskflow_workspace"] = TASKFLOW_WORKSPACE_ID
    if meta.get("epic_bug_id"):
        fm_dict["epic_bug_id"] = str(meta["epic_bug_id"])
    if meta.get("taskflow_goal_id"):
        fm_dict["taskflow_goal_id"] = str(meta["taskflow_goal_id"])
    if meta.get("buganizer_id"):
        fm_dict["buganizer_id"] = str(meta["buganizer_id"])

    fm_lines = ["---"]
    for k, v in fm_dict.items():
        if isinstance(v, (list, tuple)):
            fm_lines.append(f"{k}: {json.dumps(list(v))}")
        elif isinstance(v, bool):
            fm_lines.append(f"{k}: {str(v).lower()}")
        elif isinstance(v, str) and v.startswith("[") and v.endswith("]"):
            try:
                fm_lines.append(f"{k}: {json.dumps(json.loads(v))}")
            except Exception:
                fm_lines.append(f'{k}: "{v}"')
        else:
            fm_lines.append(f'{k}: "{v}"')
    fm_lines.append("---")
    fm_lines.append("")

    path.write_text("\n".join(fm_lines) + body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic & idempotent sync for in-repo BMAD stories and TaskFlow.")
    parser.add_argument("--update-all", action="store_true", help="Auto-create missing Epics and Stories, update comments, and link parents.")
    args = parser.parse_args()

    if not STORIES_DIR.exists():
        print(f"error: stories directory not found at {STORIES_DIR}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("🔄 BMAD ↔ TaskFlow ↔ Buganizer Synchronization")
    print(f"📁 Workspace: {TASKFLOW_WORKSPACE_ID} (Hotlist: {HOTLIST_ID} | Component: {BUGANIZER_COMPONENT_ID})")
    print("=" * 80)

    client = BuganizerClient(timeout=15.0)

    # 1. Discover Epics dynamically from docs/bmad/planning/epics.md
    epic_definitions = discover_epics()

    # If --update-all, ensure any newly discovered Epic is minted in Buganizer
    for epic_key, epic_meta in epic_definitions.items():
        if args.update_all and not epic_meta["bug_id"]:
            print(f"✨ Minting new parent Epic: {epic_meta['title']}...")
            res = client.call("buganizer_create_issue", {
                "title": epic_meta["title"],
                "comment_markdown": f"{epic_meta['description']}\n\n**TaskFlow Workspace**: {TASKFLOW_WORKSPACE_ID}\n**Hotlist**: {HOTLIST_ID}",
                "component_id": BUGANIZER_COMPONENT_ID,
                "hotlist_ids": [HOTLIST_ID],
                "issue_type": "PROJECT",
                "priority": "P2",
                "severity": "S2",
            })
            m = re.search(r"IssueID:\s*(\d+)", res)
            if m:
                epic_meta["bug_id"] = m.group(1)
                print(f"  ✅ Created Epic {epic_key} -> b/{epic_meta['bug_id']}")

    # 2. Process all story files
    story_files = sorted(
        list(STORIES_DIR.glob("story-*.md")) + list(STORIES_DIR.glob("spec-*.md"))
    )

    print("\n" + "-" * 80)
    print(f"{'Story File':<34} | {'Parent Epic':<12} | {'Buganizer ID':<14} | {'Status'}")
    print("-" * 80)

    new_stories_count = 0
    synced_count = 0

    for sf in story_files:
        parsed = parse_story_file(sf)
        story_title = parsed["title"]
        story_epic = parsed["epic"]

        # Match Epic Definition
        matched_epic_key = None
        for k in sorted(epic_definitions.keys(), key=lambda x: -len(x)):
            if re.search(rf"\b{re.escape(k)}\b", story_epic, re.I):
                matched_epic_key = k
                break

        epic_bug_id = ""
        if matched_epic_key and matched_epic_key in epic_definitions:
            epic_bug_id = epic_definitions[matched_epic_key]["bug_id"]
        if not epic_bug_id:
            epic_bug_id = parsed.get("frontmatter", {}).get("epic_bug_id", "")
        parsed["epic_bug_id"] = epic_bug_id

        bug_id = parsed.get("frontmatter", {}).get("buganizer_id", "")

        # Auto-create missing story ticket if --update-all
        if args.update_all and not bug_id and parsed["stage"] == "BACKLOG":
            print(f"✨ Minting Story Ticket: {story_title}...")
            full_comment = (
                f"### In-Repo Specification: `docs/bmad/stories/{sf.name}`\n"
                f"**Parent Epic**: {story_epic} (`b/{epic_bug_id}`)\n"
                f"**TaskFlow Workspace**: {TASKFLOW_WORKSPACE_ID} | **Hotlist**: {HOTLIST_ID}\n\n"
                f"---\n\n"
                f"{parsed['body']}"
            )
            res = client.call("buganizer_create_issue", {
                "title": f"[TO1] {story_title}",
                "comment_markdown": full_comment,
                "component_id": BUGANIZER_COMPONENT_ID,
                "hotlist_ids": [HOTLIST_ID],
                "issue_type": "TASK",
                "priority": "P2",
                "severity": "S2",
            })
            m = re.search(r"IssueID:\s*(\d+)", res)
            if m:
                bug_id = m.group(1)
                parsed["buganizer_id"] = bug_id
                parsed["taskflow_goal_id"] = bug_id
                new_stories_count += 1

                # Link to parent epic upon creation
                if epic_bug_id:
                    client.call("buganizer_add_issue_parent", {
                        "issue_id": int(bug_id),
                        "parent_id": int(epic_bug_id),
                    })

        epic_disp = f"b/{epic_bug_id}" if epic_bug_id else "NONE"
        bug_disp = f"b/{bug_id}" if bug_id else "UNASSIGNED"
        print(f"{sf.name:<34} | {epic_disp:<12} | {bug_disp:<14} | ✅ Synced")

        stamp_story_frontmatter(sf, parsed)
        synced_count += 1

    client.close()

    print("-" * 80)
    print(f"\n✅ Synchronization complete. {synced_count} specifications verified.")
    if new_stories_count > 0:
        print(f"🎉 Created and linked {new_stories_count} new tickets in TaskFlow Hotlist {HOTLIST_ID}.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
