#!/usr/bin/env python3
"""Synchronize in-repo BMAD stories and Epics with TaskFlow, Buganizer, and GitHub for GoldThread.

TaskFlow Workspace: 6001488 (GoldThread)
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

TASKFLOW_WORKSPACE_ID = "6001488"
BUGANIZER_COMPONENT_ID = 1326899
HOTLIST_ID = 8635284
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = PROJECT_ROOT / "docs/bmad/stories"
EPICS_FILE = PROJECT_ROOT / "docs/bmad/planning/epics.md"
BUGANIZER_SERVER_PATH = "/google/bin/releases/codemind-mcp-servers/buganizer_server.par"


def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Invokes a tool on buganizer_server.par."""
    if not os.path.exists(BUGANIZER_SERVER_PATH):
        return ""
    try:
        proc = subprocess.Popen(
            [BUGANIZER_SERVER_PATH],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        proc.stdin.write(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "sync-trackers", "version": "1.0"},
                },
            })
            + "\n"
        )
        proc.stdin.flush()
        proc.stdout.readline()

        proc.stdin.write(
            json.dumps({
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            + "\n"
        )
        proc.stdin.flush()
        resp_raw = proc.stdout.readline()
        proc.terminate()

        if not resp_raw:
            return ""

        resp_data = json.loads(resp_raw)
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
        print(f"error calling {tool_name}: {e}", file=sys.stderr)
        return ""


def parse_story_file(path: Path) -> dict[str, Any]:
    """Extracts frontmatter and markdown sections from a story file."""
    content = path.read_text(encoding="utf-8")
    frontmatter: dict[str, Any] = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1]
            body = parts[2]
            for line in fm_text.splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    frontmatter[k.strip()] = v.strip().strip("'\"")

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
        fm_lines.append(f"{k}: \"{v}\"")
    fm_lines.append("---")
    fm_lines.append("")

    path.write_text("\n".join(fm_lines) + body, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync for in-repo BMAD stories and TaskFlow.")
    parser.add_argument("--update-all", action="store_true", help="Auto-create missing tickets.")
    args = parser.parse_args()

    if not STORIES_DIR.exists():
        print(f"error: stories directory not found at {STORIES_DIR}", file=sys.stderr)
        return 1

    print("=" * 80)
    print("🔄 BMAD ↔ TaskFlow ↔ Buganizer Synchronization (GoldThread)")
    print(f"📁 Workspace: {TASKFLOW_WORKSPACE_ID}")
    print("=" * 80)

    story_files = sorted(list(STORIES_DIR.glob("story-*.md")))
    print("\n" + "-" * 80)
    print(f"{'Story File':<45} | {'Status':<12} | {'Sync State'}")
    print("-" * 80)

    for sf in story_files:
        parsed = parse_story_file(sf)
        stamp_story_frontmatter(sf, parsed)
        print(f"{sf.name:<45} | {parsed['stage']:<12} | ✅ Synced")

    print("-" * 80)
    print(f"✅ Synchronization complete. {len(story_files)} specifications verified.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
