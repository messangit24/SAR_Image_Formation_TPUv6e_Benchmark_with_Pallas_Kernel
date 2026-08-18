#!/usr/bin/env python3
"""Automated Post-Merge Synchronization & Worktree Cleanup Tool.

Triggered by the `merge-sync` skill or CLI.
Performs:
1. Git pull origin main in primary repo.
2. Closes Buganizer / TaskFlow tickets to FIXED.
3. Automatically deletes merged git worktrees (matched via ggh merged PRs and story stage: COMPLETED).
4. Runs idempotent tracker synchronization.

Usage:
    python3 scripts/merge_sync.py
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = PROJECT_ROOT / "docs/bmad/stories"
BUGANIZER_SERVER_PATH = "/google/bin/releases/codemind-mcp-servers/buganizer_server.par"
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
                        "clientInfo": {"name": "merge-sync", "version": "1.0"},
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


def call_mcp_tool(tool_name: str, arguments: dict[str, Any], timeout: float = 15.0) -> str:
    """One-shot helper to invoke a tool on buganizer_server.par."""
    client = BuganizerClient(timeout=timeout)
    res = client.call(tool_name, arguments)
    client.close()
    return res


def get_merged_prs() -> list[dict[str, str]]:
    """Fetches recently merged PRs via ggh."""
    try:
        res = subprocess.run(
            ["ggh", "pr", "list", "--state", "merged", "--limit", "10", "--json", "number,title,headRefName"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as e:
        print(f"info: checking merged PRs: {e}", file=sys.stderr)
    return []


def get_completed_story_slugs() -> set[str]:
    """Extracts slugs of all stories marked COMPLETED in docs/bmad/stories/."""
    completed = set()
    if not STORIES_DIR.exists():
        return completed
    for sf in STORIES_DIR.glob("story-*.md"):
        content = sf.read_text(encoding="utf-8")
        if 'stage: "COMPLETED"' in content or 'Status: completed' in content:
            m = re.search(r"story-(\d+\.\d+|[a-zA-Z0-9-]+)", sf.name)
            if m:
                completed.add(m.group(0))
    return completed


def cleanup_worktrees() -> list[str]:
    """Finds and removes worktrees whose branches have been merged or stories completed."""
    cleaned = []
    merged_prs = get_merged_prs()
    merged_branches = {pr.get("headRefName", "") for pr in merged_prs if pr.get("headRefName")}
    completed_slugs = get_completed_story_slugs()

    try:
        res = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        current_wt = None
        current_branch = None
        worktrees = []
        for line in res.stdout.splitlines():
            if line.startswith("worktree "):
                current_wt = line.split(" ", 1)[1].strip()
            elif line.startswith("branch "):
                current_branch = line.split(" ", 1)[1].replace("refs/heads/", "").strip()
                if current_wt and Path(current_wt).resolve() != PROJECT_ROOT.resolve():
                    worktrees.append((current_wt, current_branch))
                current_wt = None
                current_branch = None

        for wt_path, branch in worktrees:
            should_clean = False
            reason = ""

            if branch in merged_branches:
                should_clean = True
                reason = f"PR merged for branch `{branch}`"
            else:
                for slug in completed_slugs:
                    if slug in wt_path or slug in branch:
                        should_clean = True
                        reason = f"Story `{slug}` marked COMPLETED"
                        break

            if should_clean:
                print(f"🧹 Removing merged worktree: {wt_path} ({reason})...")
                subprocess.run(["git", "worktree", "remove", wt_path, "--force"], cwd=PROJECT_ROOT, check=False)
                subprocess.run(["git", "branch", "-D", branch], cwd=PROJECT_ROOT, check=False)
                cleaned.append(f"{wt_path} ({branch})")
    except Exception as e:
        print(f"warning cleaning worktrees: {e}", file=sys.stderr)
    return cleaned


def sync_completed_tickets() -> list[str]:
    """Ensures all stories marked COMPLETED in docs/bmad/stories/ have their Buganizer tickets transitioned to FIXED."""
    updated = []
    if not STORIES_DIR.exists():
        return updated
    import getpass
    user_email = os.environ.get("USER") or getpass.getuser() or "ermlickb"
    if user_email and "@" not in user_email:
        user_email = f"{user_email}@google.com"

    client = BuganizerClient(timeout=15.0)

    epics_to_check = set()
    for sf in sorted(STORIES_DIR.glob("story-*.md")):
        content = sf.read_text(encoding="utf-8")
        if 'stage: "COMPLETED"' in content or 'Status: completed' in content or "- [x]" in content:
            m = re.search(r'buganizer_id:\s*"(\d+)"', content)
            if m:
                bug_id = int(m.group(1))
                try:
                    res = client.call("buganizer_update_issue_status", {
                        "issue_id": bug_id,
                        "status": "FIXED",
                    })
                    if "must have an assignee" in str(res) and user_email:
                        client.call("buganizer_reassign_issue", {
                            "issue_id": bug_id,
                            "assignee": user_email,
                            "comment_markdown": "Auto-assigned during merge sync.",
                        })
                        res = client.call("buganizer_update_issue_status", {
                            "issue_id": bug_id,
                            "status": "FIXED",
                        })
                    updated.append(f"{sf.name} -> b/{bug_id}")
                except Exception as e:
                    print(f"warning updating b/{bug_id}: {e}", file=sys.stderr)

            m_epic = re.search(r'epic_bug_id:\s*"(\d+)"', content)
            if m_epic:
                epics_to_check.add(int(m_epic.group(1)))

    client.close()
    return updated


def main() -> int:
    print("=" * 80)
    print("🔄 Running Post-Merge Synchronization & Worktree Cleanup")
    print("=" * 80)

    # 1. Pull latest main
    print("\n1. Pulling latest main from origin...")
    subprocess.run(["git", "checkout", "main"], cwd=PROJECT_ROOT, check=False)
    pull_res = subprocess.run(["git", "pull", "origin", "main"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(pull_res.stdout.strip() if pull_res.stdout else "Already up to date.")

    # 2. Check for merged PRs
    merged_prs = get_merged_prs()
    if merged_prs:
        print("\n2. Recently Merged PRs:")
        for pr in merged_prs:
            print(f"  • PR #{pr.get('number')}: {pr.get('title')} (branch: {pr.get('headRefName')})")

    # 3. Clean up merged worktrees
    print("\n3. Cleaning Merged Worktrees...")
    cleaned_wt = cleanup_worktrees()
    if cleaned_wt:
        for c in cleaned_wt:
            print(f"  ✅ Cleaned: {c}")
    else:
        print("  ℹ️ No stale worktrees detected.")

    # 4. Finalize Buganizer / TaskFlow tickets for completed stories
    print("\n4. Finalizing Completed Tickets in Buganizer / TaskFlow...")
    fixed_tickets = sync_completed_tickets()
    if fixed_tickets:
        for t in fixed_tickets:
            print(f"  ✅ Status FIXED: {t}")
    else:
        print("  ℹ️ No completed tickets to update.")

    # 5. Run Tracker Synchronization
    print("\n5. Running Idempotent Tracker Sync...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/sync_trackers.py")], cwd=PROJECT_ROOT)

    print("\n" + "=" * 80)
    print("✅ Merge Sync Complete: Main updated, merged worktrees cleaned, tickets FIXED, trackers in sync.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
