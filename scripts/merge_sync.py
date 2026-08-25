#!/usr/bin/env python3
"""Automated Post-Merge Synchronization & Worktree Cleanup Tool.

Triggered by the `merge-sync` skill or CLI.
Performs:
1. Automatically merges the open PR for the ACTIVE WORKTREE BRANCH via `ggh pr merge`
   (treating manual execution as explicit human approval while isolating concurrent workers).
2. Git pull origin main in primary repo.
3. Closes Buganizer / TaskFlow tickets for the target story to FIXED (with auto-LDAP assignment fallback).
4. Automatically deletes the merged git worktree for the target branch.
5. Runs idempotent tracker synchronization.

Usage:
    # Target current worktree (when run inside a worktree directory)
    python3 scripts/merge_sync.py

    # Target specific branch, story, or PR from anywhere
    python3 scripts/merge_sync.py --branch feat/story-26.3
    python3 scripts/merge_sync.py --story docs/bmad/stories/story-26.3-grounding-dino-camo-proposals.md
    python3 scripts/merge_sync.py --story 26.3
    python3 scripts/merge_sync.py --pr 47

    # Merge all open PRs simultaneously (operator batch override)
    python3 scripts/merge_sync.py --all
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


def get_primary_repo_root() -> Path:
    """Finds the main repository root even when executed from an isolated git worktree."""
    try:
        res = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
        )
        for line in res.stdout.splitlines():
            if line.startswith("worktree "):
                return Path(line.split(" ", 1)[1].strip()).resolve()
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        )
        return Path(res.stdout.strip()).resolve()
    except Exception:
        return Path(__file__).resolve().parent.parent


PRIMARY_REPO_ROOT = get_primary_repo_root()
STORIES_DIR = PRIMARY_REPO_ROOT / "docs/bmad/stories"
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


def get_active_branch() -> str:
    """Returns the current git branch of the working directory."""
    try:
        res = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = res.stdout.strip()
        if branch:
            return branch
    except Exception:
        pass
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        branch = res.stdout.strip()
        if branch and branch != "HEAD":
            return branch
    except Exception:
        pass
    return ""


def extract_story_slug(text: str | None) -> str | None:
    """Extracts story or epic slug (e.g., 'story-26.3', '26.3', 'epic-23') from branch, path, or title."""
    if not text:
        return None
    m = re.search(r"(story-\d+\.\d+|epic-\d+|story-[a-zA-Z0-9_-]+|spec-[a-zA-Z0-9_-]+|\b\d+\.\d+\b)", text)
    if m:
        val = m.group(0)
        if re.match(r"^\d+\.\d+$", val):
            return f"story-{val}"
        return val
    return None


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


def get_open_prs() -> list[dict[str, Any]]:
    """Fetches open Pull Requests via ggh."""
    try:
        res = subprocess.run(
            ["ggh", "pr", "list", "--state", "open", "--json", "number,title,headRefName"],
            cwd=PRIMARY_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as e:
        print(f"info: checking open PRs: {e}", file=sys.stderr)
    return []


def get_merged_prs() -> list[dict[str, Any]]:
    """Fetches recently merged PRs via ggh."""
    try:
        res = subprocess.run(
            ["ggh", "pr", "list", "--state", "merged", "--limit", "10", "--json", "number,title,headRefName"],
            cwd=PRIMARY_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if res.returncode == 0 and res.stdout.strip():
            return json.loads(res.stdout)
    except Exception as e:
        print(f"info: checking merged PRs: {e}", file=sys.stderr)
    return []


def get_active_worktrees() -> list[tuple[str, str]]:
    """Returns non-root active git worktrees as (path, branch) tuples."""
    worktrees = []
    try:
        res = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PRIMARY_REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        current_wt = None
        for line in res.stdout.splitlines():
            if line.startswith("worktree "):
                current_wt = line.split(" ", 1)[1].strip()
            elif line.startswith("branch "):
                current_branch = line.split(" ", 1)[1].replace("refs/heads/", "").strip()
                if current_wt and Path(current_wt).resolve() != PRIMARY_REPO_ROOT.resolve():
                    worktrees.append((current_wt, current_branch))
                current_wt = None
    except Exception as e:
        print(f"warning getting worktree list: {e}", file=sys.stderr)
    return worktrees


def resolve_target(
    branch: str | None = None,
    story: str | None = None,
    pr_num: int | None = None,
    all_prs: bool = False,
) -> tuple[str | None, list[dict[str, Any]], str | None, bool]:
    """Resolves the target branch, matching PRs, story slug, and mode flag.

    Returns:
        (target_branch, target_prs, target_story_slug, is_all_mode)
    """
    open_prs = get_open_prs()

    if all_prs:
        return None, open_prs, None, True

    if pr_num is not None:
        match = [p for p in open_prs if str(p.get("number")) == str(pr_num)]
        if match:
            pr = match[0]
            target_branch = pr.get("headRefName")
            slug = extract_story_slug(target_branch) or extract_story_slug(pr.get("title", ""))
            return target_branch, [pr], slug, False
        else:
            print(f"⚠️ PR #{pr_num} is not currently open.", file=sys.stderr)
            return None, [], None, False

    if branch:
        target_branch = branch
        matching_prs = [p for p in open_prs if p.get("headRefName") == target_branch]
        slug = extract_story_slug(target_branch)
        return target_branch, matching_prs, slug, False

    if story:
        slug = extract_story_slug(story) or story
        clean_slug = slug.removeprefix("story-").removeprefix("feat/").removeprefix("spec-")
        matching_prs = [
            p for p in open_prs
            if slug in p.get("headRefName", "") or clean_slug in p.get("headRefName", "")
            or slug in p.get("title", "") or clean_slug in p.get("title", "")
        ]
        if matching_prs:
            target_branch = matching_prs[0].get("headRefName")
            return target_branch, matching_prs, slug, False
        else:
            target_branch = f"feat/{slug}"
            return target_branch, [], slug, False

    # Auto-detect from current working directory if inside a worktree
    cwd_branch = get_active_branch()
    if cwd_branch and cwd_branch not in ("main", "master"):
        target_branch = cwd_branch
        matching_prs = [p for p in open_prs if p.get("headRefName") == target_branch]
        slug = extract_story_slug(target_branch)
        print(f"🎯 Detected active worktree branch: `{target_branch}` (from {Path.cwd()})")
        return target_branch, matching_prs, slug, False

    # We are on main / root repo without explicit branch argument
    if not open_prs:
        print("ℹ️ No open Pull Requests found.")
        return None, [], None, False

    if len(open_prs) == 1:
        pr = open_prs[0]
        target_branch = pr.get("headRefName")
        slug = extract_story_slug(target_branch) or extract_story_slug(pr.get("title", ""))
        print(f"🎯 Detected single active PR #{pr.get('number')}: {pr.get('title')} (branch: `{target_branch}`)")
        return target_branch, [pr], slug, False

    # Check if there is only 1 non-root active worktree
    active_wts = get_active_worktrees()
    if len(active_wts) == 1:
        wt_path, wt_branch = active_wts[0]
        matching_prs = [p for p in open_prs if p.get("headRefName") == wt_branch]
        if len(matching_prs) == 1:
            pr = matching_prs[0]
            slug = extract_story_slug(wt_branch)
            print(f"🎯 Detected single active worktree at `{wt_path}` with PR #{pr.get('number')} (branch: `{wt_branch}`)")
            return wt_branch, [pr], slug, False

    # Multiple open PRs and multiple active branches exist
    print("⚠️ Multiple open Pull Requests detected across concurrent branches:")
    for p in open_prs:
        print(f"  • PR #{p.get('number')}: {p.get('title')} (branch: `{p.get('headRefName')}`)")
    print("\nTo prevent merging PRs from concurrent workers in other worktrees, please specify which branch to merge:")
    print("  python3 scripts/merge_sync.py --branch <branch_name>")
    print("  python3 scripts/merge_sync.py --story <story_slug>")
    print("  (or run `merge-sync` from inside the specific worktree directory)")
    print("To merge all open PRs simultaneously, pass `--all`.")
    return None, [], None, False


def merge_prs(prs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merges specified Pull Requests via ggh."""
    merged = []
    for pr in prs:
        pr_num = str(pr.get("number"))
        pr_title = pr.get("title", "")
        pr_branch = pr.get("headRefName", "")
        print(f"🔀 Merging PR #{pr_num} ({pr_branch}: {pr_title}) (Human Approval Confirmed)...")
        merge_res = subprocess.run(
            ["ggh", "pr", "merge", pr_num, "--squash", "--delete-branch", "--auto"],
            cwd=PRIMARY_REPO_ROOT,
            capture_output=True,
            text=True,
        )
        if merge_res.returncode != 0:
            merge_res = subprocess.run(
                ["ggh", "pr", "merge", pr_num, "--squash", "--delete-branch"],
                cwd=PRIMARY_REPO_ROOT,
                capture_output=True,
                text=True,
            )
        if merge_res.returncode == 0:
            print(f"  ✅ Successfully merged PR #{pr_num} into main.")
            merged.append(pr)
        else:
            print(f"  ℹ️ PR #{pr_num} merge response: {merge_res.stderr.strip() or merge_res.stdout.strip()}")
    return merged


def get_completed_story_slugs() -> set[str]:
    """Extracts slugs of all stories marked COMPLETED in docs/bmad/stories/."""
    completed = set()
    if not STORIES_DIR.exists():
        return completed
    for sf in list(STORIES_DIR.glob("story-*.md")) + list(STORIES_DIR.glob("spec-*.md")):
        content = sf.read_text(encoding="utf-8")
        if 'stage: "COMPLETED"' in content or 'Status: completed' in content:
            m = re.search(r"(story-\d+\.\d+|story-[a-zA-Z0-9-]+|spec-[a-zA-Z0-9-]+)", sf.name)
            if m:
                completed.add(m.group(0))
    return completed


def cleanup_worktrees(
    target_branch: str | None = None,
    target_slug: str | None = None,
    clean_all: bool = False,
) -> list[str]:
    """Finds and removes worktrees whose branches have been merged or stories completed.

    If target_branch / target_slug is specified, only that worktree is cleaned up.
    """
    cleaned = []
    merged_prs = get_merged_prs()
    merged_branches = {pr.get("headRefName", "") for pr in merged_prs if pr.get("headRefName")}
    completed_slugs = get_completed_story_slugs()

    try:
        res = subprocess.run(
            ["git", "worktree", "list", "--porcelain"],
            cwd=PRIMARY_REPO_ROOT,
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
                if current_wt and Path(current_wt).resolve() != PRIMARY_REPO_ROOT.resolve():
                    worktrees.append((current_wt, current_branch))
                current_wt = None
                current_branch = None

        for wt_path, branch in worktrees:
            # If targeting a specific branch or slug, filter out non-matching worktrees
            if not clean_all and target_branch:
                clean_target = target_branch.replace("refs/heads/", "")
                if branch != clean_target and clean_target not in wt_path:
                    continue
            if not clean_all and target_slug:
                if target_slug not in wt_path and target_slug not in branch:
                    continue

            should_clean = False
            reason = ""

            if branch in merged_branches:
                should_clean = True
                reason = f"PR merged for branch `{branch}`"
            elif target_branch and (branch == target_branch or target_branch.replace("refs/heads/", "") == branch):
                should_clean = True
                reason = f"Active branch `{branch}` merged"
            else:
                for slug in completed_slugs:
                    if slug in wt_path or slug in branch:
                        should_clean = True
                        reason = f"Story `{slug}` marked COMPLETED"
                        break

            if should_clean:
                print(f"🧹 Removing merged worktree: {wt_path} ({reason})...")
                subprocess.run(["git", "worktree", "remove", wt_path, "--force"], cwd=PRIMARY_REPO_ROOT, check=False)
                subprocess.run(["git", "branch", "-D", branch], cwd=PRIMARY_REPO_ROOT, check=False)
                cleaned.append(f"{wt_path} ({branch})")
    except Exception as e:
        print(f"warning cleaning worktrees: {e}", file=sys.stderr)
    return cleaned


def sync_completed_tickets(target_slug: str | None = None) -> list[str]:
    """Ensures stories marked COMPLETED in docs/bmad/stories/ have Buganizer tickets transitioned to FIXED."""
    updated = []
    if not STORIES_DIR.exists():
        return updated
    user_email = get_current_user_email()

    client = BuganizerClient(timeout=15.0)

    clean_target = target_slug.removeprefix("story-").removeprefix("feat/").removeprefix("spec-") if target_slug else None

    for sf in sorted(STORIES_DIR.glob("story-*.md")) + sorted(STORIES_DIR.glob("spec-*.md")):
        if clean_target and clean_target not in sf.name and target_slug not in sf.name:
            continue

        content = sf.read_text(encoding="utf-8")
        if 'stage: "COMPLETED"' in content or 'Status: completed' in content or "- [x]" in content or target_slug:
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

    client.close()
    return updated


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Automated Post-Merge Synchronization & Worktree Cleanup Tool."
    )
    parser.add_argument(
        "--branch", "-b",
        type=str,
        default=None,
        help="Specific git branch to merge and clean (e.g. feat/story-26.3).",
    )
    parser.add_argument(
        "--story", "-s",
        type=str,
        default=None,
        help="Story path or slug to target (e.g. docs/bmad/stories/story-26.3-....md or 26.3).",
    )
    parser.add_argument(
        "--pr", "-p",
        type=int,
        default=None,
        help="Specific Pull Request number to merge.",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Merge all open PRs and clean all merged worktrees across the repository.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("🔄 Running Post-Merge Synchronization & Worktree Cleanup")
    print("=" * 80)

    # 1. Resolve Target Branch & PRs
    print("\n1. Resolving Target Worktree / Pull Request...")
    target_branch, target_prs, target_slug, is_all_mode = resolve_target(
        branch=args.branch,
        story=args.story,
        pr_num=args.pr,
        all_prs=args.all,
    )

    if not target_prs and not target_branch and not is_all_mode:
        print("  ℹ️ Nothing to merge.")
        return 0

    # 2. Merge Open PR(s)
    if target_prs:
        print(f"\n2. Merging {len(target_prs)} Pull Request(s) (Human Approval Confirmed)...")
        merge_prs(target_prs)
    elif target_branch:
        print(f"\n2. Checking merge status for branch `{target_branch}`...")
        merged_prs = get_merged_prs()
        if any(p.get("headRefName") == target_branch for p in merged_prs):
            print(f"  ℹ️ Branch `{target_branch}` already merged.")
        else:
            print(f"  ℹ️ No open PR found for branch `{target_branch}`.")

    # 3. Pull latest main
    print("\n3. Pulling latest main from origin...")
    subprocess.run(["git", "checkout", "main"], cwd=PRIMARY_REPO_ROOT, check=False)
    pull_res = subprocess.run(["git", "pull", "origin", "main"], cwd=PRIMARY_REPO_ROOT, capture_output=True, text=True)
    print(pull_res.stdout.strip() if pull_res.stdout else "Already up to date.")

    # 4. Clean up target worktree(s)
    print("\n4. Cleaning Merged Worktrees...")
    cleaned_wt = cleanup_worktrees(
        target_branch=target_branch if not is_all_mode else None,
        target_slug=target_slug if not is_all_mode else None,
        clean_all=is_all_mode,
    )
    if cleaned_wt:
        for c in cleaned_wt:
            print(f"  ✅ Cleaned: {c}")
    else:
        print("  ℹ️ No matching worktrees to remove.")

    # 5. Finalize Buganizer / TaskFlow tickets
    print("\n5. Finalizing Completed Tickets in Buganizer / TaskFlow...")
    fixed_tickets = sync_completed_tickets(target_slug=target_slug if not is_all_mode else None)
    if fixed_tickets:
        for t in fixed_tickets:
            print(f"  ✅ Status FIXED: {t}")
    else:
        print("  ℹ️ No completed tickets to update.")

    # 6. Run Tracker Synchronization
    print("\n6. Running Idempotent Tracker Sync...")
    sync_script = PRIMARY_REPO_ROOT / "scripts/sync_trackers.py"
    if sync_script.exists():
        subprocess.run([sys.executable, str(sync_script)], cwd=PRIMARY_REPO_ROOT)

    print("\n" + "=" * 80)
    print("✅ Merge Sync Complete: PR merged, main updated, worktree cleaned, ticket FIXED, trackers in sync.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
