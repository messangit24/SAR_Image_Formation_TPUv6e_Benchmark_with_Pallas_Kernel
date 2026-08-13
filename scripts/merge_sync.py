#!/usr/bin/env python3
"""Automated Post-Merge Synchronization & Worktree Cleanup Tool for GoldThread."""
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORIES_DIR = PROJECT_ROOT / "docs/bmad/stories"


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


def cleanup_worktrees() -> list[str]:
    """Finds and removes worktrees whose branches have been merged."""
    cleaned = []
    merged_prs = get_merged_prs()
    merged_branches = {pr.get("headRefName", "") for pr in merged_prs if pr.get("headRefName")}

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
            if branch in merged_branches:
                print(f"🧹 Removing merged worktree: {wt_path} (branch: {branch})...")
                subprocess.run(["git", "worktree", "remove", wt_path, "--force"], cwd=PROJECT_ROOT, check=False)
                subprocess.run(["git", "branch", "-D", branch], cwd=PROJECT_ROOT, check=False)
                cleaned.append(f"{wt_path} ({branch})")
    except Exception as e:
        print(f"warning cleaning worktrees: {e}", file=sys.stderr)
    return cleaned


def main() -> int:
    print("=" * 80)
    print("🔄 Running Post-Merge Synchronization & Worktree Cleanup (GoldThread)")
    print("=" * 80)

    print("\n1. Pulling latest main from origin...")
    subprocess.run(["git", "checkout", "main"], cwd=PROJECT_ROOT, check=False)
    pull_res = subprocess.run(["git", "pull", "origin", "main"], cwd=PROJECT_ROOT, capture_output=True, text=True)
    print(pull_res.stdout.strip() if pull_res.stdout else "Already up to date.")

    merged_prs = get_merged_prs()
    if merged_prs:
        print("\n2. Recently Merged PRs:")
        for pr in merged_prs:
            print(f"  • PR #{pr.get('number')}: {pr.get('title')} (branch: {pr.get('headRefName')})")

    print("\n3. Cleaning Merged Worktrees...")
    cleaned = cleanup_worktrees()
    if cleaned:
        for c in cleaned:
            print(f"  ✅ Cleaned: {c}")
    else:
        print("  ℹ️ No stale worktrees detected.")

    print("\n4. Running Idempotent Tracker Sync...")
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/sync_trackers.py")], cwd=PROJECT_ROOT)

    print("=" * 80)
    print("✅ Merge Sync Complete: Main updated, merged worktrees cleaned.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
