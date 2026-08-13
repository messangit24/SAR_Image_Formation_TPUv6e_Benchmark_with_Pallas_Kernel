#!/usr/bin/env python3
"""Story Development Scaffolder & Lifecycle Manager for GoldThread.

Usage:
    python3 scripts/dev_story.py --story docs/bmad/stories/<story-file>.md --setup-worktree
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def setup_worktree(story_path: Path) -> Path:
    """Creates isolated worktree and symlinks heavy dependencies."""
    stem = story_path.stem
    m = re.search(r"story-([a-zA-Z0-9.-]+)", stem)
    slug = m.group(0) if m else stem
    worktree_name = f"goldthread-{slug}"
    worktree_path = PROJECT_ROOT.parent / worktree_name
    branch_name = f"feat/{slug}"

    print(f"🌲 Creating isolated worktree at: {worktree_path}")
    if worktree_path.exists():
        print(f"  ℹ️ Worktree directory already exists at {worktree_path}")
    else:
        cmd = ["git", "worktree", "add", str(worktree_path), "-b", branch_name, "origin/main"]
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        print(f"  ✅ Worktree created on branch `{branch_name}`.")

    # Symlinks
    links = [
        (".venv", worktree_path / ".venv"),
        (".env", worktree_path / ".env"),
        ("data", worktree_path / "data"),
    ]

    for src_rel, target in links:
        src = PROJECT_ROOT / src_rel
        if src.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(src)
            print(f"  🔗 Symlinked `{src_rel}`")

    return worktree_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Story dev manager for GoldThread.")
    parser.add_argument("--story", help="Path to story markdown file")
    parser.add_argument("--setup-worktree", action="store_true", help="Create worktree and symlink env")
    args = parser.parse_args()

    if args.setup_worktree and args.story:
        story_file = Path(args.story)
        if not story_file.is_absolute():
            story_file = PROJECT_ROOT / story_file
        setup_worktree(story_file)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
