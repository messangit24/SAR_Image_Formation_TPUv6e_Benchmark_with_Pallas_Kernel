#!/usr/bin/env python3
"""Story Development Scaffolder & Lifecycle Manager.

Automates:
1. Pre-flight dependency check (depends_on) and execution mode inspection (AFK vs HUMAN_MONITORED).
2. Worktree creation with zero-copy symlinks (.venv, node_modules, .env, data).
3. Buganizer / TaskFlow ticket status updates (NEW -> IN_PROGRESS / ASSIGNED).
4. Verification test execution.

Usage:
    # Setup isolated worktree for a story:
    python3 scripts/dev_story.py --story docs/bmad/stories/story-1.1-starter.md --setup-worktree

    # Verify gates on current worktree:
    python3 scripts/dev_story.py --verify
"""
import argparse
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
                    "clientInfo": {"name": "dev-story", "version": "1.0"},
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


def parse_story_frontmatter(path: Path) -> dict[str, Any]:
    """Reads frontmatter from story file."""
    content = path.read_text(encoding="utf-8")
    meta = {}
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    val = v.strip()
                    if (val.startswith("[") and val.endswith("]")) or (val.startswith("{") and val.endswith("}")):
                        try:
                            meta[k.strip()] = json.loads(val)
                        except Exception:
                            meta[k.strip()] = val.strip("'\"")
                    else:
                        meta[k.strip()] = val.strip("'\"")
    return meta


def check_dependencies(story_path: Path, meta: dict[str, Any]) -> bool:
    """Validates that all prerequisite stories in depends_on are COMPLETED."""
    deps = meta.get("depends_on", [])
    if isinstance(deps, str):
        if deps.startswith("[") and deps.endswith("]"):
            try:
                deps = json.loads(deps)
            except Exception:
                deps = [deps]
        elif deps:
            deps = [deps]
        else:
            deps = []

    print("\n📋 Pre-Flight Readiness Inspection:")
    exec_mode = meta.get("execution_mode", "AFK")
    print(f"  • Execution Mode: {exec_mode}")
    if exec_mode == "HUMAN_MONITORED":
        rationale = meta.get("human_monitoring_rationale", "Developer presence required.")
        print(f"    ⚠️ Monitoring Rationale: {rationale}")
    else:
        print("    🤖 Autonomous Execution: Unattended implementation permitted after plan approval.")

    if not deps:
        print("  • Dependencies: None (`depends_on: []`). Ready immediately.")
        return True

    print(f"  • Upstream Dependencies: {deps}")
    all_met = True
    if STORIES_DIR.exists():
        for dep in deps:
            # Match by slug or filename
            matching_files = list(STORIES_DIR.glob(f"*{dep}*"))
            if not matching_files:
                print(f"    ⚠️ Dependency `{dep}`: File not found in {STORIES_DIR}")
                all_met = False
                continue
            dep_file = matching_files[0]
            dep_content = dep_file.read_text(encoding="utf-8")
            is_completed = 'stage: "COMPLETED"' in dep_content or 'Status: completed' in dep_content or '- [x]' in dep_content
            if is_completed:
                print(f"    ✅ Dependency `{dep}`: COMPLETED")
            else:
                print(f"    ❌ Dependency `{dep}`: INCOMPLETE / BLOCKED")
                all_met = False

    if not all_met:
        print("  ⚠️ WARNING: Some upstream dependencies are not completed. Proceed with caution.\n")
    else:
        print("  ✅ All prerequisite dependencies satisfied.\n")
    return all_met


def setup_worktree(story_path: Path) -> Path:
    """Creates isolated worktree and symlinks heavy dependencies."""
    meta = parse_story_frontmatter(story_path)
    check_dependencies(story_path, meta)

    stem = story_path.stem
    m = re.search(r"(story|spec)-(\d+\.\d+|[a-zA-Z0-9.-]+)", stem)
    slug = m.group(0) if m else stem
    repo_name = PROJECT_ROOT.name
    worktree_name = f"{repo_name}-{slug}"
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
        ("web/node_modules", worktree_path / "web/node_modules"),
        (".env", worktree_path / ".env"),
        ("data", worktree_path / "data"),
    ]

    for src_rel, target in links:
        src = PROJECT_ROOT / src_rel
        if src.exists() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(src)
            print(f"  🔗 Symlinked `{src_rel}`")

    # Update Buganizer status if ID present
    bug_id = meta.get("buganizer_id")
    if bug_id:
        print(f"  🔄 Updating Buganizer ticket b/{bug_id} to IN_PROGRESS (ASSIGNED)...")
        call_mcp_tool("buganizer_update_issue_status", {
            "issue_id": int(bug_id),
            "status": "ASSIGNED",
        })

    return worktree_path


def run_verifications(workdir: Path) -> bool:
    """Runs ruff and pytest verification suite."""
    print("🔍 Running Verification Gates...")

    print("\n1. Linting with Ruff:")
    res_ruff = subprocess.run(["uv", "run", "ruff", "check", "src", "tests", "scripts"], cwd=workdir)
    if res_ruff.returncode != 0:
        print("❌ Ruff checks failed!")
        return False
    print("✅ Ruff checks passed cleanly.")

    print("\n2. Running Pytest Suite:")
    res_pytest = subprocess.run(["uv", "run", "pytest", "-q"], cwd=workdir)
    if res_pytest.returncode != 0:
        print("❌ Pytest suite failed!")
        return False
    print("✅ Pytest suite green.")

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Story dev manager.")
    parser.add_argument("--story", help="Path to story markdown file")
    parser.add_argument("--setup-worktree", action="store_true", help="Create worktree and symlink env")
    parser.add_argument("--verify", action="store_true", help="Run verification gates")
    args = parser.parse_args()

    if args.setup_worktree and args.story:
        story_file = Path(args.story)
        if not story_file.is_absolute():
            story_file = PROJECT_ROOT / story_file
        setup_worktree(story_file)
        return 0

    if args.verify:
        success = run_verifications(PROJECT_ROOT)
        return 0 if success else 1

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
