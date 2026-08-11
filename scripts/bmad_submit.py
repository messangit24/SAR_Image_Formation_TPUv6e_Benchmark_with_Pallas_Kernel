#!/usr/bin/env python3
"""Post-completion hook to request parent agent to submit the story.

Reads the current git commit, finds the matching BMAD story spec file that has this
commit hash as final_revision, parses the Auto Run Result, and sends a request to the
parent conversation ID using agentapi to automate the GitHub PR/Issue and Buganizer actions.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd: list[str], cwd: str = ".") -> str:
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd, check=True)
    return res.stdout.strip()

def main() -> int:
    conv_id = os.environ.get("ANTIGRAVITY_CONVERSATION_ID")
    if not conv_id:
        print("error: ANTIGRAVITY_CONVERSATION_ID environment variable not found.", file=sys.stderr)
        return 0  # fail silently or warning, don't break the loop exit

    project_root = "/usr/local/google/home/ermlickb/Desktop/Workspace/TPUbench"
    
    try:
        head_commit = run_cmd(["git", "rev-parse", "HEAD"], cwd=project_root)
        branch_name = run_cmd(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=project_root)
    except Exception as e:
        print(f"error getting git info: {e}", file=sys.stderr)
        return 1

    # Find the spec file containing head_commit as final_revision
    spec_file = None
    stories_dir = Path(project_root) / "docs/bmad/stories"
    if stories_dir.exists():
        for f in stories_dir.glob("spec-*.md"):
            content = f.read_text()
            # check if final_revision is in frontmatter
            if f"final_revision: '{head_commit}'" in content or f'final_revision: "{head_commit}"' in content:
                spec_file = f
                break

    if not spec_file:
        # Fallback: search the whole docs folder or check the most recently modified spec file
        print(f"warning: no spec file found with final_revision '{head_commit}'. Searching for any modified spec...", file=sys.stderr)
        for f in stories_dir.glob("spec-*.md"):
            content = f.read_text()
            if "status: 'done'" in content or 'status: "done"' in content:
                # check if it was modified in the last commit
                try:
                    diff = run_cmd(["git", "diff", "HEAD~1", "HEAD", "--", str(f)], cwd=project_root)
                    if diff:
                        spec_file = f
                        break
                except Exception:
                    pass

    if not spec_file:
        print("error: could not identify the completed story spec file.", file=sys.stderr)
        return 1

    content = spec_file.read_text()

    # Parse title from frontmatter
    title_m = re.search(r"^title:\s*['\"]?(.+?)['\"]?\s*$", content, re.MULTILINE)
    title = title_m.group(1).strip() if title_m else "Unknown Story"

    # Parse story number and slug from title
    num, story_title = "Unknown", title
    m = re.match(r"^(?:Story\s+([\w.]+)\s*:\s*)?(.+)$", title)
    if m:
        num, story_title = m.group(1) or "Unknown", m.group(2).strip()

    # Parse acceptance criteria
    acs = []
    sec = re.search(r"^##\s+Acceptance Criteria\s*$(.*?)(?=^##\s|\Z)", content, re.MULTILINE | re.DOTALL)
    if sec:
        for line in sec.group(1).splitlines():
            item = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
            if item and not item.startswith("["):
                acs.append(item)

    # Parse Auto Run Result section
    auto_run_result = ""
    arr_sec = re.search(r"^##\s+Auto Run Result\s*$(.*)", content, re.MULTILINE | re.DOTALL)
    if arr_sec:
        auto_run_result = arr_sec.group(1).strip()

    # Construct request payload
    payload = {
        "action": "submit_story",
        "story_key": spec_file.stem.replace("spec-", ""),
        "title": story_title,
        "story_number": num,
        "spec_file_path": spec_file.relative_to(project_root).as_posix(),
        "branch_name": branch_name,
        "commit_hash": head_commit,
        "acceptance_criteria": acs,
        "auto_run_result": auto_run_result
    }

    message_body = f"Please complete the submission steps for the following story:\n```json\n{json.dumps(payload, indent=2)}\n```"

    # Send the message
    try:
        run_cmd(["agentapi", "send-message", "--title", f"Submit Story {num}: {story_title}", conv_id, message_body], cwd=project_root)
        print(f"Sent submission request for Story {num} to parent conversation {conv_id}.")
    except Exception as e:
        print(f"error sending message via agentapi: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
