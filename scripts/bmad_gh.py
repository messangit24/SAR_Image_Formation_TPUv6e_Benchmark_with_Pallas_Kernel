"""Bridge a BMAD story file to a GitHub issue so commits and branches trace back
to the feature/story.

BMAD (docs/bmad.md) produces story files under docs/bmad/stories/; this script
turns one into a GitHub issue (via the `gh` CLI), stamps the issue number back
into the story, and prints the matching branch name + commit trailer. The result
is a stable three-way link: story file <-> issue <-> branch/commits.

Usage:
    # Preview the gh command + branch/commit convention (no auth needed):
    python scripts/bmad_gh.py docs/bmad/stories/story-1.1-foo.md --dry-run

    # Create the issue for real (needs `gh auth login` first) and stamp it back:
    python scripts/bmad_gh.py docs/bmad/stories/story-1.1-foo.md

Convention (see docs/bmad.md "Linking to GitHub issues"):
    branch  : story/<epic>.<story>-<slug>   (falls back to story/<slug>)
    commit  : "<subject>\\n\\nRefs #<issue>"  while WIP; "Closes #<issue>" to finish
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

LABEL_DEFAULT = "bmad-story"


def _slug(text: str) -> str:
    """A branch-safe slug: lowercase, alnum runs joined by '-'."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def parse_story(md: str) -> dict:
    """Pull the title, story number, and acceptance criteria out of a BMAD story.

    Matches the bmad-create-story template ('# Story <e>.<s>: <title>', a 'Status:'
    line, and a '## Acceptance Criteria' section). Everything is optional so a
    hand-written story still works — we degrade to sane defaults, not errors.
    """
    title, num = "Untitled story", None
    for line in md.splitlines():
        # Epic/story id may be alphanumeric (e.g. "1.2" or "UI.1"); require the
        # literal "Story " prefix so a plain "# Title" heading isn't misread.
        m = re.match(r"^#\s+(?:Story\s+([\w.]+)\s*:\s*)?(.+)$", line.strip())
        if m:
            num, title = m.group(1), m.group(2).strip()
            break

    status_m = re.search(r"^Status:\s*(.+)$", md, re.MULTILINE)
    status = status_m.group(1).strip() if status_m else "ready-for-dev"

    # Acceptance Criteria section → the bullet/numbered lines under it (up to the
    # next '## ' heading). These become the issue's checklist so review has teeth.
    acs: list[str] = []
    sec = re.search(r"^##\s+Acceptance Criteria\s*$(.*?)(?=^##\s|\Z)", md, re.MULTILINE | re.DOTALL)
    if sec:
        for line in sec.group(1).splitlines():
            item = re.sub(r"^\s*(?:[-*]|\d+\.)\s+", "", line).strip()
            if item and not item.startswith("["):  # skip the template placeholder
                acs.append(item)
    return {"title": title, "num": num, "status": status, "acceptance": acs}


def issue_body(story: dict, story_path: Path) -> str:
    """Issue body: a short pointer to the story of record + its acceptance criteria
    as a checklist. The story file stays the source of truth; the issue tracks it."""
    lines = [
        f"Tracks BMAD story **{story['title']}**"
        + (f" (story {story['num']})" if story["num"] else "") + ".",
        "",
        f"Story of record: `{story_path.as_posix()}`",
        "",
    ]
    if story["acceptance"]:
        lines.append("### Acceptance criteria")
        lines += [f"- [ ] {a}" for a in story["acceptance"]]
        lines.append("")
    lines.append("_Created from a BMAD story via `scripts/bmad_gh.py`._")
    return "\n".join(lines)


def branch_name(story: dict) -> str:
    slug = _slug(story["title"])
    return f"story/{story['num'].lower()}-{slug}" if story["num"] else f"story/{slug}"


def stamp_issue(md: str, number: int, url: str) -> str:
    """Record the issue link in the story, right after the 'Status:' line, so the
    story <-> issue link lives in the file (idempotent — replaces an old stamp)."""
    stamp = f"GitHub-Issue: #{number} ({url})"
    if re.search(r"^GitHub-Issue:.*$", md, re.MULTILINE):
        return re.sub(r"^GitHub-Issue:.*$", stamp, md, count=1, flags=re.MULTILINE)
    if re.search(r"^Status:.*$", md, re.MULTILINE):
        return re.sub(r"^(Status:.*)$", r"\1\n" + stamp, md, count=1, flags=re.MULTILINE)
    return stamp + "\n\n" + md  # no Status line — prepend


def main() -> int:
    ap = argparse.ArgumentParser(description="Create/link a GitHub issue from a BMAD story file.")
    ap.add_argument("story", type=Path, help="path to the BMAD story markdown file")
    ap.add_argument("--label", default=LABEL_DEFAULT, help=f"issue label (default: {LABEL_DEFAULT})")
    ap.add_argument("--repo", default=None, help="owner/name (default: the origin remote)")
    ap.add_argument("--dry-run", action="store_true", help="print the gh command, don't run it")
    args = ap.parse_args()

    if not args.story.is_file():
        print(f"error: no such story file: {args.story}", file=sys.stderr)
        return 2
    md = args.story.read_text()
    story = parse_story(md)
    branch = branch_name(story)
    body = issue_body(story, args.story)

    import shutil
    gh_bin = shutil.which("ggh") or shutil.which("gh") or "ggh"
    target_repo = args.repo or "personal/goldthread"

    gh = [gh_bin, "issue", "create", "--repo", target_repo, "--title", f"[TO2] {story['title']}", "--body", body]

    if args.dry_run:
        # Shell-safe preview (real run uses argv, no shell — this is just to read).
        preview = " ".join(f"'{a}'" if (" " in a or "\n" in a) else a for a in gh)
        print("# would run:\n" + preview.replace("\n", "\\n"))
        print(f"\n# branch:  git switch -c {branch}")
        print("# commit:  <subject>\\n\\nRefs #<issue>   (use 'Closes #<issue>' on the final commit)")
        return 0

    try:
        out = subprocess.run(gh, capture_output=True, text=True, check=True)
    except FileNotFoundError:
        print(f"error: `{gh_bin}` CLI not found — install it or use --dry-run.", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as e:  # not authed / no repo / label missing
        print(f"error: {gh_bin} issue create failed:\n{e.stderr}", file=sys.stderr)
        print("hint: run `ggh auth login`, or use --dry-run.",
              file=sys.stderr)
        return 1

    url = [line.strip() for line in out.stdout.strip().splitlines() if "http" in line][-1]
    number = int(url.rstrip("/").rsplit("/", 1)[-1])
    args.story.write_text(stamp_issue(md, number, url))
    print(f"created issue #{number}: {url}")
    print(f"stamped {args.story}")
    print(f"\nnext:\n  git switch -c {branch}")
    print(f'  git commit -m "<subject>" -m "Refs #{number}"   # "Closes #{number}" on the last commit')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
