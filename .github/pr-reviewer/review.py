#!/usr/bin/env python3
"""
AI Code Reviewer for GitHub PRs.

Triggered by GitHub Actions on pull_request event. Fetches the PR diff and linked
GitHub issue, sends them to Gemini or Claude for review, then posts the results 
back as PR comments.

Exit codes (advisory only — does not block merges):
  0 — Review passed (no Critical issues, or reviewer itself errored)
  1 — Review found Critical issues (shown as warning in pipeline)

Authentication:
  Support for multiple providers (configured in config.json):
  - "gemini": Google GenAI SDK (requires GEMINI_API_KEY in env)
  - "anthropic": Direct Claude API (requires ANTHROPIC_API_KEY in env)
  - "vertex-ai": GCP Vertex AI (requires GCP_PROJECT_ID, GCP_REGION, and WIF or Service Account credentials)
"""

import json
import os
import pathlib
import re
import sys
import time
import requests

# Conditional imports for dynamic provider support
try:
    from anthropic import Anthropic, AnthropicVertex, APIError
except ImportError:
    Anthropic = None
    AnthropicVertex = None
    APIError = Exception

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

BOT_MARKER = "<!-- ai-code-reviewer -->"
COLLAPSED_MARKER = "<!-- ai-code-reviewer-collapsed -->"
NO_TICKET_SENTINEL = "No ticket number found"
MAX_DIFF_CHARS = 80_000  # truncate large diffs to avoid token overflow
MAX_PAGES = 50  # safety limit for paginated API calls
MAX_CONTEXT_CHARS = 20_000  # cap previous review context to avoid token overflow

# Repo root: prefer an explicit env var (useful when running outside the repo tree),
# otherwise walk up three levels from this file's location:
#   review.py  →  pr-reviewer/  →  .github/  →  repo root
_REPO_ROOT = pathlib.Path(
    os.environ.get("REPO_ROOT", pathlib.Path(__file__).resolve().parent.parent.parent)
)

# ---------------------------------------------------------------------------
# Generic Configuration Loader
# ---------------------------------------------------------------------------

def load_config() -> dict:
    """Load configuration from .github/pr-reviewer/config.json."""
    config_path = pathlib.Path(__file__).resolve().parent / "config.json"
    if not config_path.exists():
        print(f"Warning: config.json not found at {config_path}. Using default configuration.")
        return {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "workstreams": []
        }
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Warning: Failed to parse config.json at {config_path}: {e}. Using fallback configuration.")
        return {
            "provider": "gemini",
            "model": "gemini-3.5-flash",
            "workstreams": []
        }

_CONFIG = load_config()
PROVIDER = _CONFIG.get("provider", "gemini")
MODEL = _CONFIG.get("model", "gemini-3.5-flash" if PROVIDER == "gemini" else "claude-sonnet-4-6")
_WORKSTREAMS_CONFIG = _CONFIG.get("workstreams", [])

class GitHubClient:
    def __init__(self, server_url: str, repository: str, token: str):
        if "github.com" in server_url.lower() and "api/v3" not in server_url:
            self.base = f"https://api.github.com/repos/{repository}"
        else:
            self.base = f"{server_url}/api/v3/repos/{repository}"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }

    def _get(self, path: str, params: dict = None, headers: dict = None) -> list | dict:
        h = {**self.headers, **(headers or {})}
        r = requests.get(f"{self.base}{path}", headers=h, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self.headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _patch(self, path: str, payload: dict) -> dict:
        r = requests.patch(f"{self.base}{path}", headers=self.headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def get_pr(self, pr_number: str) -> dict:
        pr = self._get(f"/pulls/{pr_number}")
        return {
            "title": pr.get("title", ""),
            "description": pr.get("body", "")
        }

    def get_issue(self, issue_number: str) -> dict | None:
        try:
            issue = self._get(f"/issues/{issue_number}")
            return {
                "title": issue.get("title", ""),
                "description": issue.get("body", "")
            }
        except Exception:
            return None

    def get_pr_diffs(self, pr_number: str) -> list:
        try:
            diff_url = f"{self.base}/pulls/{pr_number}"
            h = {**self.headers, "Accept": "application/vnd.github.v3.diff"}
            r = requests.get(diff_url, headers=h, timeout=30)
            r.raise_for_status()
            raw_diff = r.text
        except Exception as e:
            print(f"Error fetching GitHub diff: {e}")
            return []

        if not raw_diff.strip():
            return []

        blocks = []
        current_block = []
        for line in raw_diff.splitlines():
            if line.startswith("diff --git "):
                if current_block:
                    blocks.append(current_block)
                current_block = [line]
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        files = []
        for block in blocks:
            header = block[0]
            header_body = header[11:]
            
            old_path = None
            new_path = None
            renamed_file = False
            deleted_file = False
            collapsed = False
            too_large = False
            
            diff_lines = []
            has_hunks = False
            is_binary = False
            
            for line in block[1:]:
                if line.startswith("rename from "):
                    old_path = line[12:].strip().strip('"')
                    renamed_file = True
                elif line.startswith("rename to "):
                    new_path = line[10:].strip().strip('"')
                    renamed_file = True
                elif line.startswith("deleted file mode "):
                    deleted_file = True
                elif line.startswith("--- a/"):
                    old_path = line[6:].strip().strip('"')
                elif line.startswith("+++ b/"):
                    new_path = line[6:].strip().strip('"')
                elif line.startswith("--- /dev/null"):
                    pass
                elif line.startswith("+++ /dev/null"):
                    deleted_file = True
                elif line.startswith("Binary files "):
                    is_binary = True
                elif line.startswith("@@ "):
                    has_hunks = True
                    diff_lines.append(line)
                elif has_hunks:
                    diff_lines.append(line)
                else:
                    if "suppressed" in line.lower() or "too large" in line.lower() or "collapsed" in line.lower():
                        too_large = True
                    diff_lines.append(line)

            if not old_path or not new_path:
                parts = header_body.split(" b/")
                if len(parts) == 2:
                    if not old_path:
                        old_path = parts[0].replace("a/", "", 1).strip().strip('"')
                    if not new_path:
                        new_path = parts[1].strip().strip('"')

            if not new_path:
                new_path = old_path

            if not has_hunks and not is_binary and not deleted_file:
                is_100_rename = renamed_file and not any(line.startswith("---") or line.startswith("+++") for line in block)
                is_empty_file = any(line.startswith("new file mode") for line in block) and not any(line.startswith("---") or line.startswith("+++") for line in block)
                if not is_100_rename and not is_empty_file:
                    if too_large:
                        too_large = True
                    else:
                        collapsed = True

            files.append({
                "new_path": new_path,
                "old_path": old_path,
                "renamed_file": renamed_file,
                "deleted_file": deleted_file,
                "diff": "\n".join(diff_lines),
                "collapsed": collapsed,
                "too_large": too_large
            })
        return files

    def get_pr_discussions(self, pr_number: str) -> list:
        comments = []
        page = 1
        while page <= MAX_PAGES:
            try:
                page_comments = self._get(
                    f"/issues/{pr_number}/comments",
                    params={"per_page": 100, "page": page}
                )
            except Exception as e:
                print(f"Error fetching GitHub comments (page {page}): {e}")
                break
            if not page_comments:
                break
            comments.extend(page_comments)
            if len(page_comments) < 100:
                break
            page += 1

        bot_comments = []
        for c in comments:
            body = c.get("body") or ""
            if BOT_MARKER in body:
                bot_comments.append(c)

        bot_comments.sort(key=lambda x: x["id"])

        bot_review_nums = {}
        for idx, bc in enumerate(bot_comments):
            bot_review_nums[bc["id"]] = idx + 1

        discussions_map = {}
        for bc in bot_comments:
            author_login = bc.get("user", {}).get("login", "unknown")
            bc_id = bc["id"]
            discussions_map[bc_id] = {
                "id": str(bc_id),
                "notes": [
                    {
                        "id": bc_id,
                        "body": bc.get("body") or "",
                        "author": {"username": author_login}
                    }
                ],
                "resolved": False
            }

        for c in comments:
            body = c.get("body") or ""
            if BOT_MARKER in body:
                continue

            c_id = c["id"]
            author_login = c.get("user", {}).get("login", "unknown")
            note_obj = {
                "id": c_id,
                "body": body,
                "author": {"username": author_login}
            }

            target_bc_id = None

            match = re.search(r"\bReview\s*(\d+)\b", body, re.IGNORECASE)
            if match:
                ref_num = int(match.group(1))
                for bc_id, rev_num in bot_review_nums.items():
                    if rev_num == ref_num:
                        if bc_id < c_id:
                            target_bc_id = bc_id
                            break
                if target_bc_id:
                    print(f"Associated comment {c_id} with Review {ref_num} based on explicit reference.")

            if not target_bc_id:
                quotes = []
                for line in body.splitlines():
                    line_stripped = line.strip()
                    if line_stripped.startswith(">"):
                        quote_content = line_stripped[1:].strip()
                        if len(quote_content) > 10:
                            quotes.append(quote_content)
                
                if quotes:
                    best_match_bc_id = None
                    for bc in reversed(bot_comments):
                        if bc["id"] >= c_id:
                            continue
                        bc_body = bc.get("body") or ""
                        if any(quote in bc_body for quote in quotes):
                            best_match_bc_id = bc["id"]
                            break
                    if best_match_bc_id:
                        target_bc_id = best_match_bc_id
                        print(f"Associated comment {c_id} with bot comment {target_bc_id} based on quoted text.")

            if not target_bc_id:
                preceding_bc_id = None
                for bc in bot_comments:
                    if bc["id"] < c_id:
                        preceding_bc_id = bc["id"]
                    else:
                        break
                if preceding_bc_id:
                    target_bc_id = preceding_bc_id

            if target_bc_id and target_bc_id in discussions_map:
                discussions_map[target_bc_id]["notes"].append(note_obj)

        sorted_disc_ids = sorted(discussions_map.keys())
        return [discussions_map[k] for k in sorted_disc_ids]

    def create_note(self, pr_number: str, body: str) -> dict:
        return self._post(f"/issues/{pr_number}/comments", {"body": body})

    def update_note(self, pr_number: str, comment_id: str, body: str) -> dict:
        return self._patch(f"/issues/comments/{comment_id}", {"body": body})

    def get_file_raw(self, ref: str, file_path: str) -> str | None:
        encoded = requests.utils.quote(file_path, safe="/")
        try:
            h = {**self.headers, "Accept": "application/vnd.github.v3.raw"}
            r = requests.get(f"{self.base}/contents/{encoded}", headers=h, params={"ref": ref}, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_ticket_number(branch_name: str) -> str | None:
    """Extract the ticket number from a branch following the project convention.

    Matches branches like:
      feat/387_title, feature/387-title, feat/web/244-fix-alignment, 387-fix-thing

    The number must:
      - appear at the start of the string or immediately follow a '/'
      - be followed by '_' or '-' (not a bare version component)
      - be at least 2 digits (avoids single-digit false positives like python-3-12)
    """
    match = re.search(r"(?:^|/)(\d+)(?=[_-])", branch_name)
    return match.group(1) if match else None


def extract_ticket_number_fallback(branch_name: str) -> str | None:
    """Fallback: find any 2+ digit number in the branch name.

    Used when ``extract_ticket_number`` fails to match the standard
    convention.  Returns the first 2+ digit sequence found anywhere in
    the branch name, which the caller should validate against the issue
    tracker before trusting.
    """
    match = re.search(r"(\d{2,})", branch_name)
    return match.group(1) if match else None


def extract_ticket_from_description(description: str) -> str | None:
    """Extract a ticket/issue number from PR text.

    Matches (in priority order):
      #123, Closes #123, Fixes #123, Related to #123
      ticket 123, ticket #123, issue 123, issue #123, task 123
      markdown-linked references like: ticket [**1285**](.../work_items/1285)
      direct GitHub issue/work item URLs
    """
    if not description:
        return None

    normalized = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 \2", description)
    normalized = re.sub(r"[*`~]", "", normalized)

    match = re.search(r"#(\d+)", normalized)
    if match:
        return match.group(1)

    match = re.search(
        r"(?:ticket|issue|task|work\s*item)\s+#?(\d+)",
        normalized,
        re.IGNORECASE,
    )
    if match:
        return match.group(1)

    match = re.search(r"/(?:issues|work_items)/(\d+)(?:\b|[/?#])", normalized)
    return match.group(1) if match else None


def format_review_label(review_number: int) -> str:
    """Return a review label."""
    return f"Review {review_number}"


def format_diffs(diffs: list) -> str:
    all_paths = [d["new_path"] for d in diffs if d.get("diff")]
    skipped = [d["new_path"] for d in diffs if not d.get("diff") and d.get("new_path")]
    if skipped:
        print(f"Note: {len(skipped)} file(s) had no diff content (possibly too large): {', '.join(skipped)}")
    parts = []
    for d in diffs:
        if not d.get("diff"):
            continue
        header = f"### {d['new_path']}"
        if d.get("_collapsed_fallback"):
            header += "  ⚠️ (diff collapsed by GitHub — showing full file content)"
        if d.get("renamed_file"):
            header += f"  (renamed from {d['old_path']})"
        parts.append(f"{header}\n```diff\n{d['diff']}\n```")
    result = ""
    for part in parts:
        if result and len(result) + len(part) + 2 > MAX_DIFF_CHARS:
            result += (
                f"\n\n[... diff truncated due to size ...]\n\n"
                f"**All changed files ({len(all_paths)}):** "
                + ", ".join(f"`{p}`" for p in all_paths)
            )
            break
        result += ("\n\n" if result else "") + part
    return result


def find_previous_bot_reviews(discussions: list) -> list[dict]:
    """Return info on previous AI review discussions (resolved or not).

    Includes all notes in each discussion thread so that developer replies
    (e.g. dismissals or clarifications) are visible to the follow-up review.
    """
    reviews = []
    for disc in discussions:
        notes = disc.get("notes", [])
        if notes and BOT_MARKER in notes[0].get("body", ""):
            replies = [
                {"author": n.get("author", {}).get("username", "unknown"), "body": n["body"]}
                for n in notes[1:]
                if n.get("body")
            ]
            reviews.append({
                "id": disc["id"],
                "note_id": str(notes[0]["id"]) if notes[0].get("id") is not None else None,
                "resolved": disc.get("resolved", False),
                "body": notes[0]["body"],
                "replies": replies,
            })
    return reviews


def _extract_issue_summaries(body: str) -> str:
    """Extract a compact summary of issues from a bot review body.

    Parses the markdown sections and returns short titles/descriptions
    rather than the full review text, keeping token usage low.
    """
    sections_of_interest = [
        "🔴 Critical Issues",
        "🟡 Important Changes",
        "🟢 Suggestions",
        "📋 Ticket Compliance",
    ]
    lines = body.split("\n")
    summaries = []
    current_section = None
    for line in lines:
        # Detect section headers
        if line.startswith("## "):
            current_section = None
            for section in sections_of_interest:
                if section in line:
                    current_section = line.strip()
                    summaries.append(current_section)
                    break
        elif current_section:
            stripped = line.strip()
            # Skip empty lines and "None" placeholders
            if not stripped or re.match(r"^(None\.?|No new|No critical|No blocking)", stripped, re.IGNORECASE):
                continue
            summaries.append(stripped)

    return "\n".join(summaries) if summaries else "No issues were flagged."


def inject_critical_issue(review_text: str, issue_text: str) -> str:
    """Insert a Critical Issue entry into the review text.

    Handles three cases:
    - Section has "None" as its only content → replace with the new item
    - Section already has numbered items → prepend as item 1, renumber the rest
    - Section is missing entirely → insert the full section before Important Changes

    Works line-by-line rather than with cross-section regexes so that embedded
    markdown headings inside code blocks cannot corrupt the section boundaries.
    """
    section_header = "## 🔴 Critical Issues (Must Fix)"
    new_item = f"1. {issue_text}"
    lines = review_text.splitlines()

    # Locate the Critical Issues header line
    header_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == section_header),
        None,
    )

    if header_idx is None:
        # Section missing — insert it before Important Changes, or at the top
        anchor_idx = next(
            (i for i, line in enumerate(lines) if line.startswith("## 🟡")),
            None,
        )
        insertion = [section_header, new_item, ""]
        if anchor_idx is not None:
            lines[anchor_idx:anchor_idx] = insertion
        else:
            lines[0:0] = insertion
        return "\n".join(lines)

    # Locate the start of the next section (first "## " line after the header)
    next_section_idx = next(
        (i for i in range(header_idx + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )

    # Extract non-empty body lines between the header and the next section
    body = "\n".join(
        line for line in lines[header_idx + 1:next_section_idx] if line.strip()
    ).strip()

    if not body or body.lower() == "none":
        new_body_lines = [new_item]
    else:
        # Shift all existing numbered items down by one
        shifted = re.sub(
            r"^(\d+)\.",
            lambda m: f"{int(m.group(1)) + 1}.",
            body,
            flags=re.MULTILINE,
        )
        new_body_lines = [new_item] + shifted.splitlines()

    reconstructed = (
        lines[:header_idx + 1]
        + [""]
        + new_body_lines
        + [""]
        + lines[next_section_idx:]
    )
    return "\n".join(reconstructed)


def _truncate_error(exc: Exception) -> str:
    """Return the first line of an exception message, truncated with '...' if needed."""
    msg = str(exc).splitlines()[0] if str(exc) else ""
    return (msg[:197] + "...") if len(msg) > 200 else msg


def _section_has_content(lines: list[str], marker: str) -> bool:
    """Return True if the section starting with *marker* has non-'None' content.

    Handles the subsection format used in follow-up reviews (### Newly Identified,
    ### Unresolved from Previous Review) by ignoring subsection headers and checking
    whether any remaining content line has real findings.
    """
    header_idx = next(
        (i for i, line in enumerate(lines) if line.startswith(marker)),
        None,
    )
    if header_idx is None:
        return False
    next_section_idx = next(
        (i for i in range(header_idx + 1, len(lines)) if lines[i].startswith("## ")),
        len(lines),
    )
    _no_content = ("none", "no new", "no critical", "no blocking")
    content_lines = [
        line.strip() for line in lines[header_idx + 1:next_section_idx]
        if line.strip() and not line.strip().startswith("### ")
    ]
    if not content_lines:
        return False
    return any(
        not line.lower().startswith(_no_content) for line in content_lines
    )


def _ticket_issue_dismissed(previous_reviews: list[dict]) -> bool:
    """Return True if a developer has dismissed the 'no ticket found' issue.

    Scans previous bot reviews that contained the injected ticket warning
    and checks whether any developer reply indicates a dismissal.
    """
    dismissal_re = re.compile(
        r"(won'?t\s*fix|intentional|not\s*applicable|acknowledged"
        r"|disregard|no\s*(?:gitlab|github)?\s*ticket|ignore)",
        re.IGNORECASE,
    )
    for review in previous_reviews:
        if NO_TICKET_SENTINEL not in review.get("body", ""):
            continue
        for reply in review.get("replies", []):
            # Skip bot replies — only human dismissals count
            if BOT_MARKER in reply.get("body", ""):
                continue
            if dismissal_re.search(reply.get("body", "")):
                return True
    return False


def _filter_resolved_ticket_findings(issue_summary: str, ticket_number: str | None) -> str:
    """Drop stale no-ticket findings once the current review has a valid ticket."""
    if not ticket_number:
        return issue_summary

    stale_patterns = (
        NO_TICKET_SENTINEL.lower(),
        "no ticket was found",
        "no ticket available for analysis",
        "ticket compliance\nno ticket",
        "skipped — no ticket",
        "skipping acceptance criteria review",
    )
    kept_lines = [
        line for line in issue_summary.splitlines()
        if not any(pattern in line.lower() for pattern in stale_patterns)
    ]

    pruned_lines = []
    index = 0
    while index < len(kept_lines):
        line = kept_lines[index]
        if line.startswith("## "):
            next_index = index + 1
            has_section_content = False
            while next_index < len(kept_lines) and not kept_lines[next_index].startswith("## "):
                if kept_lines[next_index].strip():
                    has_section_content = True
                    break
                next_index += 1
            if has_section_content:
                pruned_lines.append(line)
        elif line.strip():
            pruned_lines.append(line)
        index += 1

    return "\n".join(pruned_lines).strip()


def build_previous_review_context(previous_reviews: list[dict], ticket_number: str | None) -> str:
    """Summarize previous bot reviews for Claude, omitting resolved no-ticket findings."""
    if not previous_reviews:
        return ""

    max_summary_chars = 1200
    context_parts = []
    all_replies = []

    for review in previous_reviews:
        status = "RESOLVED" if review["resolved"] else "STILL OPEN"
        issue_summary = _extract_issue_summaries(review["body"])
        issue_summary = _filter_resolved_ticket_findings(issue_summary, ticket_number)
        if len(issue_summary) > max_summary_chars:
            issue_summary = issue_summary[:max_summary_chars] + "\n[... summary truncated ...]"
        if issue_summary:
            context_parts.append(f"--- Review {review['id']} [{status}] ---\n{issue_summary}")

        for reply in review.get("replies", []):
            all_replies.append(
                f"[Re: Review {review['id']}] @{reply['author']}: {reply['body']}"
            )

    previous_review_context = "\n\n".join(context_parts)
    if all_replies:
        prefix = "\n\n" if previous_review_context else ""
        previous_review_context += prefix + "--- Developer Replies ---\n" + "\n\n".join(all_replies)
    if len(previous_review_context) > MAX_CONTEXT_CHARS:
        previous_review_context = (
            "[... earlier context truncated ...]\n\n"
            + previous_review_context[-MAX_CONTEXT_CHARS:]
        )
    return previous_review_context


def _normalize_empty_follow_up_sections(review_text: str) -> str:
    """Collapse empty follow-up severity sections to a single `None` body.

    Claude is asked to emit `### Newly Identified` and `### Unresolved from Previous Review`
    subsections for follow-up reviews. When both subsections are effectively empty, the UI is
    easier to scan if the section body is simply `None`.
    """
    lines = review_text.splitlines()
    section_headers = {
        "## 🔴 Critical Issues (Must Fix)",
        "## 🟡 Important Changes (Should Fix)",
        "## 🟢 Suggestions (Nice to Have)",
    }
    empty_markers = (
        "none",
        "no new",
        "no critical",
        "no blocking",
        "all previous issues resolved.",
    )

    normalized: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line not in section_headers:
            normalized.append(line)
            index += 1
            continue

        next_section_idx = next(
            (i for i in range(index + 1, len(lines)) if lines[i].startswith("## ")),
            len(lines),
        )
        body_lines = lines[index + 1:next_section_idx]
        has_follow_up_subsections = any(
            body_line.startswith("### Newly Identified") or body_line.startswith("### Unresolved from Previous Review")
            for body_line in body_lines
        )
        body_content = [
            body_line.strip()
            for body_line in body_lines
            if body_line.strip() and not body_line.startswith("### ")
        ]
        is_empty_follow_up = has_follow_up_subsections and (
            not body_content
            or all(content.lower().startswith(empty_markers) for content in body_content)
        )

        normalized.append(line)
        normalized.append("")
        if is_empty_follow_up:
            normalized.append("None")
            normalized.append("")
        else:
            normalized.extend(body_lines)
        index = next_section_idx

    return "\n".join(normalized)


def collapse_previous_reviews(
    client: GitHubClient, pr_number: str, previous_reviews: list[dict], current_review_number: int
) -> None:
    """Wrap previous bot review notes in <details> tags so only the latest is expanded."""
    for i, review in enumerate(previous_reviews):
        note_id = review.get("note_id")
        body = review.get("body", "")
        # Guard: skip notes already collapsed. This prevents nested <details>
        # blocks if collapse runs twice (e.g. retry after partial failure).
        if not note_id or COLLAPSED_MARKER in body:
            continue
        review_number = i + 1
        content = body.replace(BOT_MARKER, "", 1).strip()
        collapsed_body = (
            f"{BOT_MARKER}\n{COLLAPSED_MARKER}\n"
            f"<details>\n"
            f"<summary><strong>🤖 AI Code Review ({format_review_label(review_number)})</strong>"
            f" — <em>superseded by {format_review_label(current_review_number)}</em></summary>\n\n"
            f"{content}\n"
            f"</details>"
        )
        try:
            client.update_note(pr_number, note_id, collapsed_body)
            print(f"Collapsed previous review #{review_number} (note {note_id}).")
        except Exception as exc:
            print(f"Warning: could not collapse review #{review_number} (note {note_id}): {exc}")


def has_blocking_issues(review_text: str) -> bool:
    """Return True if the review contains Critical findings.

    Used to set the exit code (1 = warning in pipeline). The CI job has
    allow_failure: true, so this never actually blocks a merge.
    """
    lines = review_text.splitlines()
    return _section_has_content(lines, "## 🔴 Critical Issues")


# ---------------------------------------------------------------------------
# Generic Workstream & Technology Criteria Detection
# ---------------------------------------------------------------------------

def detect_workstreams(diffs: list, workstreams_config: list[dict]) -> dict[str, bool]:
    """Detect which technology workstreams are touched by the diff based on config.json patterns."""
    paths = [d.get("new_path") or d.get("old_path") for d in diffs
             if d.get("new_path") or d.get("old_path")]

    detected = {}
    for ws in workstreams_config:
        name = ws["name"]
        pattern = ws["pattern"]
        try:
            regex = re.compile(pattern)
            detected[name] = any(regex.search(p) for p in paths)
        except Exception as e:
            print(f"Warning: Invalid regex pattern '{pattern}' for workstream '{name}': {e}")
            detected[name] = False
    return detected


def build_technology_context(workstreams: dict[str, bool], workstreams_config: list[dict]) -> str:
    """Return technology-specific review criteria for the detected workstreams."""
    parts = []
    for ws in workstreams_config:
        name = ws["name"]
        if workstreams.get(name):
            criteria_path = _REPO_ROOT / ws["criteria_file"]
            try:
                content = criteria_path.read_text(encoding="utf-8")
                parts.append(content)
            except FileNotFoundError:
                print(f"Warning: criteria file not found for workstream '{name}': {criteria_path}")
            except Exception as e:
                print(f"Warning: Failed to read criteria file for workstream '{name}': {e}")
    return "\n\n".join(filter(None, parts))


# ---------------------------------------------------------------------------
# Claude & Gemini AI integrations
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI code reviewer embedded in a GitHub Actions workflow.

Review every PR diff against these eight dimensions:
1. Architecture & Design — separation of concerns, dependency injection, layering, DTO usage, immutability
2. Code Quality — readability, naming, cyclomatic complexity, DRY, KISS
3. Error Handling & Validation — exception strategy, input validation, informative error messages, graceful failure
4. Testing — >80% coverage for new code, behavior-based tests, edge cases covered
5. Security — input sanitization, authentication/authorization on endpoints, no hardcoded secrets, dependency vulnerabilities
6. Performance — obvious bottlenecks, resource management, caching appropriateness, N+1 query patterns
7. Completeness — no unresolved TODOs, complex logic is documented, significant events are logged
8. Accuracy — actively cross-reference the ticket description and each acceptance criterion against the code changes; flag any criterion that is not addressed or only partially implemented as a Critical Issue

## Severity Guidelines

Use Critical sparingly. Only flag Critical Issues for things that are genuinely broken, \
insecure (e.g. leaked secrets, injection vulnerabilities), or violate a ticket acceptance criterion. \
Do NOT flag as Critical: style preferences, hypothetical edge cases, missing nice-to-have features, \
operational suggestions (e.g. "move CI variables to Settings"), or theoretical race conditions on \
ephemeral CI containers.

Important Changes are for real code quality concerns that should be addressed but do not block shipping. \
Suggestions are for everything else.

All findings are advisory — nothing blocks the merge. Critical Issues signal the highest \
priority for human review. Important Changes and Suggestions are lower priority.

Output your review using EXACTLY these section headers (include the emoji):

## ✅ Strengths
- [what the code does well]

## 🔴 Critical Issues (Must Fix)
1. **[Short title]**: `[file:line]` [Explanation of why this is critical and how to fix it]

## 🟡 Important Changes (Should Fix)
1. **[Short title]**: `[file:line]` [Explanation and suggested fix]

## 🟢 Suggestions (Nice to Have)
- Limit to the THREE most impactful suggestions only. Do not list more than three.

## 📋 Ticket Compliance
- Only list acceptance criteria that are NOT met or only partially met.
- If all criteria are satisfied, write "All acceptance criteria are satisfied."
- If no ticket was found, note that.

## 🧪 Test Automation Analysis
- Compare each ticket acceptance criterion against the test files present in the diff.
- For each AC, state whether unit tests and integration tests cover it: Covered, Partially covered, or Missing.
- End the section with exactly this subsection heading: `### Suggested Test Prompts`.
- Under that heading, include exactly these subsection headings in this order: `#### Unit Tests` and `#### Integration Tests`.
- For every missing or partially covered gap, include copy-paste-ready fenced `text` blocks that tell a coding AI exactly what tests to add, referencing relevant files, test classes, and acceptance criteria when possible.
- Include ALL suggested unit and integration test prompts at the bottom of the section, not just examples.
- If one category has no gaps, write `None` under that subsection heading.
- If no ticket was found, write "Skipped — no ticket available for analysis."

If a section has no findings, write the word "None" as the only content under that heading.
Be specific and actionable. Reference file paths and line numbers wherever possible.

## Follow-Up Review Format

For FOLLOW-UP reviews (when previous review context is provided):
- Omit the "## ✅ Strengths" section entirely (it was already covered in the initial review).
- Within each issue section (🔴 Critical, 🟡 Important, 🟢 Suggestions), use TWO subsections:
  ### Newly Identified
  [issues found for the first time in this review]
  ### Unresolved from Previous Review
  [issues from prior reviews that remain unfixed and were not dismissed by a developer]
- If a subsection has no findings, write "None" under that subsection heading.
- Resolved issues must be OMITTED entirely from "Unresolved from Previous Review" — do \
  NOT list them with a "resolved" annotation. If all previous issues in a severity level \
  have been resolved, write "All previous issues resolved." under that subsection.
- Do NOT move issues between severity levels across reviews — if an issue was Critical \
  before, it stays Critical.
- If BOTH follow-up subsections in a severity section would be empty (for example, `None` + `None`, or `None` + `All previous issues resolved.`), do NOT emit subsection headings for that severity section. Instead, write only `None` as the section body.
- If the current Ticket Information includes a valid ticket, any earlier "No ticket number found" \
  finding is resolved and must be omitted entirely. Do NOT keep it open just because the \
  branch name itself still lacks an issue number.

## Developer Dismissals

Previous review context may include developer replies to AI-raised issues. Treat an issue \
as **dismissed** — and omit it entirely from all sections — if a developer reply indicates \
any of the following:
- Will not fix: "won't fix", "intentional", "not applicable", "acknowledged"
- Scope deferral: "deferred", "out of scope", "different MR", "separate MR", "future MR", \
  "addressed in another MR", "handled separately", "next phase", "Phase 2", "Phase 3"
- General dismissal: "disregard", "ignore", "no ticket"

Do NOT re-raise dismissed issues in the follow-up review. Only re-raise a previously \
dismissed issue if the new code changes make it materially worse.
"""


def call_llm(
    diff_text: str,
    ticket_info: str,
    previous_review_context: str,
    tech_context: str = "",
) -> str:
    follow_up_instruction = ""
    if previous_review_context:
        follow_up_instruction = (
            "\n\nThis is a FOLLOW-UP review after new commits were pushed. "
            "Use the follow-up review format from the system prompt: within each "
            "issue section (🔴, 🟡, 🟢), use '### Newly Identified' and "
            "'### Unresolved from Previous Review' subsections. "
            "Do NOT include a '## 🔄 Previous Issues' section — issues stay in their "
            "severity sections with subsection labels instead. "
            "Omit the '## ✅ Strengths' section entirely (it was covered in the initial review). "
            "Issues that have been fixed or dismissed by a developer should be omitted "
            "entirely from 'Unresolved from Previous Review' — do not list them with a "
            "'resolved' annotation. If all previous issues in a severity level were resolved, "
            "write 'All previous issues resolved.' under that subsection. "
            "Developer replies (including scope deferrals "
            "like 'deferred to another MR', 'out of scope', or 'Phase 2/3') are shown in "
            "the 'Developer Replies' section of the previous review context. "
            "For new issues: only flag genuinely new problems introduced by the "
            "latest changes. Do NOT invent new Important or Critical issues that were not "
            "flagged in prior reviews — if the prior review did not flag it, it is not "
            "a new blocker. Focus on convergence, not perfection.\n"
        )

    tech_section = (
        f"\n\n## Technology-Specific Review Criteria\n{tech_context}"
        if tech_context
        else ""
    )

    user_content = f"""\
Please review the following code changes.{follow_up_instruction}

## Ticket Information
{ticket_info}

## Previous Review Context
{previous_review_context if previous_review_context else "No previous review — this is the initial review."}

## Code Diff
{diff_text}{tech_section}
"""

    if PROVIDER == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured in the environment.")
        if genai is None:
            raise ImportError("google-genai is not installed. Run 'uv pip install --require-hashes -r .github/pr-reviewer/requirements.txt'.")
        
        client = genai.Client(api_key=api_key)
        print(f"Calling Gemini ({MODEL}) via direct API...")
        response = client.models.generate_content(
            model=MODEL,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.2,
            )
        )
        return response.text

    elif PROVIDER == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured in the environment.")
        if Anthropic is None:
            raise ImportError("anthropic is not installed. Run 'uv pip install --require-hashes -r .github/pr-reviewer/requirements.txt'.")
        
        client = Anthropic(api_key=api_key)
        print(f"Calling Claude ({MODEL}) via direct Anthropic API...")
        message = client.messages.create(
            model=MODEL,
            max_tokens=4096,
            temperature=0.2,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}]
        )
        return message.content[0].text

    elif PROVIDER == "vertex-ai":
        gcp_project_id = os.environ.get("GCP_PROJECT_ID")
        gcp_region = os.environ.get("GCP_REGION", "us-east5")
        if not gcp_project_id:
            raise ValueError("GCP_PROJECT_ID is required for Vertex AI provider.")
        if AnthropicVertex is None:
            raise ImportError("anthropic[vertex] is not installed. Run 'uv pip install --require-hashes -r .github/pr-reviewer/requirements.txt'.")
        
        client = AnthropicVertex(project_id=gcp_project_id, region=gcp_region)
        print(f"Calling Claude ({MODEL}) via GCP Vertex AI...")

        last_exc = None
        for attempt in range(3):
            try:
                message = client.messages.create(
                    model=MODEL,
                    max_tokens=4096,
                    temperature=0.2,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_content}],
                )
                return message.content[0].text
            except Exception as exc:
                last_exc = exc
                # simple backoff
                time.sleep(2 ** attempt)
        raise last_exc
    else:
        raise ValueError(f"Unsupported provider: {PROVIDER}")


# ---------------------------------------------------------------------------
# MR posting
# ---------------------------------------------------------------------------

def build_review_body(review_text: str, is_follow_up: bool, review_number: int = 1) -> str:
    follow_up_tag = "_Follow-up review after new commits._\n\n" if is_follow_up else ""
    return (
        f"{BOT_MARKER}\n## 🤖 AI Code Review ({format_review_label(review_number)})\n"
        f"{follow_up_tag}{review_text}"
    )


def post_review(client: GitHubClient, pr_number: str, review_text: str, is_follow_up: bool, review_number: int = 1) -> None:
    body = build_review_body(review_text, is_follow_up, review_number)
    # Always post as a regular comment so the review is purely advisory and never
    # blocks merges.
    try:
        client.create_note(pr_number, body)
        print("Posted review as a regular PR comment.")
    except requests.HTTPError as exc:
        if getattr(exc, 'response', None) is not None and exc.response.status_code == 403:
            print("Warning: Forbidden (403) when posting review. Check GITHUB_TOKEN permissions.", file=sys.stderr)
            return
        raise


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    if "GITHUB_ACTIONS" not in os.environ:
        raise RuntimeError("This reviewer can only be executed in a GitHub Actions environment.")

    server_url = os.environ.get("GITHUB_SERVER_URL", "https://github.com")
    repository = os.environ.get("GITHUB_REPOSITORY")
    if not repository:
        raise ValueError("GITHUB_REPOSITORY environment variable is not configured.")
    branch_name = os.environ.get("GITHUB_REF_NAME", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise ValueError("GITHUB_TOKEN is not configured (empty string)")
    
    pr_number = os.environ.get("GITHUB_PR_NUMBER")
    if not pr_number and "GITHUB_EVENT_PATH" in os.environ:
        try:
            with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
                event = json.load(f)
            if "pull_request" in event:
                pr_number = str(event["pull_request"]["number"])
            elif "issue" in event:
                pr_number = str(event["issue"]["number"])
        except Exception as e:
            print(f"Warning: Failed to parse GITHUB_EVENT_PATH: {e}")
    
    if not pr_number:
        raise ValueError("Could not determine GITHUB_PR_NUMBER or pull request number from event.")
        
    client = GitHubClient(server_url, repository, token)

    # Initialised early so the error handler always has context, even if the
    # exception is raised before the value is computed from the API response.
    is_follow_up = False
    no_ticket_found = False
    review_number = 1

    try:
        # --- Ticket ---
        # 1. Try the standard branch naming convention
        ticket_number = extract_ticket_number(branch_name)
        ticket_source = "branch convention"

        # 2. Fallback: look for any number in the branch name and validate it
        if not ticket_number:
            candidate = extract_ticket_number_fallback(branch_name)
            if candidate and client.get_issue(candidate):
                ticket_number = candidate
                ticket_source = "branch name (fallback)"

        # 3. Fallback: check the PR description for issue references.
        if not ticket_number:
            pr = client.get_pr(pr_number)
            if pr:
                pr_text = "\n\n".join(filter(None, [pr.get("title", ""), pr.get("description", "")]))
                candidate = extract_ticket_from_description(pr_text)
                if candidate:
                    ticket_number = candidate
                    ticket_source = "PR title/description"

        no_ticket_found = ticket_number is None
        if ticket_number:
            issue = client.get_issue(ticket_number)
            if issue:
                ticket_info = (
                    f"**Issue #{ticket_number}: {issue['title']}**\n\n"
                    f"{issue.get('description') or 'No description provided.'}"
                )
                print(f"Ticket #{ticket_number} found via {ticket_source}.")
            else:
                ticket_info = f"Ticket #{ticket_number} was not found in this project."
        else:
            ticket_info = f"No ticket number detected in branch name `{branch_name}` or PR description. Skip acceptance criteria review."

        # --- Diff ---
        diffs = client.get_pr_diffs(pr_number)
        head_sha = os.environ.get("GITHUB_SHA", "HEAD")
        if "GITHUB_EVENT_PATH" in os.environ:
            try:
                with open(os.environ["GITHUB_EVENT_PATH"], "r") as f:
                    event = json.load(f)
                if "pull_request" in event:
                    head_sha = event["pull_request"]["head"]["sha"]
            except Exception:
                pass
        for d in diffs:
            if (d.get("collapsed") or d.get("too_large")) and not d.get("deleted_file"):
                raw = client.get_file_raw(head_sha, d["new_path"])
                if raw:
                    d["diff"] = "\n".join(f"+{line}" for line in raw.splitlines())
                    d["_collapsed_fallback"] = True
                    print(f"Note: {d['new_path']} was collapsed by GitHub — fetched full file content.")
                else:
                    d["diff"] = "[file content could not be retrieved]\n"
                    print(f"Warning: {d['new_path']} was collapsed and raw content could not be fetched.")
        diff_text = format_diffs(diffs)
        if not diff_text.strip():
            print("No diff content found. Skipping review.")
            return 0

        # --- Generic Workstream detection ---
        workstreams = detect_workstreams(diffs, _WORKSTREAMS_CONFIG)
        tech_context = build_technology_context(workstreams, _WORKSTREAMS_CONFIG)
        active = [k for k, v in workstreams.items() if v]
        if active:
            print(f"Detected workstreams: {', '.join(active)}")
        else:
            print("No specific workstream detected — using generic review criteria.")

        # --- Previous reviews ---
        discussions = client.get_pr_discussions(pr_number)
        previous_reviews = find_previous_bot_reviews(discussions)
        is_follow_up = bool(previous_reviews)

        previous_review_context = build_previous_review_context(previous_reviews, ticket_number)

        # --- Review ---
        print(f"Running AI code review for PR #{pr_number} (branch: {branch_name})...")
        review_text = call_llm(diff_text, ticket_info, previous_review_context, tech_context)

        if no_ticket_found and not _ticket_issue_dismissed(previous_reviews):
            review_text = inject_critical_issue(
                review_text,
                f"**{NO_TICKET_SENTINEL}**: No recognizable issue number was found in branch "
                f"`{branch_name}` or the PR description, so no review was performed against a "
                f"ticket's description or acceptance criteria. Rename the branch to include the "
                f"GitHub issue number (e.g., `feat/123_short-description`) or reference the "
                f"issue in the PR description (e.g., `#123`).",
            )

        review_text = _normalize_empty_follow_up_sections(review_text)

        # --- Post ---
        review_number = len(previous_reviews) + 1

        # Collapse older reviews before posting so readers never see two expanded
        if previous_reviews:
            collapse_previous_reviews(client, pr_number, previous_reviews, review_number)

        post_review(client, pr_number, review_text, is_follow_up, review_number)

        # --- Exit code drives CI pass/fail ---
        if has_blocking_issues(review_text):
            print("\nCritical issues found — CI job will show as a warning (merge not blocked).")
            return 1

        print("\nReview complete. No blocking issues found.")
        return 0

    except Exception as exc:
        # Configuration errors (401/403) should fail the pipeline so they're noticed immediately.
        is_config_error = False

        # GitHub API auth errors (GITHUB_TOKEN issues)
        resp = getattr(exc, 'response', None)
        if isinstance(exc, requests.HTTPError) and resp is not None and resp.status_code in (401, 403):
            is_config_error = True

        # Check for permission/authentication exceptions across providers
        exc_name = type(exc).__name__
        if exc_name in ('APIError', 'APIConnectionError', 'ClientError', 'HTTPError'):
            error_str = str(exc).lower()
            if any(keyword in error_str for keyword in ['permission', 'denied', '403', 'unauthorized', '401']):
                is_config_error = True

        print(f"ERROR: AI code reviewer encountered an unexpected error: {exc}", file=sys.stderr)
        follow_up_tag = "_Follow-up review after new commits._\n\n" if is_follow_up else ""
        try:
            client.create_note(
                pr_number,
                f"{BOT_MARKER}\n## 🤖 AI Code Review ({format_review_label(review_number)})\n"
                f"{follow_up_tag}"
                f"⚠️ The reviewer failed with an error and could not complete the review.\n\n"
                f"```\n{type(exc).__name__}: {_truncate_error(exc)}\n```\n\n"
                f"Please check the CI job logs for details.",
            )
        except Exception:
            pass

        if is_config_error:
            print("\nConfiguration error detected (401/403) — failing the CI job.", file=sys.stderr)
            return 1
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)