#!/usr/bin/env python3
"""Unit tests for review.py helper functions."""

import sys
import types
import unittest
import urllib.parse
from unittest.mock import patch, MagicMock
from itertools import product

# Stub out dependencies that might not be installed locally during test run
anthropic_stub = types.ModuleType("anthropic")


class _AnthropicVertex:
    pass


class _APIError(Exception):
    status_code = None


anthropic_stub.AnthropicVertex = _AnthropicVertex
anthropic_stub.APIError = _APIError
sys.modules.setdefault("anthropic", anthropic_stub)


class MockHTTPError(Exception):
    def __init__(self, *args, **kwargs):
        self.response = kwargs.pop("response", None)
        super().__init__(*args, **kwargs)

requests_stub = types.ModuleType("requests")
requests_stub.HTTPError = MockHTTPError
requests_stub.utils = types.SimpleNamespace(quote=urllib.parse.quote)
requests_stub.get = lambda *args, **kwargs: None
requests_stub.post = lambda *args, **kwargs: None
requests_stub.patch = lambda *args, **kwargs: None
sys.modules.setdefault("requests", requests_stub)

from review import (  # noqa: E402
    _ticket_issue_dismissed,
    _normalize_empty_follow_up_sections,
    _truncate_error,
    build_previous_review_context,
    build_review_body,
    collapse_previous_reviews,
    extract_ticket_number,
    extract_ticket_number_fallback,
    extract_ticket_from_description,
    find_previous_bot_reviews,
    format_diffs,
    has_blocking_issues,
    inject_critical_issue,
    detect_workstreams,
    SYSTEM_PROMPT,
    BOT_MARKER,
    COLLAPSED_MARKER,
    NO_TICKET_SENTINEL,
    GitHubClient,
    post_review,
)


class TestExtractTicketNumber(unittest.TestCase):
    def test_standard_feature_branch(self):
        self.assertEqual(extract_ticket_number("feat/387_mr-code-reviewer"), "387")

    def test_feature_prefix(self):
        self.assertEqual(extract_ticket_number("feature/387-mr-code-reviewer"), "387")

    def test_nested_path(self):
        self.assertEqual(extract_ticket_number("feat/web/244-fix-alignment"), "244")

    def test_fix_branch(self):
        self.assertEqual(extract_ticket_number("fix/112-radio-button"), "112")

    def test_no_ticket(self):
        self.assertIsNone(extract_ticket_number("main"))

    def test_no_separator_after_number(self):
        self.assertIsNone(extract_ticket_number("feat/v2"))

    def test_single_digit_ignored(self):
        self.assertIsNone(extract_ticket_number("feat/python-3-12"))

    def test_single_digit_branch(self):
        self.assertEqual(extract_ticket_number("feat/1_gemini-3.5-flash"), "1")

    def test_number_not_after_slash(self):
        self.assertIsNone(extract_ticket_number("feat-387-thing"))

    def test_bare_number_at_start(self):
        self.assertEqual(extract_ticket_number("387-fix-thing"), "387")


class TestExtractTicketNumberFallback(unittest.TestCase):
    def test_number_not_after_slash(self):
        """Primary pattern misses this, but fallback catches it."""
        self.assertEqual(extract_ticket_number_fallback("feat-387-thing"), "387")

    def test_number_in_middle(self):
        self.assertEqual(extract_ticket_number_fallback("some-branch-42-name"), "42")

    def test_no_number(self):
        self.assertIsNone(extract_ticket_number_fallback("main"))

    def test_single_digit_ignored(self):
        self.assertIsNone(extract_ticket_number_fallback("v2"))

    def test_version_like_number(self):
        """Returns the first multi-digit number; caller must validate against tracker."""
        self.assertEqual(extract_ticket_number_fallback("feat/python-312-update"), "312")


class TestExtractTicketFromDescription(unittest.TestCase):
    def test_non_matches(self):
        cases = [
            ("no reference", "No ticket here"),
            ("empty description", ""),
            ("none description", None),
        ]

        for label, description in cases:
            with self.subTest(label=label, description=description):
                self.assertIsNone(extract_ticket_from_description(description))

    def test_hash_reference_matrix(self):
        cases = [
            ("single digit hash", "See item #5", "5"),
            ("relates", "Relates to #456", "456"),
            ("closes", "Closes #789", "789"),
            ("multiple hashes returns first", "Fix #100 and #200", "100"),
            ("hash preferred over keyword", "See #999 and ticket 1285", "999"),
        ]

        for label, description, expected in cases:
            with self.subTest(label=label, description=description):
                self.assertEqual(extract_ticket_from_description(description), expected)

    def test_keyword_reference_matrix(self):
        keyword_cases = [
            ("ticket", "ticket"),
            ("issue", "issue"),
            ("task", "task"),
            ("work item", "work item"),
        ]
        case_prefixes = ["See", "Review"]
        number_formats = [("plain", "1285"), ("hash", "#1285")]

        for (keyword_label, keyword), prefix, (number_label, number) in product(
            keyword_cases,
            case_prefixes,
            number_formats,
        ):
            description = f"{prefix} {keyword} {number}"
            with self.subTest(keyword=keyword_label, prefix=prefix, number_format=number_label):
                self.assertEqual(extract_ticket_from_description(description), "1285")

    def test_keyword_reference_is_case_insensitive(self):
        cases = [
            "See Ticket 1285",
            "See ISSUE 1285",
            "fix reviewer UX (TASK 1285)",
            "Review Work Item 1285",
        ]

        for description in cases:
            with self.subTest(description=description):
                self.assertEqual(extract_ticket_from_description(description), "1285")

    def test_link_and_url_reference_matrix(self):
        cases = [
            (
                "markdown linked work item",
                "See ticket [**1285**]"
                "(https://gitlab.com/generic/appdev/generic-appdev/-/work_items/1285)"
                "** for requirements**",
                "1285",
            ),
            (
                "work item url",
                "Requirements live at https://gitlab.com/generic/appdev/generic-appdev/-/work_items/1285",
                "1285",
            ),
            (
                "issue url",
                "Requirements live at https://gitlab.com/generic/appdev/generic-appdev/-/issues/1285",
                "1285",
            ),
        ]

        for label, description, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(extract_ticket_from_description(description), expected)


class TestFormatDiffs(unittest.TestCase):
    def test_basic_diff(self):
        diffs = [{"new_path": "foo.py", "diff": "+hello"}]
        result = format_diffs(diffs)
        self.assertIn("### foo.py", result)
        self.assertIn("+hello", result)

    def test_renamed_file(self):
        diffs = [{"new_path": "new.py", "old_path": "old.py", "diff": "+x", "renamed_file": True}]
        result = format_diffs(diffs)
        self.assertIn("renamed from old.py", result)

    def test_empty_diff_skipped(self):
        diffs = [{"new_path": "foo.py", "diff": ""}, {"new_path": "bar.py", "diff": "+ok"}]
        result = format_diffs(diffs)
        self.assertNotIn("foo.py", result)
        self.assertIn("bar.py", result)

    def test_no_diff_key_skipped(self):
        diffs = [{"new_path": "foo.py"}]
        result = format_diffs(diffs)
        self.assertEqual(result, "")


class TestHasBlockingIssues(unittest.TestCase):
    def test_critical_with_content(self):
        text = "## 🔴 Critical Issues (Must Fix)\n1. Something bad\n## 🟡 Important Changes\nNone"
        self.assertTrue(has_blocking_issues(text))

    def test_important_alone_does_not_block(self):
        text = "## 🔴 Critical Issues (Must Fix)\nNone\n## 🟡 Important Changes (Should Fix)\n1. Fix this"
        self.assertFalse(has_blocking_issues(text))

    def test_both_none(self):
        text = "## 🔴 Critical Issues (Must Fix)\nNone\n## 🟡 Important Changes (Should Fix)\nNone"
        self.assertFalse(has_blocking_issues(text))

    def test_no_sections(self):
        self.assertFalse(has_blocking_issues("All good"))

    def test_code_block_with_heading_not_confused(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "1. **Problem**: Code has:\n"
            "```\n## some markdown heading in code\n```\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertTrue(has_blocking_issues(text))


class TestInjectCriticalIssue(unittest.TestCase):
    def test_replace_none(self):
        text = "## 🔴 Critical Issues (Must Fix)\nNone\n## 🟡 Important Changes (Should Fix)\nNone"
        result = inject_critical_issue(text, "**Test**: Something")
        self.assertIn("1. **Test**: Something", result)
        self.assertNotIn("None", result.split("## 🟡")[0])

    def test_prepend_to_existing(self):
        text = "## 🔴 Critical Issues (Must Fix)\n1. Existing issue\n## 🟡 Important Changes (Should Fix)\nNone"
        result = inject_critical_issue(text, "**New**: First")
        lines = result.splitlines()
        critical_start = next(i for i, line in enumerate(lines) if "Critical Issues" in line)
        body_lines = [line for line in lines[critical_start + 1:] if line.strip() and not line.startswith("## ")]
        self.assertTrue(body_lines[0].startswith("1. **New**: First"))
        self.assertTrue(body_lines[1].startswith("2. Existing issue"))

    def test_missing_section(self):
        text = "## 🟡 Important Changes (Should Fix)\n1. Something"
        result = inject_critical_issue(text, "**Added**: Critical")
        self.assertIn("## 🔴 Critical Issues (Must Fix)", result)
        self.assertIn("1. **Added**: Critical", result)
        # Critical section should appear before Important
        self.assertTrue(result.index("Critical Issues") < result.index("Important Changes"))


class TestFindPreviousBotReviews(unittest.TestCase):
    def test_finds_bot_reviews(self):
        discussions = [
            {"id": "1", "resolved": False, "notes": [{"body": f"{BOT_MARKER}\n## Review"}]},
            {"id": "2", "resolved": True, "notes": [{"body": "Human comment"}]},
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0]["id"], "1")
        self.assertFalse(reviews[0]["resolved"])

    def test_resolved_review(self):
        discussions = [
            {"id": "1", "resolved": True, "notes": [{"body": f"{BOT_MARKER}\nDone"}]},
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(len(reviews), 1)
        self.assertTrue(reviews[0]["resolved"])

    def test_empty_discussions(self):
        self.assertEqual(find_previous_bot_reviews([]), [])

    def test_no_notes(self):
        discussions = [{"id": "1", "notes": []}]
        self.assertEqual(find_previous_bot_reviews(discussions), [])

    def test_includes_developer_replies(self):
        discussions = [
            {
                "id": "1",
                "resolved": False,
                "notes": [
                    {"body": f"{BOT_MARKER}\n## Review\n1. Missing ticket"},
                    {"body": "Won't fix — intentional", "author": {"username": "dev1"}},
                ],
            },
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(len(reviews), 1)
        self.assertEqual(len(reviews[0]["replies"]), 1)
        self.assertEqual(reviews[0]["replies"][0]["author"], "dev1")
        self.assertIn("Won't fix", reviews[0]["replies"][0]["body"])

    def test_multiple_replies(self):
        discussions = [
            {
                "id": "1",
                "resolved": False,
                "notes": [
                    {"body": f"{BOT_MARKER}\n## Review"},
                    {"body": "Can you clarify?", "author": {"username": "dev1"}},
                    {"body": "Agreed, not applicable", "author": {"username": "dev2"}},
                ],
            },
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(len(reviews[0]["replies"]), 2)

    def test_reply_missing_author(self):
        discussions = [
            {
                "id": "1",
                "resolved": False,
                "notes": [
                    {"body": f"{BOT_MARKER}\n## Review"},
                    {"body": "Acknowledged"},
                ],
            },
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(reviews[0]["replies"][0]["author"], "unknown")


class TestTicketIssueDismissed(unittest.TestCase):
    def _make_review(self, body, replies=None):
        return {"id": "1", "resolved": False, "body": body, "replies": replies or []}

    def _bot_body_with_ticket_issue(self):
        return (
            f"{BOT_MARKER}\n## 🤖 AI Code Review\n"
            "## 🔴 Critical Issues (Must Fix)\n"
            f"1. **{NO_TICKET_SENTINEL}**: No recognizable issue number was found"
        )

    def test_dismissed_with_disregard(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "No GitLab ticket exists for this, so disregard that comment."}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_dismissed_with_wont_fix(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "Won't fix — this is intentional"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_dismissed_with_no_ticket(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "No ticket for this change"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_dismissed_with_ignore(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "Please ignore this finding"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_dismissed_with_not_applicable(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "Not applicable here"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_dismissed_with_acknowledged(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "Acknowledged"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_not_dismissed_without_reply(self):
        review = self._make_review(self._bot_body_with_ticket_issue())
        self.assertFalse(_ticket_issue_dismissed([review]))

    def test_not_dismissed_with_unrelated_reply(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "Good catch, I'll fix the other issue"}],
        )
        self.assertFalse(_ticket_issue_dismissed([review]))

    def test_not_dismissed_when_no_ticket_issue_in_body(self):
        review = self._make_review(
            f"{BOT_MARKER}\n## 🤖 AI Code Review\n## 🔴 Critical Issues (Must Fix)\nNone",
            [{"author": "dev1", "body": "disregard"}],
        )
        self.assertFalse(_ticket_issue_dismissed([review]))

    def test_empty_reviews(self):
        self.assertFalse(_ticket_issue_dismissed([]))

    def test_no_gitlab_ticket_phrase(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "No GitLab ticket exists for this"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_no_github_ticket_phrase(self):
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "dev1", "body": "No GitHub ticket exists for this"}],
        )
        self.assertTrue(_ticket_issue_dismissed([review]))

    def test_bot_reply_does_not_self_dismiss(self):
        """Bot follow-up echoing 'no ticket' should not count as a dismissal."""
        review = self._make_review(
            self._bot_body_with_ticket_issue(),
            [{"author": "bot", "body": f"{BOT_MARKER}\nNo ticket number found remains unresolved."}],
        )
        self.assertFalse(_ticket_issue_dismissed([review]))


class TestBuildPreviousReviewContext(unittest.TestCase):
    def test_omits_stale_no_ticket_issue_when_ticket_now_exists(self):
        previous_reviews = [
            {
                "id": "1",
                "resolved": False,
                "body": (
                    f"{BOT_MARKER}\n"
                    "## 🔴 Critical Issues (Must Fix)\n"
                    f"1. **{NO_TICKET_SENTINEL}**: No recognizable issue number was found\n"
                    "## 📋 Ticket Compliance\n"
                    "No ticket was found — skipped.\n"
                    "## 🧪 Test Automation Analysis\n"
                    "Skipped — no ticket available for analysis."
                ),
                "replies": [],
            }
        ]

        context = build_previous_review_context(previous_reviews, "1285")
        self.assertNotIn(NO_TICKET_SENTINEL, context)
        self.assertNotIn("No ticket was found", context)
        self.assertNotIn("Skipped — no ticket available", context)
        self.assertEqual(context, "")

    def test_keeps_no_ticket_issue_when_ticket_still_missing(self):
        previous_reviews = [
            {
                "id": "1",
                "resolved": False,
                "body": (
                    f"{BOT_MARKER}\n"
                    "## 🔴 Critical Issues (Must Fix)\n"
                    f"1. **{NO_TICKET_SENTINEL}**: No recognizable issue number was found"
                ),
                "replies": [],
            }
        ]

        context = build_previous_review_context(previous_reviews, None)
        self.assertIn(NO_TICKET_SENTINEL, context)


class TestSystemPrompt(unittest.TestCase):
    def test_follow_up_prompt_resolves_no_ticket_once_ticket_exists(self):
        self.assertIn('any earlier "No ticket number found"', SYSTEM_PROMPT)
        self.assertIn("must be omitted entirely", SYSTEM_PROMPT)

    def test_test_coverage_prompt_requires_structured_suggestions(self):
        self.assertIn("### Suggested Test Prompts", SYSTEM_PROMPT)
        self.assertIn("#### Unit Tests", SYSTEM_PROMPT)
        self.assertIn("#### Integration Tests", SYSTEM_PROMPT)
        self.assertIn("copy-paste-ready fenced `text` blocks", SYSTEM_PROMPT)

    def test_follow_up_prompt_collapses_fully_empty_sections(self):
        self.assertIn("do NOT emit subsection headings for that severity section", SYSTEM_PROMPT)
        self.assertIn("write only `None` as the section body", SYSTEM_PROMPT)


class TestNormalizeEmptyFollowUpSections(unittest.TestCase):
    def test_collapses_section_when_both_subsections_empty(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "None\n"
            "### Unresolved from Previous Review\n"
            "None\n"
            "## 🟡 Important Changes (Should Fix)\n"
            "1. **Thing**: `file.py:1` still broken"
        )

        result = _normalize_empty_follow_up_sections(text)
        self.assertIn("## 🔴 Critical Issues (Must Fix)\n\nNone\n", result)
        self.assertNotIn("## 🔴 Critical Issues (Must Fix)\n### Newly Identified", result)

    def test_collapses_section_when_previous_issues_resolved(self):
        text = (
            "## 🟡 Important Changes (Should Fix)\n"
            "### Newly Identified\n"
            "None\n"
            "### Unresolved from Previous Review\n"
            "All previous issues resolved.\n"
            "## 🟢 Suggestions (Nice to Have)\n"
            "- Keep this"
        )

        result = _normalize_empty_follow_up_sections(text)
        self.assertIn("## 🟡 Important Changes (Should Fix)\n\nNone\n", result)
        self.assertNotIn("All previous issues resolved.", result)

    def test_preserves_section_with_real_findings(self):
        text = (
            "## 🟢 Suggestions (Nice to Have)\n"
            "### Newly Identified\n"
            "- **Improve**: `review.py:1` tighten wording\n"
            "### Unresolved from Previous Review\n"
            "None"
        )

        result = _normalize_empty_follow_up_sections(text)
        self.assertIn("### Newly Identified", result)
        self.assertIn("tighten wording", result)


class TestBuildReviewBody(unittest.TestCase):
    def test_initial_review_has_number(self):
        body = build_review_body("Some review text", is_follow_up=False, review_number=1)
        self.assertIn("## 🤖 AI Code Review (Review 1)", body)
        self.assertIn(BOT_MARKER, body)
        self.assertNotIn("Follow-up", body)

    def test_follow_up_review_has_number(self):
        body = build_review_body("Some review text", is_follow_up=True, review_number=3)
        self.assertIn("## 🤖 AI Code Review (Review 3)", body)
        self.assertIn("Follow-up review after new commits", body)

    def test_default_review_number_is_one(self):
        body = build_review_body("text", is_follow_up=False)
        self.assertIn("(Review 1)", body)

    def test_number_uses_non_hash_label(self):
        body = build_review_body("text", is_follow_up=False, review_number=5)
        self.assertNotIn("(#5)", body)
        self.assertIn("(Review 5)", body)


class TestHasBlockingIssuesWithSubsections(unittest.TestCase):
    def test_both_subsections_none(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "None\n"
            "### Unresolved from Previous Review\n"
            "None\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertFalse(has_blocking_issues(text))

    def test_new_critical_issue(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "1. **Bug**: `file.py:10` Something broken\n"
            "### Unresolved from Previous Review\n"
            "None\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertTrue(has_blocking_issues(text))

    def test_persistent_critical_issue(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "None\n"
            "### Unresolved from Previous Review\n"
            "1. **Still broken**: `file.py:10` Not fixed yet\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertTrue(has_blocking_issues(text))

    def test_subsection_headers_only_no_content(self):
        """Subsection headers with no content lines at all → no blocking issues."""
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "\n"
            "### Unresolved from Previous Review\n"
            "\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertFalse(has_blocking_issues(text))

    def test_subsection_headers_with_only_none_under_second(self):
        """One empty subsection, one with 'None' → no blocking issues."""
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "\n"
            "### Unresolved from Previous Review\n"
            "\n"
            "None\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertFalse(has_blocking_issues(text))

    def test_no_new_phrase_in_subsection(self):
        text = (
            "## 🔴 Critical Issues (Must Fix)\n"
            "### Newly Identified\n"
            "No new Critical Issues introduced by these commits\n"
            "### Unresolved from Previous Review\n"
            "None\n"
            "## 🟡 Important Changes (Should Fix)\nNone"
        )
        self.assertFalse(has_blocking_issues(text))


class TestFindPreviousBotReviewsNoteId(unittest.TestCase):
    def test_includes_note_id_as_string(self):
        discussions = [
            {
                "id": "disc-1",
                "resolved": False,
                "notes": [{"id": 12345, "body": f"{BOT_MARKER}\n## Review"}],
            },
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertEqual(reviews[0]["note_id"], "12345")

    def test_missing_note_id(self):
        discussions = [
            {
                "id": "disc-1",
                "resolved": False,
                "notes": [{"body": f"{BOT_MARKER}\n## Review"}],
            },
        ]
        reviews = find_previous_bot_reviews(discussions)
        self.assertIsNone(reviews[0]["note_id"])


class TestCollapsePreviousReviews(unittest.TestCase):
    def test_collapses_review_with_details_tag(self):
        updated_bodies = []

        class FakeGL:
            def update_note(self, mr_iid, note_id, body):
                updated_bodies.append((note_id, body))

        reviews = [
            {
                "note_id": "100",
                "body": f"{BOT_MARKER}\n## 🤖 AI Code Review (Review 1)\nSome review content",
            },
        ]
        collapse_previous_reviews(FakeGL(), "1", reviews, current_review_number=2)
        self.assertEqual(len(updated_bodies), 1)
        note_id, body = updated_bodies[0]
        self.assertEqual(note_id, "100")
        self.assertIn(BOT_MARKER, body)
        self.assertIn(COLLAPSED_MARKER, body)
        self.assertIn("<details>", body)
        self.assertIn("superseded by Review 2", body)
        self.assertIn("</details>", body)

    def test_skips_already_collapsed_review(self):
        updated_bodies = []

        class FakeGL:
            def update_note(self, mr_iid, note_id, body):
                updated_bodies.append(body)

        reviews = [
            {
                "note_id": "100",
                "body": f"{BOT_MARKER}\n{COLLAPSED_MARKER}\n<details>...</details>",
            },
        ]
        collapse_previous_reviews(FakeGL(), "1", reviews, current_review_number=3)
        self.assertEqual(len(updated_bodies), 0)

    def test_skips_review_without_note_id(self):
        updated_bodies = []

        class FakeGL:
            def update_note(self, mr_iid, note_id, body):
                updated_bodies.append(body)

        reviews = [{"note_id": None, "body": f"{BOT_MARKER}\nReview"}]
        collapse_previous_reviews(FakeGL(), "1", reviews, current_review_number=2)
        self.assertEqual(len(updated_bodies), 0)

    def test_continues_on_api_error(self):
        call_count = {"n": 0}

        class FakeGL:
            def update_note(self, mr_iid, note_id, body):
                call_count["n"] += 1
                if note_id == "100":
                     raise RuntimeError("API error")

        reviews = [
            {"note_id": "100", "body": f"{BOT_MARKER}\nFirst"},
            {"note_id": "200", "body": f"{BOT_MARKER}\nSecond"},
        ]
        collapse_previous_reviews(FakeGL(), "1", reviews, current_review_number=3)
        self.assertEqual(call_count["n"], 2)


class TestTruncateError(unittest.TestCase):
    def test_short_message(self):
        exc = ValueError("short error")
        self.assertEqual(_truncate_error(exc), "short error")

    def test_long_message_truncated_with_ellipsis(self):
        exc = ValueError("x" * 300)
        result = _truncate_error(exc)
        self.assertTrue(result.endswith("..."))
        self.assertEqual(len(result), 200)

    def test_empty_exception(self):
        exc = ValueError()
        self.assertEqual(_truncate_error(exc), "")

    def test_multiline_uses_first_line(self):
        exc = ValueError("first line\nsecond line\nthird line")
        self.assertEqual(_truncate_error(exc), "first line")


class TestGenericWorkstreamDetection(unittest.TestCase):
    def setUp(self):
        self.config = [
            {
                "name": "react_frontend",
                "pattern": "^demo-app/src/.*\\.(js|jsx|ts|tsx)$",
                "criteria_file": ".github/pr-reviewer/skills/frontend-review.md"
            },
            {
                "name": "backend_python",
                "pattern": "^server/.*\\.py$",
                "criteria_file": ".github/pr-reviewer/skills/backend-review.md"
            }
        ]

    def test_detects_single_workstream(self):
        diffs = [
            {"new_path": "demo-app/src/components/Button.tsx", "diff": "+hello"},
            {"new_path": "README.md", "diff": "+docs"}
        ]
        detected = detect_workstreams(diffs, self.config)
        self.assertTrue(detected["react_frontend"])
        self.assertFalse(detected["backend_python"])

    def test_detects_multiple_workstreams(self):
        diffs = [
            {"new_path": "demo-app/src/App.tsx", "diff": "+react"},
            {"new_path": "server/app.py", "diff": "+python"}
        ]
        detected = detect_workstreams(diffs, self.config)
        self.assertTrue(detected["react_frontend"])
        self.assertTrue(detected["backend_python"])

    def test_no_matching_workstreams(self):
        diffs = [
            {"new_path": "docs/architecture.md", "diff": "+markdown"},
            {"new_path": "package.json", "diff": "+json"}
        ]
        detected = detect_workstreams(diffs, self.config)
        self.assertFalse(detected["react_frontend"])
        self.assertFalse(detected["backend_python"])

    def test_invalid_regex_gracefully_handled(self):
        bad_config = [
            {
                "name": "bad_regex",
                "pattern": "[invalid-regex-",
                "criteria_file": "some-file.md"
            }
        ]
        diffs = [{"new_path": "server/app.py", "diff": "+python"}]
        # Should not crash, should return False for bad regex workstream
        detected = detect_workstreams(diffs, bad_config)
        self.assertFalse(detected["bad_regex"])





class TestGitHubClient(unittest.TestCase):
    @patch("review.requests.get")
    def test_get_pr(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "title": "Fix crash in image loading",
            "body": "This PR fixes a nil pointer dereference when loading 0-byte images."
        }
        mock_get.return_value = mock_response

        client = GitHubClient("https://github.com", "owner/repo", "token")
        pr = client.get_pr("123")
        
        self.assertEqual(pr["title"], "Fix crash in image loading")
        self.assertEqual(pr["description"], "This PR fixes a nil pointer dereference when loading 0-byte images.")
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/pulls/123",
            headers={
                "Authorization": "token token",
                "Accept": "application/vnd.github.v3+json"
            },
            params=None,
            timeout=30
        )

    @patch("review.requests.get")
    def test_get_pr_diffs(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = """diff --git a/src/main.py b/src/main.py
index 0000000..1234567 10064
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
 def main():
-    print("hello")
+    print("hello world")
+    return 0
"""
        mock_get.return_value = mock_response

        client = GitHubClient("https://github.com", "owner/repo", "token")
        diffs = client.get_pr_diffs("123")
        
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0]["new_path"], "src/main.py")
        self.assertIn("print(\"hello world\")", diffs[0]["diff"])

    @patch("review.requests.get")
    def test_get_pr_discussions(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "id": 101,
                "body": "Some developer comment",
                "user": {"login": "dev1"}
            },
            {
                "id": 102,
                "body": "<!-- ai-code-reviewer -->\n## 🔴 Critical Issues\n1. Issue",
                "user": {"login": "gemini-bot"}
            },
            {
                "id": 103,
                "body": "Developer reply to bot",
                "user": {"login": "dev1"}
            }
        ]
        mock_get.return_value = mock_response

        client = GitHubClient("https://github.com", "owner/repo", "token")
        discussions = client.get_pr_discussions("123")
        
        self.assertEqual(len(discussions), 1)
        self.assertEqual(discussions[0]["id"], "102")
        self.assertEqual(len(discussions[0]["notes"]), 2)
        self.assertEqual(discussions[0]["notes"][0]["id"], 102)
        self.assertEqual(discussions[0]["notes"][1]["id"], 103)
        self.assertEqual(discussions[0]["notes"][1]["body"], "Developer reply to bot")



class TestPostReview(unittest.TestCase):
    @patch("review.requests.post")
    def test_post_review_success(self, mock_post):
        mock_response = MagicMock()
        mock_post.return_value = mock_response

        client = GitHubClient("https://github.com", "owner/repo", "token")
        post_review(client, "123", "Some review content", is_follow_up=False)

        mock_post.assert_called_once()

    @patch("review.requests.post")
    def test_post_review_forbidden_403(self, mock_post):
        # Create a mock response with 403 status code
        mock_response = MagicMock()
        mock_response.status_code = 403
        
        # Raise MockHTTPError with response
        from review import requests as r_mod
        mock_post.side_effect = r_mod.HTTPError("Forbidden", response=mock_response)

        client = GitHubClient("https://github.com", "owner/repo", "token")
        
        # It should not raise an exception, but handle it gracefully
        try:
            post_review(client, "123", "Some review content", is_follow_up=False)
        except Exception as e:
            self.fail(f"post_review raised an unexpected exception: {e}")

    @patch("review.requests.post")
    def test_post_review_other_error_raises(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 500
        
        from review import requests as r_mod
        mock_post.side_effect = r_mod.HTTPError("Internal Server Error", response=mock_response)

        client = GitHubClient("https://github.com", "owner/repo", "token")
        
        # It should raise the exception
        with self.assertRaises(r_mod.HTTPError):
            post_review(client, "123", "Some review content", is_follow_up=False)


class TestGitHubClientGetFileRaw(unittest.TestCase):
    @patch("review.requests.get")
    def test_get_file_raw_quotes_path_safely(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = "file content"
        mock_get.return_value = mock_response

        client = GitHubClient("https://github.com", "owner/repo", "token")
        
        # Path with spaces and slash
        content = client.get_file_raw("sha123", "src/some folder/file.py")
        self.assertEqual(content, "file content")
        
        # Check the URL called. It should encode space but not slash!
        # "src/some folder/file.py" -> "src/some%20folder/file.py"
        mock_get.assert_called_once_with(
            "https://api.github.com/repos/owner/repo/contents/src/some%20folder/file.py",
            headers={
                "Authorization": "token token",
                "Accept": "application/vnd.github.v3.raw"
            },
            params={"ref": "sha123"},
            timeout=30
        )


class TestGitHubClientGetPRDiscussionsEnhanced(unittest.TestCase):
    @patch("review.requests.get")
    def test_get_pr_discussions_pagination(self, mock_get):
        # Setup mock_get to return page 1 (100 items) and page 2 (10 items)
        page1 = [{"id": i, "body": f"Comment {i}", "user": {"login": "user1"}} for i in range(1, 101)]
        # Make one comment a bot comment
        page1[50]["body"] = "<!-- ai-code-reviewer -->\nSome bot comment"
        
        page2 = [{"id": i, "body": f"Comment {i}", "user": {"login": "user1"}} for i in range(101, 111)]
        
        mock_response_page1 = MagicMock()
        mock_response_page1.json.return_value = page1
        mock_response_page2 = MagicMock()
        mock_response_page2.json.return_value = page2
        
        mock_get.side_effect = [mock_response_page1, mock_response_page2]
        
        client = GitHubClient("https://github.com", "owner/repo", "token")
        discussions = client.get_pr_discussions("123")
        
        self.assertEqual(len(discussions), 1)
        # Verify pagination fetched comments from page 2 as well
        self.assertEqual(len(discussions[0]["notes"]), 60) # 1 bot comment + 59 replies
        self.assertEqual(mock_get.call_count, 2)

    @patch("review.requests.get")
    def test_get_pr_discussions_thread_reconstruction(self, mock_get):
        comments = [
            {
                "id": 10,
                "body": "<!-- ai-code-reviewer -->\n## 🤖 AI Code Review (Review 1)\n> Some bot comment\nline 1",
                "user": {"login": "bot"}
            },
            {
                "id": 20,
                "body": "<!-- ai-code-reviewer -->\n## 🤖 AI Code Review (Review 2)\nline 2",
                "user": {"login": "bot"}
            },
            {
                "id": 30,
                "body": "Regarding Review 1, done.",
                "user": {"login": "dev"}
            },
            {
                "id": 40,
                "body": "> Some bot comment\nFixed.",
                "user": {"login": "dev"}
            },
            {
                "id": 50,
                "body": "Unrelated reply.",
                "user": {"login": "dev"}
            }
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = comments
        mock_get.return_value = mock_response
        
        client = GitHubClient("https://github.com", "owner/repo", "token")
        discussions = client.get_pr_discussions("123")
        
        self.assertEqual(len(discussions), 2)
        
        disc1 = discussions[0]
        disc2 = discussions[1]
        
        self.assertEqual(disc1["id"], "10")
        self.assertEqual(disc2["id"], "20")
        
        self.assertEqual(len(disc1["notes"]), 3)
        self.assertEqual(disc1["notes"][0]["id"], 10)
        self.assertEqual(disc1["notes"][1]["id"], 30)
        self.assertEqual(disc1["notes"][2]["id"], 40)
        
        self.assertEqual(len(disc2["notes"]), 2)
        self.assertEqual(disc2["notes"][0]["id"], 20)
        self.assertEqual(disc2["notes"][1]["id"], 50)


class TestGitHubClientGetPRDiffsEnhanced(unittest.TestCase):
    @patch("review.requests.get")
    def test_get_pr_diffs_advanced_parsing(self, mock_get):
        mock_response = MagicMock()
        mock_response.text = """diff --git a/src/old_file.py b/src/new_file.py
similarity index 85%
rename from src/old_file.py
rename to src/new_file.py
index 1111111..2222222 100644
--- a/src/old_file.py
+++ b/src/new_file.py
@@ -1,2 +1,2 @@
-old content
+new content
diff --git a/src/deleted.py b/src/deleted.py
deleted file mode 100644
index 3333333..0000000
--- a/src/deleted.py
+++ /dev/null
@@ -1,1 +0,0 @@
-removed line
diff --git a/src/binary.png b/src/binary.png
index 4444444..5555555 100644
Binary files a/src/binary.png and b/src/binary.png differ
diff --git a/src/empty.txt b/src/empty.txt
new file mode 100644
index 0000000..e69de29
diff --git a/src/large.json b/src/large.json
index 6666666..7777777 100644
--- a/src/large.json
+++ b/src/large.json
Warning: this diff was suppressed because of its size.
"""
        mock_get.return_value = mock_response
        
        client = GitHubClient("https://github.com", "owner/repo", "token")
        diffs = client.get_pr_diffs("123")
        
        self.assertEqual(len(diffs), 5)
        
        # 1. Renamed file
        self.assertEqual(diffs[0]["new_path"], "src/new_file.py")
        self.assertEqual(diffs[0]["old_path"], "src/old_file.py")
        self.assertTrue(diffs[0]["renamed_file"])
        self.assertFalse(diffs[0]["deleted_file"])
        self.assertFalse(diffs[0]["collapsed"])
        self.assertFalse(diffs[0]["too_large"])
        
        # 2. Deleted file
        self.assertEqual(diffs[1]["new_path"], "src/deleted.py")
        self.assertFalse(diffs[1]["renamed_file"])
        self.assertTrue(diffs[1]["deleted_file"])
        self.assertFalse(diffs[1]["collapsed"])
        
        # 3. Binary file
        self.assertEqual(diffs[2]["new_path"], "src/binary.png")
        self.assertFalse(diffs[2]["collapsed"])
        
        # 4. Empty file
        self.assertEqual(diffs[3]["new_path"], "src/empty.txt")
        self.assertFalse(diffs[3]["collapsed"])
        
        # 5. Suppressed (too large) file
        self.assertEqual(diffs[4]["new_path"], "src/large.json")
        self.assertTrue(diffs[4]["too_large"])


if __name__ == "__main__":
    unittest.main()