"""Unit and contract tests for scripts/merge_sync.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure scripts directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from merge_sync import (  # noqa: E402
    cleanup_worktrees,
    extract_story_slug,
    resolve_target,
)


def test_extract_story_slug():
    """Verifies story and epic slug extraction across different formats."""
    assert extract_story_slug("feat/story-26.3") == "story-26.3"
    assert extract_story_slug("docs/bmad/stories/story-26.3-grounding-dino-camo-proposals.md") == "story-26.3"
    assert extract_story_slug("[Story 12.5] Grounding DINO Zero-Shot Proposal Backend") == "story-12.5"
    assert extract_story_slug("feat/epic-23") == "epic-23"
    assert extract_story_slug("26.3") == "story-26.3"
    assert extract_story_slug("story-14.1") == "story-14.1"
    assert extract_story_slug("") is None
    assert extract_story_slug(None) is None


@patch("merge_sync.get_open_prs")
def test_resolve_target_all_flag(mock_open_prs):
    """Verifies --all flag selects all open PRs in batch mode."""
    mock_open_prs.return_value = [
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
        {"number": 53, "title": "Story 23.4", "headRefName": "feat/story-23.4"},
    ]
    target_branch, prs, slug, is_all = resolve_target(all_prs=True)
    assert is_all is True
    assert target_branch is None
    assert len(prs) == 2


@patch("merge_sync.get_open_prs")
def test_resolve_target_branch_flag(mock_open_prs):
    """Verifies --branch flag isolates to only the matching PR."""
    mock_open_prs.return_value = [
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
        {"number": 53, "title": "Story 23.4", "headRefName": "feat/story-23.4"},
    ]
    target_branch, prs, slug, is_all = resolve_target(branch="feat/story-23.4")
    assert is_all is False
    assert target_branch == "feat/story-23.4"
    assert len(prs) == 1
    assert prs[0]["number"] == 53
    assert slug == "story-23.4"


@patch("merge_sync.get_open_prs")
def test_resolve_target_story_flag(mock_open_prs):
    """Verifies --story flag matches story slug in branch or title."""
    mock_open_prs.return_value = [
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
        {"number": 53, "title": "Story 23.4", "headRefName": "feat/story-23.4"},
    ]
    target_branch, prs, slug, is_all = resolve_target(story="12.5")
    assert is_all is False
    assert target_branch == "feat/story-12.5"
    assert len(prs) == 1
    assert prs[0]["number"] == 50
    assert slug == "story-12.5"


@patch("merge_sync.get_open_prs")
def test_resolve_target_pr_flag(mock_open_prs):
    """Verifies --pr flag selects specific PR by number."""
    mock_open_prs.return_value = [
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
        {"number": 53, "title": "Story 23.4", "headRefName": "feat/story-23.4"},
    ]
    target_branch, prs, slug, is_all = resolve_target(pr_num=50)
    assert is_all is False
    assert target_branch == "feat/story-12.5"
    assert len(prs) == 1
    assert prs[0]["number"] == 50


@patch("merge_sync.get_active_branch")
@patch("merge_sync.get_open_prs")
def test_resolve_target_cwd_worktree_autodetect(mock_open_prs, mock_active_branch):
    """Verifies running inside a worktree branch automatically isolates that branch."""
    mock_active_branch.return_value = "feat/story-26.3"
    mock_open_prs.return_value = [
        {"number": 47, "title": "Story 26.3", "headRefName": "feat/story-26.3"},
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
    ]
    target_branch, prs, slug, is_all = resolve_target()
    assert is_all is False
    assert target_branch == "feat/story-26.3"
    assert len(prs) == 1
    assert prs[0]["number"] == 47
    assert slug == "story-26.3"


@patch("merge_sync.get_active_worktrees")
@patch("merge_sync.get_active_branch")
@patch("merge_sync.get_open_prs")
def test_resolve_target_guard_multiple_prs_on_main(mock_open_prs, mock_active_branch, mock_active_wts):
    """Verifies running on main with multiple open PRs refuses to batch merge without --all."""
    mock_active_branch.return_value = "main"
    mock_active_wts.return_value = [
        ("/path/to/wt1", "feat/story-12.5"),
        ("/path/to/wt2", "feat/story-23.4"),
    ]
    mock_open_prs.return_value = [
        {"number": 50, "title": "Story 12.5", "headRefName": "feat/story-12.5"},
        {"number": 53, "title": "Story 23.4", "headRefName": "feat/story-23.4"},
    ]
    target_branch, prs, slug, is_all = resolve_target()
    # Guard triggers: returns empty so it doesn't accidentally merge everything
    assert target_branch is None
    assert prs == []
    assert is_all is False


@patch("merge_sync.subprocess.run")
@patch("merge_sync.get_completed_story_slugs")
@patch("merge_sync.get_merged_prs")
def test_cleanup_worktrees_targeted(mock_merged_prs, mock_completed_slugs, mock_subproc):
    """Verifies cleanup_worktrees only cleans the target worktree when target_branch is specified."""
    mock_merged_prs.return_value = [{"headRefName": "feat/story-26.3"}]
    mock_completed_slugs.return_value = {"story-26.3"}

    mock_subproc.return_value = MagicMock(
        stdout="worktree /path/to/main\nbranch refs/heads/main\n\nworktree /path/to/orbweaver-story-26.3\nbranch refs/heads/feat/story-26.3\n\nworktree /path/to/orbweaver-story-12.5\nbranch refs/heads/feat/story-12.5\n"
    )

    cleaned = cleanup_worktrees(target_branch="feat/story-26.3")
    assert len(cleaned) == 1
    assert "orbweaver-story-26.3" in cleaned[0]
