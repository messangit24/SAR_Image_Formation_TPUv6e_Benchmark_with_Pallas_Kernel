---
name: bmad-code-review
description: Standalone adversarial red-team code review. Cynically inspects git diff against acceptance criteria, language standards (write-python/write-typescript), memory leaks, and boundary edge cases.
---

# Adversarial Red-Team Code Review (`bmad-code-review`)

Use this skill to conduct a cynical, rigorous pre-PR review on the worktree's `git diff`.

---

## 🎯 Review Mission

Your objective is to find bugs, security leaks, edge-case failures, unhandled exceptions, performance regressions, and violations of modular language standards (`write-python` / `write-typescript`) **before** the code is submitted to GitHub.

---

## 🔍 The 5-Layer Adversarial Inspection

1. **Acceptance Criteria & Vertical Slice Verification**:
   - Does the implementation satisfy EVERY acceptance criterion declared in the story?
   - Is there an executable end-to-end integration test asserting the complete vertical slice?
2. **Language Standards Compliance**:
   - **Python (`write-python`)**: Built-in generics (PEP 585/604), lazy imports for heavy ML libraries (`torch`, `jax`, `transformers`, `cv2`, `pandas`), hermetic unit tests with zero network calls.
   - **TypeScript (`write-typescript`)**: Zero `any` policy, Zod runtime boundary validation, typed return signatures, clean `useEffect` cleanup handlers.
3. **Robustness & Edge Cases**:
   - Unhandled exceptions, missing null checks (`None` / `undefined`), boundary off-by-one errors.
   - Silent error swallowing (`pass` in `except:`, un-awaited async tasks).
4. **Performance & Memory Leaks**:
   - Accidental unbounded memory growth, unclosed file/socket descriptors, duplicate expensive computations.
5. **Security & Trust Boundaries**:
   - Path traversal vulnerabilities (`Path.resolve()` checks), SQL injection / raw string formatting, unsafe deserialization (`pickle.loads`, `eval`).

---

## 📋 Structured Review Output Format

Present findings categorized by severity:

```markdown
## 🛡️ Adversarial Red-Team Review Findings

### 🔴 Critical Blockers (Must Fix Before PR)
- `path/to/file.py:L45`: Description of blocker and concrete remediation.

### 🟡 Important Improvements (Should Fix)
- `path/to/file.ts:L120`: Unvalidated API boundary response; add Zod schema parsing.

### 🟢 Suggestions & Cleanups (Nice to Have)
- `path/to/module.py:L10`: Consider lazy import for heavy dependency.

### 🏁 Verdict
**[ PASS / REVISIONS_REQUIRED ]**
```

---

## 💡 Recommended Next Actions for the Developer:
- If revisions are required: Fix findings in the worktree and re-verify with `uv run pytest -q`.
- If clean: Proceed to Step 6 of `dev-story` to push the branch and open the PR with `ggh pr create`.
