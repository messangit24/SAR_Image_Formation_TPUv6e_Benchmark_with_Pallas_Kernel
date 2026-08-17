---
name: bmad-code-review
description: Standalone adversarial red-team code review. Cynically inspects git diff against acceptance criteria, deep module design principles (where applicable), language standards (write-python/write-typescript), memory leaks, and boundary edge cases.
---

# Adversarial Red-Team Code Review (`bmad-code-review`)

Use this skill to conduct a cynical, rigorous pre-PR review on the worktree's `git diff`.

---

## 🎯 Review Mission

Your objective is to find bugs, security leaks, edge-case failures, unhandled exceptions, performance regressions, shallow spaghetti abstractions, and violations of modular language standards (`write-python` / `write-typescript`) **before** the code is submitted to GitHub.

---

## 🔍 The 6-Layer Adversarial Inspection

1. **Acceptance Criteria & Deliverable Verification**:
   - Does the implementation satisfy EVERY acceptance criterion declared in the story?
   - Is there an executable verification test or proof for the deliverables?
2. **Deep Modules & Interface Encapsulation (Where Code/APIs are Modified)**:
   - **Is the module deep?** Where new classes or services are introduced, do they offer a simple, elegant public interface that hides internal complexity?
   - **Are abstractions leaky?** Are internal implementation details exposed needlessly?
   - **Are tests interface-focused?** Are unit/integration tests asserting behavior through the public interface?
3. **Language Standards Compliance**:
   - **Python (`write-python`)**: Built-in generics (PEP 585/604), lazy imports for heavy ML libraries (`torch`, `jax`, `transformers`, `cv2`, `pandas`), hermetic unit tests with zero network calls.
   - **TypeScript (`write-typescript`)**: Zero `any` policy, Zod runtime boundary validation, typed return signatures, clean `useEffect` cleanup handlers.
4. **Robustness & Edge Cases**:
   - Unhandled exceptions, missing null checks (`None` / `undefined`), boundary off-by-one errors.
   - Silent error swallowing (`pass` in `except:`, un-awaited async tasks).
5. **Performance & Memory Leaks**:
   - Accidental unbounded memory growth, unclosed file/socket descriptors, duplicate expensive computations.
6. **Security & Trust Boundaries**:
   - Path traversal vulnerabilities (`Path.resolve()` checks), SQL injection / raw string formatting, unsafe deserialization (`pickle.loads`, `eval`).

---

## 📋 Structured Review Output Format

Present findings categorized by severity:

```markdown
## 🛡️ Adversarial Red-Team Review Findings

### 🔴 Critical Blockers (Must Fix Before PR)
- `path/to/file.py:L45`: Description of blocker and concrete remediation.

### 🟡 Important Improvements (Should Fix)
- `path/to/module.py:L12`: Consider deeper encapsulation for helper steps.

### 🟢 Suggestions & Cleanups (Nice to Have)
- `path/to/module.py:L10`: Consider lazy import for heavy dependency.

### 🏁 Verdict
**[ PASS / REVISIONS_REQUIRED ]**
```

---

## 💡 Recommended Next Actions for the Developer:
- If revisions are required: Fix findings in the worktree and re-verify with `uv run pytest -q`.
- If clean: Proceed to Step 6 of `dev-story` to push the branch and open the PR with `ggh pr create`.
