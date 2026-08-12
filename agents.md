# AGENTS.md — Working on GoldThread / TPUbench (AGO TO2)

Guidance for AI agents (and humans) contributing to **GoldThread / TPUbench** under **Apollo-Gov Task Order 2 (AGO-TO2)**.
Follow this guide so all code development, benchmarking, infrastructure, and task tracking remain consistent, verified, and synchronized across Google-internal tools.

---

## 🎯 Project & Infrastructure Context

- **Project / Repository**: `personal/goldthread` (`https://depot.code.corp.goog/personal/goldthread`)
- **Task Order**: **AGO-TO2** (`[NatSec] Apollo-Gov : AGO-TO2`, [`b/519644843`](https://b.corp.google.com/issues/519644843), Parent [`b/444192593`](https://b.corp.google.com/issues/444192593))
- **Primary TaskFlow Workspace**: [GoldThread Workspace 6001488](https://taskflow.corp.google.com/workspaces/6001488/goals)
- **Active Hotlist**: [`b/hotlists/8806410`](https://b.corp.google.com/hotlists/8806410) (`TPUbench Tasks`)
- **Default Component**: [`2011243`](https://b.corp.google.com/components/2011243) (`tasks`)
- **GitHub CLI**: Use `ggh` for all GitHub Enterprise (`depot.code.corp.goog`) CLI operations (e.g. `ggh issue create`, `ggh pr create`).
- **Python Environment**: Managed exclusively via **`uv`** with declarative configuration in [`pyproject.toml`](pyproject.toml) and reproducible lockfile [`uv.lock`](uv.lock).

---

## 🌳 Parallel Workspaces & Git Worktrees

When working on multiple concurrent features or spawning parallel agent sessions:
1. **Create Worktree**:
   ```bash
   git worktree add ../TPUbench-<feature-name> -b feat/<feature-name>
   ```
2. **Symlink Virtual Environment & Data (Zero-Copy)**:
   Never reinstall `.venv` or re-download datasets. Symlink them directly from the main repository:
   ```bash
   ln -s "$(git rev-parse --show-toplevel)/.venv" ../TPUbench-<feature-name>/.venv
   ln -s "$(git rev-parse --show-toplevel)/data" ../TPUbench-<feature-name>/data
   ```
3. **Verify Inside Worktree**:
   Run `uv run pytest -q` inside the worktree to ensure the linked `.venv` and tests run cleanly.
4. **Cleanup After Merge**:
   Once the PR is merged into `main`:
   ```bash
   git worktree remove ../TPUbench-<feature-name>
   ```

---

## 🔄 The Loop for Every Feature & Task

1. **Understand before editing.**
   - Read the relevant code paths and their callers before making changes (`src/sar/`, `src/modis/`, `terraform/`).
   - Use Google Code Search or codebase grep to confirm function/class names and interfaces. State verified facts from code, not assumptions.

2. **Plan non-trivial work.**
   - For multi-file or algorithmic modifications, write a concise plan under `docs/<feature>-plan.md` detailing current state, target files, mathematical/algorithmic approach, verification strategy, and potential pitfalls.
   - For major features, align with the TO2 delivery plan and milestone requirements.
   - **BMad Story Specification**: Create or update the dedicated story specification under `docs/bmad/stories/story-<module>-<id>-<title>.md` with clear Acceptance Criteria (AC1..ACn).

3. **Implement in the codebase's idiom.**
   - Python / JAX: Follow high-performance vectorization principles; avoid Python loops on hot array paths.
   - SAR Algorithms: Preserve double-precision floating-point compliance where required (`JAX_ENABLE_X64=True`).
   - Terraform: Follow modular GCP Infrastructure-as-Code standards under [terraform/](terraform/).

4. **Verify — The Definition of Done.**
   Run and report output for all applicable verification gates using `uv run`:
   - **Code Quality**: `uv run ruff check src tests scripts` → clean.
   - **Unit & Integration Tests**: `uv run pytest -q` → all green.
   - **PR Reviewer Test Suite**: `uv run python .github/pr-reviewer/test_review.py` → all green.
   - **Numerical & Coherence Parity**:
     - Fast Factorized Back-Projection (FFBP) & BPA: Complex coherence against reference SICD ground truth must exceed **>97.0%** ($>0.97$).
     - Polar Format Algorithm (PFA): Relative norm error vs CPU reference $< 0.05\%$.
   - **Terraform Validation**: `terraform fmt -check` and `terraform validate` inside [terraform/](terraform/).
   - **Live Benchmark Paths**: When modifying image formation or data ingestion pipelines, benchmark execution time and memory scaling on real data granules (`data/sar/`, `data/modis/`).

5. **Update docs + performance benchmarks + BMad stories.**
   - Update [docs/PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md), [README.md](README.md), and [README_MODIS.md](README_MODIS.md) whenever algorithms, latency figures, or CLI parameters change.
   - Update `docs/TODO.md` and `docs/ROADMAP.md` milestone progress.
   - **Update BMad Story File**: Update `docs/bmad/stories/story-<id>.md` to record completion status, check off all Acceptance Criteria, and document exact verification metrics.
   - Record durable architectural decisions and benchmark findings in project docs.

6. **Report honestly.**
   - Always report exact numerical metrics (e.g. coherence percentiles $p50, p90$, runtimes in milliseconds, throughput in MP/s).
   - If tests or coherence targets fail, state the numbers clearly without hedging or overselling unverified results.

7. **Complete, Track, and Submit.**
   A feature or bug fix is prepared for submission when:
   - Changes are pushed to a remote feature branch: `git push origin <branch-name>`.
   - The BMad story doc (`docs/bmad/stories/story-<id>.md`) is included in the PR diff.
   - A **GitHub Issue & Pull Request** are opened via `ggh`:
     ```bash
     ggh issue create --repo personal/goldthread --title "[TO2] <Title>" --body "<Description>"
     ggh pr create --repo personal/goldthread --base main --head <branch-name> --title "[TO2] <Title>" --body "<Summary & Verification>"
     ```
   - A corresponding **Buganizer Task** is created under component `2011243` and added to Hotlist `8806410` (TaskFlow Workspace **6001488**):
     - Title: `[TO2] <Feature/Bug Title>`
     - Description: Includes links to the GitHub PR/Issue, list of modified files, and test/benchmark verification output.


8. **Mandatory CI/CD & PR Checks Gate (Before Merging to Main).**
   - **Never merge with failing or pending checks**: Before merging any PR into `main`, the agent/developer MUST inspect and wait for all CI/CD checks to pass:
     ```bash
     ggh pr checks <PR_NUMBER> --repo personal/goldthread
     ```
   - **Required Passing Checks**:
     - `ci/Lint & Unit Tests`: Linting (`uv run ruff check src tests scripts`), full unit tests (`uv run pytest -q`), and PR reviewer tests (`uv run python .github/pr-reviewer/test_review.py`).
     - `ci/Terraform Validation`: Terraform formatting (`terraform fmt -check`) and validation (`terraform validate`).
     - `gemini-pr-review/review`: Automated AI code review pass (must complete with zero unaddressed Critical Issues).
     - Automated Security Scanners (Wiz, Semgrep): Must pass cleanly.
   - **Resolve CI/Review Findings**: If the AI reviewer surfaces Critical Issues or a CI test fails, resolve the finding on the branch, push the fix, re-verify with `ggh pr checks`, and only merge once all checks are green (`✓`). Update the Buganizer task to `FIXED` upon successful merge.

---

## 💻 Code Style & Conventions

### Python & Environment Management (`uv`)
- **Exclusively use `uv`**: We use `uv` instead of standard `pip`/`venv` for all package management, virtual environment creation, dependency synchronization, and script execution.
- **Environment Setup & Sync**:
  - `uv sync`: Automatically creates/updates `.venv` with dependencies and dev tools defined in [`pyproject.toml`](pyproject.toml) and locked in [`uv.lock`](uv.lock).
  - `source .venv/bin/activate`: Activate the virtual environment (optional if prefixing with `uv run`).
  - `uv add <package>` / `uv add --dev <package>`: Add new project or development dependencies.
  - `uv pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html`: Install TPU acceleration on Cloud TPU VMs.
- **Execution with `uv run`**:
  - Run linters: `uv run ruff check src tests scripts`
  - Run test suite: `uv run pytest -q`
  - Run reviewer test suite: `uv run python .github/pr-reviewer/test_review.py`
  - Run SAR / MODIS pipelines: `uv run python src/sar/run_pfa.py ...`

### Python & JAX (`src/sar/`, `src/modis/`)
- **Module Docstrings**: Every module begins with a clear docstring explaining the physical algorithm, inputs/outputs, and CLI usage.
- **Imports**: Group standard library → third-party (`numpy`, `jax`, `scipy`, `h5py`) → local modules.
- **Type Annotations**: Annotate function signatures (`jax.Array`, `np.ndarray`, `dict`, `str | None`).
- **Comments**: Explain **WHY** — physical constants, coordinate frame transforms, sub-aperture factorization steps, or numerical stability tradeoffs.

### Terraform (`terraform/`)
- Centralize variables in `variables.tf` and outputs in `outputs.tf`.
- Ensure all GKE node pools (CPU & TPU v4/v5e) and VPC network topologies are parameterized.
