# 🤖 Standalone GitHub PR AI Code Reviewer

A fully generic, configuration-driven GitHub Pull Request (PR) code reviewer powered by **Gemini**, **Claude (Direct)**, or **Claude (GCP Vertex AI)**. This folder showcases a portable, drop-in reviewer that can be adopted by any GitHub repository simply by configuring a GitHub Action.

---

## 🌟 Key Features

1. **Multi-Provider Support**:
   - **Gemini**: Powered by `google-genai` using direct API keys.
   - **Anthropic Claude**: Powered by `anthropic` using direct API keys.
   - **Vertex AI (GCP)**: Powered by Vertex AI using Workload Identity Federation or Service Account credentials.
2. **Config-Driven Workstreams**: Decoupled from hardcoded codebase structures. Automatically maps file patterns to technology-specific skill checklists (e.g. front-end guidelines, backend standards) using regex.
3. **Advisory Integrity**: Runs strictly as an advisory check so reviews never block developer merge operations.
4. **Intelligent Context-Aware Reviews**: Synthesizes previous reviews and developer replies so the AI converges toward completion instead of repeating identical comments.
5. **Auto-Collapsing Thread Management**: Automatically folds older bot reviews inside `<details>` blocks so that the PR comment thread remains clean and readable.

---

## 📂 Repository Structure

```
.github/
└── pr-reviewer/
    ├── review.py               # Fully generic, multi-provider reviewer script
    ├── test_review.py          # Comprehensive unit test suite (94+ checks)
    ├── config.json             # Mapping file regex patterns to skill md files
    └── skills/
        └── frontend-review.md  # Front-end review criteria (React/Vite)
```

---

## 📦 Single-File Configuration (`config.json`)

The reviewer is driven entirely by a single configuration file located at `.github/pr-reviewer/config.json`. To change AI providers or add custom review criteria, **this is the only file you need to modify!**

```json
{
  "provider": "gemini",
  "model": "gemini-3.5-flash",
  "workstreams": [
    {
      "name": "react_frontend",
      "pattern": "^web/src/.*\\.(js|jsx|ts|tsx)$",
      "criteria_file": ".github/pr-reviewer/skills/frontend-review.md"
    }
  ]
}
```

### Config Parameters:
- `provider`: Choose your AI backend: `"gemini"`, `"anthropic"`, or `"vertex-ai"`.
- `model`: The AI model to invoke (e.g., `"gemini-3.5-flash"`, `"claude-3-5-sonnet-20241022"`, or `"claude-sonnet-4-6"`).
- `workstreams`: Array mapping regex file scopes.
  - `name`: Logical identifier for the workstream.
  - `pattern`: Regular expression matching modified file paths in the PR.
  - `criteria_file`: Path pointing to specific review criteria markdown files.

---

## 🚀 Dropping This Into Your Existing GitHub Repository

Follow these simple steps to drop the AI reviewer directly into your GitHub Actions:

### Step 1: Copy the Code Reviewer Folder
Copy the `.github/pr-reviewer/` folder into your repository.

### Step 2: Configure Your Skills
Add or edit markdown files in `.github/pr-reviewer/skills/` with your own guidelines. Edit `.github/pr-reviewer/config.json` to set your provider, model, and match patterns.

### Step 3: Create GitHub Action Workflow
Create `.github/workflows/gemini-pr-review.yml` with the following contents:

```yaml
name: gemini-pr-review

on:
  pull_request:
    types: [opened, synchronize, reopened, ready_for_review]
  issue_comment:
    types: [created]

jobs:
  review:
    # Auto-review non-draft PRs; also re-run on demand via "@gemini-cli /review".
    if: >-
      (github.event_name == 'pull_request' && github.event.pull_request.draft == false) ||
      (github.event_name == 'issue_comment' && github.event.issue.pull_request != null &&
       contains(github.event.comment.body, '@gemini-cli /review'))
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Set up uv
        uses: astral-sh/setup-uv@v5
        with:
          enable-cache: true

      - name: Install dependencies
        run: |
          uv pip install --system ".[reviewer]"

      - name: Run AI Reviewer
        run: |
          python .github/pr-reviewer/review.py
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

### Step 4: Configure Secrets
Go to your GitHub repository's **Settings > Secrets and variables > Actions** and add `GEMINI_API_KEY` (from Google AI Studio or Vertex AI service account key).

---

## 💻 Local Testing

The reviewer contains a unit test suite. To run tests locally:

```bash
# Run unit test suite using uv
uv run python .github/pr-reviewer/test_review.py
```
