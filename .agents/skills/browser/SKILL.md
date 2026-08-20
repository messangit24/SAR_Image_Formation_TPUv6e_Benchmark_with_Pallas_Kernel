---
name: browser
description: Automated browser testing, UI verification, visual layout inspection, and end-to-end interaction testing for web applications and dashboards.
---

# Browser UI Testing & Visual Verification Skill (`browser`)

Use this skill whenever:
1. You are developing or refactoring web interfaces, dashboards, or frontend applications (React, Next.js, Vite, Deck.gl, FastAPI UI).
2. You need to verify DOM hydration, user interaction flows (forms, buttons, modals), and API network calls.
3. You need to capture visual screenshot artifacts, check console logs for JavaScript errors, or run headless end-to-end (E2E) browser tests.

---

## 🛠️ 1. Local Server Management & Port Verification

### Step 1: Start the Local Development Server
Launch the server with an explicit port (or dynamic port offset in worktrees):
* **Node/React/Vite**: `npm run dev -- --port 3000` or `npm run preview`
* **Python/FastAPI**: `uv run uvicorn src.main:app --port 8000 --host 0.0.0.0`
* **Static Dashboard**: `python3 -m http.server 8085`

### Step 2: Verify Port Availability & GFE Warmup
Before launching browser interactions, ensure the server is listening:
```bash
# Check listening status
fuser <port>/tcp
curl -I http://localhost:<port>/
```

> [!NOTE]
> **Cloudtop Access Invariant**: When creating URL artifacts for the user to view in their local browser, always use `http://taipei.c.googlers.com:<PORT>` instead of `http://localhost:<PORT>`.

---

## 🌐 2. Browser Automation & Verification Protocol

### Step 2.1: Navigation & DOM Hydration
1. Navigate to the local URL (e.g. `http://localhost:<PORT>/`).
2. Wait for initial DOM hydration and network idle (allow 3–5 seconds for WebAssembly / dynamic chunks to load).
3. Verify that the primary root container (`#root`, `#app`, or main component) has rendered.

### Step 2.2: UI Interaction Verification
* **Form Inputs**: Locate the input selector, focus/click the element, and type the input value.
* **Buttons & Controls**: Click interactive trigger elements and observe asynchronous state transitions (loading spinners, alert toasts, modal popups).
* **Navigation & Routing**: Click navigation links and verify that URL paths and viewports transition correctly without page reloads.

### Step 2.3: Console & Network Error Diagnostics
Inspect browser runtime telemetry:
1. **Console Logs**: Check for JavaScript runtime exceptions, React hydration mismatches, WebAssembly panics, or uncaught promise rejections.
2. **Network Activity**: Inspect API fetch requests, verifying that HTTP response codes are `200 OK` (not `500 Internal Error` or `404 Not Found`).

### Step 2.4: Visual Screenshot Capture
Capture screenshots to verify visual layouts, responsive breakpoints, and styling:
1. Capture viewport or full-page screenshot to `/tmp/ui_verification.png` (or the conversation artifact directory).
2. Inspect the captured image using the `view_file` tool to verify typography, alignment, color palettes, and contrast.

---

## 🧪 3. Automated E2E Testing with Playwright / Pytest

When running programmatic test suites:

### Playwright Python E2E
```bash
uv run pytest tests/e2e/ --headed=false
```

### Playwright / Vitest Web E2E
```bash
npm run test:e2e
# or
npx playwright test --reporter=list
```

---

## 📋 4. UI Story Acceptance Criteria Checklist

When completing a frontend story (`docs/bmad/stories/story-X.Y`):
- [ ] Local dev server boots cleanly with zero build/bundle warnings.
- [ ] Root application container mounts and hydrates without console errors.
- [ ] All critical user journeys (CUJs) execute and produce expected state updates.
- [ ] API network requests return valid schemas and payload contracts.
- [ ] Visual screenshots confirm clean styling, responsiveness, and zero layout overflow.
