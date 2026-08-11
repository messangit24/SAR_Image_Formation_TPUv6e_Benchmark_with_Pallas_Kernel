---
name: validate-mui-component
description: Validates MUI component implementation against GitLab ticket requirements and design tokens. Use when user asks to "validate component", "check MUI override", or "/validate-mui-component".
phase: review
user-invocable: true
---

# MUI Component Validation

Validate a theme override implementation in `shared-ui` against its GitLab ticket acceptance criteria.

## Usage

```
/validate-mui-component <TicketACsAndComponentName>
```

---

## Where to Find Files

All theme overrides live under:

```
shared-ui/src/theme/components/
```

Each component typically has its own folder named `MuiXxx/` containing:
- `MuiXxx.ts` — the override implementation
- `MuiXxx.test.tsx` — unit tests
- `MuiXxx.stories.tsx` — Storybook stories

Read all three files before doing any analysis.

---

## How Design Tokens Work

There are two distinct mechanisms for consuming design tokens in override files:

**Token accessors** — imported from `../../tokens/accessors`. Used for dimensional values that have a design token:
- `shape("BorderRadiusXxx")` → border radius and border width values
- `sizing("InputXxx")` / `sizing("IconXxx")` → component heights, icon sizes
- `spacing(n)` → returns a `calc(var(--mui-spacing-static-root) * n)` string

**CSS custom properties** — used directly as strings for values the browser resolves at runtime:
- `var(--mui-palette-*)` → all colors
- `var(--mui-typography-*)` → all typography properties (font-size, font-weight, line-height)
- `var(--mui-palette-action-*)` → interactive state colors

Hardcoded numeric or string values are acceptable **only** when no design token exists for that value, and should be accompanied by a comment explaining why.

---

## Testing Architecture

There are three test layers. Understand which layer covers which concern before evaluating gaps.

**Layer 1 — Override object tests** (`MuiXxx.test.tsx`)

Reads the override object directly without a browser. Validates:
- Token accessors called with the correct key: `expect(root.borderRadius).toBe(shape("BorderRadiusDefault"))`
- CSS variable strings are exactly correct: `expect(root.color).toBe("var(--mui-palette-text-primary)")`
- `validateCssVariable()` passes for variables that exist in the registry
- No hardcoded hex colors: `JSON.stringify(overrides)` matches against `/#[0-9a-fA-F]{3,8}/g`
- Basic rendering: component mounts without error via `renderWithTheme()`

**Layer 2 — Resolved token manifest** (`resolvedTokenManifest.test.ts`)

Located at `src/theme/components/resolvedTokenManifest.test.ts`. This is the alert mechanism for token value changes. Every component that uses `shape()`, `sizing()`, or `spacing()` must have an entry here. The manifest snapshots resolved pixel values using `toMatchInlineSnapshot()` — if an upstream token changes its value, this test breaks.

Check: does the component under review have an entry in this file? If it uses token accessors and has no entry, that is a gap.

**Layer 3 — Storybook interaction tests** (`Test_*` stories)

Uses `window.getComputedStyle()`, `getComputedSpacing()`, and `getResolvedCssVariables()` to verify actual browser-rendered values. Test stories are tagged `tags: ["!autodocs"]` and prefixed `Test_`. These catch CSS variable resolution failures that Layer 1 cannot catch.

---

## Validation Steps

### Step 1: Read the implementation

Read `MuiXxx.ts` and verify against each AC:
- Every dimensional value that has a token uses the correct accessor
- Every color uses a CSS variable (`var(--mui-palette-*)`)
- Every typography property uses a CSS variable (`var(--mui-typography-*)`)
- Any hardcoded value has a comment explaining why no token exists
- Uses `satisfies Components<Theme>["MuiXxx"]` pattern
- No `ownerState` function callbacks (use named slots like `sizeSmall` instead)

Map each AC to specific lines in the file. Quote the line. If an AC cannot be traced to a line, flag it as unimplemented.

### Step 2: Check the resolved token manifest

Open `resolvedTokenManifest.test.ts` and search for the component name.

- If the component uses `shape()`, `sizing()`, or `spacing()` and has **no entry**: flag as a gap — the component needs an inline snapshot test added.
- If it has an entry: verify the snapshot covers all accessor-based properties in the implementation.

### Step 3: Run the unit tests with coverage

```bash
cd shared-ui
npx vitest run --project unit --coverage --coverage.include="src/theme/components/MuiXxx/**" src/theme/components/MuiXxx/MuiXxx.test.tsx
```

Coverage target for the component file: **100% statements, branches, functions, lines**. These files are pure static objects — anything less than 100% means a property is untested.

If other test files are run together and pull in shared utilities, scope coverage to the component directory to isolate the result.

### Step 4: Verify Storybook stories against ACs

Read `MuiXxx.stories.tsx`. For each AC that describes a visual state or variant, confirm a story or `Test_*` play function exercises it.

Check that:
- `Test_*` stories have `tags: ["!autodocs"]`
- `Test_*` stories use `step` from the play function's destructured parameters, not imported from a helper
- Stories cover the states and variants called out in the ticket

### Step 5: Check design token integration

Read the implementation and answer these questions — do not grep blindly for `px`:

| Question | Pass condition |
|----------|---------------|
| Are border radius / border width values using `shape()`? | Yes for all values that have a shape token |
| Are component heights / icon sizes using `sizing()`? | Yes for all values that have a sizing token |
| Are padding / margin / gap values using `spacing()`? | Yes, producing `calc(var(--mui-spacing-static-root) * n)` strings |
| Are all colors CSS variables? | `var(--mui-palette-*)` — no hex, no rgb(), no named colors |
| Are all typography properties CSS variables? | `var(--mui-typography-*)` |
| Are hardcoded values commented? | Comment explains why no token exists |

---

## Report Format

Save to `.claude/scratch/<ticket>-<component-slug>/mui-validation-report.md`. Derive `<ticket>` from the ticket number in the argument or git branch name; derive `<component-slug>` from the component name lowercased (e.g., `mui-button`). Example: `.claude/scratch/1234-mui-button/mui-validation-report.md`.

```markdown
# MuiXxx Validation Report

**Date**: [date]
**Component**: [name]
**Ticket**: [title]
**Status**: [✅ COMPLETE | ⚠️ PARTIALLY COMPLETE | ❌ NEEDS WORK]

---

## AC Verification

| AC | Requirement | Status | Evidence |
|----|-------------|--------|----------|
| AC1 | [requirement summary] | ✅/❌ | [MuiXxx.ts:line] |
| AC2 | ... | | |

---

## Test Results

### Unit Tests
- Result: X/X passing
- Component file coverage: X% statements / X% branches / X% functions / X% lines

### Resolved Token Manifest
- Entry exists: ✅/❌
- Properties covered: [list]

### Storybook Interaction Tests
- Test_ stories present: ✅/❌
- States covered: [list]

---

## Token Integration

| Check | Status | Notes |
|-------|--------|-------|
| shape() for border values | ✅/❌ | |
| sizing() for heights | ✅/❌ | |
| spacing() for spacing | ✅/❌ | |
| CSS vars for colors | ✅/❌ | |
| CSS vars for typography | ✅/❌ | |
| Hardcoded values commented | ✅/❌/N/A | |

---

## Gaps

[For each gap, quote the AC requirement, show the current state, and state what is missing. No time estimates.]

---

## Conclusion

**Can ticket be closed?** [Yes / No — one sentence reason]
```
