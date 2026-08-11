---
name: audit-component-implementation
description: Quality audit for MUI component override implementation. Checks coding standards, token usage, test coverage, and reports rules violations.
phase: review
argument-hint: <MuiComponentName>
disable-model-invocation: true
user-invocable: true
---

# Audit Component Override

Audit the implementation of: **$ARGUMENTS**

## Workflow

### 1. Check File Structure

Verify all required files exist:

```
shared-ui/src/theme/components/{ComponentName}/
  Mui{Component}.ts
  Mui{Component}.test.tsx
  Mui{Component}.stories.tsx
```

Check component is registered in `components/index.ts`.

If files missing: report incomplete structure, recommend running `/generate-component-override`.

### 2. Audit Override File

Read the override file. Check against rules in `rules/theme-overrides.md`:

- Uses `satisfies` pattern (not type annotation)
- No hardcoded values (hex colors, pixel sizes, opacity decimals)
- No callback functions in `styleOverrides`
- Proper token accessor usage (shape, sizing, spacing, typographyStyle)
- CSS variables for runtime values (colors, typography, action states)
- No importing from generated files (palette.ts, typography.ts, shape.ts)

Flag each violation with file:line, current code, and recommended fix.

### 3. Audit Tests

Read the test file. Verify five-category structure:

1. Token Accessor Integration
2. CSS Variable References
3. Custom Style Properties
4. No Hardcoded Values
5. Rendering

Flag missing categories or incomplete slot coverage.

### 4. Audit Stories

Read the stories file. Check:

- Figma source documented in meta description
- Configured and NOT configured variants listed
- Test stories prefixed with `Test_` and tagged `['!autodocs']`
- Test stories use `step` from play parameters (not imported)
- `Test_CssVariableResolution` story exists

### 5. Check Resolved Token Manifest

If component uses accessor functions: verify entry exists in `resolvedTokenManifest.test.ts`.
If component uses only CSS variables: verify it is NOT in the manifest and `Test_CssVariableResolution` has exact value assertions.

### 6. Figma Alignment (Optional)

If Figma nodeId available in story parameters, delegate to **figma-design-analyzer** subagent and compare output with implementation. Flag color, sizing, spacing, and typography discrepancies.

## Report Format

```markdown
# Audit: Mui{Component}

**Status**: PASS / WARNINGS / CRITICAL

## Passing
- [what's correct]

## Violations
### [Title]
**Location**: `file.ts:42`
**Rule**: [which rule from theme-overrides.md]
**Current**: `[code]`
**Fix**: `[corrected code]`

## Coverage
| Category | Status |
|----------|--------|
| File Structure | ... |
| Type Pattern | ... |
| No Hardcoded Values | ... |
| Token Usage | ... |
| Test Structure | ... |
| Storybook Docs | ... |
| Test Stories | ... |
| Token Manifest | ... |
```
