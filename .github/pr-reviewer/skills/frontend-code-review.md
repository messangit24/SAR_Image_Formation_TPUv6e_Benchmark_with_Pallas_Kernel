---
name: frontend-code-review
description: Reviews frontend Next.js/React code for quality, conventions, design tokens, and completeness. Use when user asks to "review frontend code", "code review react", or "/frontend-code-review".
phase: review
user-invocable: true
---

# Frontend Code Review

Perform a focused code review for frontend TypeScript/React code in the project.

## Scope

**IMPORTANT**: This review should ONLY examine code in the frontend workspace under `frontend/`.
Do not review code in other directories (Java services, scripts, etc.).

When invoked, prefer glob patterns like:
- `web-app/src/**/*.tsx`
- `web-app/src/**/*.ts`
- `shared-ui/src/**/*.tsx`
- `shared-ui/src/**/*.ts`

> **Note on workstream scope**: When loaded by the CI bot, this file applies only to the
> `frontend_webapp` workstream (`web-app/**`). Changes in
> `shared-ui/src/theme/components/Mui*` use `audit-component-implementation/SKILL.md`
> instead, and changes in `shared-ui/src/components/` use
> `validate-mui-component/SKILL.md`. A single MR may trigger more than one criteria file if it
> touches multiple workstreams.

**CRITICAL**: Verify all code follows the conventions established for the frontend:
- **App Router**: Server Components by default; `'use client'` only when truly needed
- **MUI + design tokens**: No hardcoded colors, spacing, or typography — use `sx`/theme tokens
- **Accessibility**: Labels, keyboard nav, focus order, and semantic headings are correctness constraints, not suggestions
- **Tree-shaking**: Direct imports only; no barrel file re-exports in client-side code (except exempted paths)
- **No secret leakage**: Only `NEXT_PUBLIC_*` env vars may reach the client
- **No console logs**: Remove `console.log` / `console.error` from production code paths

---

## Primary Focus Areas

### 1. Server vs Client Boundary
Check for incorrect or unnecessary client-side rendering:
- **Unnecessary `'use client'`**: Flag components that have no state, effects, or event handlers — they should be Server Components
- **Missing `'use client'`**: Flag components that use hooks (`useState`, `useEffect`, etc.) without the directive
- **Data fetching in Client Components**: Data fetching should happen in Server Components or route handlers where possible
- **Over-fetching on the client**: Prefer passing data as props from Server Components rather than fetching client-side

### 2. MUI Theme & Design Token Usage
- **Hardcoded colors**: Flag any raw hex, `rgb()`, or named CSS colors — use `theme.palette.*` or token variables
- **Hardcoded spacing**: Flag raw pixel values in `sx` or `style` props — use `theme.spacing()` or token-based values
- **Hardcoded typography**: Flag font sizes/weights not from `theme.typography.*`
- **Inline `style` prop**: Prefer `sx` for MUI components; `style` bypasses the theme system
- **Missing `sx` shorthand**: Verbose `style={{ marginTop: '8px' }}` should be `sx={{ mt: 1 }}`

### 3. Directory & Routing Conventions
Check that code is placed in the correct location:
- **`src/app/`**: Routes and colocated feature code (App Router)
- **`src/app/_components/`**: Shared application components (underscore = not a route)
- **`src/app/[feature]/_components/`**: Feature-private components — not shared across features
- **`src/components/`**: Design-system components (from `shared-ui`)
- **`src/styles/`**: Styling and design token integration
- **`src/data/`**: Mock/demo data only
- Flag components placed in the wrong layer (e.g., a feature-specific component in `src/components/`)

### 4. Tree-Shaking & Import Hygiene
- **Barrel file re-exports in client code**: Flag `export { Foo } from './foo'` in `index.ts` for client-side modules
  - ❌ Bad: `import { Foo } from '@/lib/foo'` where `@/lib/foo` is an `index.ts` barrel
  - ✅ Good: `import { Foo } from '@/lib/foo/Foo'` (direct file import)
- **Exemptions** (OK to use barrel imports):
  - Re-exports from external packages (e.g., `shared-ui`)
  - Server-only directories: `/services/**`, `/security/**`, `/config/**`
  - Test files: `**/__tests__/**`, `*.test.ts`, `*.test.tsx`
- **Unused imports**: Flag any import not referenced in the file

### 5. Accessibility (a11y)
Treat these as **correctness bugs**, not suggestions:
- **Missing labels**: Every `<input>`, `<select>`, `<textarea>` must have an associated `<label>` or `aria-label`
- **Icon-only buttons**: Must have `aria-label` describing the action
- **Non-semantic headings**: Heading order must be logical (`h1` → `h2` → `h3`), not chosen for style
- **Focus management**: Modals/dialogs must trap focus and restore it on close
- **Keyboard navigation**: Interactive elements must be reachable and operable by keyboard
- **Color contrast**: Do not use `theme.palette` color combinations with insufficient contrast
- **`alt` on images**: All `<img>` and MUI `<Avatar>` with image `src` must have descriptive `alt`

### 6. Secret & Environment Variable Leakage
- **CRITICAL**: Any `process.env.MY_VAR` (without `NEXT_PUBLIC_` prefix) referenced in a Client Component will be `undefined` at runtime and is a bug
- **CRITICAL**: Any secret (API keys, tokens, DB URLs) must never appear in `NEXT_PUBLIC_*` vars
- Flag any client-side code reading non-public env vars

### 7. Completeness Check ("Required Stuff")
- **TODO/FIXME comments**: Flag any unresolved items
- **Console statements**: Flag `console.log`, `console.error`, `console.warn` in production code
- **Missing error boundaries**: Server Component data fetches should be wrapped in `error.tsx` / `loading.tsx` at the route level
- **Missing loading states**: Any async data fetch needs a loading UI (skeleton, spinner)
- **Unhandled promise rejections**: `async` functions in event handlers must have try/catch or `.catch()`
- **Missing test files**: Flag new components or routes with no corresponding test in `src/__tests__/` or colocated `*.test.tsx`
- **Type safety**: Flag `any` types, non-null assertions (`!`), and type casts (`as Foo`) without a comment explaining why

### 8. Testing Quality
- **React Testing Library preferred**: Tests should query by role, label, or text — not by class names or test IDs unless necessary
- **Implementation detail tests**: Flag tests that assert internal state or component structure rather than user-visible behavior
- **Missing mutation test signal**: After significant logic changes, note that `/mutation-test` should be run scoped to the changed files (target: ≥80% mutation score)

---

## Review Process

1. **Read all files** being reviewed thoroughly
2. **Check Server/Client boundaries** — verify `'use client'` usage is intentional and minimal
3. **Scan for design token violations** — list every hardcoded value with line numbers
4. **Check a11y** — flag every missing label, focus trap, or semantic issue
5. **Check imports** — flag barrel re-exports and unused imports
6. **Check for secret leakage** — verify no non-public env vars in client code
7. **Verify completeness** — TODOs, error handling, loading states, tests
8. **Flag issues by severity**
9. **Save the report** — write output to `.claude/scratch/<ticket>-<slug>/frontend-code-review.md`. Derive `<ticket>` from the ticket number in the argument or git branch name; derive `<slug>` from the feature name. Example: `.claude/scratch/1234-case-nav-drawer/frontend-code-review.md`.

### Priority Checks (In Order)
1. ⚠️ **CRITICAL**: Non-`NEXT_PUBLIC_*` env var in a Client Component (runtime bug)
2. ⚠️ **CRITICAL**: Secret value in a `NEXT_PUBLIC_*` env var
3. ⚠️ **CRITICAL**: Missing `'use client'` on a component using hooks/events
4. 🟡 **Important**: Hardcoded colors/spacing (breaks theme consistency)
5. 🟡 **Important**: Missing a11y labels or keyboard nav
6. 🟡 **Important**: Barrel file re-exports in client-side modules (breaks tree-shaking)
7. 🟡 **Important**: `console.log` in production code paths
8. 🟢 **Suggestion**: Unnecessary `'use client'` (move data up to Server Component)
9. 🟢 **Suggestion**: `any` type or non-null assertion without explanation
10. 🟢 **Suggestion**: Missing loading/error boundary at route level

---

## Output Format

```markdown
## Frontend Code Review: [Filename]

### Summary
[1-2 sentence assessment]

### Critical Issues
🔴 **[Line X]**: [Issue — e.g., non-public env var read in Client Component]
   - Problem: [Why this is broken]
   - Fix: [How to address]

### Design Token Violations
🟡 **[Line X]**: Hardcoded `color: '#1A73E8'`
   - Replace with: `color: 'primary.main'` or `theme.palette.primary.main`

### Accessibility Issues
🟡 **[Line X]**: Icon button missing `aria-label`
   - Fix: `<IconButton aria-label="Delete record">`

### Import / Tree-Shaking Issues
🟡 **[Line X]**: Barrel import from `@/lib/utils` in client component
   - Replace with direct import: `import { formatDate } from '@/lib/utils/formatDate'`

### Server / Client Boundary Issues
- **[Line X]**: [e.g., data fetch inside a Client Component — move to parent Server Component]

### Completeness Issues
⚠️ **[Line X]**: [Missing item — e.g., no error boundary for async route]
   - Required: [What's needed]
   - Fix: [How to address]

### Strengths
- [What's done well]

### Action Items
1. [Fix env var leak at line X]
2. [Replace hardcoded color at line Y with theme token]
3. [Add aria-label to icon button at line Z]
```

---

## Common Anti-Patterns to Flag

### Server vs Client Boundary
```tsx
// ❌ WRONG — fetching data in a Client Component
'use client'
export function UserList() {
  const [users, setUsers] = useState([]);
  useEffect(() => { fetch('/api/users').then(...).then(setUsers) }, []);
}

// ✅ CORRECT — fetch in Server Component, pass as props
// page.tsx (Server Component, no directive)
export default async function Page() {
  const users = await getUsers(); // server-side fetch
  return <UserList users={users} />;
}
```

```tsx
// ❌ WRONG — missing 'use client' on interactive component
export function SearchInput({ onChange }) {  // uses onChange (event handler)
  const [value, setValue] = useState('');   // uses state — needs directive!
  ...
}

// ✅ CORRECT
'use client'
export function SearchInput({ onChange }) { ... }
```

### Design Tokens
```tsx
// ❌ WRONG — hardcoded values
<Box sx={{ color: '#1A73E8', mt: '16px', fontSize: '14px' }} />

// ✅ CORRECT — theme tokens
<Box sx={{ color: 'primary.main', mt: 2, typography: 'body2' }} />
```

### Accessibility
```tsx
// ❌ WRONG — no label
<input type="text" placeholder="Search..." />

// ✅ CORRECT
<label htmlFor="search">Search</label>
<input id="search" type="text" />
// or with MUI:
<TextField label="Search" />

// ❌ WRONG — icon button with no accessible name
<IconButton onClick={handleDelete}><DeleteIcon /></IconButton>

// ✅ CORRECT
<IconButton onClick={handleDelete} aria-label="Delete record">
  <DeleteIcon />
</IconButton>
```

### Tree-Shaking
```ts
// ❌ WRONG — barrel import in client-side code
import { formatDate, formatCurrency } from '@/lib/utils'; // index.ts barrel

// ✅ CORRECT — direct imports
import { formatDate } from '@/lib/utils/formatDate';
import { formatCurrency } from '@/lib/utils/formatCurrency';
```

### Secret / Env Leakage
```tsx
// ❌ WRONG — private env var in client code (will be undefined at runtime)
'use client'
export function ApiClient() {
  const key = process.env.API_SECRET_KEY; // undefined in browser!
}

// ✅ CORRECT — keep secrets server-side
// Server Component or API route only:
const key = process.env.API_SECRET_KEY;
```

### Console Statements
```tsx
// ❌ WRONG — console in production code
const handleSubmit = () => {
  console.log('Submitting form:', data);
  submitForm(data);
};

// ✅ CORRECT — remove or guard with NODE_ENV check
const handleSubmit = () => {
  submitForm(data);
};
```
