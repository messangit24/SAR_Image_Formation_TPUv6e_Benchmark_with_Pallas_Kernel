---
name: write-typescript
description: Authoritative TypeScript and React/Web engineering standards, deep module design rules, strict typing rules, Zod validation, and frontend idioms. Enforced by dev-story, /owl, and review agents.
---

# TypeScript & Web Engineering Standards (`write-typescript`)

Use this skill when authoring, refactoring, or reviewing TypeScript and web application code in this repository.

---

## 🏛️ 1. Deep Modules & Interface-Centric Architecture

Modules, UI components, and service layers MUST be **deep**:
- **Simple Public Interface**: Expose a minimal, elegant set of props or API functions.
- **Hidden Complexity**: Encapsulate internal DOM manipulation, state transitions, API polling, and retry mechanisms within the module boundary.
- **Interface-Centric Testing**: Test components and services via their public inputs, outputs, and user-visible events (e.g. `@testing-library/react`), avoiding tests that reach into private implementation internals.
- **Anti-Pattern (Shallow Spaghetti Modules)**: Avoid creating leaky abstractions that force parents or consumers to manage multi-step internal state.

---

## 📐 2. Strict Type System & Zero-Any Invariant (TS 5+)

- **Zero `any` Policy**: Never use `any`. Use `unknown` for unvalidated boundaries and narrow with type guards or Zod schemas.
- **Discriminated Unions**: Model domain states, async statuses, and UI modes with explicit discriminated union types (`type State = { status: 'idle' } | { status: 'loading' } | ...`).
- **Explicit Return Types**: All exported functions, custom hooks, and API handlers MUST declare return types.
- **`satisfies` Operator**: Use `as const satisfies ...` for immutable lookup tables and configuration maps.

---

## 🛡️ 3. Runtime Boundary Validation (Zod)

All data entering from the outside world (API responses, WebSocket frames, URL query params, `localStorage`, form inputs) **MUST pass through a runtime Zod schema**:

```typescript
import { z } from 'zod';

export const UserProfileSchema = z.object({
  id: z.string().uuid(),
  email: z.string().email(),
  role: z.enum(['admin', 'member', 'viewer']),
});

export type UserProfile = z.infer<typeof UserProfileSchema>;
```

---

## ⚛️ 4. Modern React Patterns & Decoupled State

- **Custom Hooks for Business Logic**: Decouple state machines and data-fetching into custom hooks (`useStoryExecution`), keeping JSX rendering components lean and visual.
- **Explicit Cleanup in Effects**: Every `useEffect` managing event listeners, intervals, or WebSocket connections must return an explicit cleanup function.
- **Accessibility (a11y)**: Use semantic HTML (`<button>`, `<main>`, `<nav>`) and proper ARIA labels (`aria-expanded`, `aria-live`).

---

## 🧪 5. Hermetic Component & Unit Testing

- **User-Centric Assertions**: Use `@testing-library/react` and assert against screen text, accessible roles, and user interactions.
- **Mock External APIs**: Mock `fetch` / Axios using MSW (Mock Service Worker) or hermetic mock handlers. Zero external live network calls in test suites.
