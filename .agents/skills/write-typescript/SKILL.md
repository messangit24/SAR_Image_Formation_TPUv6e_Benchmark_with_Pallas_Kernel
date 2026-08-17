---
name: write-typescript
description: Authoritative TypeScript and React/Web engineering standards, strict typing rules, Zod validation, and frontend idioms. Enforced by dev-story, /owl, and review agents.
---

# TypeScript & Frontend Engineering Standards (`write-typescript`)

Use this skill when authoring, refactoring, or reviewing TypeScript, React, Next.js, or web application code.

---

## 🏛️ 1. Strict Typing & Type Safety (TS 5+)

- **Zero `any` Policy**: Never use `any`. Use `unknown` for unvalidated inputs and narrow types using type guards or Zod schemas.
- **Discriminated Unions**: Model domain states, asynchronous statuses, and event payloads as discriminated unions:
  ```typescript
  type AsyncState<T> =
    | { status: 'idle' }
    | { status: 'loading' }
    | { status: 'success'; data: T }
    | { status: 'error'; error: string };
  ```
- **Explicit Return Types**: All exported functions, hooks, and API endpoints must declare explicit return types.
- **Readonly Immuntability**: Use `readonly` arrays and properties for immutable application state.

---

## 🌐 2. Runtime Schema Validation (Zod)

- All external inputs (API request/response payloads, URL parameters, localStorage, WebSocket messages) **MUST** be validated at the boundary using Zod:
  ```typescript
  import { z } from 'zod';

  export const UserSchema = z.object({
    id: z.string().uuid(),
    email: z.string().email(),
    role: z.enum(['admin', 'user', 'viewer']),
  });

  export type User = z.infer<typeof UserSchema>;
  ```

---

## ⚛️ 3. React & UI Best Practices

- **Functional Components**: Use arrow functions or standard function declarations with typed props.
- **Custom Hooks for Logic**: Decouple business logic and data fetching from UI rendering into reusable custom hooks (`useFeatureData.ts`).
- **Clean Effect Cleanup**: Every `useEffect` with subscriptions, timers, or event listeners must return an explicit cleanup function.
- **Accessibility (a11y)**: All interactive elements must include proper ARIA attributes, semantic HTML tags (`<button>`, `<main>`, `<nav>`), and keyboard navigability.

---

## 🛡️ 4. Security & Sanitization

- **XSS Prevention**: Never use `dangerouslySetInnerHTML` unless explicitly sanitized with DOMPurify.
- **Safe Navigation**: Avoid unchecked property access; use optional chaining (`user?.profile?.avatar`) and nullish coalescing (`??`).
- **Environment Variables**: Access client/server env vars through a validated config module (e.g. `@/lib/env.ts`).

---

## 🧪 5. Testing & Verification

- **Component Tests**: Use Vitest / Jest with React Testing Library focusing on user-visible behavior (e.g. `screen.getByRole('button')`).
- **Type Checking Gate**: Every change must pass `tsc --noEmit` with zero warnings.
