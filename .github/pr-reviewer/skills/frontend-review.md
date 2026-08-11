# Front-End & React Review Criteria (Vite + TypeScript)

Apply the following design, code quality, and architectural standards to React files:

## 1. React Component Patterns
- **Functional Components**: Use modern functional components and React hooks.
- **Prop Definition**: In TypeScript, explicitly type props using interfaces. Avoid using `any`.
- **Effect & Callback Hooks**: Ensure dependencies arrays for `useEffect`, `useMemo`, and `useCallback` are fully complete and accurate. Do not omit referenced reactive variables.
- **State Isolation**: Keep local state local to avoid unnecessary component hierarchy re-renders.

## 2. User Interface & Premium Styling (Vanilla CSS)
- **Harmonious Color Palettes**: Avoid harsh generic colors (such as pure `#FF0000` red or pure `#0000FF` blue). Instead, verify that the UI adopts premium tailored design tokens, curated soft gradients, and sleek dark/light mode contrasts.
- **Google Fonts**: Prefer modern typography (e.g., Inter, Roboto, Outfit, Poppins, or Montserrat) imported via `<link>` or `@import` rather than browser default system fonts.
- **Layouts**: Enforce responsive layouts using CSS Grid or Flexbox. Avoid hardcoded pixel heights or widths that cause overflow on smaller device resolutions.
- **Sleek Elements**: Check for subtle modern elements, such as rounded corners (`border-radius`), micro-shadows (`box-shadow`), or glassmorphism effect (`backdrop-filter`).

## 3. Micro-Animations & Dynamic Hover Effects
- **Hover States**: Ensure every interactive button, link, or icon component includes smooth CSS hover transitions.
- **Easing Curves**: Use premium transitions like `transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);` instead of basic linear values.
- **Micro-feedback**: Validate that action components provide clear micro-interactions (such as a subtle scale-up on hover, scale-down on click, or glowing indicator).

## 4. Code Completeness & Console Errors
- **Clean Imports**: Verify there are no unused imports or dead variables.
- **Key Prop**: Verify that all lists rendered via `.map()` have unique, persistent `key` values (avoid using array indices as key where possible).
- **Warnings & Errors**: Clean up any console warnings or console logs (`console.log`) left over from debugging phases.
