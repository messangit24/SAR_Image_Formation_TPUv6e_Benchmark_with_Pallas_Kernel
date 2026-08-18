---
name: grill-me
description: Interactive Socratic interview engine. Explores every aspect of a task, walks down design trees, resolves decision dependencies one-by-one using ask_question, and aligns on implementation before code is written.
---

# Socratic Design Alignment Engine (`grill-me`)

The user has requested that you interview them about every aspect of their task until you've reached a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

## Guidelines
- **One Question at a Time**: Ask the questions one at a time to maintain focus.
- **Explore Codebase First**: If a question can be answered by exploring the codebase, explore the codebase instead rather than asking trivial questions.
- **Use the `ask_question` Tool**: Always present structured multiple-choice questions using the `ask_question` tool.
- **Provide Recommendations**: List your recommended option first and prefix it with `(Recommended)`.
- **User-Centric Option Phrasing**: Format options as the user's direct response (e.g. "(Recommended) Use dynamic database names").
- **Proceed When Aligned**: Once all architectural branches and key decisions are resolved, summarize the agreed-upon plan and proceed to implementation.
