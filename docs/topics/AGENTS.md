# `docs/topics/` — agent guide

Topic pages are deep dives on a single concept that spans multiple
chapters. Each page should make sense in isolation; do not chain
prerequisites across topic pages.

## File contract

A topic page **must** include, in order:

1. **H1 title** matching the filename (no chapter number).
2. **Framing paragraph** — what the concept is and why it matters.
3. **In this codebase** — bullet list of which classes / functions
   realize the concept, with paths.
4. **End-to-end snippet** — minimal runnable code demonstrating the
   concept (≤ 25 lines).
5. **Pitfalls** — actual numerical / API surprises, not theory.
6. **See also** — relative links to chapter, statistics, and reference
   pages.

## When to add a topic page

Only when the topic genuinely cannot be covered cleanly inside a single
chapter or reference page. If the topic is just *one* statistical tool,
prefer a `statistics/` page. If the topic is just *one* subpackage,
prefer extending the `reference/` page.

## Avoid

- Reproducing book passages — write summaries in our own prose.
- Long mathematical derivations — link to the book or a textbook.
- Code snippets > 25 lines — extract them into an orchestrator under
  `chapters/chapter_<N>/` instead and link to it.
