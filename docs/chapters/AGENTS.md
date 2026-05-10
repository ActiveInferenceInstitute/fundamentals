# `docs/chapters/` — agent guide

One Markdown file per book chapter. Update when chapter orchestrator
scripts are added, removed, or renamed.

## File contract

Every chapter page must contain, in this order:

1. **H1 title** including the chapter number and book title fragment.
2. **One-paragraph summary** in our own prose — never a quoted passage
   from the book.
3. **Recipe** section listing the 5 modeling steps with chapter-specific
   parameterizations.
4. **Scripts** table linking every orchestrator under
   `chapters/chapter_<N>/` to the question it answers and the library
   imports it uses.
5. **Where the book takes this next** — a one-sentence forward pointer.

## When to edit

| Change | Edit which page |
|---|---|
| New `example_*.py` script | `chapter_<N>.md` (script table). |
| New `animation_*.py` or `visualize_*.py` | `chapter_<N>.md` (script table). |
| Renamed orchestrator | `chapter_<N>.md` and any `topics/` page that references it by name. |
| New chapter folder added | New `chapter_<N>.md` here + add to `README.md`. |

## Style

- Tables, not bullet lists, for the scripts section.
- Always include a "What it shows" or "What it adds" column so the
  table doubles as an index.
- Cross-link to topic pages when introducing a concept the topic page
  already covers in depth.
