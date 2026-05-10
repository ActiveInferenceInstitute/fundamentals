# `docs/chapters/` — per-chapter overviews

One page per book chapter. Each page maps the book's exposition to the
orchestrator scripts in `chapters/chapter_<N>/` and to the library
components in `src/active_inference/` they exercise.

## Pages

| File | Mirrors | Purpose |
|---|---|---|
| [`chapter_01.md`](chapter_01.md) | Book Ch. 1 — *The Hypothesis-Testing Brain* | Box scenario, three perspectives, Bayes' theorem walk-through, inverse problem demos. |
| [`chapter_02.md`](chapter_02.md) | Book Ch. 2 — *Hidden State Estimation* | The full linear-Gaussian recipe + Examples 2.1–2.10. |
| [`chapter_03.md`](chapter_03.md) | Book Ch. 3 — *Combining Learning and Inference* | OLS / GD / Bayesian linear regression, multivariate Gaussians, LGS, factor-analysis EM. |

## How to read a chapter page

Each page has the same skeleton:

1. *What the chapter covers* — a paragraph summary in our own words.
2. *Recipe* — the canonical 5-step modeling pattern, parameterized for
   that chapter.
3. *Scripts* — table mapping each orchestrator to the question it
   answers and the library symbols it imports.
4. *Where the book takes this next* — a forward pointer to the next
   chapter's chapter-page in this folder.

## See also

- [`../topics/`](../topics/) — concept-by-concept walkthroughs that span
  more than one chapter.
- [`../reference/`](../reference/) — full API listing for the subpackages
  the chapter scripts import from.
