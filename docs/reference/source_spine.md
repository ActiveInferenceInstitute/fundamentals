# PDF source-spine contract

The companion currently uses the local source PDF:

`/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf`

The inspected metadata is recorded in `active_inference.source_spine`:

| Field | Value |
|---|---|
| Pages | 1153 |
| Build date | 2025-03-20 |
| Chapters | 1-14 |
| Appendices | A-D |
| Chapter 15 | Absent |

The ledger is static by design. It is a source-of-truth guard for repository
shape and docs, not a claim that every section has an exact numeric worked
example.

## Public API

| Symbol | Purpose |
|---|---|
| `SOURCE_PDF_PATH` | Absolute path to the inspected local PDF. |
| `SOURCE_PDF_PAGES` | Expected page count, currently 1153. |
| `SOURCE_PDF_BUILD_DATE` | Expected PDF build date, currently 2025-03-20. |
| `SourceChapter` | Dataclass for one chapter number, title, and section tuple. |
| `SourceAppendix` | Dataclass for one appendix letter, title, section tuple, and executable flag. |
| `CHAPTERS` | Tuple of Chapters 1-14 from the inspected table of contents. |
| `APPENDICES` | Tuple of Appendices A-D from the inspected table of contents. |
| `chapter_numbers()` | Returns `(1, ..., 14)`. |
| `appendix_letters()` | Returns `("A", "B", "C", "D")`. |
| `chapter_section_ids()` | Returns all chapter section identifiers. |
| `appendix_section_ids()` | Returns all appendix section identifiers. |
| `all_section_ids()` | Returns chapter plus appendix section identifiers in source order. |
| `has_chapter(number)` | True only for source-backed chapter numbers. |
| `has_section(identifier)` | True only for source-backed chapter or appendix section IDs. |

## Validation

Run:

```bash
uv run python scripts/validate_source_spine.py --require-pdf
```

The validator checks that:

- the PDF exists when `--require-pdf` is supplied;
- the page count remains 1153;
- `chapters/` and `docs/chapters/` expose Chapters 1-14 only;
- `chapter_15` is absent;
- every appendix section in `APPENDICES` appears in
  `docs/reference/book_topic_matrix.md`.
