# Generalized Notation Notation (GNN) adoption

[Generalized Notation Notation](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation)
(GNN) is a plain-text format for declaring Active Inference generative models:
variables and dimensions, connections, parameters, and equations in named
sections a program can read.

The Chapter 9 §9.1 weather model — the discrete perception example this
repository implements in
[`chapters/chapter_09/example_9_1_state_inference.py`](../chapters/chapter_09/example_9_1_state_inference.py)
and tests in [`tests/core/test_pomdp.py`](../tests/core/test_pomdp.py) — is
declared as a GNN document under `specs/`. This page describes the adoption, the
commands, and exactly what each check proves.

## Files

| Path | Role |
|---|---|
| [`specs/weather_pomdp.gnn.md`](../specs/weather_pomdp.gnn.md) | The GNN declaration: `A` (3×4 likelihood), `D` (uniform prior), the four declared variables, the equations, and the ontology annotations. Hand-authored; it is the canonical statement of the model's structure and parameters. |
| [`specs/gnn-pin.json`](../specs/gnn-pin.json) | The pin record: upstream repository, tag `v5.0.3`, commit `8f293d9ba810e6eda5c09842d9c7ed9ffa5f92c1`, declared language version, recording date, and SHA-256 digests of the parser files the document was reviewed against. |
| [`src/active_inference/gnn_spec.py`](../src/active_inference/gnn_spec.py) | Dependency-free reader (standard library + NumPy): sections, declared variables, `A`/`D` reshaped to the declared dimensions, `ModelParameters` scalars, and the pin. Not part of the package `__all__`. |
| [`scripts/gnn_parser_probe.py`](../scripts/gnn_parser_probe.py) | The only module that imports the upstream `gnn` package; runs as a subprocess under the pinned revision. |
| [`scripts/validate_gnn_spec.py`](../scripts/validate_gnn_spec.py) | The pinned-parser validator: materializes the pinned commit, verifies parser digests, and requires the pinned parser and the dependency-free reader to agree. |
| [`tests/test_gnn_spec.py`](../tests/test_gnn_spec.py) | The drift guard: binds the declared `A`/`D` to the chapter script's own literals and to the runtime posterior, and runs the validator against the pinned revision. |

## What the declaration is bound to

The model is stated twice, deliberately, and the tests make the two copies
cannot drift:

1. **Declaration** — `specs/weather_pomdp.gnn.md` (`A`, `D`, variables,
   equations).
2. **Implementation** — the module-level `A` and `D` literals in
   `chapters/chapter_09/example_9_1_state_inference.py`, which the chapter's
   `POMDPModel(A=A, D=D)` consumes.

`tests/test_gnn_spec.py` reads the chapter script's assignments from its source
and asserts they equal the declared arrays element by element, then runs the
declared arrays through `infer_states` and asserts the book's Eq. 15 posterior
`[0.18, 0.40, 0.36, 0.06]` for the observation "hot" (and the exact
`[0.1807, 0.3976, 0.3614, 0.0602]` values). Editing either side alone fails the
suite. The model itself comes from the book; nothing here is estimated or
fabricated.

## Commands

```bash
# Validate with the pinned upstream parser from a GNN checkout
GNN_ROOT=/path/to/Generalized_Notation_Notation \
GNN_PYTHON=/path/to/parser-runtime/bin/python \
uv run python scripts/validate_gnn_spec.py --json

# Installed pinned runtime (what CI uses)
uv run python scripts/validate_gnn_spec.py \
  --gnn-runtime /path/to/venv/bin/python --json

# The drift guard and the runtime posterior
uv run pytest tests/test_gnn_spec.py -v
```

`GNN_ROOT` is a `Generalized_Notation_Notation` checkout: the validator resolves
the pinned commit with `git archive`, verifies every recorded parser digest, and
runs the parser from that materialized tree, so a live working copy under
development is never imported. `GNN_PYTHON` (or `--gnn-runtime`) selects the
interpreter that has the parser dependencies; the GNN checkout's own `.venv`
works.

Exit codes are `0` pass, `1` conformance failure, and `2` pinned runtime or
checkout unavailable. An unavailable run is reported as *not run* and is never a
pass; the same applies to the pytest module, which prints an explicit skip when
no pinned parser is available.

The receipt is written to `docs/_build/gnn/validation-receipt.json`
(`--receipt` overrides it); `docs/_build/` is generated and gitignored.

## What each check proves

| Check | Proves |
|---|---|
| `test_declared_arrays_equal_the_chapter_literals` | The GNN declaration and the implementation's literals are the same model, field by field. |
| `test_infer_states_reproduces_the_book_posterior` | The declared model, run through this repository's inference, reproduces the book's Eq. 15 numbers. |
| `validate_gnn_spec.py` (pinned parser) | The document is well-formed GNN under the pinned revision, and the pinned parser and the dependency-free reader agree on sections, variables, model name, and `A`/`D`. The parser files are digest-verified before the document is parsed, and the pin's `tag` is resolved to its `commit` in the GNN object store (local tags, or `--verify-tag-remote`). For an installed runtime the installer's recorded VCS commit is compared with the pin; an install without that metadata, or a run with no object store, is reported as unresolvable rather than assumed. |

Nothing above proves the model is scientifically adequate, that the book's
numbers are correct beyond the asserted Eq. 15 posterior, or that a future
runtime will read the same document — the pin names the reviewed revision, not
what a consumer executes.

## Adoption context

The patterns and pin discipline this adoption follows are documented upstream in
the [GNN downstream adoption guide](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/integration/downstream_adoption.md)
(main branch; the guide postdates the pinned `v5.0.3` revision, so it is not part
of the pin). This repository uses **pattern D, structure-only declaration**, with
the book's authored `A` and `D` declared alongside the structure: the document
declares the model, no runtime code reads it, the chapter orchestrator owns the
behavior, and a conformance test binds the two sides so editing either alone
fails the suite. What that proves: the declared model is reviewable in one place
and cannot drift from the implementation. Two sibling repositories show the other
directions:

- [ActInf_RxInfer](https://github.com/ActiveInferenceInstitute/ActInf_RxInfer) —
  a conjugate Beta-Bernoulli model read by a dependency-free Julia reader;
- [ActiveBlockference](https://github.com/ActiveInferenceInstitute/ActiveBlockference) —
  a discrete grid-world POMDP whose document is generated from the implementation.

The language itself is specified in the
[GNN syntax reference](https://github.com/ActiveInferenceInstitute/Generalized_Notation_Notation/blob/main/docs/gnn/gnn_syntax.md)
and the exemplar corpus under `input/gnn_files/` in the GNN repository; the
courses' [GNN syntax track](https://github.com/ActiveInferenceInstitute/courses)
reads the same corpus at the same pin.
