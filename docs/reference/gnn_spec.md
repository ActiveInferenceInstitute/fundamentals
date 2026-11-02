# `gnn_spec` — GNN declaration reader

`active_inference.gnn_spec` reads the committed GNN declaration of the
Chapter 9 weather model, [`specs/weather_pomdp.gnn.md`](../../specs/weather_pomdp.gnn.md),
and its pin record [`specs/gnn-pin.json`](../../specs/gnn-pin.json). It uses the
standard library and NumPy only, so the model can be inspected without the
upstream `gnn` package present.

Like `extra_topics`, the module is not re-exported through the top-level
package `__all__`; the chapter conformance test and `scripts/validate_gnn_spec.py`
import it directly. The adoption, the commands, and what each check proves are
documented in [`docs/gnn.md`](../../docs/gnn.md).

## Module

| File | Role |
|---|---|
| `src/active_inference/gnn_spec.py` | Section, variable, parameter, and pin parsing for the committed GNN document. |

## Public API

| Symbol | Role |
| --- | --- |
| `ROOT` | Repository root as resolved from this file's location. |
| `GNN_SPEC_PATH` / `GNN_PIN_PATH` | Committed document and pin-record paths. |
| `REQUIRED_SECTIONS` | GNN sections the document must declare, in order. |
| `SPEC_MATRIX_KEYS` | Declared matrices the conformance test binds, currently `("A", "D")`. |
| `GNNVariable` | Frozen declaration of one variable: name, dimension tuple, dtype, section. |
| `parse_sections(text)` | Section name to body, in document order. |
| `parse_variables(text)` | Declared variables from `StateSpaceBlock` and the ontology section. |
| `parse_parameterization(text)` | Raw parameter tuples per declared name. |
| `declared_arrays(text=None)` | `A` and `D` reshaped to their declared dimensions. |
| `load_pin(path=None)` | Pin record as a dict, with the expected fields checked. |
| `spec_summary(text=None)` | Sections, variables, declared shapes, and pin fields for reporting. |

## Conventions

- The document is the declaration; the reader never rewrites it, and no code
  path generates it from the implementation.
- `declared_arrays` verifies the tuple count against the declared dimensions
  and raises `ValueError` on a mismatch, so a malformed document fails loudly.
- `parse_parameterization` tolerates both the `{...}` and `([...])` bodies of
  the GNN grammar, including `**` exponent notation.
- Reading the pin never asserts the upstream revision itself; the validator
  digests the parser files it materializes from the pinned commit.

## See also

- [`docs/gnn.md`](../../docs/gnn.md) — adoption, commands, and evidence.
- [`docs/reference/statistics/`](../statistics/) for the statistical tools the
  weather model is built from, and [`docs/reference/core.md`](core.md) for the
  `POMDPModel` API the declared matrices feed.
