# Source spine and appendices

The companion is grounded in the inspected local PDF:

`/Users/4d/Documents/Namjoshi_2025_v5_Fundamentals_of_Active_Inference.pdf`

That source has Chapters 1-14 and Appendices A-D. It has no Chapter 15. The
ledger is represented in `active_inference.source_spine` and validated by
`scripts/validate_source_spine.py --require-pdf`.

## Appendix treatment

Appendix A is historical and contextual. The repo maps it to documented extras
topics:

- `active_inference_history`
- `active_inference_future`
- `deep_generative_models`
- `cybernetics_control`
- `information_theory_lineage`
- `reinforcement_learning_lineage`

Those topics produce static source-map visualizations and raw-data manifests;
they do not pretend Appendix A is an algorithmic specification.

Appendices B-D define executable mathematical scaffolding used throughout the
companion:

- Appendix B: notation/model setup and categorical orientation checks.
- Appendix C: probability, Gamma/Dirichlet helpers, information theory,
  colored noise, dynamical-system identities, and model comparison.
- Appendix D: static, dynamic, expected, and variant free-energy forms.

## Implementation summary

| Area | Main API |
|---|---|
| Source ledger | `active_inference.source_spine` |
| Appendix B/C probability helpers | `active_inference.core.appendix_math` |
| Appendix C colored noise | `active_inference.core.noise` |
| Appendix C model comparison | `active_inference.core.model_comparison` |
| Appendix D free-energy forms | `active_inference.core.free_energy_forms` |
| Extras registry | `active_inference.extra_topics` |

## Validation

```bash
uv run python scripts/validate_source_spine.py --require-pdf
uv run python scripts/validate_book_topic_coverage.py --require-rendered
uv run python scripts/validate_raw_data_exports.py --root output/data
```

The first command prevents invented chapters and appendix drift. The second
and third commands ensure that the documented extras topics have rendered PNG
or GIF artifacts plus paired NPZ/JSON raw-data sidecars.

## See also

- [`../reference/source_spine.md`](../reference/source_spine.md)
- [`../reference/book_topic_matrix.md`](../reference/book_topic_matrix.md)
- [`../reference/core.md`](../reference/core.md)
