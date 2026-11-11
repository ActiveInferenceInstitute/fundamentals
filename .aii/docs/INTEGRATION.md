# InstituteOS integration — fundamentals

## Identity

| Field | Value |
| --- | --- |
| Repository | `ActiveInferenceInstitute/fundamentals` |
| Affiliation | `institute` |
| Category | `education` |
| Status | `active` |
| Maturity | `released` |
| Steward | Active Inference Institute |
| Default branch | `main` |
| Current release | — |
| License | `MIT` |
| Citation | Active Inference Institute — fundamentals. |

## Federation

| Setting | Value |
| --- | --- |
| Dashboard mode | `repos` |
| Analytics | `true` |
| Dashboard | `true` |
| Sync policy | `pull` |
| Registries | — |
| Tags | fundamentals, education, materials, python |

## Ecosystem relations

| Repository | Relation | Note |
| --- | --- | --- |
| `ActiveInferenceInstitute/InstituteOS` | `referenced-by` | Federated via the InstituteOS .aii manifest. |
| `ActiveInferenceInstitute/Active_Inference_Ontology` | `references` | Uses Active Inference Ontology terms. |

## Key artifacts

| Path | Kind | Description |
| --- | --- | --- |
| `README.md` | `doc` | Repository overview. |
| `AGENTS.md` | `doc` | Agent and contributor conventions. |
| `src` | `code` | Library source tree (`active_inference`). |
| `chapters` | `code` | Thin chapter orchestrators for Chapters 1-14. |
| `extras` | `code` | Cross-cutting extras topic orchestrators from the extra-topics registry. |
| `demo` | `code` | Application demos (eye saccades, bicycle, drone flight). |
| `specs` | `doc` | GNN declaration of the Chapter 9 weather model and its upstream parser pin. |
| `tests` | `code` | pytest suite — unit, contract, and chapter/extras/demo smoke tests. |
| `docs` | `doc` | Documentation hub, including the GNN adoption page and reference pages. |
| `scripts` | `code` | Batch renderers and artifact validators, including the pinned GNN conformance validator. |
| `output` | `data` | Generated figures, raw-data sidecars, and notebooks (gitignored, regenerable). |
| `manuscript` | `doc` | Manuscript scaffold describing the companion's artifacts and evidence. |
| `pyproject.toml` | `code` | Python package config. |
| `LICENSE` | `doc` | License. |

## Portable tasks

| Task | Command | Working directory | Description |
| --- | --- | --- | --- |
| `test` | `python -m pytest` | `.` | Run the test suite. |

## Validation

```bash
python -m aii_platform.aii_sidecar.validate <repo> --completeness
python -m aii_platform.aii_sidecar docs <repo> --overwrite
python -m aii_platform.aii_sidecar.run <repo> --list
```

The sidecar is valid when `validate` prints `OK` and reports no missing artifact,
manifest, link, or declared-doc path. Completeness is a curation target, not a
runtime requirement.

## Data boundary

This sidecar carries metadata only. It does not grant execution permission, does not
replace the repository's own documentation, and must not be used to move private
content into a public repository. Task execution is limited to trusted checkouts.
