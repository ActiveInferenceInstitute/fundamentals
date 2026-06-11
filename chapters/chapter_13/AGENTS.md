# `chapters/chapter_13/` - agent guide

Chapter 13 contains application-level active-inference demos for robotics and
social inference.

## Rules

- Keep application simulation logic in `active_inference.estimators.applications`.
- Every saved script must write raw sidecars through `save_chapter_data(13, ...)`.
- Prefer deterministic examples unless a stochastic seed is required.
