# Abstract {#sec:abstract}

The `{{PACKAGE_NAME}}` package in `{{REPOSITORY_SLUG}}` is an open-source Python companion to *Fundamentals of Active Inference*: it implements the book's algorithms, renders their figures and raw-data sidecars, and validates the rendered artifacts and orchestrator contracts that connect them. It ships {{CHAPTER_COUNT}} chapter directories with {{CHAPTER_SCRIPT_COUNT}} chapter orchestrators and {{EXTRAS_TOPIC_COUNT}} extras topic directories, each built on the tested `active_inference` library rather than on duplicated inline mathematics.

This draft is an evidence map rather than a completed paper. It records the project problem, source surfaces, methods narrative, reproducibility hooks, and open evidence gaps, so later work can promote the scaffold into a publishable manuscript without inventing unsupported claims. Every dynamic value in the prose is a declared variable recomputed from the repository, and every figure cited here is registered against the file its orchestrator generates.

The Generalized Notation Notation declaration of the Chapter 9 weather model is pinned at `{{GNN_PIN_TAG}}` (`{{GNN_PIN_COMMIT}}`), and a conformance check binds that declaration to the implementation.

Evidence boundary: the textbook itself is not open source; this manuscript describes the companion implementation and its validation without reproducing protected book content.
