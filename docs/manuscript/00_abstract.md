# Abstract {#sec:abstract}

The `{{PACKAGE_NAME}}` package in `{{REPOSITORY_SLUG}}` is an open-source Python companion to *Fundamentals of Active Inference*: it implements the book's algorithms, renders their figures and raw-data sidecars, and validates the rendered artifacts and orchestrator contracts that connect them. It ships {{CHAPTER_COUNT}} chapter directories with {{CHAPTER_SCRIPT_COUNT}} chapter orchestrators and {{EXTRAS_TOPIC_COUNT}} extras topic directories, each built on the tested `active_inference` library rather than on duplicated inline mathematics.

We describe how the companion relates a textbook method to library code, a thin orchestrator, and generated figures and data. Two executable examples show complementary outcomes: a Gaussian posterior summarizes a linear inverse problem, and a symmetric nonlinear observation map produces two supported modes. The experiments use declared synthetic observation models. The figures are registered against their producer paths, and the manuscript renderer recomputes volatile repository facts before rendering.

The Generalized Notation Notation declaration of the Chapter 9 weather model is pinned at `{{GNN_PIN_TAG}}` (`{{GNN_PIN_COMMIT}}`), and a conformance check binds that declaration to the implementation.

Evidence boundary: the textbook itself is not open source; this manuscript describes the companion implementation and its validation without reproducing protected book content.
