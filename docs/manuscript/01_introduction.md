# Introduction {#sec:introduction}

## Motivation

Namjoshi’s textbook develops the methods that anchor the companion [@namjoshi2026fundamentals]. Executable companion repositories need a clear boundary between textbook source authority, implemented examples, generated outputs, and future exact worked examples. Without that boundary, a figure can outlive the script that produced it, a count in prose can drift from the repository, and a "reproducible" claim can rest on nothing a reader can run.

## Contribution of the companion

The companion provides tested implementations of estimation, inference, and action examples, together with thin chapter orchestrators and data-producing runs. Readers can inspect the same function that generated a chart, change a model parameter, and compare the resulting raw arrays. A source-spine ledger records which parts of the textbook the repository addresses, while the artifact validators check which outputs are present. The local manuscript renderer uses the same checkout (see [@sec:reproducibility]) so its code and evidence can be replayed together.

## Reader Orientation

The examples are scoped to the companion. Source-ledger alignment identifies the relevant parts of the textbook; a passing artifact check alone does not establish numerical equivalence to every worked example in the book.
