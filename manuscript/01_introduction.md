# Introduction {#sec:introduction}

## Motivation

Executable companion repositories need a clear boundary between textbook source authority, implemented examples, generated outputs, and future exact worked examples. Without that boundary, a figure can outlive the script that produced it, a count in prose can drift from the repository, and a "reproducible" claim can rest on nothing a reader can run.

## Contribution of This Scaffold

This directory supplies the manuscript surfaces for the companion: modular sections, local render metadata, declared manuscript variables, a figure registry bound to generated files, and a source-surface supplement. The render path lives in the repository (see [@sec:reproducibility]) so the manuscript is built by the same checkout that contains the code it describes.

## Reader Orientation

The manuscript should be read as a controlled starting point. It names the project evidence surfaces and the claims they may eventually support, but it does not yet assert final publication readiness.
