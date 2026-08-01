"""Example 10.8 — hierarchical depth in POMDPs (Chapter 10, §10.4, nested time scales).

Run::

    python chapters/chapter_10/example_10_8_hierarchical.py [--save] [--regime 0|1]

A two-layer hierarchical POMDP (book §10.4, Fig 10.4.1). The **slow** top layer encodes a
context "regime"; its belief sets the **fast** bottom layer's initial-state prior through the
link map (Eq. 42, ``prior = link · s^{[l+1]}``). This demonstrates the defining hierarchical
mechanism — a slow high-level state contextualizing fast low-level dynamics over nested time
scales. (The book numbers no worked example here; this is a constructed, verified illustration.)
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    HierarchicalPOMDP,
    POMDPModel,
    get_logger,
    simulate_hierarchical_agent,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.unified import plot_hierarchical_timescales


LOG = get_logger("ch10.ex8")


def build_model() -> HierarchicalPOMDP:
    # Slow top layer: two regimes, fairly reliable self-observation.
    """Compute a chapter-local helper quantity for the orchestrated example."""
    top = POMDPModel(A=np.array([[0.9, 0.1], [0.1, 0.9]]), D=np.array([0.5, 0.5]))
    # Fast bottom layer: three states, identity likelihood (observes its own state).
    bot = POMDPModel(A=np.eye(3), D=np.full(3, 1 / 3))
    # Link: regime 0 → bottom state 0, regime 1 → bottom state 2 (top-down contextualization).
    link = np.array([[0.9, 0.05], [0.05, 0.05], [0.05, 0.9]])
    link = link / link.sum(axis=0)
    return HierarchicalPOMDP(layers=[bot, top], link=[link])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--regime", type=int, default=1, choices=[0, 1], help="true top-layer regime")
    p.add_argument("--macro", type=int, default=6)
    p.add_argument("--inner", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    model = build_model()
    res = simulate_hierarchical_agent(model, true_top=args.regime, n_macro=args.macro,
                                      inner_steps=args.inner, rng=np.random.default_rng(args.seed))
    LOG.info("true regime=%d | final top belief=%s | final bottom prior=%s",
             args.regime, [round(x, 3) for x in res.top_belief[-1]],
             [round(x, 3) for x in res.bottom_priors[-1]])

    fig = plot_hierarchical_timescales(res)

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_10")
        save_or_show(fig, out / "example_10_8_hierarchical.png")
        LOG.info("Saved to %s", out / "example_10_8_hierarchical.png")
    else:
        save_or_show(fig, None)
        plt.show()


if __name__ == "__main__":
    main()
