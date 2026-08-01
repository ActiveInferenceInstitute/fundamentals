"""§4.2 — Intuition for the G-form of variational free energy (Fig. 4.2.3).

Run::

    python chapters/chapter_04/visualize_vfe_intuition.py [--save]

The G-form ``F = ∫ q(x) log[q(x)/p(x,y)] dx`` scores how well ``q(x)`` matches the
*unnormalized* generative model ``p(x,y)``. Because the true posterior ``p(x|y)``
and the generative model ``p(x,y)`` have the same shape (they differ only by the
constant ``Z = p(y)``), driving ``q`` toward ``p(x,y)`` also drives it toward the
posterior — and the gap being minimized is the normalizer ``−log p(y)``. This
figure overlays the posterior, the (scaled-down) generative model, and a fan of
candidate Gaussian ``q(x)`` densities, reproducing Figure 4.2.3.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import save_or_show
from active_inference.visualizations.style import COLORS

LOG = get_logger("ch4.vfe_intuition")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(0.5, 4.5, 600)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)

    # Unnormalized generative model p(x, y) on the grid.
    joint = np.exp(model.log_likelihood(args.y, x_grid) + model.log_prior(x_grid))

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.fill_between(x_grid, exact.posterior, color=COLORS["prior"], alpha=0.8,
                    label="p(x | y)")
    ax.plot(x_grid, joint, color="black", ls="--", lw=2.5, label="p(x, y)")

    rng = np.random.default_rng(args.seed)
    mean, var = exact.posterior_mean, exact.posterior_variance
    for k in range(8):
        mu = mean + rng.normal(0, 0.12)
        v = var * (1.0 + rng.normal(0, 0.18))
        q = GaussianBelief(mu, max(v, 1e-3))
        ax.plot(x_grid, q.pdf(x_grid), color=COLORS["likelihood"], lw=0.9, alpha=0.7,
                label="q(x)" if k == 0 else None)

    ax.set_xlabel("hidden state x")
    ax.set_ylabel("density")
    ax.set_title("Fig. 4.2.3 · G-form intuition: q(x), p(x,y), and the posterior")
    ax.legend(loc="upper right", fontsize=10)
    LOG.info("posterior N(%.3f, %.3f); peak p(x,y)=%.3f (scaled by Z=p(y))",
             mean, var, joint.max())

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        save_or_show(fig, out / "visualize_vfe_intuition.png")
        LOG.info("Saved to %s", out)
    else:
        plt.show()


if __name__ == "__main__":
    main()