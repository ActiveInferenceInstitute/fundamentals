"""Animation — variational free energy descent (Chapter 4, bonus).

Run::

    python chapters/chapter_04/animation_vfe_descent.py [--save]

Renders a GIF: the left panel shows the variational density ``q(x)`` tightening
onto the exact posterior over the iterations of fixed-form VI, while the right
panel shows variational free energy falling toward the surprisal bound
``−log p(y)``. This is the dynamic version of Figures 4.2.2/4.6.1.
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    fixed_form_vi,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import animate_vfe_descent, save_animation

LOG = get_logger("ch4.anim_vfe")


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this executable entry point."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--y", type=float, default=7.0)
    p.add_argument("--frames", type=int, default=30)
    return p.parse_args()


def main() -> None:
    """Run the chapter orchestrator and render or display its outputs."""
    args = parse_args()
    x_grid = np.linspace(-6.0, 12.0, 1201)
    model = LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25,
                                m_x=4.0, s2_x=0.25)
    exact = GridBayesianInference(model=model, x_grid=x_grid).infer(args.y)

    ff = fixed_form_vi(model, args.y, x_grid, lr=5e-3, n_iter=1500)
    step = max(1, len(ff.mus) // args.frames)
    idx = list(range(0, len(ff.mus), step))
    # Build the belief/free-energy frames in lockstep from the same indices so
    # a non-divisible ``step`` can never silently misalign the two traces.
    beliefs = [GaussianBelief(ff.mus[i], ff.vars_[i]) for i in idx]
    fes = np.asarray([ff.free_energies[i] for i in idx], dtype=float)
    anim = animate_vfe_descent(x_grid, beliefs, fes, posterior=exact.posterior,
                               surprisal=-float(exact.log_evidence))
    LOG.info("built VFE-descent animation with %d frames", len(beliefs))

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_04")
        path = save_animation(anim, out / "animation_vfe_descent.gif", fps=10)
        LOG.info("Saved to %s", path)
    else:
        plt.show()


if __name__ == "__main__":
    main()
