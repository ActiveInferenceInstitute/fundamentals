"""Diagnostic — reliability diagram for the Bayesian linear regression posterior.

Run::

    python chapters/chapter_03/visualize_calibration.py [--save] [--n-trials 200]

Repeats the BLR experiment ``--n-trials`` times with fresh data, builds
prediction intervals at a sweep of nominal credible levels, and plots the
empirical-vs-nominal coverage curve along with the expected calibration
error (ECE).
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.special import erfinv

from active_inference import (
    BayesianLinearRegression,
    LinearGaussianProcess,
    calibration_curve,
    get_logger,
)
from active_inference.utils.io import default_figure_dir, ensure_dir
from active_inference.visualizations import plot_calibration, save_or_show

LOG = get_logger("ch3.calibration")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=200)
    p.add_argument("--n-train", type=int, default=80)
    p.add_argument("--n-test", type=int, default=200)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # The data-generating process: y = β₀ + β₁ x + ε with σ²_y = 0.25.
    sigma2_y = 0.25
    process = LinearGaussianProcess(
        beta0=3.0, beta1=2.0, sigma2_y=sigma2_y, rng=rng,
    )

    # Run T trials; in each, fit BLR on N_train samples, evaluate on N_test.
    levels = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95])

    # We accumulate (truth, lower(level), upper(level)) per test point.
    truths_all: list[float] = []
    lowers_all: dict[float, list[float]] = {float(l): [] for l in levels}
    uppers_all: dict[float, list[float]] = {float(l): [] for l in levels}

    for trial in range(args.n_trials):
        x_train = rng.uniform(0.0, 5.0, size=args.n_train)
        y_train = np.array([process.sample(float(x), n=1)[0] for x in x_train])
        x_test = rng.uniform(0.0, 5.0, size=args.n_test)
        y_test = np.array([process.sample(float(x), n=1)[0] for x in x_test])

        blr = BayesianLinearRegression(
            prior_mean=np.zeros(2),
            prior_cov=np.eye(2) * 4.0,
            sigma2_y=sigma2_y,
        )
        post = blr.fit(x_train, y_train)
        mean_pred, var_pred = post.predictive(x_test, sigma2_y=sigma2_y)
        std_pred = np.sqrt(var_pred)

        truths_all.extend(y_test.tolist())
        for lvl in levels:
            half = float(np.sqrt(2.0) * erfinv(float(lvl))) * std_pred
            lowers_all[float(lvl)].extend((mean_pred - half).tolist())
            uppers_all[float(lvl)].extend((mean_pred + half).tolist())

    truths = np.asarray(truths_all)

    def lower_fn(level: float) -> np.ndarray:
        return np.asarray(lowers_all[float(level)])

    def upper_fn(level: float) -> np.ndarray:
        return np.asarray(uppers_all[float(level)])

    curve = calibration_curve(truths, lower_fn, upper_fn, levels)
    LOG.info("Calibration error (ECE) = %.4f", curve.calibration_error())
    for lvl, emp in zip(curve.nominal, curve.empirical):
        LOG.info("  nominal = %.2f → empirical = %.4f", lvl, emp)

    fig = plot_calibration(
        curve,
        title=f"BLR predictive calibration · {args.n_trials} trials × {args.n_test} test points",
    )

    if args.save:
        out = ensure_dir(default_figure_dir() / "chapter_03")
        save_or_show(fig, out / "diagnostic_calibration.png")
        LOG.info("Saved to %s", out)
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()
