"""Extra topic: differential entropy over Gaussian variance."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    default_figure_dir,
    ensure_dir,
    gaussian_entropy_univariate,
    save_extra_data,
)
from active_inference.extra_topics import extra_topic_spec
from active_inference.visualizations import COLORS, save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Render or display the entropy topic figure."""
    args = parse_args()
    variances = np.logspace(-3.0, 2.0, 160)
    entropies = np.array([gaussian_entropy_univariate(float(v)) for v in variances])
    zero_entropy_variance = 1.0 / (2.0 * np.pi * np.e)

    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
    ax.semilogx(variances, entropies, lw=2, label="H[N(mu, sigma2)]")
    ax.axhline(0.0, color=COLORS["data"], lw=1, ls="--", label="zero entropy")
    ax.axvline(zero_entropy_variance, color=COLORS["likelihood"], lw=1.5, ls=":",
               label=f"sigma2 = {zero_entropy_variance:.4f}")
    ax.fill_between(variances, entropies, 0.0, where=entropies < 0.0,
                    color=COLORS["likelihood"], alpha=0.18,
                    label="negative differential entropy")
    ax.set_xlabel("variance sigma2")
    ax.set_ylabel("differential entropy (nats)")
    ax.set_title("Differential entropy grows with variance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "entropy")
        figure = save_or_show(fig, out / "visualize_entropy.png")
        save_extra_data(
            "entropy",
            "visualize_entropy",
            arrays={
                "variance": variances,
                "entropy": entropies,
                "zero_entropy_variance": np.array([zero_entropy_variance]),
            },
            metadata={
                "script": "visualize_entropy.py",
                "source_apis": list(extra_topic_spec("entropy").source_apis),
            },
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
