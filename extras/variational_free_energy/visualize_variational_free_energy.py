"""Extra topic: variational free-energy decompositions and bound gap."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from active_inference import default_figure_dir, ensure_dir, save_extra_data
from active_inference.orchestrator_workflows import build_variational_free_energy_visualization
from active_inference.visualizations import save_or_show


def parse_args() -> argparse.Namespace:
    """Parse command-line options for this extras orchestrator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--y", type=float, default=7.0)
    return parser.parse_args()


def main() -> None:
    """Render or display the variational free-energy topic figure."""
    args = parse_args()
    result = build_variational_free_energy_visualization(y=args.y)
    fig = result.figures["visualize_variational_free_energy"]
    if args.save:
        out = ensure_dir(default_figure_dir() / "extras" / "variational_free_energy")
        figure = save_or_show(fig, out / "visualize_variational_free_energy.png")
        save_extra_data(
            "variational_free_energy",
            "visualize_variational_free_energy",
            arrays=result.arrays,
            metadata=result.metadata,
            figures=[figure] if figure is not None else [],
        )
    else:
        plt.show()


if __name__ == "__main__":
    main()
