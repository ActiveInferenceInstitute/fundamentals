"""Registry-driven application demos under the repo-root ``demo/`` folder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import matplotlib.pyplot as plt

from .demo_workflows import build_bicycle_demo, build_drone_flight_demo, build_eye_saccades_demo
from .orchestrator_workflows import WorkflowResult
from .utils import default_demo_figure_dir, ensure_dir, save_demo_data


@dataclass(frozen=True)
class DemoTopicSpec:
    """Metadata contract for one application demo folder."""

    slug: str
    title: str
    summary: str
    source_apis: tuple[str, ...]
    chapters: tuple[int, ...]


_BUILDERS: dict[str, Callable[..., WorkflowResult]] = {
    "eye_saccades": build_eye_saccades_demo,
    "bicycle": build_bicycle_demo,
    "drone_flight": build_drone_flight_demo,
}


DEMO_TOPICS: tuple[DemoTopicSpec, ...] = (
    DemoTopicSpec(
        "eye_saccades",
        "Eye Saccades",
        "Discrete expected free energy chooses a sequence of fixations on a retinotopic grid.",
        (
            "active_inference.estimators.pomdp.make_gridworld",
            "active_inference.estimators.pomdp.simulate_pomdp_agent",
            "active_inference.estimators.pomdp.evaluate_policy",
        ),
        (9,),
    ),
    DemoTopicSpec(
        "bicycle",
        "Riding a Bicycle",
        "Multivariate active inference stabilizes a 2-D balance state; fault compensation after a wobble.",
        (
            "active_inference.orchestrator_workflows.build_multivariate_active_agent_env",
            "active_inference.estimators.active_inference.simulate_multivariate_active_inference",
            "active_inference.estimators.applications.simulate_fault_tolerant_control",
        ),
        (7, 13),
    ),
    DemoTopicSpec(
        "drone_flight",
        "Drone Flight",
        "Discrete waypoint lattice planning, smooth executed path, and LGS fusion of noisy position reads.",
        (
            "active_inference.estimators.pomdp.simulate_pomdp_agent",
            "active_inference.estimators.applications.simulate_robot_navigation",
            "active_inference.core.lgs.LinearGaussianSystem.posterior_batch",
        ),
        (3, 9, 13),
    ),
)


def demo_topic_slugs() -> tuple[str, ...]:
    """Return registered demo topic slugs in registry order."""
    return tuple(spec.slug for spec in DEMO_TOPICS)


def demo_topic_spec(slug: str) -> DemoTopicSpec:
    """Return registry metadata for one application demo topic identified by ``slug``."""
    for spec in DEMO_TOPICS:
        if spec.slug == slug:
            return spec
    raise KeyError(f"unknown demo topic: {slug!r}")


def build_demo(slug: str, **kwargs: object) -> WorkflowResult:
    """Build one registered application demo workflow and return figures plus arrays."""
    builder = _BUILDERS.get(slug)
    if builder is None:
        raise KeyError(f"no builder registered for demo topic: {slug!r}")
    return builder(**kwargs)


def demo_artifact_path(slug: str, stem: str, suffix: str) -> Path:
    """Return the output figure path for one demo artifact."""
    return ensure_dir(default_demo_figure_dir(slug)) / f"{stem}.{suffix}"


def _parse_demo_args(argv: Sequence[str] | None, description: str) -> argparse.Namespace:
    """Parse common demo CLI flags."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--save", action="store_true", help="Write PNG and raw-data sidecars.")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for stochastic demos.")
    return parser.parse_args(list(argv) if argv is not None else None)


def _builder_accepts_seed(slug: str) -> bool:
    """Return whether the demo's builder is stochastic and accepts ``seed``.

    Introspect the registered builder's signature rather than maintaining a
    hard-coded slug set, so a future stochastic demo is picked up automatically.
    """
    builder = _BUILDERS.get(slug)
    if builder is None:
        return False
    import inspect
    return "seed" in inspect.signature(builder).parameters


def main_visualize(slug: str, argv: Sequence[str] | None = None) -> int:
    """Run the static visualization CLI for one demo topic."""
    args = _parse_demo_args(argv, f"Visualize demo topic {slug}.")
    spec = demo_topic_spec(slug)
    kwargs: dict[str, object] = {}
    if args.seed is not None and _builder_accepts_seed(slug):
        kwargs["seed"] = args.seed
    result = build_demo(slug, **kwargs)
    stem = f"visualize_{slug}"
    figure_key = next(iter(result.figures))
    fig = result.figures[figure_key]
    if args.save:
        figure = demo_artifact_path(slug, stem, "png")
        fig.savefig(figure, dpi=150)
        save_demo_data(
            slug,
            stem,
            result.arrays,
            {
                "script": f"{stem}.py",
                "kind": "visualize",
                "title": spec.title,
                "source_apis": list(spec.source_apis),
                **result.metadata,
            },
            figures=[figure],
        )
        plt.close(fig)
    else:
        plt.show()
    return 0


__all__ = [
    "DEMO_TOPICS",
    "DemoTopicSpec",
    "build_demo",
    "demo_artifact_path",
    "demo_topic_slugs",
    "demo_topic_spec",
    "main_visualize",
]
