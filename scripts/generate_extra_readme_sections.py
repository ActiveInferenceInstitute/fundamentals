#!/usr/bin/env python3
"""Generate only the Scripts/Outputs sections of the 58 admitted extras READMEs."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_root in (REPO_ROOT, REPO_ROOT / "src"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from active_inference.extra_topics import extra_topic_spec  # noqa: E402
from scripts.validate_book_topic_coverage import _expected_artifacts, _expected_scripts  # noqa: E402


# Explicit section ownership, not every registry entry: bespoke READMEs stay authored.
OWNED_TOPICS = (
    "active_generalized_filtering",
    "backward_smoothing",
    "bayes_equation",
    "bayesian_linear_regression",
    "bayesian_mechanics_bridge",
    "belief_propagation",
    "cavi",
    "dirichlet_learning",
    "discrete_belief_filtering",
    "discrete_vfe",
    "enthalpy",
    "entropy",
    "ergodic_density",
    "expectation_maximization",
    "expected_free_energy",
    "exploration_exploitation",
    "factor_graphs",
    "factorial_depth",
    "fep_entropy_bounds",
    "free_energy_variants",
    "generalized_coordinates",
    "generalized_filtering",
    "generative_process_model",
    "gradient_descent",
    "gridworld_control",
    "hidden_state_estimation",
    "hierarchical_depth",
    "hierarchical_message_passing",
    "hierarchical_predictive_coding",
    "hybrid_generative_models",
    "inductive_planning",
    "inverse_problem",
    "kl_divergence",
    "learning_attention",
    "linear_gaussian_systems",
    "linear_regression",
    "mean_field_variational_inference",
    "mle_map",
    "model_comparison",
    "model_representation",
    "multiple_samples",
    "multivariate_gaussians",
    "order_and_surprisal",
    "parameter_uncertainty",
    "policy_precision_habits",
    "pomdp_arrays",
    "precision_weighting",
    "predictive_coding",
    "robotics_navigation",
    "social_robotics",
    "sophisticated_inference",
    "state_preferences",
    "structure_learning",
    "surprisal_evidence",
    "temperature",
    "tree_policy_search",
    "variational_free_energy",
    "variational_message_passing",
)


def render_sections(slug: str) -> dict[bytes, bytes]:
    """Use the coverage validator's registry-derived script/artifact convention."""
    spec = extra_topic_spec(slug)
    modes = {"simulation": spec.has_simulation, "animation": spec.has_animation}
    scripts = _expected_scripts(slug, **modes)
    descriptions = {
        "visualize": "Static visualization using shared library APIs.",
        "simulate": "Parameter-sweep simulation using shared library APIs.",
        "interactive": "Host GUI sliders; no saved media or data sidecars.",
        "animation": "Animated trajectory using shared library APIs.",
    }
    script_lines = ["## Scripts", ""]
    for name in scripts:
        mode = name.split("_", 1)[0]
        script_lines.append(f"- `{name}` - {descriptions[mode]}")
    output_lines = [
        "## Outputs", "",
        "Run non-interactive scripts with `--save`. The default artifact paths are",
        f"`output/figures/extras/{slug}/` and `output/data/extras/{slug}/`.",
        "These are declared outputs, not evidence that rendering has run.", "",
        "| Script | Figure | Raw data and manifest |",
        "|---|---|---|",
    ]
    for stem, suffix in _expected_artifacts(slug, **modes):
        output_lines.append(
            f"| `{stem}.py --save` | `{stem}{suffix}` | `{stem}.npz`, `{stem}.json` |"
        )
    if spec.has_simulation:
        output_lines.extend([
            "", "Interactive mode requires a host display. Its `--save` option closes",
            "the GUI without writing artifacts; use the commands below as shown.",
        ])
    output_lines.extend(["", "```bash"])
    for name in scripts:
        flag = "" if name.startswith("interactive_") else " --save"
        output_lines.append(f"uv run python extras/{slug}/{name}{flag}")
    output_lines.extend(["```", "", ""])
    return {
        b"Scripts": ("\n".join(script_lines) + "\n\n").encode("utf-8"),
        b"Outputs": "\n".join(output_lines).encode("utf-8"),
    }


def replace_sections(raw: bytes, sections: dict[bytes, bytes]) -> bytes:
    """Replace two H2 sections, preserving every byte outside their boundaries."""
    headings = []
    fence = None
    offset = 0
    for line in raw.splitlines(keepends=True):
        text = line.rstrip(b"\r\n")
        marker = re.match(rb"^ {0,3}(`{3,}|~{3,})(.*)$", text)
        if fence is not None:
            if marker and marker[1][:1] == fence[:1] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            fence = marker[1]
        else:
            heading = re.match(rb"^(#{1,2})[ \t]+(.+?)[ \t]*$", text)
            if heading:
                newline = b"\r\n" if line.endswith(b"\r\n") else b"\n"
                headings.append((offset, len(heading[1]), heading[2], newline))
        offset += len(line)
    ranges = []
    for name in sections:
        matches = [index for index, (_, level, title, _) in enumerate(headings) if level == 2 and title == name]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one H2 {name.decode()} section, found {len(matches)}")
        index = matches[0]
        start, _, _, newline = headings[index]
        end = headings[index + 1][0] if index + 1 < len(headings) else len(raw)
        ranges.append((start, end, sections[name].replace(b"\n", newline)))
    pieces = []
    offset = 0
    for start, end, replacement in sorted(ranges):
        pieces.extend((raw[offset:start], replacement))
        offset = end
    pieces.append(raw[offset:])
    return b"".join(pieces)


def plan_updates(extras_root: Path) -> list[tuple[Path, bytes, bytes]]:
    """Read and validate the complete owned set before any write is possible."""
    plans = []
    errors = []
    for slug in OWNED_TOPICS:
        directory = extras_root / slug
        path = directory / "README.md"
        try:
            if directory.is_symlink() or path.is_symlink():
                raise ValueError("owned topic and README must not be symlinks")
            spec = extra_topic_spec(slug)
            missing = [name for name in _expected_scripts(
                slug, simulation=spec.has_simulation, animation=spec.has_animation,
            ) if not (directory / name).is_file()]
            if missing:
                raise ValueError(f"missing declared wrappers: {', '.join(missing)}")
            before = path.read_bytes()
            plans.append((path, before, replace_sections(before, render_sections(slug))))
        except (OSError, ValueError) as error:
            errors.append(f"{path}: {error}")
    if errors:
        raise ValueError("\n".join(errors))
    return plans


def main(argv: list[str] | None = None) -> int:
    """Check without mutation, or explicitly write the complete admitted section set."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras-root", type=Path, default=REPO_ROOT / "extras")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        plans = plan_updates(args.extras_root)
        changed = [(path, before, after) for path, before, after in plans if before != after]
        if args.write:
            for path, before, _ in plans:
                if path.is_symlink() or path.parent.is_symlink() or path.read_bytes() != before:
                    raise ValueError(f"{path}: custody changed after planning; no writes admitted")
            for path, _, after in changed:
                path.write_bytes(after)
        for path, _, _ in changed:
            print(f"{'updated' if args.write else 'stale'}: {path}")
        print(f"{len(plans)} owned README(s); {len(changed)} {'updated' if args.write else 'stale'}")
        return int(args.check and bool(changed))
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
