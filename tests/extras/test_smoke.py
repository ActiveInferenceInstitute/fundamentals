"""Headless smoke tests for extras topic orchestrators."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import numpy as np

from active_inference.extra_topics import build_topic_animation, extra_topic_spec, extra_topic_slugs
from active_inference.menu.runner import discover_extra_scripts

REPO_ROOT = Path(__file__).resolve().parents[2]
EXTRAS_ROOT = REPO_ROOT / "extras"
TOPICS = list(extra_topic_slugs())
SCRIPT_CASES = [
    (topic, script.name)
    for topic in TOPICS
    for script in discover_extra_scripts(topic)
]


def _fresh_data_pairs(topic: str, stem: str, started_ns: int) -> list[tuple[Path, Path]]:
    """Return JSON/NPZ sidecars for ``topic`` written after ``started_ns``."""
    data_dir = REPO_ROOT / "output" / "data" / "extras" / topic
    pairs: list[tuple[Path, Path]] = []
    for json_path in data_dir.glob(f"{stem}.json"):
        npz_path = json_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        if json_path.stat().st_mtime_ns >= started_ns or npz_path.stat().st_mtime_ns >= started_ns:
            pairs.append((npz_path, json_path))
    return pairs


@pytest.mark.parametrize("topic,script_name", SCRIPT_CASES, ids=[f"{t}/{s}" for t, s in SCRIPT_CASES])
def test_extra_topic_script_runs_and_exports_raw_data(topic: str, script_name: str) -> None:
    """Every extras script runs headlessly and writes fresh NPZ+JSON sidecars."""
    script = EXTRAS_ROOT / topic / script_name
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    env["PYTHONWARNINGS"] = "error"
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    started_ns = time.time_ns()

    result = subprocess.run(
        [sys.executable, str(script), "--save"],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )

    assert result.returncode == 0, result.stderr
    assert _fresh_data_pairs(topic, script.stem, started_ns), (
        f"{topic}/{script_name} did not export fresh raw data"
    )


@pytest.mark.parametrize("topic", TOPICS, ids=TOPICS)
def test_declared_extra_topic_scripts_exist(topic: str) -> None:
    """Registry-declared extras modes have matching thin script wrappers."""
    spec = extra_topic_spec(topic)
    assert (EXTRAS_ROOT / topic / "README.md").is_file()
    assert (EXTRAS_ROOT / topic / f"visualize_{topic}.py").is_file()
    if spec.has_simulation:
        assert (EXTRAS_ROOT / topic / f"simulate_{topic}.py").is_file()
        assert (EXTRAS_ROOT / topic / f"interactive_{topic}.py").is_file()
    if spec.has_animation:
        assert (EXTRAS_ROOT / topic / f"animation_{topic}.py").is_file()


@pytest.mark.parametrize(
    "topic",
    [topic for topic in TOPICS if extra_topic_spec(topic).has_animation],
    ids=[topic for topic in TOPICS if extra_topic_spec(topic).has_animation],
)
def test_declared_extra_topic_animations_have_meaningful_raw_trajectories(topic: str) -> None:
    """Animation builders expose finite, non-static raw trajectories plus provenance metadata."""
    anim, raw, metadata = build_topic_animation(topic)
    try:
        assert metadata["trajectory_kind"]
        assert metadata["source_apis"] == list(extra_topic_spec(topic).source_apis)
        assert raw
        found_dynamic = False
        for values in raw.values():
            array = np.asarray(values)
            assert array.size > 0
            assert np.all(np.isfinite(array))
            if array.ndim >= 2 and array.shape[0] >= 2:
                found_dynamic = found_dynamic or not np.allclose(array[0], array[-1])
        assert found_dynamic
    finally:
        anim._draw_was_started = True
        import matplotlib.pyplot as plt

        plt.close(anim._fig)
