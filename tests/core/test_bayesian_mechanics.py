"""Tests for Chapter 14 Bayesian-mechanics helpers."""

from __future__ import annotations

import numpy as np

from active_inference import (
    bayesian_mechanics_summary,
    blanket_coupling_matrix,
    entropy_vfe_bound_curve,
    phase1_fep_bridge,
    simulate_markov_blanket_flow,
    survival_probability,
    viability_indicator,
)
from active_inference.core.ergodic import ergodic_ou_trajectory


def test_bayesian_mechanics_summary_has_positive_bound_gap() -> None:
    trajectory = ergodic_ou_trajectory(n_steps=300)
    summary = bayesian_mechanics_summary(trajectory, bins=60, vfe_margin=0.7)
    assert summary.grid.shape == summary.density.shape
    assert np.all(np.isfinite(summary.density))
    assert summary.upper_bound == summary.entropy + summary.gap
    assert summary.gap == 0.7


def test_markov_blanket_flow_is_finite_and_coupled() -> None:
    flow = simulate_markov_blanket_flow(n_steps=80)
    assert flow.time.shape == flow.internal.shape == flow.blanket.shape == flow.external.shape
    assert np.all(np.isfinite(flow.internal))
    corr = blanket_coupling_matrix(flow)
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(corr, corr.T)
    np.testing.assert_allclose(np.diag(corr), np.ones(3))
    assert corr[0, 1] > corr[0, 2]


def test_viability_and_phase1_bridge_are_bounded() -> None:
    states = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
    indicator = viability_indicator(states, -1.0, 1.0)
    np.testing.assert_allclose(indicator, [0.0, 1.0, 1.0, 1.0, 0.0])
    assert survival_probability(states, -1.0, 1.0) == 0.6
    entropy, bound = entropy_vfe_bound_curve([0.5, 0.8], margin=0.3)
    np.testing.assert_allclose(bound - entropy, [0.3, 0.3])
    bridge = phase1_fep_bridge(0.5, 0.9, 0.8)
    np.testing.assert_allclose(bridge, [0.5, 0.9, 0.2])
