"""Tests for Appendix D free-energy decomposition helpers."""

from __future__ import annotations

import pytest

from active_inference import (
    action_perception_divergence_form,
    constrained_bethe_free_energy_form,
    dynamic_vfe_decomposition,
    expected_free_energy_decomposition,
    free_energy_of_expected_future,
    static_vfe_decomposition,
)


def test_static_vfe_uses_energy_minus_entropy_sign() -> None:
    form = static_vfe_decomposition(energy=3.0, entropy=1.25, log_evidence=-2.0, kl_divergence=0.25)
    assert form.total == pytest.approx(1.75)
    assert form.terms["negative_entropy"] == pytest.approx(-1.25)
    assert form.terms["negative_log_evidence"] == pytest.approx(2.0)


def test_dynamic_vfe_and_efe_terms_have_expected_direction() -> None:
    dynamic = dynamic_vfe_decomposition(0.5, 0.75, 0.25)
    assert dynamic.total == pytest.approx(1.5)
    efe = expected_free_energy_decomposition(1.0, 0.4, information_gain=0.25, preference_value=0.1)
    assert efe.total == pytest.approx(1.05)
    assert efe.terms["information_gain"] < 0.0
    assert efe.terms["preference_value"] < 0.0


def test_part_iii_variant_forms_are_finite_and_penalize_constraints() -> None:
    feef = free_energy_of_expected_future(1.0, 0.5, 0.25)
    assert feef.total == pytest.approx(1.25)
    bethe = constrained_bethe_free_energy_form(2.0, 0.5, constraint_residual=0.5, penalty_weight=2.0)
    assert bethe.total == pytest.approx(2.0)
    divergence = action_perception_divergence_form(0.2, 0.3, coupling=0.1)
    assert divergence.total == pytest.approx(0.6)
