"""Tests for ``core.variational`` — variational free energy and its forms.

The generative model is the book's running Example 4.1 (Chapter 4): a linear
Gaussian ``y = beta0 + beta1 x + noise`` with ``beta0=3, beta1=2, sigma2_y=0.25``
and a Gaussian prior ``N(4, 0.25)`` on ``x``. With ``y = 7`` the exact posterior is
``N(2.4, 0.05)`` (precision 20 = likelihood precision 16 + prior precision 4), and
the surprisal ``-log p(y)`` is the tight lower bound on VFE.
"""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.generative_model import LinearGaussianModel
from active_inference.core.inference import GridBayesianInference
from active_inference.core.variational import (
    GaussianBelief,
    VFEComponents,
    evaluate_q,
    free_energy_bound_gap,
    log_model_evidence,
    surprisal,
    variational_free_energy,
    vfe_c_form,
    vfe_d_form,
    vfe_e_form,
    vfe_g_form,
    vfe_map_form,
    vfe_mle_form,
)

Y_OBS = 7.0
POST_MEAN = 2.4
POST_VAR = 0.05


@pytest.fixture
def x_grid() -> np.ndarray:
    # Wide, fine grid so trapezoid integrals of q, prior and likelihood are tight.
    return np.linspace(-6.0, 12.0, 2001)


def make_model(**overrides) -> LinearGaussianModel:
    kwargs = dict(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25,
                  prior_kind="gaussian")
    kwargs.update(overrides)
    return LinearGaussianModel(**kwargs)


# ---------------------------------------------------------------------------
# GaussianBelief
# ---------------------------------------------------------------------------


class TestGaussianBelief:
    def test_pdf_normalizes(self, x_grid: np.ndarray) -> None:
        q = GaussianBelief(POST_MEAN, POST_VAR)
        assert np.trapezoid(q.pdf(x_grid), x_grid) == pytest.approx(1.0, abs=1e-4)

    def test_entropy_matches_closed_form(self) -> None:
        var = 0.3
        q = GaussianBelief(1.0, var)
        expected = 0.5 * np.log(2 * np.pi * np.e * var)
        assert q.entropy() == pytest.approx(expected, abs=1e-12)

    def test_std_is_sqrt_var(self) -> None:
        assert GaussianBelief(0.0, 0.04).std == pytest.approx(0.2)

    def test_logpdf_consistent_with_pdf(self) -> None:
        q = GaussianBelief(2.0, 0.5)
        xs = np.array([-1.0, 0.0, 2.0, 5.0])
        assert np.allclose(np.exp(q.logpdf(xs)), q.pdf(xs))

    def test_sample_is_reproducible(self) -> None:
        q = GaussianBelief(2.4, 0.05)
        a = q.sample(1000, rng=np.random.default_rng(0))
        b = q.sample(1000, rng=np.random.default_rng(0))
        assert np.array_equal(a, b)
        assert a.mean() == pytest.approx(2.4, abs=0.05)

    @pytest.mark.parametrize("bad_var", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_nonpositive_or_nonfinite_var(self, bad_var: float) -> None:
        with pytest.raises(ValueError):
            GaussianBelief(0.0, bad_var)

    @pytest.mark.parametrize("bad_mu", [np.nan, np.inf, -np.inf])
    def test_rejects_nonfinite_mu(self, bad_mu: float) -> None:
        with pytest.raises(ValueError):
            GaussianBelief(bad_mu, 1.0)


# ---------------------------------------------------------------------------
# evaluate_q
# ---------------------------------------------------------------------------


class TestEvaluateQ:
    def test_accepts_belief(self, x_grid: np.ndarray) -> None:
        vals = evaluate_q(GaussianBelief(2.0, 0.5), x_grid)
        assert np.trapezoid(vals, x_grid) == pytest.approx(1.0, abs=1e-6)

    def test_accepts_callable(self, x_grid: np.ndarray) -> None:
        q = GaussianBelief(2.0, 0.5)
        vals = evaluate_q(q.pdf, x_grid)
        assert np.trapezoid(vals, x_grid) == pytest.approx(1.0, abs=1e-6)

    def test_accepts_array(self, x_grid: np.ndarray) -> None:
        raw = GaussianBelief(2.0, 0.5).pdf(x_grid)
        vals = evaluate_q(raw, x_grid)
        assert np.trapezoid(vals, x_grid) == pytest.approx(1.0, abs=1e-6)

    def test_normalize_false_preserves_unnormalized(self, x_grid: np.ndarray) -> None:
        raw = 2.0 * GaussianBelief(2.0, 0.5).pdf(x_grid)
        vals = evaluate_q(raw, x_grid, normalize=False)
        assert np.trapezoid(vals, x_grid) == pytest.approx(2.0, abs=1e-4)

    def test_rejects_wrong_shape(self, x_grid: np.ndarray) -> None:
        with pytest.raises(ValueError):
            evaluate_q(np.ones(len(x_grid) - 1), x_grid)

    def test_rejects_negative_density(self, x_grid: np.ndarray) -> None:
        bad = np.full_like(x_grid, -1.0)
        with pytest.raises(ValueError):
            evaluate_q(bad, x_grid)


# ---------------------------------------------------------------------------
# variational_free_energy — forms agree, bound holds, tight at posterior
# ---------------------------------------------------------------------------


class TestVFEForms:
    def test_all_forms_agree(self, x_grid: np.ndarray) -> None:
        c = variational_free_energy(GaussianBelief(3.0, 0.2), make_model(),
                                    Y_OBS, x_grid)
        c.check(atol=1e-6)  # raises if any form disagrees
        assert c.g_form == pytest.approx(c.free_energy, abs=1e-6)
        assert c.d_form == pytest.approx(c.free_energy, abs=1e-6)
        assert c.c_form == pytest.approx(c.free_energy, abs=1e-6)
        assert c.e_form == pytest.approx(c.free_energy, abs=1e-6)

    def test_forms_agree_across_many_beliefs(self, x_grid: np.ndarray) -> None:
        model = make_model()
        for mu in (0.5, 1.5, 2.4, 4.0, 6.0):
            for var in (0.02, 0.1, 0.5, 1.5):
                c = variational_free_energy(GaussianBelief(mu, var), model,
                                            Y_OBS, x_grid)
                c.check(atol=1e-5)

    def test_decomposition_identities(self, x_grid: np.ndarray) -> None:
        c = variational_free_energy(GaussianBelief(3.0, 0.2), make_model(),
                                    Y_OBS, x_grid)
        assert c.c_form == pytest.approx(c.complexity - c.accuracy, abs=1e-12)
        assert c.e_form == pytest.approx(c.neg_entropy - c.energy, abs=1e-12)
        assert c.d_form == pytest.approx(c.divergence + c.surprisal, abs=1e-12)
        assert c.bound_gap == pytest.approx(c.divergence, abs=1e-6)

    def test_bound_holds_for_arbitrary_q(self, x_grid: np.ndarray) -> None:
        model = make_model()
        for mu in (-2.0, 0.0, 2.4, 5.0, 9.0):
            for var in (0.05, 0.3, 1.0):
                c = variational_free_energy(GaussianBelief(mu, var), model,
                                            Y_OBS, x_grid)
                # F >= -log p(y) for any q (Eq. 26).
                assert c.free_energy >= c.surprisal - 1e-6
                assert c.bound_gap >= -1e-6

    def test_bound_tight_at_posterior(self, x_grid: np.ndarray) -> None:
        model = make_model()
        # q = exact posterior N(2.4, 0.05): divergence -> 0, F -> -log p(y).
        c = variational_free_energy(GaussianBelief(POST_MEAN, POST_VAR), model,
                                    Y_OBS, x_grid)
        assert c.divergence == pytest.approx(0.0, abs=1e-3)
        assert c.bound_gap == pytest.approx(0.0, abs=1e-3)
        assert c.free_energy == pytest.approx(c.surprisal, abs=1e-3)

    def test_posterior_grid_makes_gap_vanish(self, x_grid: np.ndarray) -> None:
        model = make_model()
        exact = GridBayesianInference(model=model, x_grid=x_grid).infer(Y_OBS)
        # Feeding the exact grid posterior as q drives the divergence to ~0.
        c = variational_free_energy(np.asarray(exact.posterior), model, Y_OBS,
                                    x_grid, log_evidence=float(exact.log_evidence),
                                    posterior=np.asarray(exact.posterior))
        assert c.bound_gap == pytest.approx(0.0, abs=1e-9)

    def test_surprisal_matches_grid_evidence(self, x_grid: np.ndarray) -> None:
        model = make_model()
        c = variational_free_energy(GaussianBelief(3.0, 0.2), model, Y_OBS, x_grid)
        le = log_model_evidence(model, Y_OBS, x_grid)
        assert c.surprisal == pytest.approx(-le, abs=1e-9)
        assert c.log_evidence == pytest.approx(le, abs=1e-9)

    def test_precomputed_oracle_matches_internal(self, x_grid: np.ndarray) -> None:
        model = make_model()
        exact = GridBayesianInference(model=model, x_grid=x_grid).infer(Y_OBS)
        q = GaussianBelief(3.0, 0.2)
        c_auto = variational_free_energy(q, model, Y_OBS, x_grid)
        c_given = variational_free_energy(
            q, model, Y_OBS, x_grid,
            log_evidence=float(exact.log_evidence),
            posterior=np.asarray(exact.posterior),
        )
        assert c_auto.free_energy == pytest.approx(c_given.free_energy, abs=1e-12)
        assert c_auto.divergence == pytest.approx(c_given.divergence, abs=1e-12)


# ---------------------------------------------------------------------------
# Thin form wrappers
# ---------------------------------------------------------------------------


class TestFormWrappers:
    def test_wrappers_match_components(self, x_grid: np.ndarray) -> None:
        model = make_model()
        q = GaussianBelief(3.0, 0.2)
        c = variational_free_energy(q, model, Y_OBS, x_grid)

        assert vfe_g_form(q, model, Y_OBS, x_grid) == pytest.approx(c.free_energy)

        f_d, div, surp = vfe_d_form(q, model, Y_OBS, x_grid)
        assert (f_d, div, surp) == pytest.approx(
            (c.free_energy, c.divergence, c.surprisal))

        f_c, comp, acc = vfe_c_form(q, model, Y_OBS, x_grid)
        assert (f_c, comp, acc) == pytest.approx(
            (c.free_energy, c.complexity, c.accuracy))

        f_e, neg_h, energy = vfe_e_form(q, model, Y_OBS, x_grid)
        assert (f_e, neg_h, energy) == pytest.approx(
            (c.free_energy, c.neg_entropy, c.energy))

    def test_map_form_peaks_at_posterior_mode(self) -> None:
        model = make_model()
        grid = np.linspace(1.0, 4.0, 601)
        vals = np.array([vfe_map_form(model, Y_OBS, float(m)) for m in grid])
        # MAP objective log p(x,y) is maximized at the posterior mode = 2.4.
        assert grid[int(np.argmax(vals))] == pytest.approx(POST_MEAN, abs=2e-2)

    def test_mle_form_peaks_at_inverse(self) -> None:
        model = make_model()
        grid = np.linspace(0.0, 4.0, 801)
        vals = np.array([vfe_mle_form(model, Y_OBS, float(m)) for m in grid])
        # Likelihood alone peaks where beta0 + beta1 x = y -> x = (7-3)/2 = 2.
        assert grid[int(np.argmax(vals))] == pytest.approx(2.0, abs=2e-2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_surprisal_is_negative_log_evidence(self, x_grid: np.ndarray) -> None:
        model = make_model()
        assert surprisal(model, Y_OBS, x_grid) == pytest.approx(
            -log_model_evidence(model, Y_OBS, x_grid), abs=1e-12)

    def test_bound_gap_helper_equals_divergence(self, x_grid: np.ndarray) -> None:
        model = make_model()
        q = GaussianBelief(3.0, 0.2)
        gap = free_energy_bound_gap(q, model, Y_OBS, x_grid)
        c = variational_free_energy(q, model, Y_OBS, x_grid)
        assert gap == pytest.approx(c.divergence, abs=1e-9)
        assert gap >= -1e-9

    def test_check_passes_on_real_components(self, x_grid: np.ndarray) -> None:
        # The strengthened .check() (form agreement + KL >= 0) holds for real q.
        variational_free_energy(GaussianBelief(3.0, 0.2), make_model(),
                                Y_OBS, x_grid).check(atol=1e-6)

    def test_check_rejects_negative_divergence(self) -> None:
        # A fabricated component with KL < 0 violates the Gibbs inequality guard.
        bad = VFEComponents(
            free_energy=1.0, divergence=-0.5, surprisal=1.5, complexity=0.0,
            accuracy=-1.0, neg_entropy=0.0, energy=-1.0, log_evidence=-1.5,
        )
        with pytest.raises(AssertionError, match="Gibbs"):
            bad.check(atol=1e-6)

    def test_components_is_frozen(self, x_grid: np.ndarray) -> None:
        c = variational_free_energy(GaussianBelief(3.0, 0.2), make_model(),
                                    Y_OBS, x_grid)
        assert isinstance(c, VFEComponents)
        with pytest.raises(Exception):
            c.free_energy = 0.0  # type: ignore[misc]


class TestBoundedSupport:
    @pytest.mark.filterwarnings("error")
    def test_exact_posterior_ignores_zero_density_outside_prior_support(self):
        grid = np.linspace(-2.0, 3.0, 2001)
        model = LinearGaussianModel(
            prior_kind="uniform", uniform_low=0.0, uniform_high=1.0,
        )
        exact = GridBayesianInference(model, grid).infer(0.4)
        with np.errstate(invalid="raise"):
            components = variational_free_energy(exact.posterior, model, 0.4, grid)
        assert np.isfinite(components.free_energy)
        assert components.divergence == pytest.approx(0.0, abs=1e-10)
        assert components.free_energy == pytest.approx(-exact.log_evidence)
        components.check()

    @pytest.mark.filterwarnings("error")
    def test_positive_mass_outside_prior_support_retains_infinite_divergence(self):
        grid = np.linspace(-2.0, 3.0, 2001)
        model = LinearGaussianModel(
            prior_kind="uniform", uniform_low=0.0, uniform_high=1.0,
        )
        with np.errstate(invalid="raise"):
            components = variational_free_energy(np.ones_like(grid), model, 0.4, grid)
        assert np.isposinf(components.free_energy)
        assert np.isposinf(components.divergence)
        assert np.isposinf(components.complexity)
        assert np.isfinite(components.accuracy)
