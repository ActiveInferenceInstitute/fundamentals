"""Tests for Chapter 4 variational visualization helpers."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from active_inference import (
    GaussianBelief,
    GridBayesianInference,
    LinearGaussianModel,
    make_grid,
    variational_free_energy,
)
from active_inference.visualizations.variational import (
    plot_density_evolution,
    plot_surprisal_relationship,
    plot_vfe_contour,
    plot_vfe_decomposition,
    vfe_surface,
)


def _model_grid() -> tuple[LinearGaussianModel, np.ndarray]:
    return (
        LinearGaussianModel(beta0=3.0, beta1=2.0, sigma2_y=0.25, m_x=4.0, s2_x=0.25),
        make_grid(-2.0, 7.0, 401),
    )


def test_vfe_surface_returns_finite_meshes() -> None:
    model, grid = _model_grid()
    mu, var, free_energy = vfe_surface(
        model,
        7.0,
        grid,
        mu_lo=1.5,
        mu_hi=3.0,
        var_lo=0.04,
        var_hi=0.3,
        n_mu=8,
        n_var=7,
    )
    assert mu.shape == var.shape == free_energy.shape == (7, 8)
    assert np.all(np.isfinite(free_energy))


def test_plot_vfe_contour_returns_figure_with_path_and_truth() -> None:
    model, grid = _model_grid()
    fig = plot_vfe_contour(
        model,
        7.0,
        grid,
        mu_lo=1.5,
        mu_hi=3.0,
        var_lo=0.04,
        var_hi=0.3,
        n_mu=8,
        n_var=7,
        path_mus=[4.0, 3.0, 2.4],
        path_vars=[0.25, 0.1, 0.05],
        truth=(2.4, 0.05),
    )
    assert len(fig.axes) >= 2
    assert fig.axes[0].get_xlabel()
    plt.close(fig)


def test_plot_density_evolution_draws_beliefs_and_posterior() -> None:
    model, grid = _model_grid()
    posterior = GridBayesianInference(model, grid).infer(7.0).posterior
    beliefs = [GaussianBelief(4.0, 0.25), GaussianBelief(2.4, 0.05)]
    fig = plot_density_evolution(grid, beliefs, posterior=posterior)
    assert len(fig.axes[0].lines) == 3
    assert fig.axes[0].get_ylabel() == "density"
    plt.close(fig)


def test_plot_vfe_decomposition_draws_three_panels() -> None:
    model, grid = _model_grid()
    components = [
        variational_free_energy(GaussianBelief(4.0, 0.25), model, 7.0, grid),
        variational_free_energy(GaussianBelief(3.0, 0.1), model, 7.0, grid),
        variational_free_energy(GaussianBelief(2.4, 0.05), model, 7.0, grid),
    ]
    fig = plot_vfe_decomposition(components)
    assert len(fig.axes) == 3
    assert all(ax.lines for ax in fig.axes)
    plt.close(fig)


def test_plot_surprisal_relationship_returns_three_panels() -> None:
    fig = plot_surprisal_relationship(n=40)
    assert len(fig.axes) == 3
    assert all(ax.get_xlabel() for ax in fig.axes)
    plt.close(fig)
