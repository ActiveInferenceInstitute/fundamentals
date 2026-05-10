"""Point-estimate alternatives to full Bayesian inference: MLE, MAP, gradient descent.

Also exports the multivariate-flavored regression and EM utilities used in
Chapter 3.
"""

from .em import (
    FactorAnalysisResult,
    factor_analysis_e_step,
    factor_analysis_m_step,
    fit_factor_analysis,
    incomplete_log_likelihood,
)
from .gradient_descent import gradient_descent, GradientDescentResult
from .linear_regression import (
    BLRPosterior,
    BayesianLinearRegression,
    GDRegressionResult,
    add_intercept,
    gd_linear_regression,
    mle_linear_regression,
    squared_loss,
    squared_loss_grad,
)
from .map import map_analytic_linear, map_grad_x, map_loss
from .mle import mle_analytic_linear, mle_grad_x, mle_loss

__all__ = [
    "mle_analytic_linear",
    "mle_loss",
    "mle_grad_x",
    "map_analytic_linear",
    "map_loss",
    "map_grad_x",
    "gradient_descent",
    "GradientDescentResult",
    "add_intercept",
    "mle_linear_regression",
    "squared_loss",
    "squared_loss_grad",
    "gd_linear_regression",
    "GDRegressionResult",
    "BayesianLinearRegression",
    "BLRPosterior",
    "fit_factor_analysis",
    "factor_analysis_e_step",
    "factor_analysis_m_step",
    "incomplete_log_likelihood",
    "FactorAnalysisResult",
]
