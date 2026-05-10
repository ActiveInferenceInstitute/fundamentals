"""Active Inference Fundamentals — Python companion to Namjoshi (MIT Press).

The package mirrors the conceptual structure of the book:

    core         — generative process, generative model, exact Bayesian inference
    estimators   — MLE / MAP via analytic and gradient-descent updates
    visualizations — reusable plotting helpers for static and interactive figures
    utils        — grids, logging, reproducibility helpers

The chapter orchestrators in ``chapters/`` are intentionally thin:
they wire configurable components from this package together,
producing the figures and numerical results discussed in the text.
"""

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - metadata lookup is environment dependent
    __version__ = version("active-inference-fundamentals")
except PackageNotFoundError:  # editable / source checkouts
    __version__ = "0.1.0"

from .core.compose import (
    Pipeline,
    RunningPosteriorStats,
    running_stats,
)
from .core.diagnostics import (
    CalibrationCurve,
    PosteriorPredictiveCheck,
    calibration_curve,
    coverage_from_intervals,
    crps_gaussian,
    effective_sample_size,
    gaussian_entropy_mvn,
    gaussian_entropy_univariate,
    gaussian_kl_mvn,
    gaussian_kl_univariate,
    grid_entropy,
    grid_kl_divergence,
    log_score_gaussian,
    logsumexp,
    normal_ci,
    posterior_predictive_check,
    standardize,
)
from .core.distributions import (
    diagonal_cov,
    dirac_like_pdf,
    gaussian_log_pdf,
    gaussian_pdf,
    isotropic_cov,
    mahalanobis_squared,
    mvn_log_pdf,
    mvn_pdf,
    mvn_sample,
    uniform_pdf,
)
from .core.generative_model import (
    GenerativeModel,
    LinearGaussianModel,
    LinearGaussianMVModel,
)
from .core.generative_process import (
    GenerativeProcess,
    LinearGaussianMVProcess,
    LinearGaussianProcess,
)
from .core.inference import GridBayesianInference, InferenceResult
from .core.lgs import LGSPosterior, LinearGaussianSystem
from .core.posterior import (
    Posterior,
    has_credible_interval,
    has_mean_cov,
    posterior_mean,
    posterior_std,
    summarize_posterior,
)
from .core.types import assert_cov, assert_probabilities
from .core.validators import (
    require_1d,
    require_2d,
    require_design_matrix,
    require_finite_array,
    require_in_unit_interval,
    require_int_at_least,
    require_monotone,
    require_non_negative_scalar,
    require_positive_scalar,
    require_same_length,
)
from .estimators.em import (
    FactorAnalysisResult,
    factor_analysis_e_step,
    factor_analysis_m_step,
    fit_factor_analysis,
)
from .estimators.gradient_descent import GradientDescentResult, gradient_descent
from .estimators.linear_regression import (
    BLRPosterior,
    BayesianLinearRegression,
    GDRegressionResult,
    add_intercept,
    gd_linear_regression,
    mle_linear_regression,
)
from .estimators.map import map_analytic_linear, map_loss
from .estimators.mle import mle_analytic_linear, mle_loss
from .utils.grids import make_grid
from .utils.logging import get_logger

# Menu subpackage is imported lazily inside ``run_menu`` so importing
# ``active_inference`` itself stays free of subprocess imports.


def run_menu(argv: list[str] | None = None) -> int:
    """Entry point for the chapter-runner text menu (see ``run.sh``).

    Equivalent to ``python -m active_inference.menu``. Returns a shell
    exit code.
    """
    from .menu import main

    return main(argv)


def run_web(argv: list[str] | None = None) -> int:
    """Entry point for the local web UI (see ``run.sh --web``).

    Equivalent to ``python -m active_inference.web``. Blocks until the
    server is interrupted; returns a shell exit code.
    """
    from .web import main

    return main(argv)

__all__ = [
    "__version__",
    "diagonal_cov",
    "dirac_like_pdf",
    "gaussian_log_pdf",
    "gaussian_pdf",
    "isotropic_cov",
    "mahalanobis_squared",
    "mvn_log_pdf",
    "mvn_pdf",
    "mvn_sample",
    "uniform_pdf",
    "GenerativeProcess",
    "LinearGaussianProcess",
    "LinearGaussianMVProcess",
    "GenerativeModel",
    "LinearGaussianModel",
    "LinearGaussianMVModel",
    "GridBayesianInference",
    "InferenceResult",
    "LinearGaussianSystem",
    "LGSPosterior",
    "BayesianLinearRegression",
    "BLRPosterior",
    "GDRegressionResult",
    "add_intercept",
    "gd_linear_regression",
    "mle_linear_regression",
    "FactorAnalysisResult",
    "fit_factor_analysis",
    "factor_analysis_e_step",
    "factor_analysis_m_step",
    "mle_analytic_linear",
    "mle_loss",
    "map_analytic_linear",
    "map_loss",
    "gradient_descent",
    "GradientDescentResult",
    "make_grid",
    "get_logger",
    "run_menu",
    "run_web",
    # Composition
    "Pipeline",
    "RunningPosteriorStats",
    "running_stats",
    # Diagnostics
    "CalibrationCurve",
    "PosteriorPredictiveCheck",
    "calibration_curve",
    "coverage_from_intervals",
    "crps_gaussian",
    "effective_sample_size",
    "gaussian_entropy_mvn",
    "gaussian_entropy_univariate",
    "gaussian_kl_mvn",
    "gaussian_kl_univariate",
    "grid_entropy",
    "grid_kl_divergence",
    "log_score_gaussian",
    "logsumexp",
    "normal_ci",
    "posterior_predictive_check",
    "standardize",
    # Posterior protocol
    "Posterior",
    "has_credible_interval",
    "has_mean_cov",
    "posterior_mean",
    "posterior_std",
    "summarize_posterior",
    # Type asserts
    "assert_cov",
    "assert_probabilities",
    # Validators
    "require_1d",
    "require_2d",
    "require_design_matrix",
    "require_finite_array",
    "require_in_unit_interval",
    "require_int_at_least",
    "require_monotone",
    "require_non_negative_scalar",
    "require_positive_scalar",
    "require_same_length",
]
