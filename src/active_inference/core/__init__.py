"""Core building blocks: distributions, generative process / model, exact inference, diagnostics, composition."""

from .compose import (
    Pipeline,
    RunningPosteriorStats,
    running_stats,
)
from .diagnostics import (
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
from .distributions import (
    diagonal_cov,
    dirac_like_pdf,
    gaussian_log_pdf,
    gaussian_pdf,
    isotropic_cov,
    mahalanobis_squared,
    mvn_log_pdf,
    mvn_pdf,
    mvn_sample,
    normalize_density,
    uniform_pdf,
)
from .generative_process import (
    GenerativeProcess,
    LinearGaussianMVProcess,
    LinearGaussianProcess,
)
from .generative_model import (
    GenerativeModel,
    LinearGaussianModel,
    LinearGaussianMVModel,
)
from .inference import GridBayesianInference, InferenceResult
from .lgs import LinearGaussianSystem, LGSPosterior
from .posterior import (
    Posterior,
    has_credible_interval,
    has_mean_cov,
    posterior_mean,
    posterior_std,
    summarize_posterior,
)
from .types import assert_cov, assert_probabilities
from .validators import (
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

__all__ = [
    "diagonal_cov",
    "dirac_like_pdf",
    "gaussian_log_pdf",
    "gaussian_pdf",
    "isotropic_cov",
    "mahalanobis_squared",
    "mvn_log_pdf",
    "mvn_pdf",
    "mvn_sample",
    "normalize_density",
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
    # Composition
    "Pipeline",
    "RunningPosteriorStats",
    "running_stats",
    # Type / shape helpers
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
    # Posterior protocol
    "Posterior",
    "has_credible_interval",
    "has_mean_cov",
    "posterior_mean",
    "posterior_std",
    "summarize_posterior",
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
]
