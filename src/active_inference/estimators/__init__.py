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
from .continuous_learning import (
    LearningAttentionResult,
    simulate_learning_attention,
)
from .generalized_filtering import (
    GeneralizedVectorFilterResult,
    generalized_measurements_from_series,
    generalized_vector_filter,
)
from .active_inference import (
    MultivariateActiveInferenceResult,
    simulate_multivariate_active_inference,
)
from .applications import (
    FaultTolerantControlResult,
    NavigationResult,
    RoboticsTheoryResult,
    SocialInferenceResult,
    robotics_theory_landscape,
    simulate_fault_tolerant_control,
    simulate_robot_navigation,
    simulate_social_inference,
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
from .pomdp_extensions import (
    ParameterForgettingResult,
    PathPolicyResult,
    PreferenceHabitLearningResult,
    StatePreferenceResult,
    StructureLearningResult,
    make_line_world,
    simulate_parameter_forgetting,
    simulate_path_policy_computation,
    simulate_preference_habit_learning,
    simulate_sophisticated_planning,
    simulate_state_preference_schedule,
    simulate_structure_learning,
)

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
    "LearningAttentionResult",
    "simulate_learning_attention",
    "generalized_measurements_from_series",
    "generalized_vector_filter",
    "GeneralizedVectorFilterResult",
    "simulate_multivariate_active_inference",
    "MultivariateActiveInferenceResult",
    "NavigationResult",
    "FaultTolerantControlResult",
    "RoboticsTheoryResult",
    "SocialInferenceResult",
    "robotics_theory_landscape",
    "simulate_fault_tolerant_control",
    "simulate_robot_navigation",
    "simulate_social_inference",
    "ParameterForgettingResult",
    "PathPolicyResult",
    "PreferenceHabitLearningResult",
    "StatePreferenceResult",
    "StructureLearningResult",
    "make_line_world",
    "simulate_parameter_forgetting",
    "simulate_path_policy_computation",
    "simulate_preference_habit_learning",
    "simulate_sophisticated_planning",
    "simulate_state_preference_schedule",
    "simulate_structure_learning",
]
