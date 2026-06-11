"""Small categorical factor-graph helpers used by extras and Chapter 12 orchestrators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _finite_array(name: str, value: np.ndarray | list[float]) -> np.ndarray:
    """Validate and return a finite numeric array."""
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def normalize_message(message: np.ndarray | list[float]) -> np.ndarray:
    """Normalize a non-negative one-dimensional categorical message.

    The helper rejects empty, negative, non-finite, or zero-mass messages before
    returning a unit-sum vector.
    """
    msg = _finite_array("message", message)
    if msg.ndim != 1:
        raise ValueError("message must be one-dimensional")
    if np.any(msg < 0.0):
        raise ValueError("message must be non-negative")
    total = float(np.sum(msg))
    if total <= 0.0:
        raise ValueError("message must have positive mass")
    return msg / total


def categorical_factor_message(
    factor: np.ndarray | list[float],
    incoming: list[np.ndarray] | tuple[np.ndarray, ...],
    target_axis: int,
) -> np.ndarray:
    """Compute a sum-product message from a categorical factor to one variable.

    ``incoming`` must provide one message for every factor axis except
    ``target_axis``. Axes are multiplied by the corresponding incoming message
    and then all non-target axes are summed out.
    """
    fac = _finite_array("factor", factor)
    axis = int(target_axis)
    if axis < 0:
        axis += fac.ndim
    if axis < 0 or axis >= fac.ndim:
        raise ValueError("target_axis is out of bounds")
    if len(incoming) != fac.ndim - 1:
        raise ValueError("incoming must have one message for every non-target axis")
    work = fac.copy()
    incoming_i = 0
    for dim in range(fac.ndim):
        if dim == axis:
            continue
        msg = normalize_message(incoming[incoming_i])
        if msg.shape[0] != fac.shape[dim]:
            raise ValueError("incoming message length does not match factor axis")
        shape = [1] * fac.ndim
        shape[dim] = msg.shape[0]
        work = work * msg.reshape(shape)
        incoming_i += 1
    summed = np.sum(work, axis=tuple(dim for dim in range(fac.ndim) if dim != axis))
    return normalize_message(summed)


def sum_product_chain(
    prior: np.ndarray | list[float],
    transitions: np.ndarray | list[float],
    likelihoods: np.ndarray | list[float],
) -> np.ndarray:
    """Run normalized forward messages for a categorical state-space chain.

    Parameters
    ----------
    prior : array, shape (S,)
        Initial state message.
    transitions : array, shape (T-1, S, S) or (S, S)
        Column-stochastic transition matrices ``P(s_t | s_{t-1})``.
    likelihoods : array, shape (T, S)
        Per-time likelihood messages over states.
    """
    belief = normalize_message(prior)
    like = _finite_array("likelihoods", likelihoods)
    if like.ndim != 2:
        raise ValueError("likelihoods must have shape (T, S)")
    if like.shape[1] != belief.size:
        raise ValueError("likelihood state dimension must match prior")
    trans = _finite_array("transitions", transitions)
    if trans.ndim == 2:
        trans = np.repeat(trans[None, :, :], max(0, like.shape[0] - 1), axis=0)
    if trans.ndim != 3 or trans.shape[1:] != (belief.size, belief.size):
        raise ValueError("transitions must have shape (T-1, S, S)")
    if trans.shape[0] != max(0, like.shape[0] - 1):
        raise ValueError("transition count must be T-1")
    out = []
    for t in range(like.shape[0]):
        if t > 0:
            belief = normalize_message(trans[t - 1] @ belief)
        belief = normalize_message(belief * normalize_message(like[t]))
        out.append(belief)
    return np.asarray(out)


def variational_message_update(
    log_factor: np.ndarray | list[float],
    incoming_expectations: list[np.ndarray] | tuple[np.ndarray, ...],
    target_axis: int,
) -> np.ndarray:
    """Compute a categorical VMP-style softmax update for one variable.

    Incoming expectations weight every non-target axis before marginalization,
    and the returned categorical vector is normalized through
    :func:`normalize_message`.
    """
    log_fac = _finite_array("log_factor", log_factor)
    axis = int(target_axis)
    if axis < 0:
        axis += log_fac.ndim
    if axis < 0 or axis >= log_fac.ndim:
        raise ValueError("target_axis is out of bounds")
    if len(incoming_expectations) != log_fac.ndim - 1:
        raise ValueError("incoming_expectations must cover non-target axes")
    expected = log_fac.copy()
    incoming_i = 0
    for dim in range(log_fac.ndim):
        if dim == axis:
            continue
        msg = normalize_message(incoming_expectations[incoming_i])
        if msg.shape[0] != log_fac.shape[dim]:
            raise ValueError("incoming expectation length does not match factor axis")
        shape = [1] * log_fac.ndim
        shape[dim] = msg.shape[0]
        expected = expected * msg.reshape(shape)
        incoming_i += 1
    logits = np.sum(expected, axis=tuple(dim for dim in range(log_fac.ndim) if dim != axis))
    logits = logits - float(np.max(logits))
    probs = np.exp(logits)
    return normalize_message(probs)


@dataclass(frozen=True)
class FactorGraphChain:
    """A categorical state-space chain represented as factor-graph messages."""

    prior: np.ndarray
    transition: np.ndarray
    likelihoods: np.ndarray

    def __post_init__(self) -> None:
        """Validate factor-graph arrays after dataclass initialization."""
        prior = normalize_message(self.prior)
        transition = _finite_array("transition", self.transition)
        likelihoods = _finite_array("likelihoods", self.likelihoods)
        if transition.ndim != 2 or transition.shape != (prior.size, prior.size):
            raise ValueError("transition must have shape (S, S)")
        if likelihoods.ndim != 2 or likelihoods.shape[1] != prior.size:
            raise ValueError("likelihoods must have shape (T, S)")
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "transition", transition)
        object.__setattr__(self, "likelihoods", likelihoods)


def backward_smoothing_messages(
    transitions: np.ndarray | list[float],
    likelihoods: np.ndarray | list[float],
) -> np.ndarray:
    """Return normalized backward messages for a categorical chain."""
    like = _finite_array("likelihoods", likelihoods)
    if like.ndim != 2:
        raise ValueError("likelihoods must have shape (T, S)")
    trans = _finite_array("transitions", transitions)
    if trans.ndim != 2 or trans.shape != (like.shape[1], like.shape[1]):
        raise ValueError("transitions must have shape (S, S)")
    messages = np.ones_like(like, dtype=float)
    messages[-1] = normalize_message(messages[-1])
    for t in range(like.shape[0] - 2, -1, -1):
        future = normalize_message(like[t + 1] * messages[t + 1])
        messages[t] = normalize_message(trans.T @ future)
    return messages


def forward_backward_smoothing(
    prior: np.ndarray | list[float],
    transitions: np.ndarray | list[float],
    likelihoods: np.ndarray | list[float],
) -> np.ndarray:
    """Combine forward and backward messages into smoothed state beliefs."""
    forward = sum_product_chain(prior, transitions, likelihoods)
    backward = backward_smoothing_messages(transitions, likelihoods)
    return np.asarray([normalize_message(f * b) for f, b in zip(forward, backward)])


def hybrid_model_bridge(
    continuous_state: np.ndarray | list[float],
    discrete_belief: np.ndarray | list[float],
) -> np.ndarray:
    """Return a joint hybrid belief from continuous features and a categorical belief."""
    x = _finite_array("continuous_state", continuous_state)
    if x.ndim != 1:
        raise ValueError("continuous_state must be one-dimensional")
    belief = normalize_message(discrete_belief)
    return np.outer(belief, x)


def marginal_message_passing(
    factors: list[np.ndarray] | tuple[np.ndarray, ...],
    incoming: np.ndarray | list[float],
) -> np.ndarray:
    """Repeatedly pass a categorical marginal through a chain of pairwise factors."""
    message = normalize_message(incoming)
    out = [message]
    for factor in factors:
        fac = _finite_array("factor", factor)
        if fac.ndim != 2 or fac.shape[1] != message.size:
            raise ValueError("each factor must have shape (S_next, S_current)")
        message = normalize_message(fac @ message)
        out.append(message)
    return np.asarray(out)


def active_inference_factor_messages(
    likelihood: np.ndarray | list[float],
    transition: np.ndarray | list[float],
    prior: np.ndarray | list[float],
) -> dict[str, np.ndarray]:
    """Return core messages for a one-step discrete active-inference factor graph."""
    prior_msg = normalize_message(prior)
    prediction = categorical_factor_message(transition, [prior_msg], target_axis=0)
    observation = categorical_factor_message(likelihood, [prediction], target_axis=0)
    posterior = normalize_message(prediction * categorical_factor_message(likelihood, [observation], target_axis=1))
    return {
        "prior": prior_msg,
        "prediction": prediction,
        "observation": observation,
        "posterior": posterior,
    }


def learning_attention_message(
    prediction_error: np.ndarray | list[float],
    log_precision: float,
) -> np.ndarray:
    """Weight continuous prediction errors by an exponentiated precision message."""
    errors = _finite_array("prediction_error", prediction_error)
    precision = float(np.exp(float(log_precision)))
    if not np.isfinite(precision):
        raise ValueError("log_precision must be finite")
    return precision * errors


__all__ = [
    "FactorGraphChain",
    "active_inference_factor_messages",
    "backward_smoothing_messages",
    "normalize_message",
    "categorical_factor_message",
    "forward_backward_smoothing",
    "hybrid_model_bridge",
    "learning_attention_message",
    "marginal_message_passing",
    "sum_product_chain",
    "variational_message_update",
]
