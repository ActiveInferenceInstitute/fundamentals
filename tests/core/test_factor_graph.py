"""Tests for categorical factor-graph helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference import (
    FactorGraphChain,
    active_inference_factor_messages,
    backward_smoothing_messages,
    categorical_factor_message,
    forward_backward_smoothing,
    hybrid_model_bridge,
    learning_attention_message,
    marginal_message_passing,
    normalize_message,
    sum_product_chain,
    variational_message_update,
)


def test_normalize_message_returns_categorical_vector() -> None:
    """Message normalization preserves ratios and sums to one."""
    msg = normalize_message([2.0, 1.0, 1.0])
    assert msg.sum() == pytest.approx(1.0)
    assert msg.tolist() == pytest.approx([0.5, 0.25, 0.25])
    with pytest.raises(ValueError):
        normalize_message([0.0, 0.0])


def test_categorical_factor_message_sums_out_non_target_axes() -> None:
    """Factor-to-variable messages agree with direct marginalization."""
    factor = np.array([[0.9, 0.1], [0.2, 0.8]])
    incoming = [np.array([0.75, 0.25])]
    msg = categorical_factor_message(factor, incoming, target_axis=0)
    direct = normalize_message(factor @ normalize_message(incoming[0]))
    assert msg.tolist() == pytest.approx(direct.tolist())


def test_sum_product_chain_filters_each_time_step() -> None:
    """Forward messages remain normalized over a categorical state chain."""
    prior = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.3], [0.2, 0.7]])
    likelihoods = np.array([[0.9, 0.1], [0.4, 0.7], [0.2, 0.9]])
    beliefs = sum_product_chain(prior, transition, likelihoods)
    assert beliefs.shape == (3, 2)
    assert np.allclose(beliefs.sum(axis=1), 1.0)
    assert beliefs[-1, 1] > beliefs[0, 1]


def test_variational_message_update_softmaxes_expected_log_factor() -> None:
    """VMP-style update returns a normalized categorical message."""
    log_factor = np.log(np.array([[0.8, 0.2], [0.1, 0.9]]))
    msg = variational_message_update(log_factor, [np.array([0.5, 0.5])], target_axis=0)
    assert msg.shape == (2,)
    assert msg.sum() == pytest.approx(1.0)


def test_factor_graph_chain_validates_shapes() -> None:
    """Chapter 12 chain container normalizes priors and rejects bad shapes."""
    chain = FactorGraphChain(
        prior=np.array([2.0, 1.0]),
        transition=np.array([[0.8, 0.3], [0.2, 0.7]]),
        likelihoods=np.array([[0.9, 0.1], [0.4, 0.7]]),
    )
    np.testing.assert_allclose(chain.prior.sum(), 1.0)
    with pytest.raises(ValueError):
        FactorGraphChain(chain.prior, np.eye(3), chain.likelihoods)


def test_forward_backward_smoothing_uses_future_evidence() -> None:
    """Backward messages sharpen early beliefs when later evidence is strong."""
    prior = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    likelihoods = np.array([[0.55, 0.45], [0.2, 0.8], [0.1, 0.9]])
    forward = sum_product_chain(prior, transition, likelihoods)
    backward = backward_smoothing_messages(transition, likelihoods)
    smoothed = forward_backward_smoothing(prior, transition, likelihoods)
    assert backward.shape == forward.shape
    np.testing.assert_allclose(smoothed.sum(axis=1), np.ones(3))
    assert smoothed[0, 1] > forward[0, 1]


def test_hybrid_model_bridge_outer_products_discrete_and_continuous_state() -> None:
    """Hybrid bridge preserves categorical rows and continuous feature columns."""
    bridge = hybrid_model_bridge([2.0, -1.0], [0.25, 0.75])
    assert bridge.shape == (2, 2)
    np.testing.assert_allclose(bridge[0], [0.5, -0.25])
    np.testing.assert_allclose(bridge[1], [1.5, -0.75])


def test_marginal_and_active_inference_messages_are_normalized() -> None:
    """Marginal, active-inference, and attention messages stay finite."""
    transition = np.array([[0.8, 0.2], [0.2, 0.8]])
    trace = marginal_message_passing([transition, transition], [0.9, 0.1])
    assert trace.shape == (3, 2)
    np.testing.assert_allclose(trace.sum(axis=1), np.ones(3))
    likelihood = np.array([[0.9, 0.2], [0.1, 0.8]])
    messages = active_inference_factor_messages(likelihood, transition, [0.6, 0.4])
    assert set(messages) == {"prior", "prediction", "observation", "posterior"}
    for value in messages.values():
        np.testing.assert_allclose(value.sum(), 1.0)
    weighted = learning_attention_message([1.0, -2.0], np.log(2.0))
    np.testing.assert_allclose(weighted, [2.0, -4.0])
