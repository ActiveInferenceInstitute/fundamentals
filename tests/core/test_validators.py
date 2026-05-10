"""Tests for ``core.validators`` — runtime input checks."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.core.validators import (
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


class TestPositiveScalar:
    def test_accepts_positive(self) -> None:
        assert require_positive_scalar(1.5) == 1.5

    def test_rejects_zero(self) -> None:
        with pytest.raises(ValueError, match=">"):
            require_positive_scalar(0.0)

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError, match=">"):
            require_positive_scalar(-0.5)

    def test_rejects_non_finite(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            require_positive_scalar(np.inf)
        with pytest.raises(ValueError, match="finite"):
            require_positive_scalar(np.nan)


class TestNonNegativeScalar:
    def test_accepts_zero(self) -> None:
        assert require_non_negative_scalar(0.0) == 0.0

    def test_accepts_positive(self) -> None:
        assert require_non_negative_scalar(2.0) == 2.0

    def test_rejects_negative(self) -> None:
        with pytest.raises(ValueError):
            require_non_negative_scalar(-0.1)


class TestUnitInterval:
    def test_strict_open(self) -> None:
        assert require_in_unit_interval(0.5) == 0.5
        with pytest.raises(ValueError):
            require_in_unit_interval(0.0)
        with pytest.raises(ValueError):
            require_in_unit_interval(1.0)

    def test_inclusive_endpoints(self) -> None:
        assert require_in_unit_interval(0.0, inclusive=True) == 0.0
        assert require_in_unit_interval(1.0, inclusive=True) == 1.0


class TestIntAtLeast:
    def test_accepts_int(self) -> None:
        assert require_int_at_least(3, minimum=1) == 3

    def test_rejects_below_minimum(self) -> None:
        with pytest.raises(ValueError):
            require_int_at_least(0, minimum=1)

    def test_rejects_float(self) -> None:
        with pytest.raises(ValueError, match="integer"):
            require_int_at_least(2.5)


class TestFiniteArray:
    def test_accepts_finite(self) -> None:
        out = require_finite_array(np.array([1.0, 2.0, 3.0]))
        assert out.dtype == float

    def test_rejects_nan(self) -> None:
        with pytest.raises(ValueError):
            require_finite_array(np.array([1.0, np.nan]))

    def test_rejects_inf(self) -> None:
        with pytest.raises(ValueError):
            require_finite_array(np.array([1.0, np.inf]))


class Test1D:
    def test_accepts_1d(self) -> None:
        out = require_1d(np.zeros(5))
        assert out.shape == (5,)

    def test_rejects_2d(self) -> None:
        with pytest.raises(ValueError, match="1-D"):
            require_1d(np.zeros((2, 3)))

    def test_length_check(self) -> None:
        with pytest.raises(ValueError, match="length"):
            require_1d(np.zeros(3), length=5)


class Test2D:
    def test_accepts_2d(self) -> None:
        out = require_2d(np.zeros((2, 3)))
        assert out.shape == (2, 3)

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2-D"):
            require_2d(np.zeros(5))

    def test_shape_check(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            require_2d(np.zeros((2, 3)), shape=(2, 4))


class TestSameLength:
    def test_matching(self) -> None:
        require_same_length(np.zeros(5), np.zeros(5), np.zeros(5))

    def test_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length mismatch"):
            require_same_length(np.zeros(5), np.zeros(4))


class TestMonotone:
    def test_increasing(self) -> None:
        out = require_monotone(np.array([1.0, 2.0, 3.0]))
        assert out.size == 3

    def test_strict_rejects_plateau(self) -> None:
        with pytest.raises(ValueError):
            require_monotone(np.array([1.0, 1.0, 2.0]), strict=True)

    def test_decreasing(self) -> None:
        require_monotone(np.array([3.0, 2.0, 1.0]), increasing=False)


class TestDesignMatrix:
    def test_1d_promotes_to_2d(self) -> None:
        out = require_design_matrix(np.zeros(7))
        assert out.shape == (7, 1)

    def test_n_features_check(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            require_design_matrix(np.zeros((10, 2)), n_features=3)

    def test_n_samples_check(self) -> None:
        with pytest.raises(ValueError, match="rows"):
            require_design_matrix(np.zeros((10, 2)), n_samples=11)

    def test_rejects_non_finite(self) -> None:
        bad = np.zeros((3, 2))
        bad[1, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            require_design_matrix(bad)
