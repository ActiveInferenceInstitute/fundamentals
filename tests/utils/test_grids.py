"""Tests for ``utils.grids`` — evenly-spaced 1-D and 2-D grid constructors."""

from __future__ import annotations

import numpy as np
import pytest

from active_inference.utils.grids import make_2d_grid, make_grid


class TestMakeGrid:
    def test_endpoints_inclusive(self) -> None:
        g = make_grid(0.0, 1.0, n_points=11)
        assert g[0] == 0.0
        assert g[-1] == 1.0
        assert g.size == 11

    def test_uniform_spacing(self) -> None:
        g = make_grid(-2.0, 3.0, n_points=51)
        diffs = np.diff(g)
        np.testing.assert_allclose(diffs, diffs[0])

    def test_invalid_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_grid(1.0, 1.0)
        with pytest.raises(ValueError):
            make_grid(2.0, 1.0)

    def test_invalid_n_points(self) -> None:
        with pytest.raises(ValueError):
            make_grid(0.0, 1.0, n_points=1)

    def test_non_finite_bounds(self) -> None:
        with pytest.raises(ValueError):
            make_grid(0.0, np.inf)
        with pytest.raises(ValueError):
            make_grid(-np.inf, 0.0)


class TestMake2DGrid:
    def test_shapes(self) -> None:
        x, y = make_2d_grid(0.0, 1.0, -1.0, 1.0, n_x=10, n_y=20)
        assert x.shape == (10,)
        assert y.shape == (20,)

    def test_endpoints(self) -> None:
        x, y = make_2d_grid(-1.0, 2.0, 0.0, 5.0)
        assert x[0] == -1.0 and x[-1] == 2.0
        assert y[0] == 0.0 and y[-1] == 5.0
