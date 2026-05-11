"""Tests for the obstruction tensor.

On round S^4 (conformally flat), the obstruction tensor vanishes: O = B = 0.
"""
import pytest
from conformal_toolkit.core.conformal_structure import ConformalStructure
from tests.conftest_sage import (
    _make_round_sphere_2, _make_round_sphere_4, _make_flat_rn,
)


def test_obstruction_equals_bach_4d():
    """In 4D, obstruction tensor = Bach tensor."""
    data = _make_round_sphere_4()
    cs = ConformalStructure(data['metric'])
    O = cs.obstruction_tensor()
    B = cs.bach()
    frame = list(data['manifold'].frames())[0]
    for i in range(4):
        for j in range(4):
            o_val = O[frame, i, j]
            b_val = B[frame, i, j]
            if hasattr(o_val, 'expr'):
                o_val = o_val.expr()
            if hasattr(b_val, 'expr'):
                b_val = b_val.expr()
            assert (o_val - b_val).simplify_full() == 0, \
                f"O[{i},{j}] != B[{i},{j}]"


def test_obstruction_vanishes_conformally_flat():
    """Obstruction tensor vanishes on flat R^4."""
    data = _make_flat_rn(4)
    cs = ConformalStructure(data['metric'])
    O = cs.obstruction_tensor()
    frame = list(data['manifold'].frames())[0]
    for i in range(4):
        for j in range(4):
            val = O[frame, i, j]
            if hasattr(val, 'expr'):
                val = val.expr()
            assert val == 0, f"O[{i},{j}] should vanish on flat space"


def test_obstruction_odd_dimension_raises():
    """Obstruction tensor is undefined in odd dimensions."""
    data = _make_flat_rn(3)
    cs = ConformalStructure(data['metric'])
    with pytest.raises(ValueError, match="even dimensions"):
        cs.obstruction_tensor()


def test_obstruction_2d_raises():
    """Obstruction tensor requires dimension >= 4."""
    data = _make_round_sphere_2()
    cs = ConformalStructure(data['metric'])
    with pytest.raises(ValueError, match="dimension >= 4"):
        cs.obstruction_tensor()
