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


# ---------- 6D obstruction tensor tests ----------

def test_obstruction_6d_flat_vanishes():
    """Obstruction tensor should vanish on flat R^6."""
    data = _make_flat_rn(6)
    cs = ConformalStructure(data['metric'])
    O = cs.obstruction_tensor()
    frame = list(data['manifold'].frames())[0]
    for a in range(6):
        for b in range(6):
            comp = O[frame, a, b]
            if hasattr(comp, 'expr'):
                comp = comp.expr()
            assert comp == 0, f"O[{a},{b}] should vanish on flat R^6, got {comp}"


def test_obstruction_6d_is_not_notimplemented():
    """compute_obstruction should no longer raise NotImplementedError for n=6."""
    data = _make_flat_rn(6)
    cs = ConformalStructure(data['metric'])
    # Should not raise
    O = cs.obstruction_tensor()
    assert O is not None


def test_obstruction_8d_raises():
    """Obstruction tensor in dimension 8 should still raise NotImplementedError."""
    data = _make_flat_rn(8)
    cs = ConformalStructure(data['metric'])
    with pytest.raises(NotImplementedError, match="n=4 and n=6"):
        cs.obstruction_tensor()
