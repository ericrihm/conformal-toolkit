"""Tests for conformal hypersurface invariant enumeration."""
import pytest


def test_count_weight2_any_dimension():
    """Exactly one invariant at weight 2 in any dimension >= 3."""
    from conformal_toolkit.hypersurface.invariant_enumeration import count_invariants
    for d in range(3, 8):
        assert count_invariants(2, d) == 1, f"Should be 1 invariant at weight 2 in dim {d}"


def test_list_weight2():
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants
    invs = list_invariants(2, 4)
    assert len(invs) == 1
    assert invs[0]['name'] == '|L_1|^2'
    assert invs[0]['order'] == 2


def test_weight4_ambient3():
    """In ambient dim 3: tr(L_1^4) dependent (n=2<=3), Weyl terms vanish (dim<=3).
    So only |L_2|^2 survives.
    """
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants, count_invariants
    invs = list_invariants(4, 3)
    assert count_invariants(4, 3) == 1
    names = [i['name'] for i in invs]
    assert '|L_2|^2' in names


def test_weight4_ambient4():
    """In ambient dim 4: hypersurface dim n=3, tr(L_1^4) still dependent (n<=3).
    Weyl terms present (dim>3) but |W_{nabc}|^2 dependent (dim==4).
    So: |L_2|^2, W_{nanb} L_1^{ab}.
    """
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants, count_invariants
    invs = list_invariants(4, 4)
    names = [i['name'] for i in invs]
    assert '|L_2|^2' in names
    assert 'W_{nanb} L_1^{ab}' in names
    assert 'tr(L_1^4)' not in names  # dependent for n=3
    assert '|W_{nabc}|^2' not in names  # dependent for ambient_dim=4
    assert count_invariants(4, 4) == 2


def test_weight4_ambient5():
    """In ambient dim 5: n=4, all invariants independent."""
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants, count_invariants
    invs = list_invariants(4, 5)
    names = [i['name'] for i in invs]
    assert '|L_2|^2' in names
    assert 'tr(L_1^4)' in names
    assert 'W_{nanb} L_1^{ab}' in names
    assert '|W_{nabc}|^2' in names
    assert count_invariants(4, 5) == 4


def test_invalid_ambient_dim():
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants
    with pytest.raises(ValueError):
        list_invariants(2, 2)


def test_unsupported_order():
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants
    with pytest.raises(NotImplementedError):
        list_invariants(6, 4)


def test_invariant_dict_keys():
    """Every invariant dict has the required keys."""
    from conformal_toolkit.hypersurface.invariant_enumeration import list_invariants
    for inv in list_invariants(4, 5):
        assert 'name' in inv
        assert 'formula' in inv
        assert 'order' in inv
