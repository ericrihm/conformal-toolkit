"""Convert SageManifolds tensors to NumPy arrays."""
import numpy as np


def tensor_to_numpy(tensor, chart=None):
    """Convert a SageManifolds tensor to a NumPy array of component values.

    For a (0,2) tensor on an n-dim manifold, returns an n×n array of
    symbolic expressions (or floats if the components are numeric).
    For a scalar field, returns a float or symbolic expression.

    Parameters
    ----------
    tensor : SageManifolds tensor field or scalar field
        The tensor to convert.
    chart : SageManifolds chart, optional
        The coordinate chart to use.  Defaults to the first top-level chart.

    Returns
    -------
    np.ndarray of shape (n,)*(p+q) for a (p,q) tensor, or scalar expression.
    """
    # Check for scalar fields (they don't have tensor_type)
    try:
        p, q = tensor.tensor_type()
    except AttributeError:
        # It's a scalar field
        expr = tensor.expr()
        if hasattr(expr, 'is_numeric') and expr.is_numeric():
            return float(expr)
        return expr

    M = tensor.domain()
    n = M.dim()

    if chart is None:
        chart = list(M.top_charts())[0]
    frame = chart.frame()

    rank = p + q
    if rank == 0:
        expr = tensor.expr()
        if hasattr(expr, 'is_numeric') and expr.is_numeric():
            return float(expr)
        return expr

    shape = (n,) * rank
    result = np.empty(shape, dtype=object)
    for idx in np.ndindex(shape):
        comp = tensor[frame, *idx]
        if hasattr(comp, 'expr'):
            comp = comp.expr()
        result[idx] = comp
    return result


def scalar_at_point(scalar_field, point_dict, chart=None):
    """Evaluate a scalar field at a specific coordinate point.

    Parameters
    ----------
    scalar_field : SageManifolds scalar field
        The scalar field to evaluate.
    point_dict : dict
        Mapping from coordinate symbols to numeric values,
        e.g. {x: 1.0, y: 2.0}.
    chart : SageManifolds chart, optional
        Chart whose coordinates are used.  Defaults to the first top-level chart.

    Returns
    -------
    float
        Numeric value of the scalar field at the given point.
    """
    M = scalar_field.domain()

    if chart is None:
        chart = list(M.top_charts())[0]

    expr = scalar_field.expr(chart)
    val = expr.subs(point_dict)
    return float(val)
