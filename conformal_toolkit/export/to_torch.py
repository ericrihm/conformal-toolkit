"""Convert SageManifolds tensors to PyTorch tensors."""


def tensor_to_torch(tensor, point_dict=None, chart=None):
    """Convert a SageManifolds tensor to a PyTorch tensor.

    For tensors with coordinate-dependent components, ``point_dict`` must be
    provided to obtain a numeric result.  If all components are constants,
    ``point_dict`` may be omitted.

    Parameters
    ----------
    tensor : SageManifolds tensor field
        The tensor to convert.
    point_dict : dict, optional
        Mapping {coord_symbol: value} for numeric substitution.
    chart : SageManifolds chart, optional
        Chart to read components from.  Defaults to first top-level chart.

    Returns
    -------
    torch.Tensor
        Float tensor of the same shape as the component array.
    """
    import torch
    import numpy as np
    from conformal_toolkit.export.to_numpy import tensor_to_numpy

    arr = tensor_to_numpy(tensor, chart=chart)

    if isinstance(arr, np.ndarray):
        if point_dict is not None:
            numeric = np.empty(arr.shape, dtype=float)
            for idx in np.ndindex(arr.shape):
                val = arr[idx]
                if hasattr(val, 'subs'):
                    val = float(val.subs(point_dict))
                else:
                    val = float(val)
                numeric[idx] = val
            return torch.tensor(numeric, dtype=torch.float64)
        else:
            # Try direct float conversion (only works for numeric arrays)
            try:
                return torch.tensor(arr.astype(float), dtype=torch.float64)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Tensor components are symbolic; provide point_dict for numeric evaluation."
                ) from exc
    else:
        # Scalar
        if point_dict is not None and hasattr(arr, 'subs'):
            arr = float(arr.subs(point_dict))
        return torch.tensor(float(arr), dtype=torch.float64)


def scalar_to_torch(scalar_field, point_dict=None, chart=None):
    """Convert a SageManifolds scalar field to a PyTorch scalar tensor.

    Parameters
    ----------
    scalar_field : SageManifolds scalar field
    point_dict : dict, optional
        Coordinate substitution dict for numeric evaluation.
    chart : SageManifolds chart, optional

    Returns
    -------
    torch.Tensor  (0-dimensional)
    """
    import torch
    from conformal_toolkit.export.to_numpy import scalar_at_point

    if point_dict is not None:
        val = scalar_at_point(scalar_field, point_dict, chart=chart)
        return torch.tensor(float(val), dtype=torch.float64)

    expr = scalar_field.expr()
    if hasattr(expr, 'is_numeric') and expr.is_numeric():
        return torch.tensor(float(expr), dtype=torch.float64)

    raise ValueError(
        "Scalar field is symbolic; provide point_dict for numeric evaluation."
    )
