"""Obstruction tensor computation.

In dimension n=4 the obstruction tensor equals the Bach tensor.
For general even n >= 6, the obstruction tensor is a higher-order
conformally invariant symmetric 2-tensor; not yet implemented.
"""


def compute_obstruction(cs):
    """Compute the obstruction tensor O_ab.

    In 4D this is the Bach tensor. In higher even dimensions
    it involves higher-order covariant derivatives of curvature.
    """
    n = cs.dimension
    if n % 2 != 0:
        raise ValueError(f"Obstruction tensor only defined in even dimensions, got {n}")
    if n < 4:
        raise ValueError(f"Obstruction tensor requires dimension >= 4, got {n}")
    if n == 4:
        return cs.bach()
    raise NotImplementedError(
        f"Obstruction tensor in dimension {n} not yet implemented (only n=4)"
    )
