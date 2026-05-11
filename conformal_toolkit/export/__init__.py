"""Export module for conformal toolkit.

Provides conversion utilities from SageManifolds tensors to NumPy arrays,
PyTorch tensors, and feature vectors for machine learning.

Public API
----------
tensor_to_numpy, scalar_at_point
tensor_to_torch, scalar_to_torch
conformal_feature_vector
"""
from conformal_toolkit.export.to_numpy import tensor_to_numpy, scalar_at_point
from conformal_toolkit.export.to_torch import tensor_to_torch, scalar_to_torch
from conformal_toolkit.export.feature_vector import conformal_feature_vector

__all__ = [
    "tensor_to_numpy",
    "scalar_at_point",
    "tensor_to_torch",
    "scalar_to_torch",
    "conformal_feature_vector",
]
