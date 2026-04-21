"""
nwo-eml-regression
==================

Gradient-based symbolic regression over EML trees.

Implements the `eml(x, y) = exp(x) - ln(y)` operator from

    Andrzej Odrzywołek,
    "All elementary functions from a single binary operator",
    arXiv:2603.21852 (2026).

Quick start
-----------

    >>> import numpy as np
    >>> from nwo_eml import EMLRegressor
    >>> x = np.linspace(0.1, 2.0, 100)
    >>> y = np.log(x)
    >>> reg = EMLRegressor(depth=3, n_epochs=1500).fit(x, y)
    >>> print(reg.summary())
"""
from .operator import eml, eml_pure
from .regressor import EMLRegressor
from .simplify import simplify_tree, to_sympy
from .tree import EMLTree, SymbolicNode, TreeConfig

__all__ = [
    "EMLRegressor",
    "EMLTree",
    "SymbolicNode",
    "TreeConfig",
    "eml",
    "eml_pure",
    "simplify_tree",
    "to_sympy",
]

__version__ = "0.1.0"
__paper__ = "arXiv:2603.21852"
