"""
Differentiable EML tree.

A complete binary tree whose internal nodes compute the EML operator and
whose leaves are a learned convex combination over the atom set
`{1, x_0, x_1, ..., x_{d-1}}`. Training the convex weights with Adam and
then picking `argmax` at each leaf recovers a concrete symbolic tree
following the paper's grammar

    S -> 1 | x_i | eml(S, S)

This is a straight PyTorch `nn.Module`. No custom autograd.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operator import eml


@dataclass
class TreeConfig:
    depth: int
    n_features: int
    tau: float = 1.0          # softmax temperature for leaf weights
    include_one: bool = True  # whether the constant 1 is an atom


class EMLTree(nn.Module):
    """Complete binary EML tree of fixed depth.

    A depth-`d` tree has `2**d` leaves and `2**d - 1` internal nodes.
    Each leaf owns a logit vector over atoms; `forward` softmaxes it and
    mixes the atom values. Internal nodes have no parameters — they apply
    `eml` to their two children.

    Shape contract
    --------------
    `forward(X)` takes `X` of shape `(batch, n_features)` and returns a
    tensor of shape `(batch,)`.
    """

    def __init__(self, cfg: TreeConfig):
        super().__init__()
        if cfg.depth < 1:
            raise ValueError("depth must be >= 1")
        self.cfg = cfg

        self.n_leaves = 2 ** cfg.depth
        self.atoms = cfg.n_features + (1 if cfg.include_one else 0)

        # leaf_logits[leaf_idx, atom_idx]
        self.leaf_logits = nn.Parameter(
            torch.randn(self.n_leaves, self.atoms) * 0.1
        )

    # -- leaf mixing --------------------------------------------------------

    def _leaf_values(self, X: torch.Tensor) -> torch.Tensor:
        """Compute value of each leaf for each batch row.

        Returns shape (batch, n_leaves).
        """
        batch = X.shape[0]

        # Atom table shape (batch, atoms). Atom 0 is the constant 1 if
        # include_one is set, otherwise atoms are exactly the features.
        if self.cfg.include_one:
            one = torch.ones(batch, 1, device=X.device, dtype=X.dtype)
            atom_values = torch.cat([one, X], dim=1)
        else:
            atom_values = X

        # Softmax over atoms per leaf. Shape (n_leaves, atoms).
        weights = F.softmax(self.leaf_logits / self.cfg.tau, dim=-1)

        # leaves[b, l] = sum_a weights[l, a] * atom_values[b, a]
        return atom_values @ weights.T

    # -- tree evaluation ---------------------------------------------------

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        leaves = self._leaf_values(X)            # (batch, n_leaves)
        current = leaves
        # Bottom-up reduction: pair adjacent siblings, apply eml, halve.
        while current.shape[-1] > 1:
            left = current[..., 0::2]
            right = current[..., 1::2]
            current = eml(left, right)
        return current.squeeze(-1)

    # -- discrete extraction -----------------------------------------------

    def extract_tree(self, feature_names: list[str] | None = None) -> "SymbolicNode":
        """Convert trained soft tree into a discrete symbolic tree.

        At each leaf we pick the argmax atom. Internal structure is fixed
        by the tree layout.
        """
        if feature_names is None:
            feature_names = [f"x{i}" for i in range(self.cfg.n_features)]

        atom_names: list[str] = []
        if self.cfg.include_one:
            atom_names.append("1")
        atom_names.extend(feature_names)

        # Leaves
        picks = self.leaf_logits.argmax(dim=-1).tolist()
        nodes: list[SymbolicNode] = [
            SymbolicNode.leaf(atom_names[p]) for p in picks
        ]

        # Pair up
        while len(nodes) > 1:
            new_nodes = []
            for i in range(0, len(nodes), 2):
                new_nodes.append(SymbolicNode.node(nodes[i], nodes[i + 1]))
            nodes = new_nodes
        return nodes[0]


# ---------------------------------------------------------------------------
# Symbolic tree (discrete)
# ---------------------------------------------------------------------------

@dataclass
class SymbolicNode:
    """Node in the extracted symbolic tree.

    Either a leaf holding an atom name, or an internal node with two
    children. Internal nodes always denote `eml(left, right)`.
    """

    atom: str | None
    left: "SymbolicNode | None"
    right: "SymbolicNode | None"

    @classmethod
    def leaf(cls, atom: str) -> "SymbolicNode":
        return cls(atom=atom, left=None, right=None)

    @classmethod
    def node(cls, left: "SymbolicNode", right: "SymbolicNode") -> "SymbolicNode":
        return cls(atom=None, left=left, right=right)

    @property
    def is_leaf(self) -> bool:
        return self.atom is not None

    def to_expr(self) -> str:
        """Render as a string in `eml(...)` form."""
        if self.is_leaf:
            return self.atom  # type: ignore[return-value]
        return f"eml({self.left.to_expr()}, {self.right.to_expr()})"  # type: ignore[union-attr]

    def size(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + self.left.size() + self.right.size()  # type: ignore[union-attr]


__all__ = ["EMLTree", "TreeConfig", "SymbolicNode"]
