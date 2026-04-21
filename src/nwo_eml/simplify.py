"""
Symbolic simplification of EML trees.

An extracted `SymbolicNode` tree uses the raw `eml(a, b) = exp(a) - ln(b)`
grammar. That form is not human-readable. This module converts the tree
into a SymPy expression and then applies `simplify`/`trigsimp` to try to
recover a canonical closed form.

This is where the paper's practical payoff shows up: trained trees come
out as deeply nested eml calls, but the underlying math is usually a
familiar elementary function.
"""
from __future__ import annotations

try:
    import sympy as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore[assignment]

from .tree import SymbolicNode


def to_sympy(tree: SymbolicNode, symbol_map: dict[str, "sp.Symbol"] | None = None):
    """Convert the raw EML tree into a SymPy expression.

    `eml(a, b)` becomes `exp(a) - log(b)`. Leaves become SymPy symbols or
    the integer 1.
    """
    if sp is None:
        raise ImportError("sympy is required for symbolic simplification")

    symbol_map = symbol_map or {}

    def rec(node: SymbolicNode):
        if node.is_leaf:
            atom = node.atom
            if atom == "1":
                return sp.Integer(1)
            if atom not in symbol_map:
                symbol_map[atom] = sp.Symbol(atom, real=True)  # type: ignore[assignment]
            return symbol_map[atom]
        left = rec(node.left)    # type: ignore[arg-type]
        right = rec(node.right)  # type: ignore[arg-type]
        return sp.exp(left) - sp.log(right)

    return rec(tree)


def simplify_tree(tree: SymbolicNode) -> str:
    """Return a simplified string form of the tree.

    Falls back to the raw `eml(...)` expression if SymPy is unavailable
    or if simplification fails.
    """
    if sp is None:
        return tree.to_expr()
    try:
        expr = to_sympy(tree)
        simplified = sp.simplify(expr)
        # Try trig simplification as a second pass — eml constructions of
        # sin/cos come out as large exponential combinations that trigsimp
        # recognises.
        simplified = sp.trigsimp(simplified)
        return str(simplified)
    except Exception:
        return tree.to_expr()


__all__ = ["to_sympy", "simplify_tree"]
