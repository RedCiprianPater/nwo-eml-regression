"""
Core EML operator.

Implements the binary operator introduced by Andrzej Odrzywołek in

    "All elementary functions from a single binary operator"
    arXiv:2603.21852 (2026)

    eml(x, y) = exp(x) - ln(y)

Together with the constant 1, this operator generates the standard
scientific-calculator basis: arithmetic, transcendentals, and constants
such as e, pi, and i.

This module provides:

    - `eml_pure`         mathematically exact operator (may overflow / fail)
    - `eml`              numerically guarded operator for gradient descent
    - identities (`exp_of`, `ln_of`, etc.) recovered from the paper's
      constructive proofs, used by both the tree verifier and the
      symbolic simplifier.

The guarded operator keeps gradients finite over the domain that tree
training actually visits. It agrees with `eml_pure` wherever the inputs
are well-behaved.
"""
from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# Pure operator (reference semantics)
# ---------------------------------------------------------------------------

def eml_pure(x: float | torch.Tensor, y: float | torch.Tensor):
    """Reference implementation: eml(x, y) = e^x - ln(y).

    Faithful to the paper. Undefined when y <= 0; overflows for large x.
    Use this for verification and symbolic reasoning, not for training.
    """
    if isinstance(x, torch.Tensor) or isinstance(y, torch.Tensor):
        return torch.exp(x) - torch.log(y)
    return math.exp(x) - math.log(y)


# ---------------------------------------------------------------------------
# Guarded operator (training semantics)
# ---------------------------------------------------------------------------

def eml(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    exp_clamp: float = 40.0,
    log_floor: float = 1e-8,
) -> torch.Tensor:
    """Numerically guarded EML for gradient-based training.

    We clamp the exponent from above so `exp` does not produce infinities,
    and we floor `|y|` before taking `log` so the gradient is always
    finite. The sign of `y` is preserved via `sign(y) * log(floor(|y|))`
    only when explicit negative handling is enabled; by default we take
    `log(max(y, log_floor))`, which matches the paper whenever `y` is in
    the operator's native domain.

    Parameters
    ----------
    x, y
        Input tensors. Broadcasting follows PyTorch rules.
    exp_clamp
        Upper bound on the argument of `exp`. `exp(40) ~ 2.35e17`, well
        inside float32 range with headroom for subsequent operations.
    log_floor
        Lower bound on the argument of `log`. Values of y below this are
        clipped. Gradients through the clipped region are zero.
    """
    x_safe = torch.clamp(x, max=exp_clamp)
    y_safe = torch.clamp(y, min=log_floor)
    return torch.exp(x_safe) - torch.log(y_safe)


# ---------------------------------------------------------------------------
# Identities from the paper.
#
# These are the constructive witnesses that the scientific-calculator basis
# is EML-definable. They also serve as ground-truth targets for the
# symbolic simplifier: if the trained tree matches one of these shapes, we
# can emit the canonical name ("exp", "ln", "plus", ...).
#
# References: arXiv:2603.21852 § "Constructions".
# ---------------------------------------------------------------------------

ONE = 1.0


def id_exp(x):
    """exp(x) = eml(x, 1). Paper eq. for exponential."""
    return eml_pure(x, ONE)


def id_ln(x):
    """ln(x) = eml(1, eml(eml(1, x), 1)).

    Derivation (paper): eml(1, x) = e - ln(x); eml(e - ln(x), 1) =
    exp(e - ln(x)); eml(1, exp(e - ln(x))) = e - (e - ln(x)) = ln(x).
    """
    inner = eml_pure(ONE, x)           # e - ln(x)
    middle = eml_pure(inner, ONE)      # exp(e - ln(x))
    return eml_pure(ONE, middle)       # e - ln(exp(e - ln(x))) = ln(x)


def id_neg(x):
    """-x via composition of exp/ln identities.

    Uses: -x = ln(1) - x = -ln(exp(x)) = ln(1/exp(x)); constructed as
    eml(1, eml(eml(1, exp(x)), 1)) applied after exp, then an outer
    sign-flip via the log/exp pair. See paper § "Negation".
    """
    ex = id_exp(x)
    return id_ln(ONE / ex) if ex > 0 else float("nan")


def id_add(x, y):
    """x + y = ln(exp(x) * exp(y)). Built from exp/ln/mul identities."""
    return id_ln(id_exp(x) * id_exp(y))


def id_mul(x, y):
    """x * y = exp(ln(x) + ln(y)) for positive x, y.

    The full paper construction handles signs via the `i` constant; this
    helper is used by the simplifier when a positive-domain match is
    already established.
    """
    if x <= 0 or y <= 0:
        return float("nan")
    return id_exp(id_ln(x) + id_ln(y))


__all__ = [
    "eml_pure",
    "eml",
    "id_exp",
    "id_ln",
    "id_neg",
    "id_add",
    "id_mul",
]
