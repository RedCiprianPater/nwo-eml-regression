"""Verify the operator's identities from the paper."""
import math

import pytest

from nwo_eml.operator import eml_pure, id_exp, id_ln


def test_eml_pure_matches_definition():
    # eml(x, y) = e^x - ln(y)
    assert math.isclose(eml_pure(0.0, 1.0), 1.0 - 0.0)  # e^0 - ln(1) = 1
    assert math.isclose(eml_pure(1.0, 1.0), math.e)     # e^1 - ln(1) = e
    assert math.isclose(eml_pure(0.0, math.e), 1.0 - 1.0, abs_tol=1e-12)


@pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.5, -0.3])
def test_identity_exp_from_eml(x):
    # Paper: exp(x) = eml(x, 1)
    assert math.isclose(id_exp(x), math.exp(x), rel_tol=1e-12)


@pytest.mark.parametrize("x", [0.25, 0.5, 1.0, 2.0, 7.3])
def test_identity_ln_from_eml(x):
    # Paper: ln(x) = eml(1, eml(eml(1, x), 1))
    # This is the nontrivial one — it witnesses that the operator is
    # self-inverting up to composition.
    assert math.isclose(id_ln(x), math.log(x), rel_tol=1e-9)
