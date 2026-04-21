# Paper reference and derivations

This document collects the identities and derivations from the source
paper that are used in the code.

## Primary reference

Andrzej Odrzywołek. _All elementary functions from a single binary
operator._ arXiv:2603.21852 [cs.SC], 2026.
<https://arxiv.org/abs/2603.21852>

## The operator

```
eml(x, y) = exp(x) − ln(y)
```

## Key identities used in `operator.py`

### Exponential

```
exp(x) = eml(x, 1)
```

Derivation: `eml(x, 1) = exp(x) − ln(1) = exp(x) − 0 = exp(x)`.

Used by: `id_exp`.

### Natural logarithm

```
ln(x) = eml(1, eml(eml(1, x), 1))
```

Step-by-step:

1. Let `A = eml(1, x) = e − ln(x)`.
2. Let `B = eml(A, 1) = exp(A) − ln(1) = exp(e − ln(x))`.
3. `eml(1, B) = e − ln(B) = e − ln(exp(e − ln(x))) = e − (e − ln(x)) = ln(x)`.

Used by: `id_ln`. Verified by `tests/test_operator.py::test_identity_ln_from_eml`
against `math.log` for several positive inputs.

### Negation (via exp/ln pair)

```
−x = ln(1 / exp(x))
```

Used by: `id_neg`.

### Addition

```
x + y = ln(exp(x) · exp(y))
```

Built from `id_exp`, `id_ln`, and multiplication.

### Multiplication (positive domain)

```
x · y = exp(ln(x) + ln(y))    for x, y > 0
```

The paper's construction handles signs via the `i` constant; in this
codebase we only use `id_mul` on positive intermediates and leave sign
handling to the differentiable tree itself.

## Grammar used in `tree.py`

The paper's pure grammar is `S -> 1 | eml(S, S)`. For regression we
extend the leaf alphabet with input variables:

```
S -> 1 | x_0 | x_1 | ... | x_{d-1} | eml(S, S)
```

Each leaf of the complete binary tree is a learned softmax mixture over
this alphabet. After training, `argmax` at each leaf recovers a
concrete symbolic tree that satisfies the extended grammar.

## Numerical guards

The training-time operator `eml(x, y)` in `operator.py` clamps the
exponent argument to 40 and floors the log argument at `1e-8`. This
deviates from the paper's pure semantics only in regions where the
gradient would otherwise be `NaN` or `inf`. Inside the well-defined
domain the two agree exactly.

## Simplification

After extracting the discrete tree we convert to SymPy using
`eml(a, b) = exp(a) − log(b)` and run `sp.simplify` followed by
`sp.trigsimp`. This is a best-effort pass — SymPy does not always
canonicalise deep exp/log compositions, so the raw `eml(...)` form is
always preserved alongside the simplified output.
