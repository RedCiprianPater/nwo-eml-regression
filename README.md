# nwo-eml-regression

Gradient-based symbolic regression over EML trees.

Implements the binary operator introduced by Andrzej Odrzywołek in
[*All elementary functions from a single binary operator*](https://arxiv.org/abs/2603.21852)
(arXiv:2603.21852, 2026):

```
eml(x, y) = exp(x) − ln(y)
```

Together with the constant `1`, this operator generates the entire
standard scientific-calculator basis — arithmetic, transcendentals, and
constants including `e`, `π`, and `i`. This repo uses that property to
fit differentiable EML trees by gradient descent and recover closed-form
elementary expressions from numerical data.

This package is the math core used by two downstream NWO projects:

- **[mcp-server-robotics](https://github.com/RedCiprianPater/mcp-server-robotics)** — exposes `eml_regress` as an MCP tool.
- **[nwo-timesfm-integration](https://github.com/RedCiprianPater/nwo-timesfm-integration)** — fits EML trees to TimesFM forecast residuals to recover symbolic systematic-error laws.

## Install

```bash
pip install nwo-eml-regression            # runtime only
pip install "nwo-eml-regression[sympy]"   # + symbolic simplification
pip install "nwo-eml-regression[dev]"     # + tests and tooling
```

## Quick start

```python
import numpy as np
from nwo_eml import EMLRegressor

x = np.linspace(0.1, 2.0, 100).reshape(-1, 1)
y = np.log(x).ravel()

reg = EMLRegressor(depth=3, n_epochs=1500).fit(x, y, feature_names=["x"])
print(reg.expression_)      # raw eml(...) form
print(reg.summary())
```

## CLI

The `nwo-eml` CLI accepts JSON on stdin and writes JSON on stdout. This
is what the MCP server shells out to:

```bash
echo '{
  "data":   [[0.1],[0.5],[1.0],[1.5],[2.0]],
  "target": [-2.302, -0.693, 0.0, 0.405, 0.693],
  "feature_names": ["x"],
  "depth": 3,
  "n_epochs": 1500
}' | nwo-eml regress --pretty
```

## What it does, specifically

The paper shows that every elementary function has a representation as a
nested tree of `eml` applications over the leaf alphabet `{1}`. We extend
that alphabet to `{1, x_0, x_1, ..., x_{d-1}}` for regression and train a
complete binary tree of depth *d* by:

1. Each leaf holds a softmaxed logit vector over the atom set.
2. Internal nodes apply the guarded `eml` operator.
3. The tree is trained end-to-end with Adam against MSE, annealing the
   leaf temperature from high to low so the discrete choice emerges late.
4. After training, each leaf is `argmax`-projected to a single atom,
   producing a concrete symbolic tree.
5. Optional SymPy simplification converts the raw `eml(...)` form into a
   canonical elementary function when possible.

The paper reports exact recovery at depths ≤ 4 for elementary generating
laws. This repo confirms that behaviour — see `tests/` and `examples/`.

## Numerical stability

`exp(x) − ln(y)` is unfriendly to gradient descent as written:
`exp` overflows, `ln` is undefined for `y ≤ 0`. The module `operator.py`
provides `eml_pure` (reference semantics) and `eml` (training semantics).
The training variant clamps `exp`'s argument to 40 and floors `ln`'s
argument at `1e-8`. Inside the `eml_pure` domain the two agree.

## Citation

If you use this in published work, cite the paper:

```
@article{odrzywolek2026eml,
  title   = {All elementary functions from a single binary operator},
  author  = {Odrzywołek, Andrzej},
  journal = {arXiv preprint arXiv:2603.21852},
  year    = {2026}
}
```

And, optionally, this software via its Zenodo DOI once assigned.
See `CITATION.cff`.

## License

MIT. The mathematical results are due to Odrzywołek (arXiv:2603.21852).
This repo implements them; it does not claim them.
