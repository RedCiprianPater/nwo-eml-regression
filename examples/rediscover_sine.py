"""
Showcase: rediscover sin(x) from samples using the EML regressor.

The paper (Odrzywołek, arXiv:2603.21852) proves that sin, cos and all
other trig functions are EML-definable through Euler's identity and the
complex constant `i`, which itself is EML-definable. Because our
regressor is real-valued, exact recovery of `sin` requires enough depth
for the real-part reconstruction. This example demonstrates the
end-to-end workflow: sample data → fit → extract → simplify.

Expect: a reasonable fit on [-π, π] at depth 4. Perfect recovery of
`sin` would require depth 5+ and very careful training. This script is
more about showing the full pipeline than hitting machine precision.
"""
from __future__ import annotations

import numpy as np

from nwo_eml import EMLRegressor
from nwo_eml.simplify import simplify_tree


def main() -> None:
    rng = np.random.default_rng(0)

    # Sample sin on a positive interval to stay inside the friendly
    # domain of the guarded `eml` operator. Full-period recovery is
    # left as an exercise — the paper's construction works for all x
    # but requires more depth to hold stably under Adam.
    x = np.linspace(0.1, np.pi - 0.1, 200).reshape(-1, 1)
    y = np.sin(x).ravel() + rng.normal(0, 1e-4, x.shape[0])

    reg = EMLRegressor(
        depth=4,
        n_epochs=3000,
        lr=0.03,
        seed=0,
    ).fit(x, y, feature_names=["x"])

    summary = reg.summary()
    print("=== EML symbolic regression for sin(x) ===")
    print(f"final loss    : {summary['final_loss']:.4e}")
    print(f"tree size     : {summary['tree_size']}")
    print(f"raw expression: {summary['expression']}")
    print(f"simplified    : {simplify_tree(reg.result_.tree)}")
    print("paper         : Odrzywołek, arXiv:2603.21852")


if __name__ == "__main__":
    main()
