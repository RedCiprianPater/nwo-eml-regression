"""Smoke tests for the differentiable tree and regressor."""
import numpy as np
import torch

from nwo_eml import EMLRegressor, EMLTree, TreeConfig


def test_tree_forward_shape():
    cfg = TreeConfig(depth=3, n_features=2)
    tree = EMLTree(cfg)
    X = torch.randn(17, 2)
    out = tree(X)
    assert out.shape == (17,)


def test_tree_extract_structure():
    cfg = TreeConfig(depth=2, n_features=1)
    tree = EMLTree(cfg)
    sym = tree.extract_tree(feature_names=["x"])
    # Depth 2 complete binary tree: 4 leaves, 3 internal, size = 7.
    assert sym.size() == 7
    # Root is an internal node (eml call).
    assert not sym.is_leaf


def test_regressor_converges_on_constant():
    """Cheapest possible sanity check: fit a constant target."""
    X = np.linspace(0.5, 1.5, 64).reshape(-1, 1)
    y = np.ones(64)  # target = 1
    reg = EMLRegressor(depth=2, n_epochs=400, seed=0).fit(X, y)
    # Not asking for exact recovery here — just that loss goes down.
    assert reg.result_.losses[0] > reg.result_.losses[-1]
    assert reg.result_.final_loss < reg.result_.losses[0]
