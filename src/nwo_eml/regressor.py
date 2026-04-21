"""
EML symbolic regressor.

sklearn-style `fit` / `predict` API that wraps the differentiable tree
and the discrete-extraction step. Uses Adam as recommended by the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .tree import EMLTree, SymbolicNode, TreeConfig


@dataclass
class FitResult:
    final_loss: float
    losses: list[float] = field(default_factory=list)
    tree: SymbolicNode | None = None
    expression: str = ""


class EMLRegressor:
    """Gradient-based symbolic regression over EML trees.

    Parameters
    ----------
    depth
        Depth of the complete binary tree. The paper reports exact
        recovery of elementary functions at depths up to 4.
    lr
        Adam learning rate.
    n_epochs
        Number of optimizer steps over the full batch.
    tau_start, tau_end
        Softmax temperature at the leaves. Annealing from high to low
        encourages a smooth landscape early and a discrete choice late.
    device
        "cpu", "cuda", or None (auto).
    """

    def __init__(
        self,
        depth: int = 4,
        *,
        lr: float = 0.05,
        n_epochs: int = 2000,
        tau_start: float = 2.0,
        tau_end: float = 0.1,
        include_one: bool = True,
        device: str | None = None,
        seed: int | None = 0,
    ) -> None:
        self.depth = depth
        self.lr = lr
        self.n_epochs = n_epochs
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.include_one = include_one
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        self.model_: EMLTree | None = None
        self.result_: FitResult | None = None

    # -- fit ----------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor,
        *,
        feature_names: list[str] | None = None,
        log_every: int = 0,
    ) -> "EMLRegressor":
        if self.seed is not None:
            torch.manual_seed(self.seed)

        X_t = _to_tensor(X, device=self.device).float()
        y_t = _to_tensor(y, device=self.device).float().squeeze()
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)

        cfg = TreeConfig(
            depth=self.depth,
            n_features=X_t.shape[1],
            tau=self.tau_start,
            include_one=self.include_one,
        )
        model = EMLTree(cfg).to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()

        losses: list[float] = []
        for epoch in range(self.n_epochs):
            # Temperature anneal (geometric).
            frac = epoch / max(1, self.n_epochs - 1)
            model.cfg.tau = self.tau_start * (self.tau_end / self.tau_start) ** frac

            opt.zero_grad()
            pred = model(X_t)
            loss = loss_fn(pred, y_t)
            loss.backward()

            # Clip to keep Adam sane when eml blows gradients through exp.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            opt.step()

            losses.append(loss.item())
            if log_every and epoch % log_every == 0:
                print(f"[{epoch:5d}] loss={loss.item():.6f} tau={model.cfg.tau:.3f}")

        self.model_ = model

        tree = model.extract_tree(feature_names=feature_names)
        self.result_ = FitResult(
            final_loss=losses[-1],
            losses=losses,
            tree=tree,
            expression=tree.to_expr(),
        )
        return self

    # -- predict ------------------------------------------------------------

    def predict(self, X: np.ndarray | torch.Tensor) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("call fit() first")
        X_t = _to_tensor(X, device=self.device).float()
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(-1)
        with torch.no_grad():
            return self.model_(X_t).cpu().numpy()

    # -- inspection --------------------------------------------------------

    @property
    def expression_(self) -> str:
        return self.result_.expression if self.result_ else ""

    def summary(self) -> dict[str, Any]:
        if self.result_ is None:
            raise RuntimeError("call fit() first")
        return {
            "depth": self.depth,
            "final_loss": self.result_.final_loss,
            "tree_size": self.result_.tree.size() if self.result_.tree else 0,
            "expression": self.result_.expression,
        }


# ---------------------------------------------------------------------------

def _to_tensor(x, *, device: str) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(np.asarray(x), device=device)


__all__ = ["EMLRegressor", "FitResult"]
