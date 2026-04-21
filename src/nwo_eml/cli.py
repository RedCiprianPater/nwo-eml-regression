"""
Command-line interface for `nwo-eml`.

The TypeScript MCP server shells out to this CLI. Contract: read a JSON
blob from stdin, write a JSON blob to stdout. Any stderr output is
captured by the caller as a diagnostic.

JSON request schema:

    {
        "data":           [[x00, x01, ...], ...],  # features, shape (n, d)
        "target":         [y0, y1, ...],           # shape (n,)
        "feature_names":  ["t", "load", ...],      # optional
        "depth":          4,
        "n_epochs":       2000,
        "lr":             0.05,
        "seed":           0
    }

JSON response schema:

    {
        "ok":             true,
        "expression":     "...",       # raw eml(...) form
        "simplified":     "...",       # sympy-simplified form (or same as above)
        "final_loss":     0.0001,
        "tree_size":      31,
        "depth":          4
    }
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback

import numpy as np

from .regressor import EMLRegressor
from .simplify import simplify_tree


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nwo-eml")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_reg = sub.add_parser("regress", help="fit an EML tree from JSON on stdin")
    p_reg.add_argument("--pretty", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "regress":
        return _cmd_regress(args)
    return 2


def _cmd_regress(args) -> int:
    try:
        req = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        _emit_error(f"invalid JSON on stdin: {e}")
        return 1

    try:
        X = np.asarray(req["data"], dtype=float)
        y = np.asarray(req["target"], dtype=float)
    except Exception as e:  # noqa: BLE001
        _emit_error(f"failed to parse data/target: {e}")
        return 1

    reg = EMLRegressor(
        depth=int(req.get("depth", 4)),
        n_epochs=int(req.get("n_epochs", 2000)),
        lr=float(req.get("lr", 0.05)),
        seed=req.get("seed", 0),
    )

    try:
        reg.fit(
            X, y,
            feature_names=req.get("feature_names"),
        )
    except Exception as e:  # noqa: BLE001
        _emit_error(f"fit failed: {e}", tb=traceback.format_exc())
        return 1

    summary = reg.summary()
    out = {
        "ok": True,
        "expression": summary["expression"],
        "simplified": simplify_tree(reg.result_.tree) if reg.result_ else "",
        "final_loss": summary["final_loss"],
        "tree_size": summary["tree_size"],
        "depth": summary["depth"],
        "paper_reference": "Odrzywołek, arXiv:2603.21852",
    }

    json.dump(out, sys.stdout, indent=2 if args.pretty else None)
    sys.stdout.write("\n")
    return 0


def _emit_error(msg: str, tb: str | None = None) -> None:
    payload = {"ok": False, "error": msg}
    if tb:
        payload["traceback"] = tb
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
