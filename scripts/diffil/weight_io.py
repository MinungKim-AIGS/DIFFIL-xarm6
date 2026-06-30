"""Export the DIFF-IL SAC actor to a framework-neutral dict (server, TF) and
save/load it (so the laptop can run it via policy_runtime without TensorFlow).

Export format (consumed by policy_runtime.NumpyActor / comm.encode_weights):
    {version:int, act_dim:int, norm_mean:np|None, norm_std:np|None,
     layers:[{W:[in,out] f32, b:[out] f32, act:str}, ...]}

`export_actor` reads a sac_models.StochasticActor (its Keras Dense layers).
Run ONLY on the server (needs TF + a built actor). The laptop side uses the
pure-numpy helpers (save/load) and never imports TF.
"""
from __future__ import annotations

import numpy as np


def export_actor(actor, version: int = 0) -> dict:
    """actor: sac_models.StochasticActor (or SAC._act). Requires TF at call time.

    The actor must have been *called once* (built) so layer weights exist.
    """
    layers = []
    for layer in actor._act_layers:                      # list of keras Dense
        W, b = layer.get_weights()                       # W:[in,out], b:[out]
        act = getattr(layer.activation, "__name__", "linear")
        layers.append({"W": np.asarray(W, np.float32),
                       "b": np.asarray(b, np.float32), "act": act})
    out_dim = layers[-1]["b"].shape[0]
    nm = getattr(actor, "_norm_mean", None)
    ns = getattr(actor, "_norm_stddev", None)
    return {"version": int(version), "act_dim": int(out_dim // 2),
            "norm_mean": None if nm is None else np.asarray(nm, np.float32),
            "norm_std": None if ns is None else np.asarray(ns, np.float32),
            "layers": layers}


def save_weights(weights: dict, path: str):
    """Save export dict to .npz (portable; no TF needed to load)."""
    flat = {"version": weights["version"], "act_dim": weights["act_dim"],
            "n_layers": len(weights["layers"])}
    if weights.get("norm_mean") is not None:
        flat["norm_mean"] = weights["norm_mean"]
        flat["norm_std"] = weights["norm_std"]
    for i, L in enumerate(weights["layers"]):
        flat[f"W{i}"] = L["W"]; flat[f"b{i}"] = L["b"]; flat[f"act{i}"] = np.array(L["act"])
    np.savez(path, **flat)


def load_weights(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    n = int(z["n_layers"])
    layers = [{"W": z[f"W{i}"], "b": z[f"b{i}"], "act": str(z[f"act{i}"])} for i in range(n)]
    return {"version": int(z["version"]), "act_dim": int(z["act_dim"]),
            "norm_mean": z["norm_mean"] if "norm_mean" in z.files else None,
            "norm_std": z["norm_std"] if "norm_std" in z.files else None,
            "layers": layers}
