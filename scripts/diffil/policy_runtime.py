"""TF-free numpy forward of the DIFF-IL SAC actor (runs on the robot laptop).

Mirrors sac_models.StochasticActor exactly:
    out = x; for layer in dense_layers: out = act(layer(out))
    mean, log_std = split(out, 2)         # last layer is linear, units = 2*act_dim
    deterministic action = tanh(mean)     # what get_action(noise=0) returns
Optional obs normalization (obs - mean)/(std + 1e-7), matching _preprocess_obs.

Weights come from weight_io.export_actor (a list of {W,b,act} dicts), so this
forward is generic over architecture — no hard-coded layer sizes.
"""
from __future__ import annotations

import threading
import numpy as np


def _activate(x: np.ndarray, name: str) -> np.ndarray:
    if name in ("linear", "None", None):
        return x
    if name == "relu":
        return np.maximum(x, 0.0)
    if name == "tanh":
        return np.tanh(x)
    if name == "elu":
        return np.where(x > 0, x, np.expm1(x))
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    raise ValueError(f"unsupported activation {name!r}")


class NumpyActor:
    """Thread-safe numpy actor with hot-swappable weights."""

    def __init__(self, weights: dict):
        self._lock = threading.Lock()
        self.version = -1
        self._load(weights)

    def _load(self, w: dict):
        self.layers = w["layers"]                       # [{W:[in,out], b:[out], act}]
        self.act_dim = int(w["act_dim"])
        self.norm_mean = w.get("norm_mean")
        self.norm_std = w.get("norm_std")
        self.version = int(w.get("version", self.version))

    def update_weights(self, w: dict) -> bool:
        """Hot-swap; ignores stale versions. Returns True if applied."""
        with self._lock:
            if int(w.get("version", -1)) <= self.version:
                return False
            self._load(w)
            return True

    def _forward_mean(self, obs: np.ndarray) -> np.ndarray:
        x = obs.astype(np.float32)
        if self.norm_mean is not None and self.norm_std is not None:
            x = (x - self.norm_mean) / (self.norm_std + 1e-7)
        for L in self.layers:
            x = _activate(x @ L["W"] + L["b"], L["act"])
        mean = x[..., : self.act_dim]                   # first half = mean
        return mean

    def get_action(self, state: np.ndarray, noise_stddev: float = 0.0) -> np.ndarray:
        """state: [N] or [1,N]. Returns action in [-1,1]^act_dim (deterministic
        tanh(mean); add exploration via noise_stddev on the pre-tanh mean)."""
        with self._lock:
            obs = np.atleast_2d(state).astype(np.float32)
            mean = self._forward_mean(obs)
            if noise_stddev and noise_stddev > 0.0:
                mean = mean + np.random.randn(*mean.shape).astype(np.float32) * noise_stddev
            act = np.tanh(mean)
        return act[0] if np.ndim(state) == 1 else act


def random_actor(obs_dim: int = 21, act_dim: int = 6,
                 hidden=(256, 256), version: int = 0, seed: int = 0) -> dict:
    """A random-weight actor in export format — for dry-runs / tests without TF."""
    rng = np.random.default_rng(seed)
    dims = [obs_dim, *hidden, act_dim * 2]
    acts = ["relu"] * len(hidden) + ["linear"]
    layers = []
    for i, a in zip(range(len(dims) - 1), acts):
        layers.append({"W": (rng.standard_normal((dims[i], dims[i + 1])) * 0.1).astype(np.float32),
                       "b": np.zeros(dims[i + 1], np.float32), "act": a})
    return {"version": version, "act_dim": act_dim,
            "norm_mean": None, "norm_std": None, "layers": layers}
