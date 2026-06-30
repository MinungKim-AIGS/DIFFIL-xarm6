"""ZeroMQ transport for the DIFF-IL actor<->learner link.

Two independent channels (never on the 50 Hz control hot path — the actor sends
from a background thread, see actor_node.py):

  data  up   : laptop  PUSH  -> server  PULL   (trajectories)
  policy down: server   PUB  -> laptop   SUB    (actor weights)

Payloads are msgpack dicts. ndarrays are packed as {shape,dtype,data}; the big
image array is LZ4-compressed. Every message carries a `version` tag so the
learner can do off-policy bookkeeping and the actor can ignore stale weights.
"""
from __future__ import annotations

import time
import numpy as np
import msgpack
import lz4.frame
import zmq


# ----------------------------------------------------------------------------
# ndarray <-> portable dict
# ----------------------------------------------------------------------------
def pack_array(a: np.ndarray, compress: bool = False) -> dict:
    a = np.ascontiguousarray(a)
    raw = a.tobytes()
    if compress:
        raw = lz4.frame.compress(raw)
    return {"shape": list(a.shape), "dtype": str(a.dtype), "lz4": bool(compress), "data": raw}


def unpack_array(d: dict) -> np.ndarray:
    raw = lz4.frame.decompress(d["data"]) if d["lz4"] else d["data"]
    return np.frombuffer(raw, dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()


def encode_trajectory(traj: dict) -> bytes:
    """traj keys: obs,nobs,act,rew,don,ims,ids,step,n,version (ims compressed)."""
    out = {"n": int(traj.get("n", len(traj["act"]))), "version": int(traj.get("version", -1)),
           "ts": time.time(), "_arrays": {}}
    for k, v in traj.items():
        if isinstance(v, np.ndarray):
            out["_arrays"][k] = pack_array(v, compress=(k == "ims"))
    return msgpack.packb(out, use_bin_type=True)


def decode_trajectory(buf: bytes) -> dict:
    raw = msgpack.unpackb(buf, raw=False)
    traj = {k: unpack_array(v) for k, v in raw["_arrays"].items()}
    traj["n"] = raw["n"]; traj["version"] = raw["version"]; traj["ts"] = raw["ts"]
    return traj


def encode_weights(w: dict) -> bytes:
    """w: {version,int act_dim, norm_mean|None, norm_std|None, layers:[{W,b,act}]}."""
    out = {"version": int(w["version"]), "act_dim": int(w["act_dim"]),
           "norm_mean": None if w.get("norm_mean") is None else pack_array(np.asarray(w["norm_mean"])),
           "norm_std": None if w.get("norm_std") is None else pack_array(np.asarray(w["norm_std"])),
           "layers": [{"W": pack_array(L["W"]), "b": pack_array(L["b"]), "act": L["act"]}
                      for L in w["layers"]]}
    return msgpack.packb(out, use_bin_type=True)


def decode_weights(buf: bytes) -> dict:
    raw = msgpack.unpackb(buf, raw=False)
    return {"version": raw["version"], "act_dim": raw["act_dim"],
            "norm_mean": None if raw["norm_mean"] is None else unpack_array(raw["norm_mean"]),
            "norm_std": None if raw["norm_std"] is None else unpack_array(raw["norm_std"]),
            "layers": [{"W": unpack_array(L["W"]), "b": unpack_array(L["b"]), "act": L["act"]}
                       for L in raw["layers"]]}


# ----------------------------------------------------------------------------
# Data-up channel
# ----------------------------------------------------------------------------
class TrajectorySender:
    """Laptop side: PUSH trajectories to the server (non-blocking, drops if HWM)."""
    def __init__(self, host: str, port: int = 5557, hwm: int = 50):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUSH)
        self.sock.setsockopt(zmq.SNDHWM, hwm)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.connect(f"tcp://{host}:{port}")

    def send(self, traj: dict) -> bool:
        try:
            self.sock.send(encode_trajectory(traj), flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False  # server backpressure; caller may buffer/retry

    def close(self):
        self.sock.close(0)


class TrajectoryReceiver:
    """Server side: PULL trajectories (bind)."""
    def __init__(self, port: int = 5557, hwm: int = 200):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PULL)
        self.sock.setsockopt(zmq.RCVHWM, hwm)
        self.sock.bind(f"tcp://*:{port}")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def recv(self, timeout_ms: int = 0):
        if dict(self.poller.poll(timeout_ms)).get(self.sock) == zmq.POLLIN:
            return decode_trajectory(self.sock.recv())
        return None

    def drain(self, max_msgs: int = 1000):
        out = []
        while len(out) < max_msgs:
            t = self.recv(0)
            if t is None:
                break
            out.append(t)
        return out

    def close(self):
        self.sock.close(0)


# ----------------------------------------------------------------------------
# Policy-down channel
# ----------------------------------------------------------------------------
class WeightPublisher:
    """Server side: PUB latest actor weights (bind)."""
    def __init__(self, port: int = 5558):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.bind(f"tcp://*:{port}")

    def publish(self, weights: dict):
        self.sock.send(encode_weights(weights))

    def close(self):
        self.sock.close(0)


class WeightPuller:
    """Laptop side: SUB; keeps only the latest weights (CONFLATE), non-blocking."""
    def __init__(self, host: str, port: int = 5558):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt(zmq.SUBSCRIBE, b"")
        self.sock.setsockopt(zmq.CONFLATE, 1)   # keep only newest
        self.sock.setsockopt(zmq.RCVHWM, 1)
        self.sock.connect(f"tcp://{host}:{port}")
        self.poller = zmq.Poller()
        self.poller.register(self.sock, zmq.POLLIN)

    def latest(self):
        newest = None
        while dict(self.poller.poll(0)).get(self.sock) == zmq.POLLIN:
            newest = self.sock.recv()
        return decode_weights(newest) if newest is not None else None

    def close(self):
        self.sock.close(0)
