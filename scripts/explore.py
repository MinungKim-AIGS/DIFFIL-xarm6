"""Shared exploration + non-terminating safety for random data collection.

Used identically by the sim collector (collect_sim_demos.py, --mode random -> B^SR)
and the real collector (real_reach_collector.py -> B^TR) so the two random
distributions match (DIFF-IL then learns domain differences, not policy differences).

Solves two real-world collection problems:

  1. Episodes cut short by the safe-zone guard -> variable length.
     => SafetyRecovery is NON-TERMINATING: on a margin'd safe-zone exit it overrides
        the action with a home-ward move and the episode keeps running.

  2. White-noise random policy diffuses near home (poor coverage).
     => WaypointBabbler does joint-space GOAL BABBLING: it repeatedly picks a target
        joint config (from an FK-verified safe pool, or an OU fallback) and moves toward
        it, sweeping the reachable workspace. Staying within safe waypoints also means
        the arm rarely approaches the boundary, so problem 1 seldom triggers either.

Pure numpy (no mujoco/gym) -> runs on the robot laptop.
"""
from __future__ import annotations

import os
import numpy as np


class WaypointBabbler:
    """Joint-space goal babbling. `act(q)` returns a normalized action in [-1,1]^6
    that moves the joints toward the current target waypoint at <= action_scale rad/step."""

    def __init__(self, action_scale, joint_low, joint_high, home_qpos,
                 waypoints=None, hold_steps=(15, 50), reach_tol=0.04,
                 ou_theta=0.15, ou_sigma=0.3, ou_mix=0.2, seed=0):
        self.A = float(action_scale)
        self.jl = np.asarray(joint_low, np.float32)
        self.jh = np.asarray(joint_high, np.float32)
        self.home = np.asarray(home_qpos, np.float32)
        self.wp = None if waypoints is None or len(waypoints) == 0 else np.asarray(waypoints, np.float32)
        self.hold_lo, self.hold_hi = hold_steps
        self.tol = float(reach_tol)
        self.theta, self.sigma, self.ou_mix = ou_theta, ou_sigma, ou_mix
        self.rng = np.random.default_rng(seed)
        self._ou = np.zeros(6, np.float32)
        self._target = None
        self._hold = 0
        self._t = 0

    def _new_target(self, q):
        if self.wp is not None:
            self._target = self.wp[self.rng.integers(len(self.wp))].copy()
        else:
            # OU fallback (no pool): correlated jump within joint limits
            step = self.rng.uniform(-1.0, 1.0, size=6).astype(np.float32) * 0.8
            self._target = np.clip(np.asarray(q, np.float32) + step, self.jl, self.jh)
        self._hold = int(self.rng.integers(self.hold_lo, self.hold_hi + 1))
        self._t = 0

    def reset(self, q):
        self._ou[:] = 0.0
        self._new_target(q)

    def act(self, q) -> np.ndarray:
        q = np.asarray(q, np.float32)
        if self._target is None:
            self._new_target(q)
        # resample when the waypoint is reached or the hold elapses
        if (np.max(np.abs(self._target - q)) < self.tol) or (self._t >= self._hold):
            self._new_target(q)
        self._t += 1
        a = np.clip((self._target - q) / self.A, -1.0, 1.0)        # move toward target
        self._ou = (1.0 - self.theta) * self._ou + self.sigma * self.rng.standard_normal(6).astype(np.float32)
        a = np.clip(a + self.ou_mix * self._ou, -1.0, 1.0)         # small jitter for richer dynamics
        return a.astype(np.float32)


class SafetyRecovery:
    """Non-terminating safe-zone guard. `wrap(q, ee, action)` returns the action to
    actually apply: the proposed action while the TCP is inside a margin'd safe box,
    otherwise a home-ward override (episode keeps running)."""

    def __init__(self, safe_low, safe_high, home_qpos, action_scale, margin=0.03):
        self.lo = np.asarray(safe_low, np.float32) + margin
        self.hi = np.asarray(safe_high, np.float32) - margin
        self.home = np.asarray(home_qpos, np.float32)
        self.A = float(action_scale)

    def in_safe(self, ee) -> bool:
        ee = np.asarray(ee, np.float32)
        return bool(np.all(ee >= self.lo) and np.all(ee <= self.hi))

    def wrap(self, q, ee, action):
        if self.in_safe(ee):
            return np.asarray(action, np.float32), False
        a = np.clip((self.home - np.asarray(q, np.float32)) / self.A, -1.0, 1.0)
        return a.astype(np.float32), True


def load_waypoints(path):
    """Load a safe-waypoint pool saved by make_safe_waypoints.py. Returns [K,6] or None."""
    if not path or not os.path.exists(path):
        return None
    z = np.load(path)
    return z["waypoints"].astype(np.float32)
