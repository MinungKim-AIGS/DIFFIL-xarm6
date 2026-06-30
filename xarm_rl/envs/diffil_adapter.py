"""SimDiffilEnv — the SOURCE environment for DIFF-IL, wrapping the existing
gymnasium ``XArm6ReachEnv`` in the *old-gym* + visual interface that DIFF-IL's
Sampler expects.

    reset() -> obs(21)
    step(act) -> (obs, rew, done, info)          # collapses terminated/truncated
    get_ims() -> [past_frames, H, W, C] uint8
    action_space (gymnasium Box: .sample(), .shape), seed()

Selectable camera viewpoint (front / ob_b / ob_c / ob_d / topdown) passes straight
through to the env, so source datasets can be rendered under whichever camera you
later align to the real RealSense.

Used to collect the source datasets B^SE (trained policy) and B^SR (random) in sim,
and (optionally) as a "local_sim" target feed for offline testing of the learner.
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import xarm_rl  # noqa: F401  (registers XArm6Reach-v0)


class SimDiffilEnv:
    def __init__(self, env_id: str = "XArm6Reach-v0", render_camera: str = "front",
                 action_rate_penalty: float = 0.0, domain_rand: bool = False, seed: int = 0):
        self.env = gym.make(env_id, render_mode="rgb_array",
                            render_camera=render_camera,
                            action_rate_penalty=action_rate_penalty,
                            domain_rand=domain_rand)
        self.action_space = self.env.action_space        # gymnasium Box (.sample/.shape)
        self.observation_space = self.env.observation_space
        self._seed = seed
        self.spec_max_steps = self.env.spec.max_episode_steps

    def seed(self, s):
        self._seed = s

    def reset(self):
        obs, _ = self.env.reset(seed=self._seed)
        self._seed = None
        return obs

    def step(self, act):
        obs, rew, terminated, truncated, info = self.env.step(np.asarray(act, np.float32))
        return obs, float(rew), bool(terminated or truncated), info

    def get_ims(self) -> np.ndarray:
        return self.env.unwrapped.get_ims()

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()
