# src/rl/envs/routing_env.py
import gym
from gym import spaces
import numpy as np
import math
import random
from dataclasses import dataclass

@dataclass
class RoutingConfig:
    # cost knobs
    cost_ll: float = 0.08         # unit cost for low-latency path
    cost_batch: float = 0.02      # unit cost for batch path
    penalty_skip_pos: float = 1.0 # skip penalty if event was truly positive
    latency_base_ll: float = 0.5  # ms base per event (relative scale)
    latency_base_batch: float = 0.1
    q_ll_coef: float = 0.02       # latency growth per queued item
    q_batch_coef: float = 0.005
    gamma_decay: float = 0.99     # moving avg latency decay
    # data mix
    p_purchase: float = 0.15      # prior prob of true positive label
    noise: float = 0.05           # stochastic mismatch between score & label
    # rollout
    max_steps: int = 2048
    seed: int = 42

class RoutingEnv(gym.Env):
    """
    Simple simulator for routing:
      - We sample a "true" label y ~ Bernoulli(p_purchase) and a model score ~ y with noise
      - SHAP-ish flags: has_purchase, has_signup
      - Queues & moving avg latency form part of the state
      - Action: 0=batch, 1=low_latency, 2=skip
      - Reward: utility(score,y) - (latency + cost)
    """
    metadata = {"render.modes": []}

    def __init__(self, cfg: RoutingConfig | None = None):
        super().__init__()
        self.cfg = cfg or RoutingConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

        # obs: [score, hour01, hour24, has_purchase, has_signup, q_ll, q_batch, ma_latency]
        low = np.array([0., 0., 0., 0., 0., 0., 0., 0.], dtype=np.float32)
        high = np.array([1., 23., 1., 1., 1., 1e6, 1e6, 10.], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.q_ll = 0.0
        self.q_batch = 0.0
        self.ma_latency = 0.0
        obs = self._sample_obs()
        return obs, {}

    def _sample_event(self):
        # True label
        y = 1 if self.rng.random() < self.cfg.p_purchase else 0
        # Base score around y with noise
        mu = 0.85 if y == 1 else 0.15
        score = float(np.clip(self.rng.normal(mu, self.cfg.noise), 0, 1))
        # SHAP-ish: if score high, often 'purchase' important; mid scores may have 'signup'
        has_purchase = 1.0 if score > 0.7 and self.rng.random() < 0.8 else 0.0
        has_signup = 1.0 if (0.5 < score < 0.8 and self.rng.random() < 0.5) else 0.0
        # hour
        hour = int(self.rng.integers(0, 24))
        return score, hour, has_purchase, has_signup, y

    def _sample_obs(self):
        score, hour, has_purchase, has_signup, y = self._sample_event()
        self._cur = dict(score=score, hour=hour, has_purchase=has_purchase, has_signup=has_signup, y=y)
        obs = np.array([
            score,
            float(hour),
            1.0 if hour in (0, 6, 12, 18) else 0.0,  # simple cyclical hint
            has_purchase,
            has_signup,
            float(self.q_ll),
            float(self.q_batch),
            float(self.ma_latency),
        ], dtype=np.float32)
        return obs

    def step(self, action: int):
        self.t += 1
        score = self._cur["score"]
        y = self._cur["y"]

        # latency model
        if action == 1:  # low_latency
            lat = self.cfg.latency_base_ll + self.cfg.q_ll_coef * self.q_ll
            cost = self.cfg.cost_ll
            self.q_ll = max(0.0, self.q_ll + 1 - 2.5)  # service rate > arrival
        elif action == 0:  # batch
            lat = self.cfg.latency_base_batch + self.cfg.q_batch_coef * self.q_batch
            cost = self.cfg.cost_batch
            self.q_batch = max(0.0, self.q_batch + 1 - 1.2)
        else:  # skip
            lat = 0.0
            cost = 0.0

        # utility: reward for keeping positives on fast path, small utility even if batched
        util = 0.0
        if action == 1:
            util = score  # more reward for high score on fast path
        elif action == 0:
            util = 0.35 * score
        else:
            util = 0.0

        # penalty if we skipped a truly positive
        skip_pen = self.cfg.penalty_skip_pos if (action == 2 and y == 1) else 0.0

        reward = util - cost - 0.15 * lat - skip_pen

        # update moving average latency
        self.ma_latency = self.cfg.gamma_decay * self.ma_latency + (1 - self.cfg.gamma_decay) * lat

        terminated = self.t >= self.cfg.max_steps
        truncated = False
        info = {"score": score, "y": y, "latency": lat, "q_ll": self.q_ll, "q_batch": self.q_batch}
        obs = self._sample_obs()
        return obs, float(reward), terminated, truncated, info
