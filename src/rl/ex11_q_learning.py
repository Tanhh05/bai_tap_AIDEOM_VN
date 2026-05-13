from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces


ACTIONS = {
    0: ("Truyen thong", np.array([0.70, 0.10, 0.10, 0.10])),
    1: ("Can bang", np.array([0.40, 0.25, 0.15, 0.20])),
    2: ("So hoa nhanh", np.array([0.25, 0.45, 0.15, 0.15])),
    3: ("AI dan dat", np.array([0.20, 0.20, 0.45, 0.15])),
    4: ("Bao trum", np.array([0.30, 0.20, 0.10, 0.40])),
}


class VietnamEconomyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.MultiDiscrete([3, 3, 3, 3])
        self.T = 10

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.array(options.get("state", [1, 1, 0, 1]) if options else [1, 1, 0, 1], dtype=int)
        self.t = 0
        self.K, self.D, self.AI, self.H, self.prevY = 27500.0, 20.3, 86.0, 30.0, 12847.6
        return self.state.copy(), {}

    def step(self, action):
        a = ACTIONS[int(action)][1]
        budget = 1000.0
        self.K += a[0] * budget
        self.D += a[1] * budget / 100
        self.AI += a[2] * budget / 20
        self.H += a[3] * budget / 200
        Y = self.K**0.33 * 54.0**0.42 * self.D**0.10 * self.AI**0.08 * self.H**0.07
        gdp_gain = (Y - self.prevY / 400) / 10
        unemployment = max(0.0, 0.16 * a[2] - 0.12 * a[3] - 0.02 * a[1])
        cyber = 0.12 * a[2] + 0.04 * a[1] - 0.05 * a[3]
        emission = 0.10 * a[0] + 0.08 * a[2]
        reward = 0.40 * gdp_gain - 0.25 * unemployment - 0.20 * cyber - 0.15 * emission
        self.prevY = Y * 400
        self.state = np.array([
            2 if gdp_gain > 0.8 else 1 if gdp_gain > 0.4 else 0,
            2 if self.D > 27 else 1 if self.D > 20 else 0,
            2 if self.AI > 115 else 1 if self.AI > 95 else 0,
            2 if unemployment > 0.04 else 1 if unemployment > 0.015 else 0,
        ])
        self.t += 1
        return self.state.copy(), float(reward), self.t >= self.T, False, {}


def train(episodes: int = 10000, seed: int = 42) -> tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    env = VietnamEconomyEnv()
    Q = np.zeros((3, 3, 3, 3, 5))
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        total = 0.0
        eps = max(0.05, 1.0 - ep / 5000)
        while True:
            a = env.action_space.sample() if rng.random() < eps else int(np.argmax(Q[tuple(s)]))
            s2, r, done, _, _ = env.step(a)
            Q[tuple(s) + (a,)] += 0.1 * (r + 0.95 * Q[tuple(s2)].max() - Q[tuple(s) + (a,)])
            s = s2
            total += r
            if done:
                break
        rewards.append({"episode": ep + 1, "reward": total})
    return Q, pd.DataFrame(rewards)


def evaluate_policy(Q: np.ndarray | None = None, fixed_action: int | None = None, random_policy: bool = False, episodes: int = 200) -> float:
    rng = np.random.default_rng(7)
    env = VietnamEconomyEnv()
    vals = []
    for _ in range(episodes):
        s, _ = env.reset()
        total = 0.0
        while True:
            if random_policy:
                a = int(rng.integers(0, 5))
            elif fixed_action is not None:
                a = fixed_action
            else:
                a = int(np.argmax(Q[tuple(s)]))
            s, r, done, _, _ = env.step(a)
            total += r
            if done:
                break
        vals.append(total)
    return float(np.mean(vals))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out = root / "results"
    out.mkdir(exist_ok=True)
    Q, curve = train()
    curve.to_csv(out / "ex11_learning_curve.csv", index=False)
    plt.figure(figsize=(8, 4))
    plt.plot(curve["episode"], curve["reward"].rolling(200, min_periods=1).mean())
    plt.title("Exercise 11 - Q-learning reward curve")
    plt.tight_layout()
    plt.savefig(out / "ex11_learning_curve.png", dpi=150)
    plt.close()
    states = {"VN_2026": [1, 1, 0, 1], "low_low_highU": [0, 0, 0, 2], "high_ai_lowU": [2, 2, 2, 0], "medium_all": [1, 1, 1, 1], "digital_low": [1, 0, 1, 1]}
    policy_rows = []
    for name, s in states.items():
        a = int(np.argmax(Q[tuple(s)]))
        policy_rows.append({"state_name": name, "state": str(s), "action_id": a, "action_name": ACTIONS[a][0]})
    pd.DataFrame(policy_rows).to_csv(out / "ex11_policy_samples.csv", index=False)
    pd.DataFrame([
        {"policy": "q_learning", "avg_reward": evaluate_policy(Q=Q)},
        {"policy": "always_balanced_a1", "avg_reward": evaluate_policy(fixed_action=1)},
        {"policy": "always_ai_a3", "avg_reward": evaluate_policy(fixed_action=3)},
        {"policy": "random", "avg_reward": evaluate_policy(random_policy=True)},
    ]).to_csv(out / "ex11_policy_comparison.csv", index=False)
    print("=== Exercise 11 Completed ===")
    print(pd.DataFrame(policy_rows).to_string(index=False))


if __name__ == "__main__":
    main()
