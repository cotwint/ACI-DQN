"""
src/rl/dqn_agent.py
-------------------
Minimal-but-complete Deep Q-Network agent.

Components
----------
* QNetwork    -- MLP mapping state -> Q-values over discrete actions
* ReplayBuffer -- uniform experience replay
* DQNAgent    -- epsilon-greedy action selection, target network, soft updates

Used by all three RL variants:
    * Ordinary DQN   (no extra inputs)
    * ACI-DQN        (state augmented with ACI intervals)
    * DtACI-DQN      (state augmented with DtACI intervals + shield)

The state-feature dimension is provided externally so the same class
can serve any of the variants without modification.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: List[int]):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[Transition] = deque(maxlen=capacity)

    def push(self, tr: Transition) -> None:
        self.buf.append(tr)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = np.stack([b.state for b in batch]).astype(np.float32)
        a = np.array([b.action for b in batch], dtype=np.int64)
        r = np.array([b.reward for b in batch], dtype=np.float32)
        s2 = np.stack([b.next_state for b in batch]).astype(np.float32)
        d = np.array([b.done for b in batch], dtype=np.float32)
        return s, a, r, s2, d

    def __len__(self) -> int:
        return len(self.buf)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden: List[int],
                 lr: float = 5e-4,
                 gamma: float = 0.99,
                 batch_size: int = 128,
                 replay_size: int = 50000,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 0.995,
                 target_update_interval: int = 200,
                 learning_starts: int = 200,
                 reward_scale: float = 0.01,
                 max_grad_norm: float = 5.0,
                 device: str | None = None):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.gamma = float(gamma)
        self.batch_size = int(batch_size)
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.target_update_interval = int(target_update_interval)
        self.reward_scale = float(reward_scale)
        self.max_grad_norm = float(max_grad_norm)

        self.device = torch.device(device or
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        self.q = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.q_target = QNetwork(state_dim, action_dim, hidden).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad_(False)
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)

        self.buffer = ReplayBuffer(replay_size)
        self.learning_starts = int(learning_starts)
        self._grad_steps = 0

    # ------------------------------------------------------------------
    @torch.no_grad()
    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        s = torch.from_numpy(np.asarray(state, dtype=np.float32))[None, :].to(self.device)
        return int(self.q(s).argmax(dim=1).item())

    # ------------------------------------------------------------------
    def push(self, s, a, r, s2, done):
        self.buffer.push(Transition(np.asarray(s, dtype=np.float32),
                                    int(a),
                                    float(r) * self.reward_scale,
                                    np.asarray(s2, dtype=np.float32),
                                    bool(done)))

    # ------------------------------------------------------------------
    def train_step(self) -> float | None:
        if len(self.buffer) < max(self.batch_size, self.learning_starts):
            return None
        s, a, r, s2, d = self.buffer.sample(self.batch_size)
        s = torch.from_numpy(s).to(self.device)
        a = torch.from_numpy(a).to(self.device)
        r = torch.from_numpy(r).to(self.device)
        s2 = torch.from_numpy(s2).to(self.device)
        d = torch.from_numpy(d).to(self.device)

        q_sa = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            target = r + (1.0 - d) * self.gamma * q_next
        loss = F.smooth_l1_loss(q_sa, target)

        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.max_grad_norm)
        self.optim.step()
        self._grad_steps += 1
        if self._grad_steps % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    # ------------------------------------------------------------------
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy_net": self.q.state_dict(),
            "target_net": self.q_target.state_dict(),
            "optimizer": self.optim.state_dict(),
            "grad_steps": self._grad_steps,
        }, path)

    # ------------------------------------------------------------------
    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q.load_state_dict(ckpt["policy_net"])
        self.q_target.load_state_dict(ckpt["target_net"])
        self.optim.load_state_dict(ckpt["optimizer"])
        self._grad_steps = ckpt.get("grad_steps", 0)
        self.q.eval()
        self.q_target.eval()
