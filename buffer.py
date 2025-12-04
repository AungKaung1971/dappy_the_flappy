import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.obs_shape = obs_shape
        self.device = device

        self.observations = np.zeros(
            (buffer_size, *obs_shape), dtype=np.float32)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.logprobs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)

        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)

        self.ptr = 0  # pointer for storing transitions

    def compute_returns_and_advantages(self, last_value, gamma=0.99, lam=0.95):
        if self.ptr != self.buffer_size:
            raise ValueError(
                f"Buffer not full: ptr={self.ptr}, buffer_size={self.buffer_size}. "
            )

        advantages = np.zeros_like(self.rewards)
        last_advantage = 0.0

        for t in reversed(range(self.buffer_size)):
            if t == self.buffer_size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_advantage = delta + gamma * lam * next_non_terminal * last_advantage
            advantages[t] = last_advantage

        self.advantages = advantages
        self.returns = advantages + self.values

    def get(self, normalize_advantages=True):
        if self.ptr != self.buffer_size:
            raise ValueError(
                f"Trying to get data from buffer, but ptr={self.ptr}"
            )

        obs = torch.from_numpy(self.observations).to(self.device)
        actions = torch.from_numpy(self.actions).to(self.device)
        old_logprobs = torch.from_numpy(self.logprobs).to(self.device)
        returns = torch.from_numpy(self.returns).to(self.device)
        advantages = torch.from_numpy(self.advantages).to(self.device)
        values = torch.from_numpy(self.values).to(self.device)

        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)

        self.ptr = 0
        return obs, actions, old_logprobs, returns, advantages, values

    def reset(self):
        self.ptr = 0
