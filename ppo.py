import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from buffer import RolloutBuffer


class PPOAgent:
    def __init__(self, model: nn.Module,
                 device=torch.device,
                 lr: float = 2.5e-4,
                 gamma: float = 0.99,
                 lam: float = 0.95,
                 clip_range: float = 0.1,
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 n_epochs: int = 4,
                 batch_size: int = 64):

        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.lam = lam
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.n_epochs = n_epochs
        self.batch_size = batch_size

    def act(self, obs_np):
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits, value = self.model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)

        action_int = int(action.item())
        logprob_float = float(logprob.item())
        value_float = float(value.squeeze(-1).item())

        return action_int, logprob_float, value_float

    def _evaluate(self, obs_batch, actions_batch):
        logits, values = self.model(obs_batch)
        dist = Categorical(logits=logits)

        logprobs = dist.log_prob(actions_batch)
        entropy = dist.entropy()

        values = values.squeeze(-1)

        return logprobs, entropy, values

    def update(self, buffer: RolloutBuffer, last_value: float):

        buffer.compute_returns_and_advantages(
            last_value=last_value,
            gamma=self.gamma,
            lam=self.lam,
        )

        observations, actions, old_logprobs, returns, advantages, old_values = buffer.get(
            normalize_advantages=True
        )

        num_steps = observations.shape[0]

        for _ in range(self.n_epochs):
            indices = torch.randperm(num_steps, device=self.device)

            for start in range(0, num_steps, self.batch_size):
                end = start + self.batch_size
                mb_idx = indices[start:end]

                mb_obs = observations[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logprobs = old_logprobs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                new_logprobs, entropy, values = self._evaluate(
                    mb_obs, mb_actions)

                log_ratio = new_logprobs - mb_old_logprobs
                ratio = torch.exp(log_ratio)

                unclipped = ratio * mb_advantages
                clipped = torch.clamp(
                    ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * mb_advantages

                policy_loss = -torch.min(unclipped, clipped).mean()

                value_loss = F.mse_loss(values, mb_returns)

                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * \
                    value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
