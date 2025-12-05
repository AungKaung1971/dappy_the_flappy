import os
import torch
import numpy as np
from tqdm import tqdm

from env.flappy_env import FlappyBirdEnv
from preprocess import preprocess_frame
from model import CNNPolicy
from ppo import PPOAgent
from buffer import RolloutBuffer


def train():
    # Hyperparameters
    TOTAL_STEPS = 1_000_000
    SAVE_INTERVAL = 50_000
    ROLLOUT_STEPS = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = FlappyBirdEnv()
    model = CNNPolicy(action_dim=2).to(device)
    ppo = PPOAgent(model=model, device=device)

    obs = env.reset()

    print("obs shape:", obs.shape)

    state = obs.astype(np.float32)

    buffer = RolloutBuffer(
        buffer_size=ROLLOUT_STEPS,
        obs_shape=state.shape,
        device=device
    )

    total_steps = 0
    episode_rewards = []

    os.makedirs("checkpoints", exist_ok=True)
    pbar = tqdm(total=TOTAL_STEPS, desc="Training PPO Agent")

    while total_steps < TOTAL_STEPS:

        buffer.reset()

        for _ in range(ROLLOUT_STEPS):

            action, logprob, value = ppo.act(state.astype(np.float32))

            next_obs, reward, done, info = env.step(action)
            next_state = next_obs.astype(np.float32)

            idx = buffer.ptr
            buffer.observations[idx] = state
            buffer.actions[idx] = action
            buffer.logprobs[idx] = logprob
            buffer.rewards[idx] = reward
            buffer.dones[idx] = float(done)
            buffer.values[idx] = value
            buffer.ptr += 1

            state = next_state
            total_steps += 1
            pbar.update(1)

            if done:
                episode_rewards.append(info.get("episode_reward", 0))

                obs = env.reset()
                state = obs.astype(np.float32)

            if total_steps >= TOTAL_STEPS:
                break

        if buffer.ptr < buffer.buffer_size:
            break

        with torch.no_grad():
            state_t = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(device)
            _, last_value = model(state_t)
            last_value = float(last_value.squeeze(-1).item())

        ppo.update(buffer, last_value)

        if len(episode_rewards) > 0:
            avg_reward = float(np.mean(episode_rewards[-10:]))
            pbar.set_postfix({
                "avg_reward": round(avg_reward, 2)
            })

        if total_steps % SAVE_INTERVAL < ROLLOUT_STEPS:
            ckpt_path = f"checkpoints/ckpt_{total_steps}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
