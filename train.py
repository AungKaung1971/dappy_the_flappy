import os
import torch
import numpy as np
from tqdm import tqdm

from env.flappy_env import FlappyBirdEnv
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack
from model import CNNPolicy
from ppo import PPOAgent
from buffer import RolloutBuffer


def train():
    # Hyperparameters
    TOTAL_STEPS = 500_000
    SAVE_INTERVAL = 50_000
    ROLLOUT_STEPS = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------
    # INIT ENV + FIRST FRAME STACK
    # -----------------------------------------
    env = FlappyBirdEnv()
    env.reset()

    raw = env._get_frame()                # raw RGB frame
    processed = preprocess_frame(raw)     # grayscale, resize, normalize
    state = init_frame_stack(processed)   # 4-frame stack

    model = CNNPolicy(action_dim=2).to(device)
    ppo = PPOAgent(model=model, device=device)

    print("state shape:", state.shape)

    buffer = RolloutBuffer(
        buffer_size=ROLLOUT_STEPS,
        obs_shape=state.shape,
        device=device
    )

    total_steps = 0
    episode_rewards = []

    os.makedirs("checkpoints", exist_ok=True)
    pbar = tqdm(total=TOTAL_STEPS, desc="Training PPO Agent")

    # -----------------------------------------
    # MAIN TRAINING LOOP
    # -----------------------------------------
    while total_steps < TOTAL_STEPS:

        buffer.reset()

        for _ in range(ROLLOUT_STEPS):

            # 1️⃣ Use CURRENT STATE to get action
            action, logprob, value = ppo.act(state.astype(np.float32))

            # 2️⃣ Step environment
            next_obs, reward, done, info = env.step(action)

            # 3️⃣ Store transition USING CURRENT STATE
            buffer.add(state, action, logprob, reward, done, value)

            # 4️⃣ Build NEXT STATE from raw frame
            raw = env._get_frame()
            processed = preprocess_frame(raw)
            state = update_frame_stack(state, processed)

            
            
            # Print difference between consecutive frames debugggggggggg
            # d1 = np.sum(np.abs(state[0] - state[1]))
            # d2 = np.sum(np.abs(state[1] - state[2]))
            # d3 = np.sum(np.abs(state[2] - state[3]))

            # print("Frame diffs:", d1, d2, d3)


            total_steps += 1
            pbar.update(1)

            # 5️⃣ If episode ends: reset + rebuild frame stack
            if done:
                episode_rewards.append(info.get("episode_reward", 0))

                env.reset()
                raw = env._get_frame()
                processed = preprocess_frame(raw)
                state = init_frame_stack(processed)

            if total_steps >= TOTAL_STEPS:
                break

        # If rollout incomplete (unlikely), skip PPO update
        if buffer.ptr < buffer.buffer_size:
            continue

        # -----------------------------------------
        # PPO UPDATE
        # -----------------------------------------
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            _, last_value = model(state_t)
            last_value = float(last_value.squeeze(-1).item())

        ppo.update(buffer, last_value)

        # Track training progression
        if len(episode_rewards) > 0:
            avg_reward = float(np.mean(episode_rewards[-10:]))
            pbar.set_postfix({"avg_reward": round(avg_reward, 2)})

        # Save checkpoint
        if total_steps % SAVE_INTERVAL < ROLLOUT_STEPS:
            ckpt_path = f"checkpoints/ckpt_{total_steps}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    print("Training complete!")


if __name__ == "__main__":
    train()
