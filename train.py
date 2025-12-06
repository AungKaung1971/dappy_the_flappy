import os
import torch
import numpy as np
from tqdm import tqdm

from env.flappy_env import FlappyBirdEnv
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack
from model import CNNPolicy
from ppo import PPOAgent
from buffer import RolloutBuffer
from logger import MetricLogger


def train():

    # ------------------------------------------------
    # HYPERPARAMETERS
    # ------------------------------------------------
    TOTAL_STEPS = 500_000
    SAVE_INTERVAL = 50_000
    ROLLOUT_STEPS = 2048

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------
    # ENV + INITIAL FRAME STACK
    # ------------------------------------------------
    env = FlappyBirdEnv()
    env.reset()

    raw = env._get_frame()
    processed = preprocess_frame(raw)
    state = init_frame_stack(processed)   # shape (4, 84, 84)

    model = CNNPolicy(action_dim=2).to(device)
    ppo = PPOAgent(model=model, device=device)

    print("state shape:", state.shape)

    # ------------------------------------------------
    # ROLLOUT BUFFER
    # ------------------------------------------------
    buffer = RolloutBuffer(
        buffer_size=ROLLOUT_STEPS,
        obs_shape=state.shape,
        device=device
    )

    # ------------------------------------------------
    # LOGGER INITIALIZATION
    # ------------------------------------------------
    logger = MetricLogger(base_log_dir="logs")

    logger.save_hparams({
        "TOTAL_STEPS": TOTAL_STEPS,
        "SAVE_INTERVAL": SAVE_INTERVAL,
        "ROLLOUT_STEPS": ROLLOUT_STEPS,
        "learning_rate": ppo.optimizer.param_groups[0]["lr"],
        "gamma": getattr(ppo, "gamma", None),
        "gae_lambda": getattr(ppo, "gae_lambda", None),
        "clip_range": getattr(ppo, "clip_range", None),
        "entropy_coef": getattr(ppo, "entropy_coef", None),
        "value_loss_coef": getattr(ppo, "value_loss_coef", None),
    })

    os.makedirs("checkpoints", exist_ok=True)

    # ------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------
    total_steps = 0
    episode_rewards = []

    pbar = tqdm(total=TOTAL_STEPS, desc="Training PPO Agent")

    while total_steps < TOTAL_STEPS:

        buffer.reset()

        # --------------------------------------------
        # COLLECT ROLLOUT
        # --------------------------------------------
        for _ in range(ROLLOUT_STEPS):

            # Decide action from current state
            action, logprob, value = ppo.act(state.astype(np.float32))

            # Environment step
            next_obs, reward, done, info = env.step(action)

            # Save transition
            buffer.add(state, action, logprob, reward, done, value)

            # Build next state (from raw frame)
            raw = env._get_frame()
            processed = preprocess_frame(raw)
            state = update_frame_stack(state, processed)

            total_steps += 1
            pbar.update(1)

            # If episode ended — reset env + frame stack
            if done:
                ep_r = info.get("episode_reward", 0)
                episode_rewards.append(ep_r)

                env.reset()
                raw = env._get_frame()
                processed = preprocess_frame(raw)
                state = init_frame_stack(processed)

            if total_steps >= TOTAL_STEPS:
                break

        # Skip PPO update if rollout incomplete
        if buffer.ptr < buffer.buffer_size:
            continue

        # --------------------------------------------
        # FINAL VALUE FOR GAE
        # --------------------------------------------
        with torch.no_grad():
            s = torch.tensor(
                state, dtype=torch.float32).unsqueeze(0).to(device)
            _, last_value = model(s)
            last_value = float(last_value.squeeze().item())

        # --------------------------------------------
        # PPO UPDATE
        # --------------------------------------------
        update_stats = ppo.update(buffer, last_value)

        # --------------------------------------------
        # TRAINING METRICS
        # --------------------------------------------
        if len(episode_rewards) > 0:
            avg_reward = float(np.mean(episode_rewards[-10:]))
        else:
            avg_reward = 0.0

        pbar.set_postfix({"avg_reward": round(avg_reward, 2)})

        # --------------------------------------------
        # LOG TO DASHBOARD FILES
        # --------------------------------------------
        logger.log(
            step=total_steps,
            avg_reward=avg_reward,
            latest_episode_reward=episode_rewards[-1] if episode_rewards else 0,

            # PPO stats (safe even if missing)
            policy_loss=update_stats.get("policy_loss") if isinstance(
                update_stats, dict) else None,
            value_loss=update_stats.get("value_loss") if isinstance(
                update_stats, dict) else None,
            entropy=update_stats.get("entropy") if isinstance(
                update_stats, dict) else None,
            clip_fraction=update_stats.get("clip_fraction") if isinstance(
                update_stats, dict) else None,
            approx_kl=update_stats.get("approx_kl") if isinstance(
                update_stats, dict) else None,

            learning_rate=ppo.optimizer.param_groups[0]["lr"],
        )

        # --------------------------------------------
        # SAVE CHECKPOINT
        # --------------------------------------------
        if total_steps % SAVE_INTERVAL < ROLLOUT_STEPS:
            ckpt_path = f"checkpoints/ckpt_{total_steps}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"\nSaved checkpoint: {ckpt_path}")

    # ------------------------------------------------
    # END TRAINING
    # ------------------------------------------------
    logger.close()
    print("Training complete!")


if __name__ == "__main__":
    train()
