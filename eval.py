import cv2
import torch
import numpy as np

from env.flappy_env import FlappyBirdEnv
from model import CNNPolicy


def evaluate(checkpoint_path, output_video="evaluation.mp4", max_steps=5000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = CNNPolicy(action_dim=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # ---------------------------
    # INIT ENV (returns (4,84,84))
    # ---------------------------
    env = FlappyBirdEnv()
    state = env.reset().astype(np.float32)

    # -----------------------------------------
    # DEBUG: Check what the model outputs at start
    # -----------------------------------------
    print("DEBUG: STATE SHAPE:", state.shape)

    inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, value = model(inp)

    print("DEBUG: LOGITS:", logits.cpu().numpy())
    print("DEBUG: VALUE HEAD:", value.cpu().numpy())
    print("DEBUG: ACTION SELECTED:", torch.argmax(logits, dim=-1).item())
    print("-----------------------------------------")

    # Get raw RGB frame for video size
    raw = env._get_frame()
    h, w, _ = raw.shape

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        60,
        (w, h)
    )

    done = False
    steps = 0

    while not done and steps < max_steps:

        # Convert to tensor
        inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Deterministic action (no sampling!)
        with torch.no_grad():
            logits, _ = model(inp)
            action = torch.argmax(logits, dim=-1).item()

        # Step the environment – THIS returns the NEXT stacked state
        next_state, reward, done, info = env.step(action)

        # Update state EXACTLY like training
        state = next_state.astype(np.float32)

        # Write raw frame for video
        raw = env._get_frame()
        out.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))

        steps += 1

    out.release()
    print("Video saved:", output_video)
    print("Final score:", info.get("score", "N/A"))


if __name__ == "__main__":
    evaluate("checkpoints/ckpt_251904.pt")
