import cv2
import torch
import argparse
import numpy as np

from env.flappy_env import FlappyBirdEnv
from model import CNNPolicy
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack


def evaluate(checkpoint_path, output_video="evaluation.mp4", max_steps=5000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------
    # LOAD MODEL
    # -----------------------------------
    model = CNNPolicy(action_dim=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # -----------------------------------
    # INIT ENV + INITIAL FRAME STACK
    # -----------------------------------
    env = FlappyBirdEnv()
    env.reset()

    raw = env._get_frame()
    processed = preprocess_frame(raw)
    state = init_frame_stack(processed).astype(np.float32)

    # -----------------------------------
    # INIT VIDEO WRITER
    # -----------------------------------
    h, w, _ = raw.shape
    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        60,
        (w, h)
    )

    done = False
    steps = 0
    info = {}

    # -----------------------------------
    # MAIN LOOP
    # -----------------------------------
    while not done and steps < max_steps:

        # Convert to tensor
        inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits, _ = model(inp)

            # 🔥 PPO STOCHASTIC SAMPLING — MATCHES TRAINING
            probs = torch.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()

        # Step environment
        _, _, done, info = env.step(action)

        # New frame
        raw = env._get_frame()
        processed = preprocess_frame(raw)
        state = update_frame_stack(state, processed).astype(np.float32)

        # Write to video
        out.write(cv2.cvtColor(raw, cv2.COLOR_RGB2BGR))

        steps += 1

    out.release()
    print(f"\n🎥 Video saved as: {output_video}")
    print(f"🏆 Final score: {info.get('score', 'N/A')}")


# -----------------------------------
# CLI SUPPORT
# -----------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to checkpoint .pt file")

    parser.add_argument("--output",
                        type=str,
                        default="evaluation.mp4",
                        help="Output MP4 file name")

    args = parser.parse_args()

    evaluate(args.checkpoint, args.output)
