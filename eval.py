import cv2
import torch
import argparse
import numpy as np
import os
import re
from datetime import datetime

from env.flappy_env import FlappyBirdEnv
from model import CNNPolicy
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack


# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------

def extract_step_from_checkpoint(path):
    """
    Extracts step number from checkpoint filename.
    Expected format: ckpt_200352.pt
    """
    match = re.search(r"ckpt_(\d+)\.pt", os.path.basename(path))
    return int(match.group(1)) if match else None


def find_run_folder_for_checkpoint(step, logs_dir="logs"):
    """
    Searches all run folders for a checkpoint whose saved step matches.
    If found → return that run folder path.
    Otherwise → return None.
    """
    if not os.path.exists(logs_dir):
        return None

    for run_name in os.listdir(logs_dir):
        run_path = os.path.join(logs_dir, run_name)

        if not os.path.isdir(run_path):
            continue

        metrics_path = os.path.join(run_path, "metrics.csv")
        if not os.path.exists(metrics_path):
            continue

        # Check if metrics.csv contains this step
        try:
            with open(metrics_path, "r") as f:
                for line in f:
                    if line.startswith(str(step) + ",") or f",{step}," in line:
                        return run_path
        except:
            continue

    return None


# --------------------------------------------------------------
# MAIN EVALUATION FUNCTION
# --------------------------------------------------------------

def evaluate(checkpoint_path, output_video=None, run_override=None, max_steps=5000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------
    model = CNNPolicy(action_dim=2).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # --------------------------------------------------
    # DETERMINE RUN FOLDER
    # --------------------------------------------------
    step = extract_step_from_checkpoint(checkpoint_path)

    # Priority 1: user-specified run
    if run_override is not None:
        run_folder = run_override

    # Priority 2: auto-detect run
    elif step is not None:
        found_run = find_run_folder_for_checkpoint(step)
        if found_run:
            run_folder = found_run
        else:
            # fallback
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_folder = f"logs/run_manual_eval_{timestamp}"
            os.makedirs(run_folder, exist_ok=True)
    else:
        # fallback when filename doesn't include ckpt_XXXX
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_folder = f"logs/run_manual_eval_{timestamp}"
        os.makedirs(run_folder, exist_ok=True)

    # Ensure videos directory exists
    videos_dir = os.path.join(run_folder, "videos")
    os.makedirs(videos_dir, exist_ok=True)

    # --------------------------------------------------
    # DETERMINE OUTPUT VIDEO PATH
    # --------------------------------------------------
    if output_video is None:
        # auto-generate filename based on checkpoint step
        if step is not None:
            output_video = os.path.join(videos_dir, f"ckpt_{step}.mp4")
        else:
            output_video = os.path.join(videos_dir, "evaluation.mp4")
    else:
        # Custom filename → keep user path
        # If user only gives a filename, save it inside the run's videos folder
        if not os.path.dirname(output_video):
            output_video = os.path.join(videos_dir, output_video)

    print(f"\n🔍 Using run folder: {run_folder}")
    print(f"🎬 Saving video to: {output_video}\n")

    # --------------------------------------------------
    # INIT ENV + FRAME STACK
    # --------------------------------------------------
    env = FlappyBirdEnv()
    env.reset()

    raw = env._get_frame()
    processed = preprocess_frame(raw)
    state = init_frame_stack(processed).astype(np.float32)

    # --------------------------------------------------
    # INIT VIDEO WRITER
    # --------------------------------------------------
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

    # --------------------------------------------------
    # MAIN LOOP
    # --------------------------------------------------
    while not done and steps < max_steps:

        # Convert to tensor
        inp = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        # Policy → action
        with torch.no_grad():
            logits, _ = model(inp)
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

    print(f"\n🎥 Video saved: {output_video}")
    print(f"🏆 Final score: {info.get('score', 'N/A')}")
    print(f"📁 Video stored under run: {run_folder}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Path to checkpoint .pt file")

    parser.add_argument("--output",
                        type=str,
                        default=None,
                        help="Optional: custom output file name (mp4).")

    parser.add_argument("--run",
                        type=str,
                        default=None,
                        help="Optional: manually specify which run folder to save the video into.")

    parser.add_argument("--max_steps",
                        type=int,
                        default=5000,
                        help="Max environment steps for evaluation.")

    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        output_video=args.output,
        run_override=args.run,
        max_steps=args.max_steps,
    )
