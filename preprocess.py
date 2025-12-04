import cv2
import numpy as np


def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized


def init_frame_stack(processed_frame):
    return np.stack([processed_frame] * 4, axis=0)


def update_frame_stack(frame_stack, processed_frame):
    frame_stack[:-1] = frame_stack[1:]
    frame_stack[-1] = processed_frame

    return frame_stack
