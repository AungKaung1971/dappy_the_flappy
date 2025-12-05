import pygame
import numpy as np
import random

from .flappy_env_basic import Bird, Pipe, check_collision, WIDTH, HEIGHT, FPS
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack

print("ENV FILE LOADED")


class FlappyBirdEnv:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.bird = None
        self.pipes = []
        self.pipe_timer = 0
        self.PIPE_INTERVAL = 90

        self.score = 0
        self.done = False

        self.episode_reward = 0
        self.state = None

    def reset(self):
        """Reset the environment and return the first observation."""
        self.bird = Bird(50, HEIGHT // 2)

        self.pipes = []
        self.pipe_timer = 0
        self.score = 0
        self.done = False

        self.episode_reward = 0

        self._render_for_capture()

        frame = self._get_frame()

        processed = preprocess_frame(frame)

        self.state = init_frame_stack(processed)

        return self.state

    def step(self, action):
        """
        action = 0 → do nothing
        action = 1 → flap
        """

        reward = 1  # alive reward

        if action == 1:
            self.bird.flap()

        self.bird.update()

        for pipe in self.pipes:
            pipe.update()

            if pipe.x + pipe.width < self.bird.x and not hasattr(pipe, "scored"):
                self.score += 1
                pipe.scored = True
                reward += 10  # pipe reward

            if check_collision(self.bird, pipe):
                self.done = True
                reward = -50

        if self.bird.y < 0 or self.bird.y > HEIGHT:
            self.done = True
            reward = -50

        self.episode_reward += reward

        self.pipe_timer += 1
        if self.pipe_timer >= self.PIPE_INTERVAL:
            gap_y = random.randint(100, HEIGHT - 100)
            self.pipes.append(Pipe(WIDTH, gap_y))
            self.pipe_timer = 0

        # --- NEW PART: update frame stack and return it ---
        self._render_for_capture()
        frame = self._get_frame()
        processed = preprocess_frame(frame)
        if self.state is None:
            # safety, but normally state is set in reset()
            self.state = init_frame_stack(processed)
        else:
            self.state = update_frame_stack(self.state, processed)

        obs = self.state

        return obs, reward, self.done, {
            "score": self.score,
            "episode_reward": self.episode_reward
        }

    def _get_obs(self):
        """Return RGB frame (H,W,3)."""

        self.screen.fill((0, 150, 255))

        for pipe in self.pipes:
            pipe.draw(self.screen)

        self.bird.icon(self.screen)

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # to (H, W, 3)

        return frame

    def render(self):
        """Render to an actual visible display (debugging only)."""
        pygame.display.init()
        window = pygame.display.set_mode((WIDTH, HEIGHT))

        window.blit(self.screen, (0, 0))
        pygame.display.update()

    def _render_for_capture(self):
        self.screen.fill((0, 150, 255))

        for pipe in self.pipes:
            pipe.draw(self.screen)

        self.bird.icon(self.screen)

        pygame.display.flip()

    def _get_frame(self):
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return frame
