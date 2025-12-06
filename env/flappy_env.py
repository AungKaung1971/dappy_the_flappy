import pygame
import numpy as np
import random

from .flappy_env_basic import Bird, Pipe, check_collision, WIDTH, HEIGHT, FPS
from preprocess import preprocess_frame, init_frame_stack, update_frame_stack

print("ENV FILE LOADED")


class FlappyBirdEnv:
    def __init__(self):
        pygame.init()

        # Offscreen surface for rendering
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.bird = None
        self.pipes = []
        self.pipe_timer = 0
        # self.PIPE_INTERVAL = 10  # use if you need to debug
        self.PIPE_INTERVAL = 90

        self.score = 0
        self.done = False

        self.episode_reward = 0.0
        self.state = None

    def reset(self):
        """Reset the environment and return the first observation."""
        self.bird = Bird(50, HEIGHT // 2)

        self.pipes = []
        self.pipe_timer = 0
        self.score = 0
        self.done = False

        self.episode_reward = 0.0
        self.state = None

        # Initial frame / state stack
        self._render_for_capture()
        frame = self._get_frame()
        processed = preprocess_frame(frame)
        self.state = init_frame_stack(processed)

        return self.state

    def step(self, action):
        """
        action = 0 → do nothing
        action = 1 → flap

        Reward scheme (sparse PPO-friendly):
            +1  for each pipe passed
            -1  on death (collision or out of bounds)
             0  otherwise
        """
        reward = 0.01

        # --- ACTION ---
        if action == 1:
            self.bird.flap()

        # --- PHYSICS UPDATE ---
        self.bird.update()

        # Move pipes and check for pipes passed
        for pipe in self.pipes:
            pipe.update()

            # Pipe passed
            if (not pipe.scored) and (pipe.x + pipe.width < self.bird.x):
                pipe.scored = True
                self.score += 1
                reward += 1.0  # pipe reward

        # Remove off-screen pipes
        self.pipes = [p for p in self.pipes if p.x + p.width > 0]

        # --- COLLISION CHECK ---
        for pipe in self.pipes:
            if check_collision(self.bird, pipe):
                self.done = True
                reward -= 1.0  # death penalty (can cancel a pipe reward if same frame)
                return self._terminal_step(reward)

        # --- OUT OF BOUNDS ---
        if self.bird.y < 0 or self.bird.y > HEIGHT:
            self.done = True
            reward -= 1.0
            return self._terminal_step(reward)

        # --- STILL ALIVE: accumulate reward and continue ---
        self.episode_reward += reward

        # Spawn new pipes
        self.pipe_timer += 1
        if self.pipe_timer >= self.PIPE_INTERVAL:
            gap_y = random.randint(100, HEIGHT - 100)
            self.pipes.append(Pipe(WIDTH, gap_y))
            self.pipe_timer = 0

        # Update frame stack / observation
        self._render_for_capture()
        frame = self._get_frame()
        processed = preprocess_frame(frame)

        if self.state is None:
            self.state = init_frame_stack(processed)
        else:
            self.state = update_frame_stack(self.state, processed)

        obs = self.state

        return obs, reward, False, {
            "score": self.score,
            "episode_reward": self.episode_reward
        }

    def _terminal_step(self, reward):
        """Handle terminal transition: update state stack and return final obs, reward, done, info."""
        self.episode_reward += reward

        self._render_for_capture()
        frame = self._get_frame()
        processed = preprocess_frame(frame)
        if self.state is None:
            self.state = init_frame_stack(processed)
        else:
            self.state = update_frame_stack(self.state, processed)

        return self.state, reward, True, {
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
        """Draw current game state into self.screen."""
        self.screen.fill((0, 150, 255))

        for pipe in self.pipes:
            pipe.draw(self.screen)

        self.bird.icon(self.screen)

    def _get_frame(self):
        """Grab the current RGB frame (H,W,3) from self.screen."""
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))
        return frame
