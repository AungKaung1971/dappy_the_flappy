import pygame
import numpy as np
import random

from .flappy_env_basic import Bird, Pipe, check_collision, WIDTH, HEIGHT, FPS


class FlappyBirdEnv:
    def __init__(self):
        pygame.init()

        # Hidden internal surface (no window)
        self.screen = pygame.Surface((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        # Game state
        self.bird = None
        self.pipes = []
        self.pipe_timer = 0
        self.PIPE_INTERVAL = 90

        self.score = 0
        self.done = False

    def reset(self):
        """Reset the environment and return the first observation."""
        self.bird = Bird(50, HEIGHT // 2)

        self.pipes = []
        self.pipe_timer = 0
        self.score = 0
        self.done = False

        return self._get_obs()

    def step(self, action):
        """
        action = 0 → do nothing
        action = 1 → flap
        """

        reward = 1  # alive reward

        # ---- Apply Action ----
        if action == 1:
            self.bird.flap()

        # ---- Update Bird ----
        self.bird.update()

        # ---- Update Pipes ----
        for pipe in self.pipes:
            pipe.update()

            # Score logic (your version)
            if pipe.x + pipe.width < self.bird.x and not hasattr(pipe, "scored"):
                self.score += 1
                pipe.scored = True
                reward += 10  # pipe reward

            # Collision check
            if check_collision(self.bird, pipe):
                self.done = True
                reward = -50

        # ---- Death by ground/ceiling ----
        if self.bird.y < 0 or self.bird.y > HEIGHT:
            self.done = True
            reward = -50

        # ---- Spawn Pipes ----
        self.pipe_timer += 1
        if self.pipe_timer >= self.PIPE_INTERVAL:
            gap_y = random.randint(100, HEIGHT - 100)
            self.pipes.append(Pipe(WIDTH, gap_y))
            self.pipe_timer = 0

        # ---- Build observation ----
        obs = self._get_obs()

        return obs, reward, self.done, {"score": self.score}

    def _get_obs(self):
        """Return RGB frame (H,W,3)."""

        self.screen.fill((0, 150, 255))

        for pipe in self.pipes:
            pipe.draw(self.screen)

        self.bird.icon(self.screen)

        # Convert pygame surface → RGB array
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  # to (H, W, 3)

        return frame

    def render(self):
        """Render to an actual visible display (debugging only)."""
        pygame.display.init()
        window = pygame.display.set_mode((WIDTH, HEIGHT))

        window.blit(self.screen, (0, 0))
        pygame.display.update()
