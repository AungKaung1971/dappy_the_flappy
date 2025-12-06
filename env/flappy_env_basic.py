import pygame
import sys
import random

WIDTH = 640
HEIGHT = 360
FPS = 60

GRAVITY = 0.25
FLAP_STRENGTH = -5


class Bird:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vel = 0
        self.radius = 12

    def update(self):
        self.vel += GRAVITY
        self.y += self.vel

    def flap(self):
        self.vel = FLAP_STRENGTH

    def icon(self, screen):
        pygame.draw.circle(screen, (255, 255, 0),
                           (int(self.x), int(self.y)), self.radius)


class Pipe:
    def __init__(self, x, gap_y, gap_height=120, speed=2):
        self.x = x
        self.gap_y = gap_y
        self.gap_height = gap_height
        self.speed = speed
        self.width = 50

        self.scored = False

    def update(self):
        self.x -= self.speed

    def draw(self, screen):
        top_rect = pygame.Rect(self.x, 0, self.width,
                               self.gap_y - self.gap_height//2)
        bottom_rect = pygame.Rect(self.x, self.gap_y + self.gap_height//2,
                                  self.width, HEIGHT - (self.gap_y + self.gap_height//2))

        pygame.draw.rect(screen, (0, 255, 0), top_rect)
        pygame.draw.rect(screen, (0, 255, 0), bottom_rect)

    


def check_collision(bird, pipe):
    if bird.x + bird.radius > pipe.x and bird.x - bird.radius < pipe.x + pipe.width:
        top_pipe_bottom = pipe.gap_y - pipe.gap_height // 2
        bottom_pipe_top = pipe.gap_y + pipe.gap_height // 2

        if bird.y - bird.radius < top_pipe_bottom:
            return True
        if bird.y + bird.radius > bottom_pipe_top:
            return True

    return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    font = pygame.font.SysFont(None, 32)
    clock = pygame.time.Clock()

    bird = Bird(50, HEIGHT // 2)

    score = 0

    pipes = []
    pipe_timer = 0
    PIPE_INTERVAL = 45

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    bird.flap()

        bird.update()

        for pipe in pipes:
            pipe.update()

            if pipe.x + pipe.width < bird.x and not hasattr(pipe, "scored"):
                score += 1
                pipe.scored = True
                print("Score:", score)

            if check_collision(bird, pipe):
                print("GAME OVER")
                pygame.time.wait(800)
                return main()

        if bird.y > HEIGHT or bird.y < 0:
            print("GAME OVER")
            pygame.time.wait(800)
            return main()

        screen.fill((0, 150, 255))

        for pipe in pipes:
            pipe.draw(screen)

        bird.icon(screen)

        score_text = font.render(f"Score: {score}", True, (255, 255, 255))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()

        pipe_timer += 1
        if pipe_timer >= PIPE_INTERVAL:
            gap_y = random.randint(100, HEIGHT - 100)
            pipes.append(Pipe(WIDTH, gap_y))
            pipe_timer = 0


if __name__ == "__main__":
    print("running pygame")
    main()


# python env/flappy_env_basic.py
