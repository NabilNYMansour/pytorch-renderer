import os
import sys

# silence pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import torch
import torch.nn.functional as F

SEED = 313131

# Make sure we are using cuda
assert torch.cuda.is_available(), "CUDA GPU is not available"
print("CUDA GPU is available")
torch.set_default_device("cuda")
torch.manual_seed(SEED)

RESOLUTION = (1280, 720)
WIDTH, HEIGHT = RESOLUTION
CX, CY = (WIDTH // 2, HEIGHT // 2)

xi = torch.arange(WIDTH)
yi = torch.arange(HEIGHT)
us, vs = torch.meshgrid(xi, yi, indexing="ij")  # like uvs in a shader

screen = torch.zeros(
    (WIDTH, HEIGHT, 3),
    dtype=torch.float32,
)

# Centered coordinates
xs = (2.0 * us.float() / (WIDTH - 1) - 1.0) * WIDTH / HEIGHT
ys = 1.0 - 2.0 * vs.float() / (HEIGHT - 1)

raydirs = torch.stack(
    [xs, ys, torch.ones_like(xs)],
    dim=-1,
)

raydirs = F.normalize(raydirs, dim=-1)

# Ray directions render
screen[..., 0] = raydirs[..., 0].abs()  # red: horizontal distance
screen[..., 1] = raydirs[..., 1].abs()  # green: vertical distance
screen[..., 2] = 0.0  # no blue
screen = (screen * 255.0).clamp(0, 255).to(torch.uint8)


def render():
    image = screen.cpu().numpy()
    surface = pygame.surfarray.make_surface(image)
    window.blit(surface, (0, 0))
    pygame.display.flip()


pygame.init()
pygame.display.set_caption("Ray Directions")
window = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

MAX_FPS = 120

render()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    clock.tick(MAX_FPS)
