import os
import sys

# silence pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import torch

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

xs = 2.0 * us.float() / (WIDTH - 1) - 1.0
ys = 0.5 - vs.float() / (HEIGHT - 1)

# No pixel may be exactly on x = 0, so we use a larger tolerance for x axis
x_axis_mask = torch.isclose(xs, torch.zeros_like(xs), atol=2.0 / WIDTH)
y_axis_mask = torch.isclose(ys, torch.zeros_like(ys), atol=1.0 / HEIGHT)
screen[x_axis_mask | y_axis_mask] = torch.tensor([255, 255, 255], dtype=torch.float32)

function_mask = torch.isclose(xs.pow(3), ys, atol=1.0 / HEIGHT)
screen[function_mask] = torch.tensor([255, 0, 0], dtype=torch.float32)


def render():
    image = screen.cpu().numpy()
    surface = pygame.surfarray.make_surface(image)
    window.blit(surface, (0, 0))
    pygame.display.flip()


pygame.init()
pygame.display.set_caption("Pytorch: Conway's game of life")
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
