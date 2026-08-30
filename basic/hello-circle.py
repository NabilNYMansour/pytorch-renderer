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
screen = torch.zeros(
    (WIDTH, HEIGHT, 3),
    dtype=torch.float32,
)


def render():
    image = screen.cpu().numpy()
    surface = pygame.surfarray.make_surface(image)
    window.blit(surface, (0, 0))
    pygame.display.flip()


xi = torch.arange(WIDTH)
yi = torch.arange(HEIGHT)
xs, ys = torch.meshgrid(xi, yi, indexing="ij")
RADIUS = 80


def position_circle(pos_x, pos_y):
    screen[:] = 0
    mask_radius = torch.hypot(xs.float() - pos_x, ys.float() - pos_y) < RADIUS
    screen[mask_radius] = torch.tensor([1.0, 1.0, 1.0]) * 255


pygame.init()
pygame.display.set_caption("Pytorch: Conway's game of life")
window = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

MAX_FPS = 120

while True:
    position_circle(*pygame.mouse.get_pos())
    render()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    clock.tick(MAX_FPS)
