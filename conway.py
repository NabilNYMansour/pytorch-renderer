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

# General setup
cells = torch.randint(
    0,
    2,
    (WIDTH, HEIGHT),
    dtype=torch.bool,
)

neighbors_kernel = torch.tensor(
    [
        [
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ]
        ]
    ],
    dtype=torch.float32,
)


def render():
    # Turn the cells tensor into a 3-channel image
    image = (cells.unsqueeze(-1).expand(-1, -1, 3).to(torch.float) * 255).cpu().numpy()
    surface = pygame.surfarray.make_surface(image)
    window.blit(surface, (0, 0))
    pygame.display.flip()


def update():
    global cells
    cells_reshape = cells.float()[None, None]  # (1, 1, 5, 5)
    neighbors = F.conv2d(cells_reshape, neighbors_kernel, padding=1)
    neighbors = neighbors[0, 0]  # back to (5, 5)

    birth = ~cells & (neighbors == 3)
    survive = cells & ((neighbors == 2) | (neighbors == 3))
    cells = birth | survive


pygame.init()
pygame.display.set_caption("Pytorch: Conway's game of life")
window = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()

MAX_FPS = 120

while True:
    render()
    update()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
    clock.tick(MAX_FPS)
