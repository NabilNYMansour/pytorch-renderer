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
BOARD_PX_SIZE = 120

# General setup
cells = torch.ones(
    (WIDTH, HEIGHT),
    dtype=torch.bool,
)

xs = torch.arange(WIDTH)  // BOARD_PX_SIZE   # so [0,0,0,...,1,1,1,...,222,...]
ys = torch.arange(HEIGHT) // BOARD_PX_SIZE   # same here but for height

# We do an indexing unsqueeze here to and then see if the sum is even. If it is, the cell is black; otherwise, it is white.
cells = ((xs[:, None] + ys[None, :]) % 2) == 0


def render():
    # Turn the cells tensor into a 3-channel image
    image = (cells.unsqueeze(-1).expand(-1, -1, 3).to(torch.float) * 255).cpu().numpy()
    surface = pygame.surfarray.make_surface(image)
    window.blit(surface, (0, 0))
    pygame.display.flip()


pygame.init()
pygame.display.set_caption("Checkerboard")
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
