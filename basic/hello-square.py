import os
import sys

# silence pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import torch

RESOLUTION = (1280, 720)
WIDTH, HEIGHT = RESOLUTION
SQUARE_SIZE = 100

# Make sure we are using cuda
assert torch.cuda.is_available(), "CUDA GPU is not available"
print("CUDA GPU is available")
torch.set_default_device("cuda")

# General setup
main_screen = torch.zeros(
    (HEIGHT, WIDTH, 3),
    dtype=torch.uint8,
    device="cuda",
)

center = (WIDTH // 2, HEIGHT // 2)
coord_from = (center[0] - SQUARE_SIZE // 2, center[1] - SQUARE_SIZE // 2)
coord_to = (coord_from[0] + SQUARE_SIZE, coord_from[1] + SQUARE_SIZE)

# Setting background color
main_screen[:, :, 0] = 20
main_screen[:, :, 1] = 40
main_screen[:, :, 2] = 60

pygame.init()
pygame.display.set_caption("Hello Square")
window = pygame.display.set_mode(RESOLUTION)


def inside_square(pos):
    return (
        coord_from[0] <= pos[0] < coord_to[0] and coord_from[1] <= pos[1] < coord_to[1]
    )


def draw_square(color: tuple[int, int, int]):
    y_slice = slice(coord_from[1], coord_to[1])
    x_slice = slice(coord_from[0], coord_to[0])

    main_screen[y_slice, x_slice, 0] = color[0]  # R
    main_screen[y_slice, x_slice, 1] = color[1]  # G
    main_screen[y_slice, x_slice, 2] = color[2]  # B


def render():
    image = main_screen.cpu().numpy()
    surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
    window.blit(surface, (0, 0))
    pygame.display.flip()

while True:
    render()
    for event in pygame.event.get():
        if event.type == pygame.MOUSEMOTION:
            mouse_pos = event.pos
            if inside_square(mouse_pos):
                draw_square((0, 255, 0))
            else:
                draw_square((255, 0, 0))
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
