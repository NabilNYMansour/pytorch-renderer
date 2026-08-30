import math
import os
import sys

# Silence pygame support prompt
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import pygame
import torch
import torch.nn.functional as F

SEED = 313131

# Make sure we are using CUDA
assert torch.cuda.is_available(), "CUDA GPU is not available"
print("CUDA GPU is available")

torch.set_default_device("cuda")
torch.manual_seed(SEED)

# Constants
RESOLUTION = (1280, 720)
WIDTH, HEIGHT = RESOLUTION
CX, CY = (WIDTH // 2, HEIGHT // 2)

CAMERA_POSITION = torch.tensor(
    [0.0, 0.0, 0.0], dtype=torch.float32
)  # Basically ray origin
SPHERE_CENTER = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
SPHERE_RADIUS = 0.5

FLOOR_Y = -1

MAX_FPS = 120

# UV coordinates
xi = torch.arange(WIDTH)
yi = torch.arange(HEIGHT)
us, vs = torch.meshgrid(xi, yi, indexing="ij")

# Main fragment buffer
fragments = torch.zeros((WIDTH, HEIGHT, 3), dtype=torch.float32)

# Centered coordinates
xs = (2.0 * us.float() / (WIDTH - 1) - 1.0) * WIDTH / HEIGHT
ys = 1.0 - 2.0 * vs.float() / (HEIGHT - 1)

raydirs = F.normalize(torch.stack([xs, ys, -torch.ones_like(xs)], dim=-1), dim=-1)


def ray_sphere_intersection(ray_origins, ray_directions, sphere_center, sphere_radius):
    oc = ray_origins - sphere_center

    a = torch.sum(ray_directions * ray_directions, dim=-1)
    b = 2.0 * torch.sum(ray_directions * oc, dim=-1)
    c = torch.sum(oc * oc, dim=-1) - sphere_radius**2

    discriminant = b**2 - 4.0 * a * c

    sqrt_discriminant = torch.sqrt(torch.clamp(discriminant, min=0.0))

    t0 = (-b - sqrt_discriminant) / (2.0 * a)
    t1 = (-b + sqrt_discriminant) / (2.0 * a)
    t_sphere = torch.minimum(t0, t1)

    t_sphere = torch.where(
        (discriminant >= 0.0) & (t_sphere > 0.0),
        t_sphere,
        torch.inf,
    )

    return t_sphere


t_sphere = ray_sphere_intersection(CAMERA_POSITION, raydirs, SPHERE_CENTER, SPHERE_RADIUS)

t_floor = (FLOOR_Y - CAMERA_POSITION[1]) / raydirs[..., 1]

# Only accept floor intersections in front of the camera
t_floor = torch.where((raydirs[..., 1] < 0.0) & (t_floor > 0.0), t_floor, torch.inf)

# Select the closest hit
t = torch.minimum(t_sphere, t_floor)

hit_sphere_mask = t_sphere <= t_floor
hit_floor_mask = t_floor < t_sphere
hit_mask = torch.isfinite(t)

hit_points = t[..., None] * raydirs + CAMERA_POSITION

# Normals
sphere_normals = F.normalize(hit_points - SPHERE_CENTER, dim=-1)

floor_normals = torch.zeros_like(hit_points)
floor_normals[..., 1] = 1.0

hit_normals = torch.where(
    hit_sphere_mask[..., None],
    sphere_normals,
    floor_normals,
)


def update_frame_tensor(light_pos):
    # Lighting and shadows
    light_dirs = F.normalize(light_pos - hit_points, dim=-1)

    SHADOW_BIAS = 0.001
    shadow_origins = hit_points + hit_normals * SHADOW_BIAS

    to_light = light_pos - shadow_origins
    light_distance = torch.linalg.vector_norm(to_light, dim=-1)

    light_dirs = F.normalize(to_light, dim=-1)
    shadow_t_sphere = ray_sphere_intersection(
        shadow_origins, light_dirs, SPHERE_CENTER, SPHERE_RADIUS
    )

    in_shadow = shadow_t_sphere < light_distance

    intensity = torch.where(
        in_shadow,
        0.0,
        torch.linalg.vecdot(light_dirs, hit_normals, dim=-1).clamp_min(0.0),
    )

    # Colors
    BASE_SPHERE_COLOR = torch.tensor([1.0, 0.3, 0.1], dtype=torch.float32)
    sphere_colors = intensity[..., None] * BASE_SPHERE_COLOR * 255.0

    BASE_FLOOR_CHECKER_1_COLOR = torch.tensor([0.4, 0.4, 0.4], dtype=torch.float32)
    BASE_FLOOR_CHECKER_2_COLOR = torch.tensor([0.6, 0.6, 0.6], dtype=torch.float32)
    floor_cx = torch.floor(hit_points[..., 0])
    floot_cz = torch.floor(hit_points[..., 2])
    floor_checker_check = (floor_cx + floot_cz) % 2 != 0
    floor_colors = (
        (intensity[..., None] * 2).clamp(0.0, 1.0)
        * torch.where(
            floor_checker_check[..., None],
            BASE_FLOOR_CHECKER_1_COLOR,
            BASE_FLOOR_CHECKER_2_COLOR,
        )
        * 255.0
    )

    colors = torch.where(hit_sphere_mask[..., None], sphere_colors, floor_colors)

    # Final color
    fragments[hit_mask] = colors[hit_mask]

    # Background color of non-hits
    fragments[~hit_mask] = (
        torch.tensor(
            [0.4, 0.8, 0.9],
            dtype=torch.float32,
        )
        * 255.0
    )


# Pygame setup
pygame.init()
pygame.display.set_caption("Ray Directions")
window = pygame.display.set_mode(RESOLUTION)
clock = pygame.time.Clock()


def render():
    image = fragments.clamp(0.0, 255.0).byte().cpu().numpy()
    surface = pygame.surfarray.make_surface(image)

    window.blit(surface, (0, 0))
    pygame.display.flip()


last_move_time = pygame.time.get_ticks()
light_pos = torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float32)
while True:
    update_frame_tensor(light_pos)
    render()
    current_time = pygame.time.get_ticks()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if current_time - last_move_time >= 1:  # in milliseconds
        light_pos[0] = math.sin(current_time / 1000.0)
        light_pos[2] = math.cos(current_time / 1000.0)
        last_move_time = current_time

    clock.tick(MAX_FPS)
