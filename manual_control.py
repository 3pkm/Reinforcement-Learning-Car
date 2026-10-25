"""Manual driving interface to verify Pymunk vehicle physics, procedural track generation, and raycasts."""

import sys
import numpy as np
import pygame
from envs.car_env import RaycastCarEnv


def main():
    print("=" * 60)
    print("Autonomous 2D RL Car - Manual Control Diagnostic")
    print("=" * 60)
    print("Controls:")
    print("  [W / UP]     : Accelerate (Throttle)")
    print("  [S / DOWN]   : Reverse / Brake")
    print("  [A / LEFT]   : Steer Left")
    print("  [D / RIGHT]  : Steer Right")
    print("  [R]          : Reset & Generate New Procedural Track")
    print("  [ESC]        : Quit")
    print("=" * 60)

    env = RaycastCarEnv(render_mode="human", max_steps=3000)
    obs, info = env.reset()

    running = True
    clock = pygame.time.Clock()

    while running:
        # Handle Pygame user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    print("--> Regenerating new procedural track...")
                    obs, info = env.reset()

        keys = pygame.key.get_pressed()

        # Steering: -1.0 (left) to +1.0 (right)
        steering = 0.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            steering -= 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            steering += 1.0

        # Throttle: -1.0 (reverse/brake) to +1.0 (forward drive)
        throttle = 0.0
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            throttle += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            throttle -= 1.0

        action = np.array([steering, throttle], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            reason = "COLLISION WITH WALL" if terminated else "STEP LIMIT / STUCK"
            print(f"Episode Finished ({reason}) | Steps: {info['step']} | Total Checkpoints: {info['checkpoints_passed']}")
            obs, info = env.reset()

        clock.tick(60)

    env.close()
    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
