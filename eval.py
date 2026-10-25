"""Model evaluation and real-time visualization of trained PPO agent navigating procedural tracks."""

import argparse
import os
import sys
import numpy as np
import pygame
from stable_baselines3 import PPO

from envs.car_env import RaycastCarEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PPO policy in RaycastCarEnv")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.zip",
        help="Path to saved model zip file (default: models/best_model.zip)",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes (default: 10)")
    parser.add_argument("--seed", type=int, default=1337, help="Evaluation random seed")
    parser.add_argument("--fps", type=int, default=60, help="Target rendering FPS (default: 60)")
    parser.add_argument("--no-render", action="store_true", help="Run headlessly without opening Pygame window")
    args = parser.parse_args()

    # Fallback to final_model.zip if best_model.zip is not found
    if not os.path.exists(args.model_path):
        alt_path = "models/final_model.zip"
        if os.path.exists(alt_path):
            print(f"[i] '{args.model_path}' not found, falling back to '{alt_path}'")
            args.model_path = alt_path
        else:
            print(f"[!] Error: Model file not found at '{args.model_path}' or '{alt_path}'.")
            print("    Please train the model first by running: python train.py")
            sys.exit(1)

    print("=" * 60)
    print("  Autonomous 2D RL Car - Model Evaluation & Visualizer")
    print("=" * 60)
    print(f"  Model Path : {args.model_path}")
    print(f"  Episodes   : {args.episodes}")
    print(f"  Target FPS : {args.fps}")
    print("  Controls   : [R] Generate New Track | [ESC] Quit")
    print("=" * 60)

    # Load trained model
    print("[*] Loading PPO policy weights...")
    model = PPO.load(args.model_path)
    print("[OK] Model loaded successfully.")

    # Initialize environment with Pygame renderer (or headless if --no-render)
    render_mode = None if args.no_render else "human"
    env = RaycastCarEnv(render_mode=render_mode, fps=args.fps, max_steps=1800)

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        ep_reward = 0.0
        step = 0
        done = False

        print(f"\n---> Starting Episode {ep}/{args.episodes}")

        while not done:
            # Handle Pygame quit or track reset key if rendered
            if render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        sys.exit(0)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            env.close()
                            sys.exit(0)
                        elif event.key == pygame.K_r:
                            print("  [R pressed] Regenerating new track...")
                            obs, info = env.reset()
                            ep_reward = 0.0
                            step = 0

            # Deterministic inference from trained policy
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

        reason = "COLLISION" if info["collided"] else "STEP LIMIT / TIMEOUT"
        print(f"  Finished Episode {ep} | Result: {reason} | Steps: {step} | Total Reward: {ep_reward:.1f} | Laps: {info['laps_completed']} | Checkpoints: {info['checkpoints_passed']}")

    env.close()
    print("\n[OK] Evaluation completed.")


if __name__ == "__main__":
    main()
