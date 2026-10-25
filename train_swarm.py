"""Launch 100+ parallel cars training simultaneously on GPU with real-time swarm visualizer."""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

from envs.swarm_car_env import SwarmCarEnv


class SwarmVisualCallback(BaseCallback):
    """Orchestrates live Pygame rendering, GPU memory tracking, and keyboard hotkeys for 100+ cars."""

    def __init__(self, swarm_env: SwarmCarEnv, model_dir: str = "models", verbose: int = 0):
        super().__init__(verbose)
        self.swarm_env = swarm_env
        self.model_dir = model_dir
        self.render_enabled = True
        self.is_paused = False
        self.device_name = "CPU"
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)

    def _on_step(self) -> bool:
        if self.render_enabled and self.swarm_env.renderer is not None:
            # Live GPU VRAM
            vram_mb = 0.0
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)

            # Update live telemetry
            self.swarm_env.renderer.set_training_info({
                "device_name": self.device_name,
                "vram_mb": vram_mb,
                "timesteps": self.num_timesteps,
                "status": "PAUSED [SPACE]" if self.is_paused else "SWARM LEARNING (GPU)" if torch.cuda.is_available() else "SWARM LEARNING",
            })

            # Process interactive hotkeys
            signals = self.swarm_env.renderer.process_events()

            if signals["quit_requested"]:
                print("\n[!] Exit requested from window. Saving model and exiting...")
                return False

            if signals["pause_toggled"]:
                self.is_paused = not self.is_paused
                print(f"\n[i] Swarm training {'PAUSED' if self.is_paused else 'RESUMED'}")

            if signals["render_toggled"]:
                self.render_enabled = False
                print("\n[i] Visualization disabled. Swarm running at maximum GPU throughput! (Press [V] in window or restart to re-enable)")

            if signals["fps_change"] != 0:
                new_fps = max(15, min(240, self.swarm_env.renderer.fps + signals["fps_change"]))
                self.swarm_env.renderer.fps = new_fps
                print(f"\n[i] Swarm target FPS adjusted to: {new_fps}")

            if signals["save_requested"]:
                manual_path = os.path.join(self.model_dir, "checkpoint_swarm_manual.zip")
                self.model.save(manual_path)
                print(f"\n[OK] Manual swarm checkpoint saved to: {manual_path}")

            # Pause loop
            while self.is_paused:
                if self.swarm_env.renderer is not None:
                    signals = self.swarm_env.renderer.process_events()
                    if signals["quit_requested"]:
                        return False
                    if signals["pause_toggled"]:
                        self.is_paused = False
                        print("\n[i] Swarm training RESUMED")
                        break
                    self.swarm_env.render()
                time.sleep(0.04)

        return True


def main():
    parser = argparse.ArgumentParser(description="Train 100+ parallel cars simultaneously on GPU with live swarm rendering")
    parser.add_argument("--num-cars", type=int, default=100, help="Number of parallel cars in the swarm (default: 100)")
    parser.add_argument("--timesteps", type=int, default=250000, help="Total training timesteps (default: 250,000)")
    parser.add_argument("--fps", type=int, default=60, help="Rendering target FPS (default: 60)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"], help="Compute device (default: cuda)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard logs")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine Device
    target_device = args.device
    if target_device in ("cuda", "auto"):
        if torch.cuda.is_available():
            target_device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        else:
            print("[!] CUDA requested but unavailable. Falling back to CPU.")
            target_device = "cpu"
            device_name = "CPU"
    else:
        device_name = "CPU"

    print("=" * 70)
    print("  Autonomous 2D RL Car - 100+ Car Parallel Swarm Trainer")
    print("=" * 70)
    print(f"  Parallel Cars       : {args.num_cars:,}")
    print(f"  Device              : {target_device.upper()} ({device_name})")
    print(f"  Target Playback FPS : {args.fps}")
    print(f"  Total Timesteps     : {args.timesteps:,}")
    print(f"  Est. Throughput     : ~{args.num_cars * args.fps:,} steps/second")
    print(f"  Model Save Dir      : {args.model_dir}")
    print(f"  Log Dir             : {args.log_dir}")
    print("  Controls            : [SPACE] Pause | [V] Toggle Render | [UP/DOWN] Speed")
    print("                        [S] Save Checkpoint | [ESC] Exit & Save")
    print("=" * 70)

    # 1. Initialize Swarm Environment with 100+ cars
    swarm_env = SwarmCarEnv(
        num_cars=args.num_cars,
        render_mode="human",
        fps=args.fps,
        max_steps_per_episode=1200,
    )

    # Initial reset to populate track and spawn grid
    swarm_env.reset()

    # 2. Visual Callback
    visual_callback = SwarmVisualCallback(
        swarm_env=swarm_env,
        model_dir=args.model_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(10, 25000 // args.num_cars),
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix=f"swarm_{args.num_cars}_ppo",
    )

    # 3. PPO Configuration optimized for 100+ parallel cars on GPU
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
    )

    # For 100 cars: n_steps=64 => 64 * 100 = 6,400 samples per rollout iteration
    n_steps_per_car = 64
    batch_size = 256

    model = PPO(
        policy="MlpPolicy",
        env=swarm_env,
        learning_rate=args.lr,
        n_steps=n_steps_per_car,
        batch_size=batch_size,
        n_epochs=6,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        device=target_device,
        verbose=1,
    )

    print(f"[*] Policy loaded on {model.device}.")
    print(f"[*] Launching {args.num_cars} cars into the track arena...")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[visual_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by keyboard. Saving current checkpoint...")

    final_path = os.path.join(args.model_dir, "best_swarm_model.zip")
    model.save(final_path)
    print(f"\n[OK] Swarm model successfully saved to: {final_path}")

    swarm_env.close()


if __name__ == "__main__":
    main()
