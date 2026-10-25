"""Interactive real-time visual training launcher with live crash animations and GPU acceleration."""

import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.car_env import RaycastCarEnv


class RealTimeVisualCallback(BaseCallback):
    """Callback that drives live Pygame rendering, crash visualization, and interactive keyboard controls during SB3 training."""

    def __init__(self, target_fps: int = 60, model_dir: str = "models", verbose: int = 0):
        super().__init__(verbose)
        self.target_fps = target_fps
        self.model_dir = model_dir
        self.render_enabled = True
        self.is_paused = False
        self.episode_count = 0
        self.total_crashes = 0
        self.device_name = "CPU"
        if torch.cuda.is_available():
            self.device_name = torch.cuda.get_device_name(0)

    def _on_step(self) -> bool:
        env = self.training_env.envs[0].unwrapped

        # Update crash and episode metrics when an episode terminates
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_count += 1
            infos = self.locals.get("infos", [{}])
            if infos and infos[0].get("collided", False):
                self.total_crashes += 1
                # Play brief crash impact animation in the window
                if self.render_enabled and env.renderer is not None:
                    cpos = infos[0].get("collision_pos", None)
                    if cpos is not None:
                        env.renderer.trigger_crash(cpos)
                    for _ in range(12):
                        env.render()

        # Handle rendering & interactive controls
        if self.render_enabled:
            # Query VRAM if on GPU
            vram_mb = 0.0
            if torch.cuda.is_available():
                vram_mb = torch.cuda.memory_allocated() / (1024.0 * 1024.0)

            # Update live telemetry on the HUD
            if env.renderer is not None:
                env.renderer.set_training_info({
                    "device_name": self.device_name,
                    "vram_mb": vram_mb,
                    "episode": self.episode_count,
                    "total_crashes": self.total_crashes,
                    "timesteps": self.num_timesteps,
                    "status": "PAUSED [SPACE]" if self.is_paused else "TRAINING (GPU)" if torch.cuda.is_available() else "TRAINING (CPU)",
                })

                # Handle interactive key events
                signals = env.renderer.process_events()
                if signals["quit_requested"]:
                    print("\n[!] User requested exit from visual window. Saving model and exiting...")
                    return False
                if signals["pause_toggled"]:
                    self.is_paused = not self.is_paused
                    print(f"\n[i] Training {'PAUSED' if self.is_paused else 'RESUMED'}")
                if signals["render_toggled"]:
                    self.render_enabled = False
                    print("\n[i] Visualization disabled. Training at maximum unrendered speed! (Press [V] in window or restart to re-enable)")
                if signals["fps_change"] != 0:
                    new_fps = max(15, min(240, env.renderer.fps + signals["fps_change"]))
                    env.renderer.fps = new_fps
                    print(f"\n[i] Target FPS adjusted to: {new_fps}")
                if signals["save_requested"]:
                    manual_path = os.path.join(self.model_dir, "checkpoint_manual.zip")
                    self.model.save(manual_path)
                    print(f"\n[OK] Manual checkpoint saved to: {manual_path}")

            # Pause loop
            while self.is_paused:
                if env.renderer is not None:
                    signals = env.renderer.process_events()
                    if signals["quit_requested"]:
                        return False
                    if signals["pause_toggled"]:
                        self.is_paused = False
                        print("\n[i] Training RESUMED")
                        break
                    env.render()
                time.sleep(0.04)

            # Step frame render
            env.render()

        return True


def main():
    parser = argparse.ArgumentParser(description="Real-time Visual RL Training with GPU Acceleration and Crash Feedback")
    parser.add_argument("--timesteps", type=int, default=250000, help="Total training timesteps (default: 250,000)")
    parser.add_argument("--fps", type=int, default=60, help="Target rendering FPS (default: 60)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"], help="Compute device (default: cuda)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard logs")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Determine Device
    target_device = args.device
    if target_device == "cuda" or target_device == "auto":
        if torch.cuda.is_available():
            target_device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        else:
            print("[!] CUDA requested but not available. Falling back to CPU.")
            target_device = "cpu"
            device_name = "CPU"
    else:
        device_name = "CPU"

    print("=" * 70)
    print("  Autonomous 2D RL Car - Real-time Visual GPU Training")
    print("=" * 70)
    print(f"  Device              : {target_device.upper()} ({device_name})")
    print(f"  Target FPS          : {args.fps}")
    print(f"  Total Timesteps     : {args.timesteps:,}")
    print(f"  Learning Rate       : {args.lr}")
    print(f"  Model Save Dir      : {args.model_dir}")
    print(f"  Log Dir             : {args.log_dir}")
    print("  Controls            : [SPACE] Pause | [V] Toggle Render | [UP/DOWN] Speed")
    print("                        [S] Save Checkpoint | [ESC] Exit & Save")
    print("=" * 70)

    # Single environment with render_mode='human' for real-time visualization
    def make_visual_env():
        return RaycastCarEnv(render_mode="human", fps=args.fps, max_steps=1200)

    train_env = make_vec_env(
        make_visual_env,
        n_envs=1,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
        monitor_dir=os.path.join(args.log_dir, "train_visual"),
    )

    # Visual Callback for live rendering and keyboard events
    visual_callback = RealTimeVisualCallback(
        target_fps=args.fps,
        model_dir=args.model_dir,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix="car_ppo_visual",
    )

    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
        activation_fn=nn.Tanh,
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=args.lr,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=args.log_dir,
        device=target_device,
        verbose=1,
        seed=args.seed,
    )

    print(f"[*] Policy loaded on device: {model.device}")
    print("[*] Starting real-time visual training loop...")

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=[visual_callback, checkpoint_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by keyboard. Saving current model...")

    final_path = os.path.join(args.model_dir, "best_model.zip")
    model.save(final_path)
    print(f"\n[OK] Model successfully saved to: {final_path}")

    train_env.close()


if __name__ == "__main__":
    main()
