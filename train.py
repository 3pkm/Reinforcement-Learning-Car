"""Training pipeline for autonomous 2D racing car using Stable-Baselines3 PPO."""

import argparse
import os
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from envs.car_env import RaycastCarEnv


def make_env(seed: int = 0):
    def _init():
        env = RaycastCarEnv(render_mode=None, max_steps=1200)
        env.reset(seed=seed)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train RaycastCarEnv with PPO (Stable-Baselines3)")
    parser.add_argument("--timesteps", type=int, default=250000, help="Total training timesteps (default: 250,000)")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments (default: 4)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "auto"], help="Compute device (default: cuda)")
    parser.add_argument("--render", action="store_true", help="Launch real-time visual training window")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log-dir", type=str, default="logs", help="Directory for TensorBoard logs")
    parser.add_argument("--eval-freq", type=int, default=10000, help="Timesteps between evaluations (default: 10,000)")
    args = parser.parse_args()

    # Determine Device
    device = args.device
    if device in ("cuda", "auto"):
        if torch.cuda.is_available():
            device = "cuda"
            device_name = torch.cuda.get_device_name(0)
        else:
            print("[!] CUDA requested but unavailable. Falling back to CPU.")
            device = "cpu"
            device_name = "CPU"
    else:
        device_name = "CPU"

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    print("=" * 65)
    print("  Autonomous 2D RL Car - PPO Training Pipeline (SB3)")
    print("=" * 65)
    print(f"  Device              : {device.upper()} ({device_name})")
    print(f"  Total Timesteps     : {args.timesteps:,}")
    print(f"  Parallel Envs       : {args.num_envs}")
    print(f"  Learning Rate       : {args.lr}")
    print(f"  Model Save Dir      : {args.model_dir}")
    print(f"  Tensorboard Log Dir : {args.log_dir}")
    print("=" * 65)

    # 1. Vectorized Training Environment
    train_n_envs = 1 if args.render else args.num_envs
    train_render_mode = "human" if args.render else None

    train_env = make_vec_env(
        RaycastCarEnv,
        n_envs=train_n_envs,
        seed=args.seed,
        vec_env_cls=DummyVecEnv,
        monitor_dir=os.path.join(args.log_dir, "train"),
        env_kwargs={"render_mode": train_render_mode, "max_steps": 1200},
    )

    # 2. Evaluation Environment
    eval_env = make_vec_env(
        RaycastCarEnv,
        n_envs=1,
        seed=args.seed + 100,
        vec_env_cls=DummyVecEnv,
        monitor_dir=os.path.join(args.log_dir, "eval"),
        env_kwargs={"render_mode": None, "max_steps": 1200},
    )

    # 3. Callbacks: Save best model & regular checkpoints
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=args.model_dir,
        log_path=args.log_dir,
        eval_freq=max(1000, args.eval_freq // args.num_envs),
        n_eval_episodes=5,
        deterministic=True,
        render=False,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, 50000 // args.num_envs),
        save_path=os.path.join(args.model_dir, "checkpoints"),
        name_prefix="car_ppo",
    )

    callbacks = [eval_callback, checkpoint_callback]
    if args.render:
        from train_visual import RealTimeVisualCallback
        callbacks.append(RealTimeVisualCallback(model_dir=args.model_dir))

    # 4. PPO Agent Configuration
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
        device=device,
        verbose=1,
        seed=args.seed,
    )

    # 5. Train Policy
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n[!] Training manually interrupted by user. Saving current checkpoint...")

    # Save final model
    final_path = os.path.join(args.model_dir, "final_model.zip")
    model.save(final_path)
    print(f"\n[OK] Training complete! Model saved to: {final_path}")
    print(f"[OK] Best model checkpoint stored at: {os.path.join(args.model_dir, 'best_model.zip')}")
    print("\nTo view TensorBoard curves:")
    print(f"  tensorboard --logdir={args.log_dir}")
    print("\nTo evaluate the trained model:")
    print(f"  python eval.py --model-path={os.path.join(args.model_dir, 'best_model.zip')}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
