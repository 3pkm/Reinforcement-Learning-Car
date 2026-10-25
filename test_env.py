"""Automated verification suite for RaycastCarEnv, physics, raycasts, and SB3 compatibility."""

import numpy as np
import gymnasium as gym
import torch
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO

from envs.car_env import RaycastCarEnv


def test_environment():
    print("[1/5] Testing environment instantiation...")
    env = RaycastCarEnv(render_mode=None, max_steps=500)
    print(f"  Obs Space   : {env.observation_space}")
    print(f"  Action Space: {env.action_space}")

    print("\n[2/5] Testing reset and procedural track generation...")
    obs, info = env.reset(seed=123)
    assert obs.shape == (7,), f"Expected obs shape (7,), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Observation values out of bounds [0, 1]: {obs}"
    print(f"  Initial Obs: {np.round(obs, 3)}")
    print(f"  Num Checkpoints: {len(env.track_gen.checkpoints)}")
    print(f"  Num Wall Shapes: {len(env.track_gen.wall_shapes)}")

    print("\n[3/5] Testing multi-step simulation with random actions...")
    total_reward = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        assert obs.shape == (7,), f"Expected obs shape (7,), got {obs.shape}"
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0), f"Observation values out of bounds [0, 1]: {obs}"
        if terminated or truncated:
            obs, info = env.reset()

    print(f"  50 random steps completed successfully. Total reward: {total_reward:.2f}")

    print("\n[4/5] Running Gymnasium official check_env...")
    check_env(env.unwrapped)
    print("  Gymnasium check_env PASSED!")

    print("\n[5/6] Testing Pygame renderer with rgb_array mode...")
    render_env = RaycastCarEnv(render_mode="rgb_array", max_steps=100)
    render_env.reset(seed=42)
    frame = render_env.render()
    assert frame is not None, "Rendered frame is None!"
    assert frame.shape == (800, 1000, 3), f"Unexpected frame shape: {frame.shape}"
    render_env.close()
    print(f"  Frame rendered successfully with shape {frame.shape}!")

    print("\n[6/8] Testing Stable-Baselines3 PPO initialization on GPU...")
    if torch.cuda.is_available():
        gpu_model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, device="cuda", verbose=0)
        gpu_model.learn(total_timesteps=64)
        print(f"  PPO agent successfully learned 64 timesteps on GPU ({torch.cuda.get_device_name(0)})!")
    else:
        print("  CUDA not available, skipping GPU learn test.")

    print("\n[7/8] Testing crash trigger and shockwave rendering...")
    crash_env = RaycastCarEnv(render_mode="rgb_array", max_steps=100)
    crash_env.reset(seed=10)
    crash_env.render()
    crash_env.renderer.set_training_info({
        "device_name": "Test GPU",
        "vram_mb": 128.5,
        "episode": 1,
        "total_crashes": 1,
        "timesteps": 100,
        "status": "TESTING",
    })
    crash_env.renderer.trigger_crash((500.0, 400.0))
    crash_frame = crash_env.render()
    assert crash_frame is not None and crash_frame.shape == (800, 1000, 3)
    crash_env.close()
    print("  Crash animation and telemetry HUD rendered without errors!")

    env.close()
    print("\n" + "=" * 50)
    print("ALL ENVIRONMENT, GPU & VISUAL TESTS PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    test_environment()
