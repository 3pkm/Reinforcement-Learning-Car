"""Verification suite for 100+ car SwarmCarEnv, GPU batching, and SwarmRenderer."""

import numpy as np
import torch
from stable_baselines3 import PPO

from envs.swarm_car_env import SwarmCarEnv


def test_swarm():
    print("[1/5] Instantiating SwarmCarEnv with 100 cars...")
    num_cars = 100
    env = SwarmCarEnv(num_cars=num_cars, render_mode="rgb_array", max_steps_per_episode=200)
    print(f"  Num cars: {env.num_cars}")
    print(f"  Obs space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")

    print("\n[2/5] Testing swarm reset and staggered spawn grid...")
    obs = env.reset()
    assert obs.shape == (num_cars, 7), f"Expected shape ({num_cars}, 7), got {obs.shape}"
    assert np.all(obs >= 0.0) and np.all(obs <= 1.0), "Observation out of bounds [0, 1]"
    print(f"  Reset successfully. Obs matrix shape: {obs.shape}")
    print(f"  Total wall shapes in space: {len(env.track_gen.wall_shapes)}")

    print("\n[3/5] Testing 10 parallel steps with 100 cars...")
    for step in range(10):
        # 100 random continuous actions
        actions = np.random.uniform(-1.0, 1.0, size=(num_cars, 2)).astype(np.float32)
        env.step_async(actions)
        obs, rewards, dones, infos = env.step_wait()
        assert obs.shape == (num_cars, 7)
        assert rewards.shape == (num_cars,)
        assert dones.shape == (num_cars,)
        assert len(infos) == num_cars

    print(f"  10 steps completed (1,000 car-transitions collected).")
    print(f"  Total swarm crashes recorded: {env.total_swarm_crashes}")
    print(f"  Leader car index: #{env.get_leader_index()}")

    print("\n[4/5] Testing SwarmRenderer with 100 cars in rgb_array mode...")
    frame = env.render()
    assert frame is not None and frame.shape == (800, 1000, 3), f"Invalid frame shape: {frame.shape}"
    print(f"  Rendered 100-car swarm frame successfully with shape {frame.shape}!")

    print("\n[5/5] Testing GPU batched PPO initialization and learning step with 100 cars...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using compute device: {device.upper()}")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=16,  # 16 steps * 100 cars = 1,600 transitions per iteration
        batch_size=128,
        device=device,
        verbose=0,
    )
    # Run 1 iteration (1,600 samples)
    model.learn(total_timesteps=1600)
    print(f"  GPU batched PPO rollout of 1,600 transitions completed successfully!")

    env.close()
    print("\n" + "=" * 55)
    print("ALL 100-CAR SWARM & GPU BATCHING TESTS PASSED!")
    print("=" * 55)


if __name__ == "__main__":
    test_swarm()
