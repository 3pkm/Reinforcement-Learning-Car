"""High-performance Swarm / Multi-Agent Environment: 100+ cars simulated and trained in parallel."""

import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk
from stable_baselines3.common.vec_env import VecEnv

from .car_physics import CarPhysics
from .raycast_sensor import RaycastSensor
from .track_generator import TrackGenerator


class SwarmCarEnv(VecEnv):
    """Vectorized Gymnasium Environment simulating N (e.g. 100+) cars in parallel on the same procedural racetrack."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        num_cars: int = 100,
        render_mode: Optional[str] = "human",
        screen_width: int = 1000,
        screen_height: int = 800,
        max_steps_per_episode: int = 1200,
        fps: int = 60,
        substeps: int = 4,
    ):
        self.num_cars = num_cars
        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_steps_per_episode = max_steps_per_episode
        self.fps = fps
        self.substeps = substeps
        self.dt = 1.0 / self.fps
        self.sub_dt = self.dt / self.substeps

        # 1. Observation & Action Spaces (per car)
        observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )
        action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        super().__init__(self.num_cars, observation_space, action_space)

        # 2. Physics Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        # 3. Procedural Track Generator
        self.track_gen = TrackGenerator(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )

        # 4. Multi-Car Pool Setup
        self.cars: List[CarPhysics] = []
        self.sensors: List[RaycastSensor] = []
        self.car_collision_types: List[int] = []

        # Per-car state tracking
        self.steps = np.zeros(self.num_cars, dtype=np.int32)
        self.steps_since_cp = np.zeros(self.num_cars, dtype=np.int32)
        self.current_cp = np.zeros(self.num_cars, dtype=np.int32)
        self.total_checkpoints = np.zeros(self.num_cars, dtype=np.int32)
        self.laps = np.zeros(self.num_cars, dtype=np.int32)
        self.collided = np.zeros(self.num_cars, dtype=bool)

        # Global swarm metrics
        self.total_swarm_crashes = 0
        self.total_swarm_laps = 0
        self.recent_crashes: List[Tuple[float, float]] = []

        # Color palette for 100 cars
        self.car_colors = self._generate_car_colors(self.num_cars)

        # Instantiate 100 cars & sensors in the shared Pymunk space
        for i in range(self.num_cars):
            car = CarPhysics(space=self.space, length=32.0, width=16.0)
            # Assign unique collision type (100 + i) so we know exactly which car hit a wall
            car_col_type = 100 + i
            car.shape.collision_type = car_col_type
            self.car_collision_types.append(car_col_type)

            sensor = RaycastSensor(
                angles_deg=(-90.0, -45.0, 0.0, 45.0, 90.0),
                max_range=240.0,
            )
            self.cars.append(car)
            self.sensors.append(sensor)

            # Register collision callback for each car vs Wall (type 1)
            self._register_collision_handler(car_col_type, i)

        # Starting line spawn geometry
        self.start_pos = np.array([500.0, 400.0], dtype=np.float32)
        self.start_angle = 0.0
        self.start_tangent = np.array([1.0, 0.0], dtype=np.float32)
        self.start_normal = np.array([0.0, 1.0], dtype=np.float32)

        # Stored pending actions for VecEnv async interface
        self._pending_actions: Optional[np.ndarray] = None

        # Renderer
        self.renderer = None

    def _generate_car_colors(self, count: int) -> List[Tuple[int, int, int]]:
        """Generates visually distinct color palette for the swarm."""
        colors = []
        for i in range(count):
            hue = (i * 0.618033988749895) % 1.0  # Golden ratio hue distribution
            # Convert HSV to RGB
            import colorsys
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
            colors.append((int(r * 255), int(g * 255), int(b * 255)))
        return colors

    def _register_collision_handler(self, car_type: int, car_idx: int) -> None:
        """Registers a collision handler between a specific car and track walls."""
        def _on_collide(arbiter, space, data):
            self.collided[car_idx] = True
            pos = (float(self.cars[car_idx].body.position.x), float(self.cars[car_idx].body.position.y))
            self.recent_crashes.append(pos)
            self.total_swarm_crashes += 1
            return True

        if hasattr(self.space, "on_collision"):
            self.space.on_collision(1, car_type, begin=_on_collide)
        elif hasattr(self.space, "add_collision_handler"):
            handler = self.space.add_collision_handler(1, car_type)
            handler.begin = _on_collide

    def _compute_spawn_position(self, car_idx: int) -> Tuple[np.ndarray, float]:
        """Arranges 100 cars in a staggered starting grid behind the start line."""
        # Grid layout: 4 columns wide across track width, rows behind start line
        cols = 4
        col_idx = (car_idx % cols) - (cols - 1) / 2.0  # -1.5, -0.5, 0.5, 1.5
        row_idx = car_idx // cols  # 0, 1, 2, 3 ...

        lateral_spacing = 20.0
        row_spacing = 28.0

        offset = (
            self.start_normal * (col_idx * lateral_spacing)
            - self.start_tangent * (row_idx * row_spacing + 10.0)
        )
        spawn_pos = self.start_pos + offset
        return spawn_pos, self.start_angle

    def reset(self) -> np.ndarray:
        """Regenerates track and resets all 100 cars to the starting grid."""
        self.collided.fill(False)
        self.steps.fill(0)
        self.steps_since_cp.fill(0)
        self.current_cp.fill(0)
        self.total_checkpoints.fill(0)
        self.laps.fill(0)
        self.recent_crashes.clear()

        # Generate new procedural track geometry
        start_pos, start_angle = self.track_gen.generate(np.random.default_rng())
        self.start_pos = start_pos
        self.start_angle = start_angle

        if self.track_gen.checkpoints:
            self.start_tangent = self.track_gen.checkpoints[0].tangent.copy()
            self.start_normal = self.track_gen.checkpoints[0].normal.copy()

        self.track_gen.build_physics_walls(self.space)

        # Place all cars in staggered grid
        obs_list = []
        for i in range(self.num_cars):
            spawn_pos, angle = self._compute_spawn_position(i)
            self.cars[i].reset(spawn_pos, angle)
            # Raycast
            rays = self.sensors[i].cast_rays(self.space, self.cars[i].body)
            obs_list.append(self._get_car_observation(i, rays))

        if self.render_mode == "human":
            self.render()

        return np.array(obs_list, dtype=np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        """Stores actions for asynchronous stepping (VecEnv interface)."""
        self._pending_actions = actions

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        """Executes simulation step for all 100 cars in parallel."""
        actions = self._pending_actions
        assert actions is not None, "step_async must be called before step_wait"
        self._pending_actions = None

        self.steps += 1
        self.steps_since_cp += 1
        self.recent_crashes.clear()

        # 1. Multi-substep physics simulation for all cars
        for _ in range(self.substeps):
            for i in range(self.num_cars):
                self.cars[i].apply_control(actions[i], self.sub_dt)
            self.space.step(self.sub_dt)

        # 2. Process observations, rewards, and individual resets
        obs_list = []
        rewards = np.zeros(self.num_cars, dtype=np.float32)
        dones = np.zeros(self.num_cars, dtype=bool)
        infos: List[Dict[str, Any]] = []

        num_cp = len(self.track_gen.checkpoints)

        for i in range(self.num_cars):
            car = self.cars[i]
            sensor = self.sensors[i]

            # Raycast distances
            rays = sensor.cast_rays(self.space, car.body)

            # Checkpoint & forward velocity reward
            cp_reward = self._check_car_progress(i)
            curr_cp = self.track_gen.checkpoints[self.current_cp[i]]
            car_vel = np.array([car.body.velocity.x, car.body.velocity.y], dtype=np.float32)
            fwd_speed_on_track = float(np.dot(car_vel, curr_cp.tangent))

            reward = (fwd_speed_on_track / car.max_speed) + cp_reward + 0.05
            if fwd_speed_on_track < -5.0:
                reward -= 0.5

            is_collided = self.collided[i]
            is_timeout = self.steps[i] >= self.max_steps_per_episode
            is_stuck = self.steps_since_cp[i] > 260

            is_done = is_collided or is_timeout or is_stuck

            if is_collided:
                reward = -100.0

            rewards[i] = reward
            dones[i] = is_done

            info = {
                "step": int(self.steps[i]),
                "forward_velocity": float(car.forward_velocity),
                "checkpoints_passed": int(self.total_checkpoints[i]),
                "laps": int(self.laps[i]),
                "collided": bool(is_collided),
                "car_index": i,
            }
            infos.append(info)

            # INDEPENDENT PER-CAR RESPAWN:
            # If car i crashed or timed out, respawn ONLY car i to the starting grid!
            if is_done:
                self.collided[i] = False
                self.steps[i] = 0
                self.steps_since_cp[i] = 0
                self.current_cp[i] = 0
                spawn_pos, angle = self._compute_spawn_position(i)
                car.reset(spawn_pos, angle)
                rays = sensor.cast_rays(self.space, car.body)

            obs_list.append(self._get_car_observation(i, rays))

        if self.render_mode == "human":
            self.render()

        return np.array(obs_list, dtype=np.float32), rewards, dones, infos

    def _check_car_progress(self, car_idx: int) -> float:
        """Tracks checkpoint crossings for an individual car."""
        num_cp = len(self.track_gen.checkpoints)
        if num_cp == 0:
            return 0.0

        next_idx = (self.current_cp[car_idx] + 1) % num_cp
        next_cp = self.track_gen.checkpoints[next_idx]

        car = self.cars[car_idx]
        pos = np.array([car.body.position.x, car.body.position.y], dtype=np.float32)
        vec_to_next = pos - next_cp.center
        dist = float(np.linalg.norm(vec_to_next))
        passed_plane = float(np.dot(vec_to_next, next_cp.tangent)) > 0.0

        if dist < (self.track_gen.track_width * 0.75) and passed_plane:
            self.current_cp[car_idx] = next_idx
            self.steps_since_cp[car_idx] = 0
            self.total_checkpoints[car_idx] += 1
            if next_idx == 0:
                self.laps[car_idx] += 1
                self.total_swarm_laps += 1
                return 30.0  # Lap bonus
            return 8.0  # Checkpoint bonus

        return 0.0

    def _get_car_observation(self, car_idx: int, ray_distances: np.ndarray) -> np.ndarray:
        """Returns 7-dim normalized observation vector for a specific car."""
        car = self.cars[car_idx]
        rays = np.clip(ray_distances, 0.0, 1.0)
        norm_vel = float(np.clip(car.forward_velocity / car.max_speed, 0.0, 1.0))
        norm_steer = float(np.clip((car.steer_angle / car.max_steer_angle + 1.0) * 0.5, 0.0, 1.0))

        return np.array(
            [rays[0], rays[1], rays[2], rays[3], rays[4], norm_vel, norm_steer],
            dtype=np.float32,
        )

    def get_leader_index(self) -> int:
        """Returns the index of the car with the greatest track progress."""
        return int(np.argmax(self.total_checkpoints))

    def render(self):
        """Renders the entire 100-car swarm via SwarmRenderer."""
        if self.renderer is None:
            from utils.swarm_renderer import SwarmRenderer
            self.renderer = SwarmRenderer(
                env=self,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                fps=self.fps,
                render_mode=self.render_mode,
            )
        return self.renderer.render()

    def close(self):
        """Cleans up display."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # VecEnv abstract methods
    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        target_indices = self._get_indices(indices)
        results = []
        for _ in target_indices:
            if hasattr(self, method_name):
                fn = getattr(self, method_name)
                results.append(fn(*method_args, **method_kwargs))
            else:
                results.append(None)
        return results

    def get_attr(self, attr_name, indices=None):
        target_indices = self._get_indices(indices)
        val = getattr(self, attr_name, None)
        return [val for _ in target_indices]

    def set_attr(self, attr_name, value, indices=None):
        setattr(self, attr_name, value)
