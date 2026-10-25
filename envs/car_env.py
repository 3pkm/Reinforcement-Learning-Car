"""Custom Gymnasium Environment for Autonomous 2D Car with Pymunk and Raycast LiDAR."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pymunk

from .car_physics import CarPhysics
from .raycast_sensor import RaycastSensor
from .track_generator import TrackGenerator


class RaycastCarEnv(gym.Env):
    """Reinforcement learning environment for continuous steering and throttle control on procedural racetracks."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        screen_width: int = 1000,
        screen_height: int = 800,
        max_steps: int = 1200,
        fps: int = 60,
        substeps: int = 5,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.max_steps = max_steps
        self.fps = fps
        self.substeps = substeps
        self.dt = 1.0 / self.fps
        self.sub_dt = self.dt / self.substeps

        # 1. Observation Space: 5 LiDAR rays + forward velocity + steering angle
        # All normalized to [0.0, 1.0]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32,
        )

        # 2. Action Space: Continuous [steering (-1.0 to 1.0), throttle/brake (-1.0 to 1.0)]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # Physics Space
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)

        # Collision detection (1 = Wall, 2 = Car)
        self.collided = False
        if hasattr(self.space, "on_collision"):
            self.space.on_collision(1, 2, begin=self._handle_collision)
        elif hasattr(self.space, "add_collision_handler"):
            collision_handler = self.space.add_collision_handler(1, 2)
            collision_handler.begin = self._handle_collision

        # Core Components
        self.track_gen = TrackGenerator(
            screen_width=self.screen_width,
            screen_height=self.screen_height,
        )
        self.car = CarPhysics(space=self.space)
        self.sensor = RaycastSensor(
            angles_deg=(-90.0, -45.0, 0.0, 45.0, 90.0),
            max_range=260.0,
        )

        # State tracking
        self.step_count = 0
        self.current_cp_idx = 0
        self.steps_since_checkpoint = 0
        self.total_checkpoints_passed = 0
        self.laps_completed = 0
        self.total_crashes = 0
        self.collision_pos = None

        # Renderer (lazy-initialized when render_mode is set)
        self.renderer = None

    def _handle_collision(self, arbiter, space, data) -> bool:
        """Callback triggered when car contacts a wall."""
        self.collided = True
        self.total_crashes += 1
        self.collision_pos = (float(self.car.body.position.x), float(self.car.body.position.y))
        return True

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment with a newly generated procedural track."""
        super().reset(seed=seed)

        self.collided = False
        self.collision_pos = None
        self.step_count = 0
        self.current_cp_idx = 0
        self.steps_since_checkpoint = 0
        self.total_checkpoints_passed = 0
        self.laps_completed = 0

        # Generate new procedural track geometry and populate physics space
        start_pos, start_angle = self.track_gen.generate(self.np_random)
        self.track_gen.build_physics_walls(self.space)

        # Position car at starting line facing track direction
        self.car.reset(start_pos, start_angle)

        # Perform initial sensor raycast
        ray_distances = self.sensor.cast_rays(self.space, self.car.body)
        obs = self._get_observation(ray_distances)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Executes one simulation step."""
        self.step_count += 1
        self.steps_since_checkpoint += 1

        # Clip and sanitize action
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Multi-substep physics integration to prevent tunneling
        for _ in range(self.substeps):
            self.car.apply_control(action, self.sub_dt)
            self.space.step(self.sub_dt)
            if self.collided:
                break

        # Query LiDAR sensors
        ray_distances = self.sensor.cast_rays(self.space, self.car.body)

        # Track progress and checkpoints
        checkpoint_reward = self._check_progress()

        # Compute velocity projected along the track tangent
        curr_cp = self.track_gen.checkpoints[self.current_cp_idx]
        car_vel = np.array([self.car.body.velocity.x, self.car.body.velocity.y], dtype=np.float32)
        forward_velocity_on_track = float(np.dot(car_vel, curr_cp.tangent))

        # Reward Calculation
        # 1. Base progress reward: velocity along forward track direction
        velocity_reward = forward_velocity_on_track / self.car.max_speed

        # 2. Checkpoint reward bonus for advancing through the circuit
        # 3. Small alive bonus
        alive_bonus = 0.05
        reward = velocity_reward + checkpoint_reward + alive_bonus

        # 4. Backward driving penalty
        if forward_velocity_on_track < -5.0:
            reward -= 0.5

        # Terminations and Truncations
        terminated = False
        truncated = False

        # Collision with wall
        if self.collided:
            reward = -100.0
            terminated = True

        # Episode step limit reached
        if self.step_count >= self.max_steps:
            truncated = True

        # Stuck or driving backward for too long
        if self.steps_since_checkpoint > 300:
            truncated = True

        obs = self._get_observation(ray_distances)
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def _check_progress(self) -> float:
        """Determines if the car has progressed past the next checkpoint."""
        num_cp = len(self.track_gen.checkpoints)
        if num_cp == 0:
            return 0.0

        next_cp_idx = (self.current_cp_idx + 1) % num_cp
        next_cp = self.track_gen.checkpoints[next_cp_idx]

        car_pos = np.array([self.car.body.position.x, self.car.body.position.y], dtype=np.float32)
        vec_to_next = car_pos - next_cp.center
        dist_to_next = float(np.linalg.norm(vec_to_next))

        # Check if car has passed the checkpoint plane or is close enough
        # Dot product with next checkpoint tangent indicates if car is passing across
        passed_plane = float(np.dot(vec_to_next, next_cp.tangent)) > 0.0

        if dist_to_next < (self.track_gen.track_width * 0.75) and passed_plane:
            self.current_cp_idx = next_cp_idx
            self.steps_since_checkpoint = 0
            self.total_checkpoints_passed += 1
            if next_cp_idx == 0:
                self.laps_completed += 1
                return 30.0  # Lap completion bonus
            return 8.0  # Checkpoint cleared bonus

        return 0.0

    def _get_observation(self, ray_distances: np.ndarray) -> np.ndarray:
        """Constructs normalized 7-dimensional observation vector in [0.0, 1.0]."""
        # 1-5: LiDAR normalized distances
        rays = np.clip(ray_distances, 0.0, 1.0)

        # 6: Forward velocity normalized to [0.0, 1.0]
        norm_vel = float(np.clip(self.car.forward_velocity / self.car.max_speed, 0.0, 1.0))

        # 7: Steering angle mapped from [-max_steer, +max_steer] to [0.0, 1.0]
        norm_steer = float(np.clip(
            (self.car.steer_angle / self.car.max_steer_angle + 1.0) * 0.5,
            0.0,
            1.0,
        ))

        obs = np.array(
            [rays[0], rays[1], rays[2], rays[3], rays[4], norm_vel, norm_steer],
            dtype=np.float32,
        )
        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Auxiliary telemetry info."""
        return {
            "step": self.step_count,
            "forward_velocity": self.car.forward_velocity,
            "steer_angle": self.car.steer_angle,
            "checkpoints_passed": self.total_checkpoints_passed,
            "laps_completed": self.laps_completed,
            "collided": self.collided,
            "collision_pos": self.collision_pos,
            "total_crashes": self.total_crashes,
        }

    def render(self):
        """Renders the environment via Pygame."""
        if self.renderer is None:
            try:
                from utils.renderer import PygameRenderer
            except ImportError:
                from ..utils.renderer import PygameRenderer

            self.renderer = PygameRenderer(
                env=self,
                screen_width=self.screen_width,
                screen_height=self.screen_height,
                fps=self.fps,
                render_mode=self.render_mode,
            )
        return self.renderer.render()

    def close(self):
        """Cleans up rendering display."""
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
