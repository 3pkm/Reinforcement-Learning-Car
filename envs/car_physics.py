"""2D Top-down vehicle physics simulation using Pymunk rigid bodies and tire friction dynamics."""

from typing import Tuple
import numpy as np
import pymunk


class CarPhysics:
    """Rigid-body car with realistic top-down acceleration, steering, and lateral tire grip."""

    def __init__(
        self,
        space: pymunk.Space,
        length: float = 36.0,
        width: float = 18.0,
        mass: float = 1.0,
        max_speed: float = 300.0,
        engine_power: float = 650.0,
        braking_power: float = 500.0,
        max_steer_angle: float = np.radians(35.0),
        steer_rate: float = 5.0,
        lateral_grip: float = 0.88,
    ):
        self.space = space
        self.length = length
        self.width = width
        self.mass = mass
        self.max_speed = max_speed
        self.engine_power = engine_power
        self.braking_power = braking_power
        self.max_steer_angle = max_steer_angle
        self.steer_rate = steer_rate
        self.lateral_grip = lateral_grip

        # State tracking
        self.steer_angle: float = 0.0

        # Create Pymunk body and shape
        moment = pymunk.moment_for_box(self.mass, (self.length, self.width))
        self.body = pymunk.Body(self.mass, moment)
        self.shape = pymunk.Poly.create_box(self.body, (self.length, self.width))
        self.shape.elasticity = 0.0
        self.shape.friction = 0.5
        self.shape.collision_type = 2  # 2 = Car collision type

        # Filter: Car collides with walls (category 1), not with raycast sensors
        from .track_generator import TrackGenerator

        self.shape.filter = pymunk.ShapeFilter(
            categories=TrackGenerator.CAR_CATEGORY,
            mask=TrackGenerator.WALL_CATEGORY,
        )

        self.space.add(self.body, self.shape)

    def reset(self, position: np.ndarray, heading_angle: float) -> None:
        """Resets car position, heading, and dynamic states."""
        self.body.position = pymunk.Vec2d(float(position[0]), float(position[1]))
        self.body.angle = float(heading_angle)
        self.body.velocity = pymunk.Vec2d(0.0, 0.0)
        self.body.angular_velocity = 0.0
        self.steer_angle = 0.0

    @property
    def forward_vector(self) -> np.ndarray:
        """Unit vector in the car's forward heading direction."""
        return np.array([np.cos(self.body.angle), np.sin(self.body.angle)], dtype=np.float32)

    @property
    def lateral_vector(self) -> np.ndarray:
        """Unit vector perpendicular to the car's heading (pointing right)."""
        return np.array([-np.sin(self.body.angle), np.cos(self.body.angle)], dtype=np.float32)

    @property
    def forward_velocity(self) -> float:
        """Scalar forward velocity component."""
        vel = np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32)
        return float(np.dot(vel, self.forward_vector))

    @property
    def lateral_velocity(self) -> float:
        """Scalar lateral (sideways drift) velocity component."""
        vel = np.array([self.body.velocity.x, self.body.velocity.y], dtype=np.float32)
        return float(np.dot(vel, self.lateral_vector))

    def apply_control(self, action: np.ndarray, dt: float) -> None:
        """Applies steering and throttle/brake actions with realistic vehicle kinematics.

        Args:
            action: [steering in [-1, 1], throttle_brake in [-1, 1]]
            dt: Physics integration timestep
        """
        steer_target = float(np.clip(action[0], -1.0, 1.0)) * self.max_steer_angle
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        # Smooth steering transition
        self.steer_angle += np.clip(
            steer_target - self.steer_angle,
            -self.steer_rate * dt,
            self.steer_rate * dt,
        )

        # 1. Drive force along forward heading
        forward_dir = pymunk.Vec2d(float(np.cos(self.body.angle)), float(np.sin(self.body.angle)))
        if throttle_brake >= 0.0:
            force_mag = throttle_brake * self.engine_power
        else:
            force_mag = throttle_brake * self.braking_power

        # Cap maximum speed
        current_fwd_speed = self.forward_velocity
        if current_fwd_speed > self.max_speed and throttle_brake > 0.0:
            force_mag = 0.0
        elif current_fwd_speed < -self.max_speed * 0.4 and throttle_brake < 0.0:
            force_mag = 0.0

        self.body.apply_force_at_local_point(pymunk.Vec2d(force_mag, 0.0), (0.0, 0.0))

        # 2. Tire Grip Dynamics: cancel lateral velocity component to simulate tire grip
        vel_fwd = self.forward_velocity
        vel_lat = self.lateral_velocity

        # Dampen lateral velocity based on grip
        damped_lat = vel_lat * (1.0 - self.lateral_grip)
        new_vx = float(np.cos(self.body.angle) * vel_fwd - np.sin(self.body.angle) * damped_lat)
        new_vy = float(np.sin(self.body.angle) * vel_fwd + np.cos(self.body.angle) * damped_lat)
        self.body.velocity = pymunk.Vec2d(new_vx, new_vy)

        # 3. Ackermann/Bicycle steering model: turning angular velocity is proportional to forward velocity
        # omega = (v / wheelbase) * tan(steering_angle)
        wheelbase = self.length * 0.75
        target_angular_vel = (vel_fwd / wheelbase) * np.tan(self.steer_angle)
        self.body.angular_velocity = float(target_angular_vel)

        # Passive rolling friction
        self.body.velocity = self.body.velocity * 0.992
