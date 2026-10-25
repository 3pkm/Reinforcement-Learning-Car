"""Procedural closed-loop racetrack generator using periodic cubic splines and Pymunk physics boundaries."""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pymunk
from scipy.interpolate import CubicSpline


@dataclass
class Checkpoint:
    index: int
    center: np.ndarray  # [x, y]
    tangent: np.ndarray  # [tx, ty] unit forward vector
    normal: np.ndarray  # [nx, ny] unit perpendicular vector
    inner: np.ndarray  # [x, y]
    outer: np.ndarray  # [x, y]


class TrackGenerator:
    """Generates procedural closed-loop tracks with physics walls and checkpoints."""

    # Collision categories
    WALL_CATEGORY = 0b0001
    CAR_CATEGORY = 0b0010
    SENSOR_CATEGORY = 0b0100

    def __init__(
        self,
        screen_width: int = 1000,
        screen_height: int = 800,
        num_control_points: int = 14,
        track_width: float = 110.0,
        min_radius: float = 190.0,
        max_radius: float = 330.0,
        num_spline_points: int = 360,
        num_checkpoints: int = 36,
    ):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.center = np.array([screen_width / 2.0, screen_height / 2.0], dtype=np.float32)
        self.num_control_points = num_control_points
        self.track_width = track_width
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.num_spline_points = num_spline_points
        self.num_checkpoints = num_checkpoints

        # Track geometry
        self.centerline: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.inner_boundary: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.outer_boundary: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.checkpoints: List[Checkpoint] = []
        self.wall_shapes: List[pymunk.Segment] = []

    def generate(self, rng: np.random.Generator) -> Tuple[np.ndarray, float]:
        """Generates a new randomized track.

        Returns:
            Tuple of (start_position: np.ndarray, start_heading_angle: float)
        """
        # Step 1: Generate sorted polar control points around the center
        base_angles = np.linspace(0, 2 * np.pi, self.num_control_points, endpoint=False)
        angle_jitter = rng.uniform(
            -0.5 * (2 * np.pi / self.num_control_points),
            0.5 * (2 * np.pi / self.num_control_points),
            size=self.num_control_points,
        )
        angles = np.sort((base_angles + angle_jitter) % (2 * np.pi))

        radii = rng.uniform(self.min_radius, self.max_radius, size=self.num_control_points)

        control_x = self.center[0] + radii * np.cos(angles)
        control_y = self.center[1] + radii * np.sin(angles)

        # Step 2: Fit periodic cubic spline
        # Append first point to close the loop smoothly
        angles_closed = np.append(angles, angles[0] + 2 * np.pi)
        x_closed = np.append(control_x, control_x[0])
        y_closed = np.append(control_y, control_y[0])

        cs_x = CubicSpline(angles_closed, x_closed, bc_type="periodic")
        cs_y = CubicSpline(angles_closed, y_closed, bc_type="periodic")

        eval_angles = np.linspace(0, 2 * np.pi, self.num_spline_points, endpoint=False)
        center_x = cs_x(eval_angles)
        center_y = cs_y(eval_angles)
        self.centerline = np.column_stack((center_x, center_y)).astype(np.float32)

        # Step 3: Compute unit tangents and normals along the centerline
        # Forward finite differences with wrap-around
        next_pts = np.roll(self.centerline, -1, axis=0)
        prev_pts = np.roll(self.centerline, 1, axis=0)
        tangents = next_pts - prev_pts
        norms = np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-8
        tangents = tangents / norms

        # Normal vector: 90 degrees counter-clockwise to tangent (-ty, tx)
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))

        # Step 4: Compute inner and outer boundary vertices
        half_w = self.track_width / 2.0
        # In a counter-clockwise loop, inward points towards center
        self.inner_boundary = self.centerline - normals * half_w
        self.outer_boundary = self.centerline + normals * half_w

        # Step 5: Generate evenly spaced checkpoints
        step = max(1, self.num_spline_points // self.num_checkpoints)
        self.checkpoints = []
        for i, idx in enumerate(range(0, self.num_spline_points, step)):
            if len(self.checkpoints) >= self.num_checkpoints:
                break
            cp = Checkpoint(
                index=i,
                center=self.centerline[idx].copy(),
                tangent=tangents[idx].copy(),
                normal=normals[idx].copy(),
                inner=self.inner_boundary[idx].copy(),
                outer=self.outer_boundary[idx].copy(),
            )
            self.checkpoints.append(cp)

        # Start position and initial heading angle (facing along tangent at idx=0)
        start_pos = self.centerline[0].copy()
        start_angle = float(np.arctan2(tangents[0, 1], tangents[0, 0]))

        return start_pos, start_angle

    def build_physics_walls(self, space: pymunk.Space) -> None:
        """Constructs static Pymunk Segment shapes for inner and outer boundaries."""
        # Clean previous wall shapes if present
        for shape in self.wall_shapes:
            if shape in space.shapes:
                space.remove(shape)
        self.wall_shapes.clear()

        wall_filter = pymunk.ShapeFilter(
            categories=self.WALL_CATEGORY,
            mask=self.CAR_CATEGORY | self.SENSOR_CATEGORY,
        )

        n = len(self.centerline)
        for i in range(n):
            next_i = (i + 1) % n

            # Inner wall segment
            p1_in = tuple(self.inner_boundary[i])
            p2_in = tuple(self.inner_boundary[next_i])
            seg_in = pymunk.Segment(space.static_body, p1_in, p2_in, radius=2.5)
            seg_in.elasticity = 0.0
            seg_in.friction = 0.8
            seg_in.filter = wall_filter
            seg_in.collision_type = 1  # 1 = Wall collision type
            space.add(seg_in)
            self.wall_shapes.append(seg_in)

            # Outer wall segment
            p1_out = tuple(self.outer_boundary[i])
            p2_out = tuple(self.outer_boundary[next_i])
            seg_out = pymunk.Segment(space.static_body, p1_out, p2_out, radius=2.5)
            seg_out.elasticity = 0.0
            seg_out.friction = 0.8
            seg_out.filter = wall_filter
            seg_out.collision_type = 1
            space.add(seg_out)
            self.wall_shapes.append(seg_out)
