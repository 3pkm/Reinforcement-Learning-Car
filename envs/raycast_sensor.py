"""5-Ray LiDAR sensor using Pymunk segment queries for autonomous vehicle observation."""

from typing import List, Tuple
import numpy as np
import pymunk
from .track_generator import TrackGenerator


class RaycastSensor:
    """Simulates 5-ray LiDAR rangefinder around the vehicle."""

    def __init__(
        self,
        angles_deg: Tuple[float, ...] = (-90.0, -45.0, 0.0, 45.0, 90.0),
        max_range: float = 260.0,
    ):
        self.angles_rad = [np.radians(a) for a in angles_deg]
        self.max_range = float(max_range)

        # ShapeFilter: only queries static track walls (WALL_CATEGORY)
        self.sensor_filter = pymunk.ShapeFilter(
            categories=TrackGenerator.SENSOR_CATEGORY,
            mask=TrackGenerator.WALL_CATEGORY,
        )

        # Cached raycast results for observation and rendering
        # Each entry: (start_pos, end_pos, hit_point, normalized_dist)
        self.latest_rays: List[Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], float]] = []

    def cast_rays(self, space: pymunk.Space, car_body: pymunk.Body) -> np.ndarray:
        """Casts 5 rays from car center at specified angles.

        Returns:
            np.ndarray of shape (5,) with normalized distances in [0.0, 1.0].
        """
        car_pos = car_body.position
        car_angle = car_body.angle

        distances = []
        self.latest_rays.clear()

        for rel_angle in self.angles_rad:
            total_angle = car_angle + rel_angle
            dx = float(np.cos(total_angle) * self.max_range)
            dy = float(np.sin(total_angle) * self.max_range)

            start = (car_pos.x, car_pos.y)
            end = (car_pos.x + dx, car_pos.y + dy)

            # Query Pymunk space for intersection with walls
            first_hit = space.segment_query_first(
                start=pymunk.Vec2d(*start),
                end=pymunk.Vec2d(*end),
                radius=0.0,
                shape_filter=self.sensor_filter,
            )

            if first_hit is not None and first_hit.shape is not None:
                # alpha is in [0.0, 1.0] representing hit fraction along ray
                norm_dist = float(np.clip(first_hit.alpha, 0.0, 1.0))
                hit_pt = (float(first_hit.point.x), float(first_hit.point.y))
            else:
                norm_dist = 1.0
                hit_pt = end

            distances.append(norm_dist)
            self.latest_rays.append((start, end, hit_pt, norm_dist))

        return np.array(distances, dtype=np.float32)
