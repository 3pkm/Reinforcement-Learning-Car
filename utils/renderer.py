"""Real-time Pygame renderer, crash impact visualizer, and GPU telemetry HUD for RaycastCarEnv."""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame


class PygameRenderer:
    """Renders racetrack, car kinematics, raycast LiDAR lasers, crash animations, and GPU telemetry HUD."""

    # Color palette
    BG_COLOR = (24, 26, 33)
    TRACK_SURFACE_COLOR = (42, 45, 56)
    WALL_COLOR = (80, 160, 240)
    WALL_GLOW = (40, 90, 180)
    CAR_COLOR = (245, 65, 65)
    CAR_ACCENT = (255, 210, 50)
    CAR_GLASS = (30, 30, 40)
    CHECKPOINT_ACTIVE = (255, 215, 0)
    CHECKPOINT_INACTIVE = (60, 70, 90)
    FINISH_LINE = (50, 220, 120)
    TEXT_COLOR = (240, 240, 245)
    HUD_BG = (15, 17, 22, 220)
    CRASH_RED = (255, 45, 45)
    GPU_GREEN = (50, 255, 130)

    def __init__(
        self,
        env,
        screen_width: int = 1000,
        screen_height: int = 800,
        fps: int = 60,
        render_mode: Optional[str] = "human",
    ):
        self.env = env
        self.width = screen_width
        self.height = screen_height
        self.fps = fps
        self.render_mode = render_mode

        # Training telemetry data
        self.training_info: Dict[str, Any] = {}

        # Crash animation state
        self.crash_frame = 0
        self.crash_max_frames = 18
        self.crash_pos: Optional[Tuple[float, float]] = None
        self.crash_particles: List[Dict[str, Any]] = []

        pygame.init()
        pygame.font.init()

        if self.render_mode == "human":
            pygame.display.set_caption("Autonomous 2D RL Car - Real-time GPU Training")
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 13)
        self.header_font = pygame.font.SysFont("Consolas", 15, bold=True)
        self.crash_font = pygame.font.SysFont("Consolas", 28, bold=True)
        self.banner_font = pygame.font.SysFont("Consolas", 12, bold=True)

    def set_training_info(self, info: Dict[str, Any]) -> None:
        """Updates live training telemetry (GPU, episodes, loss, etc.)."""
        self.training_info = info

    def trigger_crash(self, pos: Tuple[float, float]) -> None:
        """Initializes a visual collision animation with shockwave and sparks."""
        self.crash_frame = self.crash_max_frames
        self.crash_pos = (float(pos[0]), float(pos[1]))
        self.crash_particles = []

        # Generate spark particles
        for _ in range(25):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2.5, 9.0)
            self.crash_particles.append({
                "x": float(pos[0]),
                "y": float(pos[1]),
                "vx": float(np.cos(angle) * speed),
                "vy": float(np.sin(angle) * speed),
                "radius": random.uniform(2.0, 5.0),
                "color": random.choice([(255, 60, 60), (255, 180, 40), (255, 255, 255)]),
            })

    def process_events(self) -> Dict[str, Any]:
        """Pumps Pygame events and returns control signals for interactive visual training."""
        signals = {
            "pause_toggled": False,
            "render_toggled": False,
            "fps_change": 0,
            "save_requested": False,
            "quit_requested": False,
            "reset_track": False,
        }

        if self.render_mode != "human":
            return signals

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                signals["quit_requested"] = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    signals["quit_requested"] = True
                elif event.key == pygame.K_SPACE:
                    signals["pause_toggled"] = True
                elif event.key == pygame.K_v:
                    signals["render_toggled"] = True
                elif event.key == pygame.K_UP:
                    signals["fps_change"] = +30
                elif event.key == pygame.K_DOWN:
                    signals["fps_change"] = -30
                elif event.key == pygame.K_s:
                    signals["save_requested"] = True
                elif event.key == pygame.K_r:
                    signals["reset_track"] = True

        return signals

    def render(self):
        """Draws current frame, handles crash effects, and renders HUDs."""
        # Auto-trigger crash animation if env collided this step
        if getattr(self.env, "collided", False) and self.crash_frame == 0:
            cpos = getattr(self.env, "collision_pos", None)
            if cpos is None:
                cpos = (float(self.env.car.body.position.x), float(self.env.car.body.position.y))
            self.trigger_crash(cpos)

        self.screen.fill(self.BG_COLOR)

        # 1. Track Boundaries & Surface
        self._draw_track()

        # 2. Checkpoints
        self._draw_checkpoints()

        # 3. LiDAR Raycasts
        self._draw_raycasts()

        # 4. Car Body
        self._draw_car()

        # 5. Active Crash Effects (Shockwave, Sparks, Alert)
        if self.crash_frame > 0:
            self._draw_crash_effects()
            self.crash_frame -= 1

        # 6. Car Telemetry HUD (Top-Left)
        self._draw_car_hud()

        # 7. GPU Training Telemetry HUD (Top-Right)
        if self.training_info:
            self._draw_training_hud()

        # 8. Interactive Control Keybindings Bar (Bottom)
        self._draw_controls_bar()

        if self.render_mode == "human":
            pygame.display.flip()
            if self.fps > 0:
                self.clock.tick(self.fps)
            return None
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)),
                axes=(1, 0, 2),
            )

    def _draw_track(self):
        """Draws track surface and boundaries with smooth borders."""
        tg = self.env.track_gen
        if len(tg.inner_boundary) < 3 or len(tg.outer_boundary) < 3:
            return

        outer_pts = [(float(p[0]), float(p[1])) for p in tg.outer_boundary]
        inner_pts = [(float(p[0]), float(p[1])) for p in tg.inner_boundary]

        pygame.draw.polygon(self.screen, self.TRACK_SURFACE_COLOR, outer_pts)
        pygame.draw.polygon(self.screen, self.BG_COLOR, inner_pts)

        # Glow & border lines
        pygame.draw.lines(self.screen, self.WALL_GLOW, True, outer_pts, width=5)
        pygame.draw.lines(self.screen, self.WALL_COLOR, True, outer_pts, width=2)
        pygame.draw.lines(self.screen, self.WALL_GLOW, True, inner_pts, width=5)
        pygame.draw.lines(self.screen, self.WALL_COLOR, True, inner_pts, width=2)

    def _draw_checkpoints(self):
        """Draws track progression checkpoints."""
        tg = self.env.track_gen
        curr_idx = self.env.current_cp_idx
        num_cp = len(tg.checkpoints)

        for i, cp in enumerate(tg.checkpoints):
            p1 = (int(cp.inner[0]), int(cp.inner[1]))
            p2 = (int(cp.outer[0]), int(cp.outer[1]))

            if i == 0:
                pygame.draw.line(self.screen, self.FINISH_LINE, p1, p2, width=3)
            elif i == (curr_idx + 1) % max(1, num_cp):
                pygame.draw.line(self.screen, self.CHECKPOINT_ACTIVE, p1, p2, width=2)
            else:
                pygame.draw.line(self.screen, self.CHECKPOINT_INACTIVE, p1, p2, width=1)

    def _draw_raycasts(self):
        """Draws LiDAR sensor rays with distance-based color transitions."""
        sensor = self.env.sensor

        for ray in sensor.latest_rays:
            start_pt, end_pt, hit_pt, norm_dist = ray
            sp = (int(start_pt[0]), int(start_pt[1]))
            hp = (int(hit_pt[0]), int(hit_pt[1]))

            if norm_dist < 0.28:
                laser_color = (255, 60, 60)
            elif norm_dist < 0.65:
                laser_color = (255, 210, 40)
            else:
                laser_color = (60, 240, 160)

            pygame.draw.line(self.screen, laser_color, sp, hp, width=2)

            if norm_dist < 0.99:
                pygame.draw.circle(self.screen, laser_color, hp, radius=4)
                pygame.draw.circle(self.screen, (255, 255, 255), hp, radius=2)

    def _draw_car(self):
        """Draws the rotated vehicle body and headlights."""
        car = self.env.car
        pos = (car.body.position.x, car.body.position.y)
        angle = car.body.angle

        l_half = car.length / 2.0
        w_half = car.width / 2.0

        local_corners = [
            (l_half, -w_half),
            (l_half, w_half),
            (-l_half, w_half),
            (-l_half, -w_half),
        ]

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        world_corners = []
        for lx, ly in local_corners:
            wx = float(pos[0] + (lx * cos_a - ly * sin_a))
            wy = float(pos[1] + (lx * sin_a + ly * cos_a))
            world_corners.append((wx, wy))

        # Main chassis
        pygame.draw.polygon(self.screen, self.CAR_COLOR, world_corners)
        pygame.draw.polygon(self.screen, (255, 255, 255), world_corners, width=1)

        # Cabin glass
        cabin_pts = [
            (l_half * 0.4, -w_half * 0.7),
            (l_half * 0.4, w_half * 0.7),
            (-l_half * 0.3, w_half * 0.7),
            (-l_half * 0.3, -w_half * 0.7),
        ]
        world_cabin = []
        for lx, ly in cabin_pts:
            wx = float(pos[0] + (lx * cos_a - ly * sin_a))
            wy = float(pos[1] + (lx * sin_a + ly * cos_a))
            world_cabin.append((wx, wy))
        pygame.draw.polygon(self.screen, self.CAR_GLASS, world_cabin)

        # Headlights
        front_left = world_corners[0]
        front_right = world_corners[1]
        pygame.draw.circle(self.screen, self.CAR_ACCENT, (int(front_left[0]), int(front_left[1])), radius=3)
        pygame.draw.circle(self.screen, self.CAR_ACCENT, (int(front_right[0]), int(front_right[1])), radius=3)

    def _draw_crash_effects(self):
        """Draws visual explosion, expanding shockwave, debris sparks, and red border vignette."""
        if self.crash_pos is None:
            return

        cx, cy = int(self.crash_pos[0]), int(self.crash_pos[1])
        progress = 1.0 - (self.crash_frame / self.crash_max_frames)  # 0.0 -> 1.0

        # 1. Red Border Vignette Alert
        border_thickness = int(12 * (1.0 - progress))
        if border_thickness > 0:
            pygame.draw.rect(
                self.screen,
                self.CRASH_RED,
                (0, 0, self.width, self.height),
                width=border_thickness,
            )

        # 2. Expanding Shockwave Ring
        shock_radius = int(15 + progress * 65)
        shock_width = max(1, int(5 * (1.0 - progress)))
        shock_surface = pygame.Surface((shock_radius * 2 + 10, shock_radius * 2 + 10), pygame.SRCALPHA)
        alpha = int(240 * (1.0 - progress))
        pygame.draw.circle(
            shock_surface,
            (255, 45, 45, alpha),
            (shock_radius + 5, shock_radius + 5),
            shock_radius,
            width=shock_width,
        )
        self.screen.blit(shock_surface, (cx - shock_radius - 5, cy - shock_radius - 5))

        # 3. Dynamic Spark Particles
        for p in self.crash_particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.94
            p["vy"] *= 0.94
            r = max(1, int(p["radius"] * (1.0 - progress)))
            pygame.draw.circle(self.screen, p["color"], (int(p["x"]), int(p["y"])), r)

        # 4. Impact Banner Alert (Centered above track)
        banner_w, banner_h = 320, 48
        bx = (self.width - banner_w) // 2
        by = 24
        banner_surface = pygame.Surface((banner_w, banner_h), pygame.SRCALPHA)
        banner_surface.fill((180, 20, 20, int(220 * (1.0 - progress))))
        pygame.draw.rect(banner_surface, (255, 255, 255), (0, 0, banner_w, banner_h), width=2, border_radius=6)

        alert_text = self.crash_font.render("! WALL COLLISION !", True, (255, 255, 255))
        text_rect = alert_text.get_rect(center=(banner_w // 2, banner_h // 2))
        banner_surface.blit(alert_text, text_rect)
        self.screen.blit(banner_surface, (bx, by))

    def _draw_car_hud(self):
        """Draws vehicle telemetry HUD in top-left."""
        hud_w, hud_h = 240, 200
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill(self.HUD_BG)
        pygame.draw.rect(hud_surface, (70, 80, 100), (0, 0, hud_w, hud_h), width=1, border_radius=4)

        title = self.header_font.render("VEHICLE TELEMETRY", True, (255, 215, 0))
        hud_surface.blit(title, (12, 10))

        car = self.env.car
        speed = abs(car.forward_velocity)
        steer_deg = np.degrees(car.steer_angle)
        cp_total = len(self.env.track_gen.checkpoints)
        cp_current = (self.env.current_cp_idx + 1) % max(1, cp_total)

        lines = [
            f"Speed:      {speed:5.1f} px/s",
            f"Steering:   {steer_deg:+5.1f} deg",
            f"Checkpoint: {cp_current:02d} / {cp_total:02d}",
            f"Laps:       {self.env.laps_completed}",
            f"Step:       {self.env.step_count:4d} / {self.env.max_steps}",
        ]

        y_offset = 32
        for line in lines:
            text = self.font.render(line, True, self.TEXT_COLOR)
            hud_surface.blit(text, (12, y_offset))
            y_offset += 17

        # 5 LiDAR range bars
        y_offset += 4
        bar_label = self.font.render("LiDAR Range (L -> R):", True, (180, 200, 220))
        hud_surface.blit(bar_label, (12, y_offset))
        y_offset += 18

        bar_w = 34
        bar_spacing = 8
        start_x = 14
        sensor_distances = [ray[3] for ray in self.env.sensor.latest_rays]

        for i, dist in enumerate(sensor_distances):
            bx = start_x + i * (bar_w + bar_spacing)
            pygame.draw.rect(hud_surface, (40, 45, 55), (bx, y_offset, bar_w, 9), border_radius=2)
            fill_w = int(bar_w * dist)
            if dist < 0.28:
                col = (255, 60, 60)
            elif dist < 0.65:
                col = (255, 210, 40)
            else:
                col = (60, 240, 160)
            if fill_w > 0:
                pygame.draw.rect(hud_surface, col, (bx, y_offset, fill_w, 9), border_radius=2)

        self.screen.blit(hud_surface, (16, 16))

    def _draw_training_hud(self):
        """Draws GPU training status, crash statistics, and VRAM in top-right."""
        hud_w, hud_h = 280, 185
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill(self.HUD_BG)
        pygame.draw.rect(hud_surface, (70, 80, 100), (0, 0, hud_w, hud_h), width=1, border_radius=4)

        # Header with pulsating active indicator
        title = self.header_font.render("GPU TRAINING MONITOR", True, self.GPU_GREEN)
        hud_surface.blit(title, (12, 10))

        device_name = self.training_info.get("device_name", "GPU")
        # Shorten long device name if needed
        if len(device_name) > 26:
            device_name = device_name[:24] + ".."
        vram_mb = self.training_info.get("vram_mb", 0.0)
        episode = self.training_info.get("episode", 0)
        crashes = self.training_info.get("total_crashes", 0)
        timesteps = self.training_info.get("timesteps", 0)
        status = self.training_info.get("status", "TRAINING")
        fps_disp = f"{self.fps} FPS" if self.fps > 0 else "MAX / UNCAPPED"

        status_col = (255, 215, 0) if "PAUSED" in status else self.GPU_GREEN

        lines = [
            ("Device:   ", device_name, self.GPU_GREEN),
            ("VRAM:     ", f"{vram_mb:.1f} MB (Active)", self.TEXT_COLOR),
            ("Status:   ", status, status_col),
            ("Episode:  ", f"#{episode:,}", self.TEXT_COLOR),
            ("Crashes:  ", f"{crashes:,}", self.CRASH_RED if crashes > 0 else self.TEXT_COLOR),
            ("Steps:    ", f"{timesteps:,}", self.TEXT_COLOR),
            ("Speed:    ", fps_disp, self.TEXT_COLOR),
        ]

        y_offset = 32
        for label, val, val_col in lines:
            lbl_surf = self.font.render(label, True, (170, 180, 195))
            val_surf = self.font.render(val, True, val_col)
            hud_surface.blit(lbl_surf, (12, y_offset))
            hud_surface.blit(val_surf, (95, y_offset))
            y_offset += 17

        self.screen.blit(hud_surface, (self.width - hud_w - 16, 16))

    def _draw_controls_bar(self):
        """Draws interactive hotkey instruction banner at screen bottom."""
        bar_h = 24
        bar_surface = pygame.Surface((self.width, bar_h), pygame.SRCALPHA)
        bar_surface.fill((10, 12, 16, 210))

        controls_str = "[SPACE] Pause/Resume  |  [V] Toggle View  |  [UP/DOWN] Speed  |  [S] Save  |  [R] New Track  |  [ESC] Exit"
        text = self.banner_font.render(controls_str, True, (200, 210, 225))
        text_rect = text.get_rect(center=(self.width // 2, bar_h // 2))
        bar_surface.blit(text, text_rect)

        self.screen.blit(bar_surface, (0, self.height - bar_h))

    def close(self):
        """Shuts down Pygame display cleanly."""
        pygame.quit()
