"""High-performance Swarm Renderer for visualizing 100+ racing cars, crash particle bursts, and GPU stats."""

import random
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pygame


class SwarmRenderer:
    """Renders 100+ cars simultaneously, highlights the race leader, manages multi-crash particles, and displays GPU swarm telemetry."""

    BG_COLOR = (20, 22, 28)
    TRACK_SURFACE_COLOR = (38, 41, 52)
    WALL_COLOR = (70, 150, 235)
    WALL_GLOW = (35, 80, 170)
    LEADER_GOLD = (255, 215, 0)
    LEADER_AURA = (255, 215, 0, 90)
    FINISH_LINE = (50, 220, 120)
    CHECKPOINT_ACTIVE = (255, 215, 0)
    CHECKPOINT_INACTIVE = (55, 65, 85)
    HUD_BG = (12, 14, 18, 225)
    GPU_GREEN = (50, 255, 130)
    CRASH_RED = (255, 55, 55)

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

        self.training_info: Dict[str, Any] = {}
        self.particles: List[Dict[str, Any]] = []

        pygame.init()
        pygame.font.init()

        if self.render_mode == "human":
            pygame.display.set_caption(f"Autonomous 2D RL Car - Swarm Arena ({self.env.num_cars} Cars Parallel)")
            self.screen = pygame.display.set_mode((self.width, self.height))
        else:
            self.screen = pygame.Surface((self.width, self.height))

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Consolas", 12)
        self.header_font = pygame.font.SysFont("Consolas", 14, bold=True)
        self.banner_font = pygame.font.SysFont("Consolas", 12, bold=True)

    def set_training_info(self, info: Dict[str, Any]) -> None:
        """Updates live training telemetry."""
        self.training_info = info

    def trigger_crash(self, pos: Tuple[float, float]) -> None:
        """Spawns spark particles for a car crash event."""
        for _ in range(12):
            angle = random.uniform(0, 2 * np.pi)
            speed = random.uniform(2.0, 7.5)
            self.particles.append({
                "x": float(pos[0]),
                "y": float(pos[1]),
                "vx": float(np.cos(angle) * speed),
                "vy": float(np.sin(angle) * speed),
                "life": 14,
                "max_life": 14,
                "radius": random.uniform(1.8, 3.8),
                "color": random.choice([(255, 60, 60), (255, 200, 40), (255, 255, 255)]),
            })

    def process_events(self) -> Dict[str, Any]:
        """Pumps Pygame events and returns control signals."""
        signals = {
            "pause_toggled": False,
            "render_toggled": False,
            "fps_change": 0,
            "save_requested": False,
            "quit_requested": False,
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

        return signals

    def render(self):
        """Draws track, 100 cars, leader raycasts, crash bursts, and dual HUDs."""
        # Auto-trigger crashes for any newly collided cars this step
        for cpos in self.env.recent_crashes:
            self.trigger_crash(cpos)

        self.screen.fill(self.BG_COLOR)

        # 1. Track & Boundaries
        self._draw_track()

        # 2. Checkpoints
        self._draw_checkpoints()

        # 3. Render 100 Cars & Leader Highlighting
        leader_idx = self.env.get_leader_index()
        self._draw_swarm_cars(leader_idx)

        # 4. Render Leader Raycasts
        self._draw_leader_raycasts(leader_idx)

        # 5. Render Crash Particle System
        self._draw_particles()

        # 6. Swarm Left HUD
        self._draw_swarm_hud(leader_idx)

        # 7. GPU Training Right HUD
        self._draw_gpu_hud()

        # 8. Bottom Controls Banner
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
        """Draws track asphalt and boundary lines."""
        tg = self.env.track_gen
        if len(tg.inner_boundary) < 3 or len(tg.outer_boundary) < 3:
            return

        outer_pts = [(float(p[0]), float(p[1])) for p in tg.outer_boundary]
        inner_pts = [(float(p[0]), float(p[1])) for p in tg.inner_boundary]

        pygame.draw.polygon(self.screen, self.TRACK_SURFACE_COLOR, outer_pts)
        pygame.draw.polygon(self.screen, self.BG_COLOR, inner_pts)

        pygame.draw.lines(self.screen, self.WALL_GLOW, True, outer_pts, width=5)
        pygame.draw.lines(self.screen, self.WALL_COLOR, True, outer_pts, width=2)
        pygame.draw.lines(self.screen, self.WALL_GLOW, True, inner_pts, width=5)
        pygame.draw.lines(self.screen, self.WALL_COLOR, True, inner_pts, width=2)

    def _draw_checkpoints(self):
        """Draws checkpoints."""
        tg = self.env.track_gen
        leader_cp = self.env.current_cp[self.env.get_leader_index()]
        num_cp = len(tg.checkpoints)

        for i, cp in enumerate(tg.checkpoints):
            p1 = (int(cp.inner[0]), int(cp.inner[1]))
            p2 = (int(cp.outer[0]), int(cp.outer[1]))

            if i == 0:
                pygame.draw.line(self.screen, self.FINISH_LINE, p1, p2, width=3)
            elif i == (leader_cp + 1) % max(1, num_cp):
                pygame.draw.line(self.screen, self.CHECKPOINT_ACTIVE, p1, p2, width=2)
            else:
                pygame.draw.line(self.screen, self.CHECKPOINT_INACTIVE, p1, p2, width=1)

    def _draw_swarm_cars(self, leader_idx: int):
        """Renders all 100 cars with distinct colors, highlighting the race leader."""
        # First draw non-leader cars
        for i in range(self.env.num_cars):
            if i == leader_idx:
                continue
            self._draw_single_car(i, is_leader=False)

        # Draw leader on top with golden glow
        if leader_idx < self.env.num_cars:
            self._draw_single_car(leader_idx, is_leader=True)

    def _draw_single_car(self, i: int, is_leader: bool = False):
        """Draws an individual rotated car."""
        car = self.env.cars[i]
        pos = (car.body.position.x, car.body.position.y)
        angle = car.body.angle

        l_half = car.length / 2.0
        w_half = car.width / 2.0

        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        corners = [
            (l_half, -w_half),
            (l_half, w_half),
            (-l_half, w_half),
            (-l_half, -w_half),
        ]

        world_pts = []
        for lx, ly in corners:
            wx = float(pos[0] + (lx * cos_a - ly * sin_a))
            wy = float(pos[1] + (lx * sin_a + ly * cos_a))
            world_pts.append((wx, wy))

        if is_leader:
            # Gold Leader Car with halo
            pygame.draw.polygon(self.screen, self.LEADER_GOLD, world_pts)
            pygame.draw.polygon(self.screen, (255, 255, 255), world_pts, width=2)
            # Headlights
            pygame.draw.circle(self.screen, (255, 255, 255), (int(world_pts[0][0]), int(world_pts[0][1])), radius=3)
            pygame.draw.circle(self.screen, (255, 255, 255), (int(world_pts[1][0]), int(world_pts[1][1])), radius=3)
        else:
            # Swarm Car in unique palette
            color = self.env.car_colors[i]
            pygame.draw.polygon(self.screen, color, world_pts)
            pygame.draw.polygon(self.screen, (30, 30, 40), world_pts, width=1)

    def _draw_leader_raycasts(self, leader_idx: int):
        """Draws LiDAR sensor rays exclusively for the leader car to keep screen uncluttered."""
        if leader_idx >= self.env.num_cars:
            return

        sensor = self.env.sensors[leader_idx]
        for ray in sensor.latest_rays:
            start_pt, end_pt, hit_pt, norm_dist = ray
            sp = (int(start_pt[0]), int(start_pt[1]))
            hp = (int(hit_pt[0]), int(hit_pt[1]))

            if norm_dist < 0.28:
                col = (255, 60, 60)
            elif norm_dist < 0.65:
                col = (255, 210, 40)
            else:
                col = (60, 240, 160)

            pygame.draw.line(self.screen, col, sp, hp, width=1)
            if norm_dist < 0.99:
                pygame.draw.circle(self.screen, col, hp, radius=3)

    def _draw_particles(self):
        """Animates and renders spark explosions from car crashes."""
        surviving = []
        for p in self.particles:
            p["x"] += p["vx"]
            p["y"] += p["vy"]
            p["vx"] *= 0.92
            p["vy"] *= 0.92
            p["life"] -= 1

            if p["life"] > 0:
                prog = p["life"] / p["max_life"]
                r = max(1, int(p["radius"] * prog))
                pygame.draw.circle(self.screen, p["color"], (int(p["x"]), int(p["y"])), r)
                surviving.append(p)

        self.particles = surviving

    def _draw_swarm_hud(self, leader_idx: int):
        """Draws Left Swarm HUD panel."""
        hud_w, hud_h = 240, 180
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill(self.HUD_BG)
        pygame.draw.rect(hud_surface, (70, 80, 100), (0, 0, hud_w, hud_h), width=1, border_radius=4)

        title = self.header_font.render(f"SWARM RACING ARENA", True, self.LEADER_GOLD)
        hud_surface.blit(title, (12, 10))

        leader = self.env.cars[leader_idx]
        speed = abs(leader.forward_velocity)
        cp_total = len(self.env.track_gen.checkpoints)
        cp_curr = (self.env.current_cp[leader_idx] + 1) % max(1, cp_total)

        lines = [
            f"Parallel Cars:   {self.env.num_cars}",
            f"Leader Car:      #{leader_idx:02d} (Gold)",
            f"Leader Speed:    {speed:5.1f} px/s",
            f"Leader Progress: {cp_curr:02d} / {cp_total:02d}",
            f"Leader Laps:     {self.env.laps[leader_idx]}",
            f"Total Crashes:   {self.env.total_swarm_crashes:,}",
            f"Total Laps Won:  {self.env.total_swarm_laps:,}",
        ]

        y_offset = 32
        for line in lines:
            text = self.font.render(line, True, (230, 235, 245))
            hud_surface.blit(text, (12, y_offset))
            y_offset += 18

        self.screen.blit(hud_surface, (16, 16))

    def _draw_gpu_hud(self):
        """Draws Right GPU Multi-Agent Training HUD."""
        hud_w, hud_h = 270, 180
        hud_surface = pygame.Surface((hud_w, hud_h), pygame.SRCALPHA)
        hud_surface.fill(self.HUD_BG)
        pygame.draw.rect(hud_surface, (70, 80, 100), (0, 0, hud_w, hud_h), width=1, border_radius=4)

        title = self.header_font.render("GPU SWARM PIPELINE", True, self.GPU_GREEN)
        hud_surface.blit(title, (12, 10))

        device_name = self.training_info.get("device_name", "GPU")
        if len(device_name) > 24:
            device_name = device_name[:22] + ".."
        vram_mb = self.training_info.get("vram_mb", 0.0)
        timesteps = self.training_info.get("timesteps", 0)
        status = self.training_info.get("status", "SWARM LEARNING")
        throughput = self.env.num_cars * (self.fps if self.fps > 0 else 60)

        lines = [
            ("Device:    ", device_name, self.GPU_GREEN),
            ("VRAM:      ", f"{vram_mb:.1f} MB (Active)", (230, 235, 245)),
            ("Status:    ", status, (255, 215, 0) if "PAUSED" in status else self.GPU_GREEN),
            ("Batch:     ", f"{self.env.num_cars} Agents Batched", (230, 235, 245)),
            ("Throughput:", f"~{throughput:,} steps/sec", self.GPU_GREEN),
            ("Total Steps:", f"{timesteps:,}", (230, 235, 245)),
            ("Render FPS:", f"{self.fps} FPS", (230, 235, 245)),
        ]

        y_offset = 32
        for label, val, col in lines:
            lbl_surf = self.font.render(label, True, (165, 175, 190))
            val_surf = self.font.render(val, True, col)
            hud_surface.blit(lbl_surf, (12, y_offset))
            hud_surface.blit(val_surf, (100, y_offset))
            y_offset += 18

        self.screen.blit(hud_surface, (self.width - hud_w - 16, 16))

    def _draw_controls_bar(self):
        """Draws bottom keybindings guide."""
        bar_h = 24
        bar_surface = pygame.Surface((self.width, bar_h), pygame.SRCALPHA)
        bar_surface.fill((10, 12, 16, 215))

        text_str = "[SPACE] Pause/Resume  |  [V] Toggle Render  |  [UP/DOWN] Speed  |  [S] Save Checkpoint  |  [ESC] Exit"
        text = self.banner_font.render(text_str, True, (200, 215, 235))
        rect = text.get_rect(center=(self.width // 2, bar_h // 2))
        bar_surface.blit(text, rect)

        self.screen.blit(bar_surface, (0, self.height - bar_h))

    def close(self):
        """Closes Pygame."""
        pygame.quit()
