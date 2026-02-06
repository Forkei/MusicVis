"""Persistent 3D loop tangle visualization (CPU side).

Generates 10-20 persistent closed-loop splines that orbit in 3D,
creating a tangled wire ball / scribble sphere effect.
"""

import math
import numpy as np


class Loop:
    """A single persistent closed-loop spline orbiting in 3D."""

    def __init__(self, n_control_points: int, rng: np.random.RandomState):
        self.n_cp = n_control_points

        # Random tilt axis for the great circle
        tilt_theta = rng.uniform(0, math.pi)
        tilt_phi = rng.uniform(0, 2 * math.pi)

        # Base control points: evenly around a tilted great circle
        angles = np.linspace(0, 2 * math.pi, n_control_points, endpoint=False)

        # Great circle in XZ plane, then rotate by tilt
        self.base_theta = np.full(n_control_points, math.pi / 2)  # equator
        self.base_phi = angles.copy()

        # Apply random tilt by storing rotation matrix
        ct, st = math.cos(tilt_theta), math.sin(tilt_theta)
        cp, sp = math.cos(tilt_phi), math.sin(tilt_phi)
        # Rotation: first around Y by phi, then around X by theta
        self.rot_matrix = np.array([
            [cp * ct, -sp, cp * st],
            [sp * ct,  cp, sp * st],
            [-st,       0,  ct],
        ])

        # Random perturbation to break perfect circle (gentler for smoother loops)
        self.base_theta += rng.uniform(-0.3, 0.3, n_control_points)
        self.base_phi += rng.uniform(-0.2, 0.2, n_control_points)

        # Noise phase offsets (unique per control point, per axis)
        self.noise_phase = rng.uniform(0, 100, (n_control_points, 3))

        # Visual properties
        self.base_thickness = rng.uniform(3.0, 5.0)
        self.base_brightness = rng.uniform(0.5, 1.0)


def _catmull_rom(p0, p1, p2, p3, n_subdiv):
    """Catmull-Rom spline interpolation between p1 and p2 (N-dimensional).

    Returns n_subdiv points (not including p1, but including p2 at t=1 if n_subdiv includes it).
    """
    ts = np.linspace(0, 1, n_subdiv + 1)[1:]  # exclude t=0 (that's p1)
    t2 = ts * ts
    t3 = t2 * ts

    n_dims = len(p0)
    out = np.empty((n_subdiv, n_dims))
    for dim in range(n_dims):
        a = -0.5 * p0[dim] + 1.5 * p1[dim] - 1.5 * p2[dim] + 0.5 * p3[dim]
        b = p0[dim] - 2.5 * p1[dim] + 2.0 * p2[dim] - 0.5 * p3[dim]
        c = -0.5 * p0[dim] + 0.5 * p2[dim]
        d = p1[dim]
        out[:, dim] = a * t3 + b * t2 + c * ts + d

    return out


def _multi_sine_noise(phase, t, speed):
    """Smooth multi-sine noise: 4 octaves for flowing light-trail motion."""
    ts = t * speed
    return (
        np.sin(phase * 0.5 + ts * 0.7) * 0.3   # slow large-scale drift
        + np.sin(phase + ts) * 0.4               # primary motion
        + np.sin(phase * 1.7 + ts * 2.3) * 0.2  # secondary detail
        + np.sin(phase * 0.3 + ts * 4.1) * 0.1  # fine detail (small)
    )


class EnergyBallGenerator:
    """Generates persistent 3D loop tangle geometry each frame."""

    def __init__(self):
        self._loops: list[Loop] = []
        self._current_radius = 50.0
        self._rotation_angle = 0.0
        self._prev_beat = 0.0
        self._initialized = False
        self._rng = np.random.RandomState(42)

    def _ensure_loops(self, count: int):
        """Create or adjust the number of persistent loops."""
        if len(self._loops) == count and self._initialized:
            return

        self._loops = []
        for _ in range(count):
            self._loops.append(Loop(24, self._rng))
        self._initialized = True

    def generate(
        self,
        width: float,
        height: float,
        time: float,
        delta_time: float,
        features: dict,
        settings: dict,
    ) -> np.ndarray:
        """Generate loop tangle geometry for one frame.

        Returns:
            numpy array of shape (N, 6): [x0, y0, x1, y1, thickness, brightness]
        """
        loop_count = int(settings.get("loop_count", 15))
        loop_count = max(5, min(25, loop_count))
        self._ensure_loops(loop_count)

        cx, cy = width / 2, height / 2
        half_min = min(width, height) * 0.5

        # Extract audio features
        rms = features.get("rms", 0.3)
        onset = features.get("onset_strength", 0.0)
        beat = features.get("beat_pulse", 0.0)
        bandwidth = features.get("spectral_bandwidth", 0.5)
        spectral_flux = features.get("spectral_flux", 0.1)
        anticipation = features.get("anticipation_factor", 0.0)
        explosion = features.get("explosion_factor", 0.0)

        # New EDM-specific features
        bass = features.get("bass_energy", 0.12)
        mid = features.get("mid_energy", 0.2)
        treble = features.get("treble_energy", 0.15)
        kick = features.get("kick_pulse", 0.0)
        snare = features.get("snare_pulse", 0.0)
        hihat = features.get("hihat_pulse", 0.0)

        energy_mult = settings.get("energy_mult", 1.0)
        noise_mult = settings.get("noise_mult", 1.0)
        rotation_speed = settings.get("rotation_speed", 1.0)

        dt = min(delta_time, 0.05)

        # --- A. Spring-damped radius (bass drives size) ---
        quiet_radius = half_min * 0.04
        loud_radius = half_min * 0.22

        target = quiet_radius + (loud_radius - quiet_radius) * (bass ** 0.7) * energy_mult

        # Anticipation compresses, explosion expands
        target *= (1.0 - anticipation * 0.25)
        target *= (1.0 + explosion * 1.5)

        # Hard cap: never exceed center third of screen width
        max_radius = width / 6.0
        target = min(target, max_radius)

        # Critically damped follow (no bounce/overshoot)
        smoothing = 1.0 - math.exp(-8.0 * dt)  # ~8 Hz response, no oscillation
        self._current_radius += (target - self._current_radius) * smoothing

        # Kick: radius snap on bass hits (replaces generic beat_pulse)
        kick_impulse = max(0.0, kick - self._prev_beat)
        if kick_impulse > 0.3:
            self._current_radius += kick_impulse * self._current_radius * 0.15
        self._prev_beat = kick

        # Hard cap radius after spring physics too
        max_radius = width / 6.0
        self._current_radius = min(self._current_radius, max_radius)
        radius = max(5.0, self._current_radius)

        # --- B. Noise parameters (mid drives chaos) ---
        noise_intensity = (0.05 + mid * 0.2 + bandwidth * 0.15 + explosion * 0.4) * noise_mult
        noise_speed = 3.0 + spectral_flux * 8.0 + anticipation * 5.0

        # --- C. Rotation ---
        self._rotation_angle += dt * (0.3 + anticipation * 0.8 + rms * 0.2) * rotation_speed

        # --- D. Volume scatter: tight on surface when small, fill interior when big ---
        radius_ratio = (radius - quiet_radius) / max(1.0, loud_radius - quiet_radius)
        radius_ratio = max(0.0, min(1.0, radius_ratio))
        volume_scatter = radius_ratio ** 2.0  # quadratic: stays tight until really loud

        # --- E. Animate control points and build segments ---
        cos_rot = math.cos(self._rotation_angle)
        sin_rot = math.sin(self._rotation_angle)
        perspective_d = 3.0  # perspective distance

        all_segments = []

        for loop in self._loops:
            n_cp = loop.n_cp

            # Compute noise offsets for theta, phi, r
            angular_boost = 1.0 + volume_scatter * 1.5
            noise_theta = _multi_sine_noise(loop.noise_phase[:, 0], time, noise_speed) * noise_intensity * angular_boost
            noise_phi = _multi_sine_noise(loop.noise_phase[:, 1], time, noise_speed) * noise_intensity * angular_boost
            noise_r = _multi_sine_noise(loop.noise_phase[:, 2], time, noise_speed)

            # Hihat jitter: kick noise phases on hi-hat transients
            if hihat > 0.5:
                loop.noise_phase += self._rng.uniform(-0.5, 0.5, loop.noise_phase.shape)

            # Animated spherical coords
            theta = loop.base_theta + noise_theta * math.pi
            phi = loop.base_phi + noise_phi * 2 * math.pi

            # Radial scatter: surface-hugging when small, volume-filling when big
            r_base = 1.0 - volume_scatter * 0.7       # center shifts inward: 1.0 → 0.3
            r_spread = noise_intensity * 0.5 + volume_scatter * 0.5
            r_factor = r_base + noise_r * r_spread
            r_factor = np.clip(r_factor, 0.1, 1.3)

            # Spherical to Cartesian
            sin_theta = np.sin(theta)
            cos_theta = np.cos(theta)
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x_local = sin_theta * cos_phi * r_factor
            y_local = cos_theta * r_factor
            z_local = sin_theta * sin_phi * r_factor

            # Apply loop's tilt rotation
            xyz = np.stack([x_local, y_local, z_local], axis=0)  # (3, n_cp)
            xyz = loop.rot_matrix @ xyz  # (3, n_cp)

            # Y-axis rotation (global)
            x_rot = xyz[0] * cos_rot - xyz[2] * sin_rot
            y_rot = xyz[1]
            z_rot = xyz[0] * sin_rot + xyz[2] * cos_rot

            # --- F. Catmull-Rom subdivision in 3D (before projection) ---
            pts_3d = np.stack([x_rot, y_rot, z_rot], axis=1)  # (n_cp, 3)

            subdiv_3d = [pts_3d[0:1]]  # start with first point
            n_subdiv = 10

            for i in range(n_cp):
                p0 = pts_3d[(i - 1) % n_cp]
                p1 = pts_3d[i]
                p2 = pts_3d[(i + 1) % n_cp]
                p3 = pts_3d[(i + 2) % n_cp]

                sub = _catmull_rom(p0, p1, p2, p3, n_subdiv)
                subdiv_3d.append(sub)

            # Concatenate all subdivision points and close the loop
            all_3d = np.vstack(subdiv_3d)  # (1 + n_cp * n_subdiv, 3)
            all_3d = np.vstack([all_3d, all_3d[0:1]])

            # --- G. Project subdivided 3D points with per-point perspective ---
            persp = perspective_d / (perspective_d + all_3d[:, 2])
            screen_x = cx + all_3d[:, 0] * radius * persp
            screen_y = cy + all_3d[:, 1] * radius * persp

            # --- H. Per-segment depth factor from average perspective of endpoints ---
            persp_avg = (persp[:-1] + persp[1:]) * 0.5
            # Normalize: persp ranges ~0.5 (far) to ~1.5 (close), map to 0..1
            depth_norm = (persp_avg - 0.5) / 1.0
            depth_norm = np.clip(depth_norm, 0.0, 1.0)
            thickness_mod = 0.4 + depth_norm * 0.6   # 0.4x (far) to 1.0x (close)
            brightness_mod = 0.3 + depth_norm * 0.7   # 0.3x (far) to 1.0x (close)

            # --- I. Curvature-based brightness (corner glow) ---
            n_pts = len(screen_x)
            if n_pts < 3:
                continue

            # Direction vectors between consecutive screen points
            dx = np.diff(screen_x)
            dy = np.diff(screen_y)
            # Segment lengths
            seg_len = np.sqrt(dx * dx + dy * dy) + 1e-8

            # Cosine of angle between consecutive direction vectors
            dot_prod = dx[:-1] * dx[1:] + dy[:-1] * dy[1:]
            len_prod = seg_len[:-1] * seg_len[1:]
            cos_angle = np.clip(dot_prod / len_prod, -1.0, 1.0)
            curvature = 1.0 - cos_angle  # 0=straight, 2=reversal
            curvature_boost = 1.0 + curvature * 0.5  # up to 1.5x at sharp turns

            # Pad curvature_boost: first segment has no previous direction
            curvature_boost = np.concatenate([[1.0], curvature_boost])

            # --- J. Assemble segments with per-segment modulation ---
            base_thickness = loop.base_thickness * (1.0 + kick * 0.3)
            base_brightness = loop.base_brightness * (0.5 + treble * 0.3 + rms * 0.2) * (1.0 + snare * 1.5)

            n_segs = n_pts - 1
            segs = np.empty((n_segs, 6), dtype=np.float32)
            segs[:, 0] = screen_x[:-1]
            segs[:, 1] = screen_y[:-1]
            segs[:, 2] = screen_x[1:]
            segs[:, 3] = screen_y[1:]
            segs[:, 4] = base_thickness * thickness_mod
            segs[:, 5] = base_brightness * brightness_mod * curvature_boost

            all_segments.append(segs)

        if not all_segments:
            return np.zeros((0, 6), dtype=np.float32)

        return np.vstack(all_segments)
