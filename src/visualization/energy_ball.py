"""Persistent 3D loop tangle visualization with GPU compute.

Generates 10-25 persistent closed-loop splines that orbit in 3D,
creating a tangled wire ball / scribble sphere effect.
Spline generation is done on GPU via compute shader (GL 4.3).
Loops are differentiated by frequency band (bass/mid/treble).
"""

import math
import os
import numpy as np
import moderngl


SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")


class Loop:
    """A single persistent closed-loop spline orbiting in 3D."""

    def __init__(self, n_control_points: int, rng: np.random.RandomState,
                 band_index: int = 1, hue_offset: float = 0.0):
        self.n_cp = n_control_points
        self.band_index = band_index  # 0=bass, 1=mid, 2=treble
        self.hue_offset = hue_offset

        # Random tilt axis for the great circle
        tilt_theta = rng.uniform(0, math.pi)
        tilt_phi = rng.uniform(0, 2 * math.pi)

        # Base control points: evenly around a tilted great circle
        angles = np.linspace(0, 2 * math.pi, n_control_points, endpoint=False)

        self.base_theta = np.full(n_control_points, math.pi / 2)
        self.base_phi = angles.copy()

        ct, st = math.cos(tilt_theta), math.sin(tilt_theta)
        cp, sp = math.cos(tilt_phi), math.sin(tilt_phi)
        self.rot_matrix = np.array([
            [cp * ct, -sp, cp * st],
            [sp * ct,  cp, sp * st],
            [-st,       0,  ct],
        ], dtype=np.float32)

        self.base_theta += rng.uniform(-0.3, 0.3, n_control_points)
        self.base_phi += rng.uniform(-0.2, 0.2, n_control_points)

        # Noise phase offsets (unique per control point, per axis)
        self.noise_phase = rng.uniform(0, 100, (n_control_points, 3)).astype(np.float32)

        # Visual properties — modulated by band
        if band_index == 0:  # bass
            self.base_thickness = float(rng.uniform(3.0, 5.0) * 1.3)
            self.base_brightness = float(rng.uniform(0.5, 1.0))
        elif band_index == 2:  # treble
            self.base_thickness = float(rng.uniform(3.0, 5.0) * 0.7)
            self.base_brightness = float(rng.uniform(0.5, 1.0))
        else:  # mid
            self.base_thickness = float(rng.uniform(3.0, 5.0))
            self.base_brightness = float(rng.uniform(0.5, 1.0))


def _pack_loop_data(loops: list[Loop]) -> np.ndarray:
    """Pack all loop data into a flat float32 array for SSBO.

    Per loop: 24 CPs × 5 floats + 9 floats (rot matrix) + 4 floats (visual) = 133 floats
    """
    n_loops = len(loops)
    data = np.zeros(n_loops * 133, dtype=np.float32)

    for i, loop in enumerate(loops):
        base = i * 133

        # 24 control points × 5 floats each
        for j in range(24):
            cp_base = base + j * 5
            data[cp_base + 0] = loop.base_theta[j]
            data[cp_base + 1] = loop.base_phi[j]
            data[cp_base + 2] = loop.noise_phase[j, 0]
            data[cp_base + 3] = loop.noise_phase[j, 1]
            data[cp_base + 4] = loop.noise_phase[j, 2]

        # Rotation matrix (9 floats at offset 120)
        # GLSL mat3() constructor is column-major, so transpose before flattening
        rm_base = base + 120
        data[rm_base:rm_base + 9] = loop.rot_matrix.T.flatten()

        # Visual properties (4 floats at offset 129)
        vp_base = base + 129
        data[vp_base + 0] = loop.base_thickness
        data[vp_base + 1] = loop.base_brightness
        data[vp_base + 2] = loop.hue_offset
        data[vp_base + 3] = float(loop.band_index)

    return data


class EnergyBallGenerator:
    """Generates persistent 3D loop tangle geometry each frame using GPU compute."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._loops: list[Loop] = []
        self._current_radius = 50.0
        self._rotation_angle = 0.0
        self._prev_beat = 0.0
        self._initialized = False
        self._rng = np.random.RandomState(42)

        # Shake offset state for kick impacts
        self._shake_offset_x = 0.0
        self._shake_offset_y = 0.0

        # Load compute shader
        comp_path = os.path.join(SHADER_DIR, "spline.comp")
        with open(comp_path, "r") as f:
            comp_src = f.read()
        self._compute = ctx.compute_shader(comp_src)

        # SSBOs — will be created on first _ensure_loops
        self._loop_ssbo: moderngl.Buffer | None = None
        self._segment_ssbo: moderngl.Buffer | None = None
        self._current_loop_count = 0

    def _ensure_loops(self, count: int):
        """Create or adjust the number of persistent loops with band assignment."""
        if self._current_loop_count == count and self._initialized:
            return

        self._loops = []
        third = count / 3.0
        for i in range(count):
            if i < third:
                band = 0       # bass
                hue_off = -0.15  # shift toward warm red
            elif i < 2 * third:
                band = 1       # mid
                hue_off = 0.0
            else:
                band = 2       # treble
                hue_off = 0.20   # shift toward cool blue
            self._loops.append(Loop(24, self._rng, band_index=band, hue_offset=hue_off))

        # Pack and upload loop data to SSBO
        loop_data = _pack_loop_data(self._loops)
        if self._loop_ssbo is not None:
            self._loop_ssbo.release()
        self._loop_ssbo = self.ctx.buffer(loop_data.tobytes())

        # Create segment output SSBO: count × 240 segments × 8 floats
        seg_size = count * 240 * 8 * 4  # bytes
        if self._segment_ssbo is not None:
            self._segment_ssbo.release()
        self._segment_ssbo = self.ctx.buffer(reserve=seg_size)

        self._current_loop_count = count
        self._initialized = True

        # Store base noise phases for decay-back (hihat jitter fix)
        self._base_noise_phases = [loop.noise_phase.copy() for loop in self._loops]

    def _update_loop_ssbo_noise(self):
        """Re-upload loop data when noise phases are mutated (hihat jitter)."""
        if not self._loops:
            return
        loop_data = _pack_loop_data(self._loops)
        self._loop_ssbo.orphan(len(loop_data) * 4)
        self._loop_ssbo.write(loop_data.tobytes())

    @property
    def segment_buffer(self) -> moderngl.Buffer | None:
        """The output SSBO that doubles as the instance VBO."""
        return self._segment_ssbo

    def generate(
        self,
        width: float,
        height: float,
        time: float,
        delta_time: float,
        features: dict,
        settings: dict,
    ) -> tuple[moderngl.Buffer | None, int, np.ndarray | None]:
        """Generate loop tangle geometry for one frame using GPU compute.

        Returns:
            (segment_buffer, compute_segment_count, ring_segments_or_None)
            The buffer is the SSBO containing compute-generated segments.
            ring_segments is a CPU-generated numpy array for the waveform ring (if enabled).
        """
        loop_count = int(settings.get("loop_count", 15))
        loop_count = max(5, min(25, loop_count))
        self._ensure_loops(loop_count)

        cx, cy = width / 2, height / 2
        half_min = min(width, height) * 0.5

        # Extract audio features
        rms = features.get("rms", 0.3)
        onset = features.get("onset_strength", 0.0)
        bandwidth = features.get("spectral_bandwidth", 0.5)
        spectral_flux = features.get("spectral_flux", 0.1)
        anticipation = features.get("anticipation_factor", 0.0)
        explosion = features.get("explosion_factor", 0.0)

        bass = features.get("bass_energy", 0.12)
        mid = features.get("mid_energy", 0.2)
        treble = features.get("treble_energy", 0.15)
        kick = features.get("kick_pulse", 0.0)
        snare = features.get("snare_pulse", 0.0)
        hihat = features.get("hihat_pulse", 0.0)

        tempo = features.get("tempo", 120.0)
        groove = features.get("groove_factor", 0.0)
        rhythmic_density = features.get("rhythmic_density", 0.0)
        arousal = features.get("arousal", 0.5)

        energy_mult = settings.get("energy_mult", 1.0)
        noise_mult = settings.get("noise_mult", 1.0)
        rotation_speed = settings.get("rotation_speed", 1.0)

        dt = min(delta_time, 0.05)

        # --- A. Spring-damped radius ---
        zoom = settings.get("zoom", 1.0)
        quiet_radius = half_min * 0.08 * zoom
        loud_radius = half_min * (0.28 + arousal * 0.14) * zoom

        target = quiet_radius + (loud_radius - quiet_radius) * (bass ** 0.7) * energy_mult
        target *= (1.0 - anticipation * 0.25)
        target *= (1.0 + explosion * 1.5)

        max_radius = width / 4.0 * zoom
        target = min(target, max_radius)

        smoothing = 1.0 - math.exp(-8.0 * dt)
        self._current_radius += (target - self._current_radius) * smoothing

        kick_impulse = max(0.0, kick - self._prev_beat)
        if kick_impulse > 0.3:
            self._current_radius += kick_impulse * self._current_radius * 0.15
            # Add shake offset on kick impact
            shake_amount = self._current_radius * 0.06
            self._shake_offset_x += self._rng.uniform(-shake_amount, shake_amount)
            self._shake_offset_y += self._rng.uniform(-shake_amount, shake_amount)
        self._prev_beat = kick

        # Decay shake offset (fast falloff ~80ms)
        shake_decay = math.exp(-12.0 * dt)
        self._shake_offset_x *= shake_decay
        self._shake_offset_y *= shake_decay

        self._current_radius = min(self._current_radius, max_radius)
        radius = max(5.0, self._current_radius)

        # --- B. Noise parameters ---
        noise_intensity = (0.05 + mid * 0.2 + bandwidth * 0.15 + explosion * 0.4 + rhythmic_density * 0.1) * noise_mult
        noise_speed = 3.0 * (tempo / 120.0) + spectral_flux * 8.0 + anticipation * 5.0

        # --- C. Rotation (beat-synchronized pulse) ---
        rotation_speed *= settings.get("director_rotation_tempo", 1.0)
        beat_sync = 1.0 + kick * 0.3 + snare * 0.15
        rot_increment = dt * (0.3 + anticipation * 0.8 + rms * 0.2) * rotation_speed * beat_sync
        rot_increment += math.sin(time * tempo / 60.0 * math.pi) * groove * rotation_speed * 0.15 * dt
        self._rotation_angle += rot_increment

        # --- D. Volume scatter ---
        radius_ratio = (radius - quiet_radius) / max(1.0, loud_radius - quiet_radius)
        radius_ratio = max(0.0, min(1.0, radius_ratio))
        volume_scatter = radius_ratio ** 2.0

        angular_boost = 1.0 + volume_scatter * 1.5

        # --- E. Hihat jitter (CPU-side noise phase mutation with decay) ---
        if hihat > 0.5:
            for loop in self._loops:
                loop.noise_phase += self._rng.uniform(-0.5, 0.5, loop.noise_phase.shape).astype(np.float32)

        # Decay noise phases back toward base values (prevents unbounded drift)
        decay_factor = 1.0 - math.exp(-3.0 * dt)
        for i, loop in enumerate(self._loops):
            loop.noise_phase += (self._base_noise_phases[i] - loop.noise_phase) * decay_factor

        # Check if any noise phase has drifted from base (avoid unnecessary uploads)
        needs_upload = hihat > 0.5
        if not needs_upload:
            for i, loop in enumerate(self._loops):
                if np.max(np.abs(loop.noise_phase - self._base_noise_phases[i])) > 0.01:
                    needs_upload = True
                    break
        if needs_upload:
            self._update_loop_ssbo_noise()

        # --- F. Dynamic loop count ---
        active = max(3, int(rms * energy_mult * loop_count) + int(rhythmic_density * 3))

        # --- G. Dispatch compute shader ---
        self._compute["u_time"] = time
        self._compute["u_radius"] = radius
        self._compute["u_center"] = (cx + self._shake_offset_x, cy + self._shake_offset_y)
        self._compute["u_rotation_angle"] = self._rotation_angle
        self._compute["u_perspective_d"] = 3.0
        self._compute["u_noise_intensity"] = noise_intensity
        self._compute["u_noise_speed"] = noise_speed
        self._compute["u_bass"] = bass
        self._compute["u_mid"] = mid
        self._compute["u_treble"] = treble
        self._compute["u_kick"] = kick
        self._compute["u_snare"] = snare
        self._compute["u_rms"] = rms
        self._compute["u_volume_scatter"] = volume_scatter
        self._compute["u_angular_boost"] = angular_boost
        self._compute["u_loop_count"] = loop_count
        self._compute["u_active_loops"] = active

        self._loop_ssbo.bind_to_storage_buffer(0)
        self._segment_ssbo.bind_to_storage_buffer(1)

        self._compute.run(group_x=loop_count, group_y=1, group_z=1)

        # Memory barrier: ensure compute writes are visible to vertex shader
        self.ctx.memory_barrier()

        compute_seg_count = loop_count * 240

        # --- H. Waveform ring (CPU, appended separately) ---
        ring_segs = None
        if settings.get("show_ring", True):
            ring_segs = self._generate_ring(
                cx + self._shake_offset_x, cy + self._shake_offset_y,
                radius, features, settings)

        return self._segment_ssbo, compute_seg_count, ring_segs

    def _generate_ring(self, cx: float, cy: float, radius: float,
                       features: dict, settings: dict) -> np.ndarray | None:
        """Generate waveform frequency ring around the ball."""
        mel_frame = features.get("mel_frame", None)
        if mel_frame is None:
            return None

        ring_opacity = settings.get("ring_opacity", 0.5)
        if ring_opacity < 0.01:
            return None

        n_bins = len(mel_frame)
        ring_radius = radius * 1.5
        angles = np.linspace(0, 2 * np.pi, n_bins, endpoint=False)

        displace = mel_frame * radius * 0.5
        r = ring_radius + displace

        x = cx + np.cos(angles) * r
        y = cy + np.sin(angles) * r

        x = np.append(x, x[0])
        y = np.append(y, y[0])

        n_segs = n_bins
        segs = np.empty((n_segs, 8), dtype=np.float32)
        segs[:, 0] = x[:-1]
        segs[:, 1] = y[:-1]
        segs[:, 2] = x[1:]
        segs[:, 3] = y[1:]
        segs[:, 4] = 3.0
        segs[:, 5] = np.clip(mel_frame * 3.0, 0.0, 1.0) * ring_opacity
        global_hue = settings.get("global_hue", 0.3)
        segs[:, 6] = global_hue + np.linspace(-0.15, 0.15, n_bins).astype(np.float32)
        segs[:, 7] = 0.5

        return segs

    def cleanup(self):
        """Release GPU resources."""
        for buf in [self._loop_ssbo, self._segment_ssbo]:
            if buf is not None:
                try:
                    buf.release()
                except Exception:
                    pass
