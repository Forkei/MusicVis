"""Particle system: CPU spawning + GPU update via compute shader.

Particles burst from the energy ball surface on kick/snare impulses.
Updated on GPU, rendered as motion-blur line segments using the same
instance format as the energy ball.
"""

import math
import os
import numpy as np
import moderngl


SHADER_DIR = os.path.join(os.path.dirname(__file__), "shaders")
MAX_PARTICLES = 2048


class ParticleSystem:
    """GPU-accelerated particle system."""

    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self._rng = np.random.RandomState(123)

        # Load compute shader
        comp_path = os.path.join(SHADER_DIR, "particle.comp")
        with open(comp_path, "r") as f:
            comp_src = f.read()
        self._compute = ctx.compute_shader(comp_src)

        # Particle state SSBO: MAX_PARTICLES × 8 floats
        # [x, y, vx, vy, life, max_life, brightness, hue]
        state_size = MAX_PARTICLES * 8 * 4
        self._state_data = np.zeros(MAX_PARTICLES * 8, dtype=np.float32)
        self._state_ssbo = ctx.buffer(self._state_data.tobytes())

        # Output segment SSBO: MAX_PARTICLES × 8 floats
        # [x0, y0, x1, y1, thickness, brightness, hue, depth]
        seg_size = MAX_PARTICLES * 8 * 4
        self._segment_ssbo = ctx.buffer(reserve=seg_size)

        # Spawn accumulator (fractional particles per frame)
        self._spawn_accum = 0.0

        # Track next free slot (ring buffer)
        self._next_slot = 0

    @property
    def segment_buffer(self) -> moderngl.Buffer:
        return self._segment_ssbo

    def update(self, delta_time: float, features: dict, settings: dict,
               ball_cx: float, ball_cy: float, ball_radius: float):
        """Update particles: spawn new ones from CPU, update all on GPU.

        Args:
            delta_time: seconds since last frame
            features: audio features dict
            settings: directed settings dict (has particle_spawn_rate, particle_energy)
            ball_cx, ball_cy: ball center in pixels
            ball_radius: ball radius in pixels
        """
        spawn_rate = settings.get("particle_spawn_rate", 0.0)
        particle_energy = settings.get("particle_energy", 0.0)

        # Boost spawn on kick/snare impulses
        kick = features.get("kick_pulse", 0.0)
        snare = features.get("snare_pulse", 0.0)
        vocal = features.get("vocal_presence", 0.0)
        climax_score = features.get("climax_score", 0.0)
        arousal = features.get("arousal", 0.5)
        explosion = features.get("explosion_factor", 0.0)
        valence = features.get("valence", 0.5)
        impulse = max(kick, snare)
        effective_rate = spawn_rate * 200.0  # base: up to 200 particles/sec
        effective_rate *= (1.0 - vocal * 0.4)   # vocal dampens spawn
        effective_rate += climax_score * 100.0   # climax boosts spawn
        if impulse > 0.3:
            effective_rate += impulse * 300.0  # burst on hits

        # Accumulate spawns
        self._spawn_accum += effective_rate * delta_time
        n_spawn = int(self._spawn_accum)
        self._spawn_accum -= n_spawn

        # Spawn particles on CPU
        if n_spawn > 0 and ball_radius > 1.0:
            n_spawn = min(n_spawn, 64)  # cap per frame
            hue_base = features.get("spectral_centroid", 0.5) * 0.85 + 0.05 + 0.35
            speed_base = (50.0 + particle_energy * 200.0 + arousal * 50.0 + explosion * 100.0) * max(1.0, ball_radius / 50.0)

            for _ in range(n_spawn):
                angle = self._rng.uniform(0, 2 * math.pi)
                # Spawn on ball surface
                sx = ball_cx + math.cos(angle) * ball_radius
                sy = ball_cy + math.sin(angle) * ball_radius
                # Velocity: outward with some spread
                spread = self._rng.uniform(-0.3, 0.3)
                v_angle = angle + spread
                speed = speed_base * self._rng.uniform(0.5, 1.5)
                vx = math.cos(v_angle) * speed
                vy = math.sin(v_angle) * speed

                life = self._rng.uniform(0.3, 1.2) + vocal * 0.3
                brightness = self._rng.uniform(0.3, 0.8) * (0.5 + particle_energy)
                hue = hue_base + self._rng.uniform(-0.1, 0.1) + (valence - 0.5) * 0.05

                slot = self._next_slot
                base = slot * 8
                self._state_data[base + 0] = sx
                self._state_data[base + 1] = sy
                self._state_data[base + 2] = vx
                self._state_data[base + 3] = vy
                self._state_data[base + 4] = life
                self._state_data[base + 5] = life  # max_life
                self._state_data[base + 6] = brightness
                self._state_data[base + 7] = hue

                self._next_slot = (self._next_slot + 1) % MAX_PARTICLES

            # Upload updated state
            self._state_ssbo.orphan(len(self._state_data) * 4)
            self._state_ssbo.write(self._state_data.tobytes())

        # Dispatch compute shader to update positions
        self._state_ssbo.bind_to_storage_buffer(0)
        self._segment_ssbo.bind_to_storage_buffer(1)

        self._compute["u_dt"] = min(delta_time, 0.05)
        self._compute["u_drag"] = 2.0

        n_groups = (MAX_PARTICLES + 255) // 256
        self._compute.run(group_x=n_groups)
        self.ctx.memory_barrier()

        # Update CPU-side life tracking (no GPU readback needed —
        # decrement life on CPU to keep spawn slot tracking in sync)
        self._state_data[4::8] = np.maximum(self._state_data[4::8] - delta_time, 0.0)

    def cleanup(self):
        """Release GPU resources."""
        for buf in [self._state_ssbo, self._segment_ssbo]:
            try:
                buf.release()
            except Exception:
                pass
