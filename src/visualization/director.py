"""Musical Director: interprets audio context and drives visuals intelligently.

Reads genre, section type, climax score, arousal, valence, and look-ahead
features from the analyzer, then modulates visualization settings per-frame
with smooth transitions. Genre-aware profiles define how each section type
and climax event should look visually.
"""

import math


# Genre name lookup
GENRE_NAMES = ["EDM", "Rock", "Jazz", "Classical", "Hip-Hop", "Ambient", "Pop"]

# Section type name lookup
SECTION_NAMES = [
    "Intro", "Verse", "Chorus", "Bridge", "Breakdown",
    "Buildup", "Drop", "Outro", "Solo",
]

# Smoothing time constants
_TAU_FAST = 0.16    # 160ms
_TAU_MEDIUM = 0.33  # 330ms
_TAU_SLOW = 0.67    # 670ms

# Genre ranges for audio-driven auto values: (low, high) per setting per genre.
# Drive signals (0-1 from audio features) map linearly through these ranges.
# fmt: off
_AUTO_GENRE_RANGES = {
    #              brightness    bloom_int     anamorphic    loops       noise         rotation      fog
    0: {  # EDM
        "brightness": (0.8, 2.5), "bloom_intensity": (0.8, 4.0), "anamorphic_flare": (0.0, 0.7),
        "loop_count": (8, 23),    "noise_mult": (0.15, 1.5),     "rotation_speed": (0.8, 2.8),
        "depth_fog": (0.05, 0.5),
        "tint_cool": (0.3, 0.1, 1.0), "tint_warm": (1.0, 0.3, 0.8),
    },
    1: {  # Rock
        "brightness": (0.9, 2.3), "bloom_intensity": (0.6, 3.5), "anamorphic_flare": (0.0, 0.5),
        "loop_count": (7, 20),    "noise_mult": (0.15, 1.2),     "rotation_speed": (0.6, 2.3),
        "depth_fog": (0.05, 0.55),
        "tint_cool": (0.4, 0.2, 0.8), "tint_warm": (1.0, 0.5, 0.2),
    },
    2: {  # Jazz
        "brightness": (0.7, 1.8), "bloom_intensity": (0.5, 2.5), "anamorphic_flare": (0.0, 0.2),
        "loop_count": (5, 14),    "noise_mult": (0.1, 0.6),      "rotation_speed": (0.3, 1.5),
        "depth_fog": (0.15, 0.7),
        "tint_cool": (0.2, 0.4, 0.7), "tint_warm": (1.0, 0.85, 0.3),
    },
    3: {  # Classical
        "brightness": (0.6, 2.2), "bloom_intensity": (0.8, 3.5), "anamorphic_flare": (0.0, 0.4),
        "loop_count": (6, 18),    "noise_mult": (0.1, 0.7),      "rotation_speed": (0.2, 1.5),
        "depth_fog": (0.1, 0.7),
        "tint_cool": (0.3, 0.5, 1.0), "tint_warm": (0.9, 0.7, 0.5),
    },
    4: {  # Hip-Hop
        "brightness": (0.9, 2.2), "bloom_intensity": (0.5, 3.0), "anamorphic_flare": (0.0, 0.35),
        "loop_count": (6, 17),    "noise_mult": (0.1, 0.8),      "rotation_speed": (0.4, 2.0),
        "depth_fog": (0.05, 0.5),
        "tint_cool": (0.4, 0.1, 0.6), "tint_warm": (1.0, 0.3, 0.4),
    },
    5: {  # Ambient
        "brightness": (0.5, 1.5), "bloom_intensity": (1.0, 3.0), "anamorphic_flare": (0.0, 0.15),
        "loop_count": (5, 10),    "noise_mult": (0.1, 0.4),      "rotation_speed": (0.1, 0.8),
        "depth_fog": (0.3, 0.85),
        "tint_cool": (0.15, 0.4, 0.8), "tint_warm": (0.4, 0.8, 0.7),
    },
    6: {  # Pop
        "brightness": (0.9, 2.3), "bloom_intensity": (0.7, 3.5), "anamorphic_flare": (0.0, 0.5),
        "loop_count": (7, 19),    "noise_mult": (0.1, 0.8),      "rotation_speed": (0.5, 2.2),
        "depth_fog": (0.05, 0.55),
        "tint_cool": (0.3, 0.4, 1.0), "tint_warm": (1.0, 0.5, 0.7),
    },
}
# fmt: on

# Keys that support auto mode
AUTO_KEYS = [
    "brightness", "bloom_intensity", "bloom_tint", "anamorphic_flare",
    "loop_count", "noise_mult", "rotation_speed", "depth_fog",
]


class _DirectorState:
    """Smooth internal state maintained across frames."""
    __slots__ = (
        "brightness_mult", "energy_mult_mult", "bloom_mult",
        "noise_mult_mult", "rotation_mult", "anamorphic_mult",
        "depth_fog_offset", "hue_shift", "loop_count_offset",
        "trail_decay", "particle_spawn_rate", "particle_energy",
        "prev_section_type", "section_transition_t",
        "saturation_offset", "vignette_offset", "chromatic_offset",
        "bg_intensity_offset", "color_temp_shift", "rotation_tempo_scale",
        # Auto base smoothing
        "auto_brightness", "auto_bloom_intensity", "auto_anamorphic_flare",
        "auto_loop_count", "auto_noise_mult", "auto_rotation_speed",
        "auto_depth_fog", "auto_tint_r", "auto_tint_g", "auto_tint_b",
    )

    def __init__(self):
        self.reset()

    def reset(self):
        self.brightness_mult = 1.0
        self.energy_mult_mult = 1.0
        self.bloom_mult = 1.0
        self.noise_mult_mult = 1.0
        self.rotation_mult = 1.0
        self.anamorphic_mult = 1.0
        self.depth_fog_offset = 0.0
        self.hue_shift = 0.0
        self.loop_count_offset = 0.0
        self.trail_decay = 0.0
        self.particle_spawn_rate = 0.0
        self.particle_energy = 0.0
        self.prev_section_type = -1
        self.section_transition_t = 0.0
        self.saturation_offset = 0.0
        self.vignette_offset = 0.0
        self.chromatic_offset = 0.0
        self.bg_intensity_offset = 0.0
        self.color_temp_shift = 0.0
        self.rotation_tempo_scale = 1.0
        # Auto values — initialize at midpoint of Pop ranges (subdued ball)
        pop = _AUTO_GENRE_RANGES[6]
        self.auto_brightness = sum(pop["brightness"]) / 2.0
        self.auto_bloom_intensity = sum(pop["bloom_intensity"]) / 2.0
        self.auto_anamorphic_flare = sum(pop["anamorphic_flare"]) / 2.0
        self.auto_loop_count = sum(pop["loop_count"]) / 2.0
        self.auto_noise_mult = sum(pop["noise_mult"]) / 2.0
        self.auto_rotation_speed = sum(pop["rotation_speed"]) / 2.0
        self.auto_depth_fog = sum(pop["depth_fog"]) / 2.0
        cool, warm = pop["tint_cool"], pop["tint_warm"]
        self.auto_tint_r = (cool[0] + warm[0]) / 2.0
        self.auto_tint_g = (cool[1] + warm[1]) / 2.0
        self.auto_tint_b = (cool[2] + warm[2]) / 2.0


class _GenreProfile:
    """Per-genre visual behavior definition."""

    def __init__(
        self,
        section_visuals: dict,
        climax_intensity: float,
        beat_reactivity: float,
        trail_preference: float,
        particle_preference: float,
    ):
        self.section_visuals = section_visuals
        self.climax_intensity = climax_intensity
        self.beat_reactivity = beat_reactivity
        self.trail_preference = trail_preference
        self.particle_preference = particle_preference


# Default section visual adjustments (multipliers/offsets)
# Keys: brightness, energy, bloom, noise, rotation, loop_offset, anamorphic, fog_offset, trail, particle
_DEFAULT_SECTION = {
    "brightness": 1.0, "energy": 1.0, "bloom": 1.0, "noise": 1.0,
    "rotation": 1.0, "loop_offset": 0, "anamorphic": 1.0, "fog_offset": 0.0,
    "trail": 0.0, "particle": 0.0,
}


def _sv(**kwargs):
    """Create section visual dict with defaults."""
    d = dict(_DEFAULT_SECTION)
    d.update(kwargs)
    return d


# --- 7 Genre Profiles ---

_PROFILES = {
    0: _GenreProfile(  # EDM
        section_visuals={
            0: _sv(brightness=0.7, energy=0.6, bloom=0.8, noise=0.5, trail=0.3),        # intro
            1: _sv(brightness=0.9, energy=0.9, bloom=1.0, noise=0.8),                    # verse
            2: _sv(brightness=1.3, energy=1.3, bloom=1.5, loop_offset=3, anamorphic=1.4), # chorus
            3: _sv(brightness=0.8, energy=0.7, bloom=0.9, noise=1.2),                    # bridge
            4: _sv(brightness=0.5, energy=0.4, bloom=0.6, noise=1.5, fog_offset=0.15, trail=0.5),  # breakdown
            5: _sv(brightness=0.9, energy=1.1, bloom=1.2, noise=1.3, rotation=1.5),      # buildup
            6: _sv(brightness=1.5, energy=1.5, bloom=2.0, loop_offset=5, anamorphic=1.8, particle=0.8), # drop
            7: _sv(brightness=0.6, energy=0.5, bloom=0.7, noise=0.4, trail=0.4),        # outro
            8: _sv(brightness=1.2, energy=1.2, bloom=1.3, noise=0.6),                    # solo
        },
        climax_intensity=2.0,
        beat_reactivity=1.5,
        trail_preference=0.2,
        particle_preference=0.8,
    ),
    1: _GenreProfile(  # Rock
        section_visuals={
            0: _sv(brightness=0.7, energy=0.6, bloom=0.8),
            1: _sv(brightness=0.9, energy=0.9, bloom=1.0),
            2: _sv(brightness=1.3, energy=1.3, bloom=1.4, loop_offset=2, anamorphic=1.3, particle=0.5),
            3: _sv(brightness=0.8, energy=0.7, bloom=0.9, noise=1.1),
            4: _sv(brightness=0.6, energy=0.5, bloom=0.7, trail=0.3),
            5: _sv(brightness=1.0, energy=1.1, bloom=1.2, rotation=1.3),
            6: _sv(brightness=1.4, energy=1.4, bloom=1.6, particle=0.7),
            7: _sv(brightness=0.6, energy=0.5, bloom=0.7, trail=0.3),
            8: _sv(brightness=1.4, energy=1.3, bloom=1.5, noise=0.5, rotation=1.4, particle=0.4),
        },
        climax_intensity=1.6,
        beat_reactivity=1.2,
        trail_preference=0.3,
        particle_preference=0.6,
    ),
    2: _GenreProfile(  # Jazz
        section_visuals={
            0: _sv(brightness=0.8, energy=0.7, bloom=0.9, trail=0.4),
            1: _sv(brightness=0.9, energy=0.8, bloom=1.0, noise=1.1),
            2: _sv(brightness=1.1, energy=1.1, bloom=1.2, loop_offset=1),
            3: _sv(brightness=0.9, energy=0.8, bloom=1.0, noise=1.2),
            4: _sv(brightness=0.7, energy=0.6, bloom=0.8, trail=0.5),
            5: _sv(brightness=1.0, energy=1.0, bloom=1.1),
            6: _sv(brightness=1.2, energy=1.2, bloom=1.3),
            7: _sv(brightness=0.7, energy=0.6, bloom=0.8, trail=0.5),
            8: _sv(brightness=1.3, energy=1.2, bloom=1.4, noise=0.6, rotation=1.2),
        },
        climax_intensity=1.0,
        beat_reactivity=0.6,
        trail_preference=0.5,
        particle_preference=0.2,
    ),
    3: _GenreProfile(  # Classical
        section_visuals={
            0: _sv(brightness=0.7, energy=0.5, bloom=0.9, trail=0.6, fog_offset=0.1),
            1: _sv(brightness=0.8, energy=0.7, bloom=1.0, trail=0.4),
            2: _sv(brightness=1.2, energy=1.2, bloom=1.5, loop_offset=2, anamorphic=1.3, trail=0.3),
            3: _sv(brightness=0.9, energy=0.8, bloom=1.1, trail=0.5),
            4: _sv(brightness=0.6, energy=0.4, bloom=0.8, trail=0.7, fog_offset=0.15),
            5: _sv(brightness=1.0, energy=1.1, bloom=1.3, rotation=1.2),
            6: _sv(brightness=1.3, energy=1.4, bloom=1.6, anamorphic=1.5),
            7: _sv(brightness=0.6, energy=0.4, bloom=0.8, trail=0.7),
            8: _sv(brightness=1.3, energy=1.2, bloom=1.4, noise=0.5),
        },
        climax_intensity=1.8,
        beat_reactivity=0.4,
        trail_preference=0.7,
        particle_preference=0.15,
    ),
    4: _GenreProfile(  # Hip-Hop
        section_visuals={
            0: _sv(brightness=0.7, energy=0.6, bloom=0.8),
            1: _sv(brightness=0.9, energy=0.9, bloom=1.0),
            2: _sv(brightness=1.2, energy=1.2, bloom=1.3, loop_offset=2, particle=0.4),
            3: _sv(brightness=0.8, energy=0.7, bloom=0.9),
            4: _sv(brightness=0.6, energy=0.5, bloom=0.7, trail=0.3),
            5: _sv(brightness=1.0, energy=1.1, bloom=1.2),
            6: _sv(brightness=1.3, energy=1.3, bloom=1.5, particle=0.6),
            7: _sv(brightness=0.6, energy=0.5, bloom=0.7),
            8: _sv(brightness=1.2, energy=1.1, bloom=1.3, noise=0.6),
        },
        climax_intensity=1.4,
        beat_reactivity=1.3,
        trail_preference=0.25,
        particle_preference=0.5,
    ),
    5: _GenreProfile(  # Ambient
        section_visuals={
            0: _sv(brightness=0.8, energy=0.5, bloom=1.2, trail=0.8, fog_offset=0.2, noise=0.6),
            1: _sv(brightness=0.9, energy=0.6, bloom=1.3, trail=0.7, fog_offset=0.15),
            2: _sv(brightness=1.1, energy=0.8, bloom=1.5, trail=0.6, anamorphic=1.2),
            3: _sv(brightness=0.9, energy=0.6, bloom=1.3, trail=0.7),
            4: _sv(brightness=0.7, energy=0.4, bloom=1.1, trail=0.9, fog_offset=0.25),
            5: _sv(brightness=1.0, energy=0.7, bloom=1.4, trail=0.6),
            6: _sv(brightness=1.1, energy=0.9, bloom=1.5, trail=0.5),
            7: _sv(brightness=0.7, energy=0.4, bloom=1.1, trail=0.9, fog_offset=0.2),
            8: _sv(brightness=1.0, energy=0.7, bloom=1.3, trail=0.6),
        },
        climax_intensity=0.5,
        beat_reactivity=0.1,
        trail_preference=0.9,
        particle_preference=0.1,
    ),
    6: _GenreProfile(  # Pop
        section_visuals={
            0: _sv(brightness=0.7, energy=0.6, bloom=0.9),
            1: _sv(brightness=0.9, energy=0.9, bloom=1.0),
            2: _sv(brightness=1.3, energy=1.3, bloom=1.5, loop_offset=3, anamorphic=1.3, particle=0.3),
            3: _sv(brightness=0.8, energy=0.7, bloom=0.9, noise=1.1),
            4: _sv(brightness=0.6, energy=0.5, bloom=0.7, trail=0.3),
            5: _sv(brightness=1.0, energy=1.1, bloom=1.2, rotation=1.2),
            6: _sv(brightness=1.4, energy=1.4, bloom=1.7, particle=0.6),
            7: _sv(brightness=0.6, energy=0.5, bloom=0.7, trail=0.3),
            8: _sv(brightness=1.2, energy=1.1, bloom=1.3),
        },
        climax_intensity=1.5,
        beat_reactivity=1.0,
        trail_preference=0.3,
        particle_preference=0.4,
    ),
}


def _exp_smooth(current: float, target: float, tau: float, dt: float) -> float:
    """Exponential smoothing toward target."""
    if tau < 0.001:
        return target
    alpha = 1.0 - math.exp(-dt / tau)
    return current + (target - current) * alpha


class MusicalDirector:
    """Interprets musical context and modulates visualization settings."""

    def __init__(self):
        self._state = _DirectorState()

    def reset(self):
        """Reset state (called on song switch)."""
        self._state.reset()

    def _compute_auto_values(self, features: dict, genre_id: int) -> dict:
        """Compute raw auto-value targets from live audio features.

        Each setting has a drive formula — a weighted sum of 0-1 audio features
        that maps through a genre-specific (low, high) range. Two moments within
        the same section produce different values if the audio differs.
        """
        gr = _AUTO_GENRE_RANGES.get(genre_id, _AUTO_GENRE_RANGES[6])

        # Extract features (all normalized 0-1 except tempo)
        rms = features.get("rms", 0.0)
        arousal = features.get("arousal", 0.5)
        valence = features.get("valence", 0.5)
        climax = features.get("climax_score", 0.0)
        bass = features.get("bass_energy", 0.0)
        mid = features.get("mid_energy", 0.0)
        treble = features.get("treble_energy", 0.0)
        vocal = features.get("vocal_presence", 0.0)
        flux = features.get("spectral_flux", 0.0)
        centroid = features.get("spectral_centroid", 0.0)
        sharpness = features.get("onset_sharpness", 0.0)
        density = features.get("rhythmic_density", 0.0)
        explosion = features.get("explosion_factor", 0.0)
        anticipation = features.get("anticipation_factor", 0.0)
        snare = features.get("snare_pulse", 0.0)
        groove = features.get("groove_factor", 0.0)
        tempo = features.get("tempo", 120.0)
        trajectory = features.get("energy_trajectory", 0.0)

        def _clamp01(x):
            return max(0.0, min(1.0, x))

        def _lerp(lo, hi, t):
            return lo + (hi - lo) * t

        def _lerp_range(key, drive):
            lo, hi = gr[key]
            return _lerp(lo, hi, _clamp01(drive))

        # 1. Brightness — "how much energy hits you"
        brightness_drive = rms * 0.35 + arousal * 0.20 + climax * 0.20 + mid * 0.10 + explosion * 0.15
        brightness = _lerp_range("brightness", brightness_drive)

        # 2. Bloom Intensity — "ethereal glow"
        bloom_drive = (rms * 0.20 + arousal * 0.15 + climax * 0.25 + treble * 0.10
                       + vocal * 0.10 + anticipation * 0.10 + centroid * 0.10)
        bloom_intensity = _lerp_range("bloom_intensity", bloom_drive)

        # 3. Bloom Tint — "color identity" (cool ↔ warm)
        warmth = (valence * 0.40 + centroid * 0.20 + bass * 0.15
                  + (1.0 - treble) * 0.10 + vocal * 0.15)
        warmth = _clamp01(warmth)
        cool = gr["tint_cool"]
        warm = gr["tint_warm"]
        tint_r = _lerp(cool[0], warm[0], warmth)
        tint_g = _lerp(cool[1], warm[1], warmth)
        tint_b = _lerp(cool[2], warm[2], warmth)

        # 4. Anamorphic Flare — "cinematic lens streaks"
        anamorphic_drive = climax * 0.30 + rms * 0.20 + explosion * 0.20 + centroid * 0.15 + snare * 0.15
        anamorphic = _lerp_range("anamorphic_flare", anamorphic_drive)

        # 5. Loop Count — "visual complexity"
        loop_drive = arousal * 0.30 + density * 0.25 + rms * 0.15 + flux * 0.15 + (1.0 - vocal) * 0.15
        loop_count = _lerp_range("loop_count", loop_drive)

        # 6. Noise Mult — "wire deformation/chaos"
        noise_drive = (flux * 0.25 + sharpness * 0.20 + density * 0.15
                       + arousal * 0.15 + anticipation * 0.15 + (1.0 - vocal) * 0.10)
        noise_mult = _lerp_range("noise_mult", noise_drive)

        # 7. Rotation Speed — "spin momentum"
        tempo_norm = _clamp01((tempo - 60.0) / 140.0)
        rotation_drive = (arousal * 0.25 + tempo_norm * 0.20 + density * 0.15
                          + flux * 0.15 + groove * 0.10 + bass * 0.15)
        rotation_speed = _lerp_range("rotation_speed", rotation_drive)

        # 8. Depth Fog — "atmospheric depth" (INVERSE energy)
        fog_drive = ((1.0 - arousal) * 0.25 + (1.0 - rms) * 0.20 + vocal * 0.15
                     + (1.0 - climax) * 0.15 + (1.0 - centroid) * 0.10
                     + max(0.0, -trajectory) * 0.15)
        depth_fog = _lerp_range("depth_fog", fog_drive)

        return {
            "brightness": brightness,
            "bloom_intensity": bloom_intensity,
            "tint_r": tint_r, "tint_g": tint_g, "tint_b": tint_b,
            "anamorphic_flare": anamorphic,
            "loop_count": loop_count,
            "noise_mult": noise_mult,
            "rotation_speed": rotation_speed,
            "depth_fog": depth_fog,
        }

    def _smooth_auto_state(self, auto_targets: dict, dt: float):
        """Smooth auto values toward their computed targets."""
        s = self._state
        s.auto_brightness = _exp_smooth(s.auto_brightness, auto_targets["brightness"], _TAU_FAST, dt)
        s.auto_bloom_intensity = _exp_smooth(s.auto_bloom_intensity, auto_targets["bloom_intensity"], _TAU_MEDIUM, dt)
        s.auto_anamorphic_flare = _exp_smooth(s.auto_anamorphic_flare, auto_targets["anamorphic_flare"], _TAU_FAST, dt)
        s.auto_loop_count = _exp_smooth(s.auto_loop_count, auto_targets["loop_count"], _TAU_SLOW, dt)
        s.auto_noise_mult = _exp_smooth(s.auto_noise_mult, auto_targets["noise_mult"], _TAU_MEDIUM, dt)
        s.auto_rotation_speed = _exp_smooth(s.auto_rotation_speed, auto_targets["rotation_speed"], _TAU_MEDIUM, dt)
        s.auto_depth_fog = _exp_smooth(s.auto_depth_fog, auto_targets["depth_fog"], _TAU_SLOW, dt)
        s.auto_tint_r = _exp_smooth(s.auto_tint_r, auto_targets["tint_r"], _TAU_SLOW, dt)
        s.auto_tint_g = _exp_smooth(s.auto_tint_g, auto_targets["tint_g"], _TAU_SLOW, dt)
        s.auto_tint_b = _exp_smooth(s.auto_tint_b, auto_targets["tint_b"], _TAU_SLOW, dt)

    def process(self, features: dict, base_settings: dict, delta_time: float) -> dict:
        """Main per-frame call. Returns modified settings dict.

        Args:
            features: dict from AnalysisResult.get_features()
            base_settings: current user settings
            delta_time: seconds since last frame

        Returns:
            New settings dict blending director adjustments with user settings.
        """
        if not base_settings.get("director_enabled", True):
            # Director disabled — pass through with trail/particle defaults
            result = dict(base_settings)
            result["trail_decay"] = base_settings.get("trail_decay_manual", 0.0)
            result["particle_spawn_rate"] = base_settings.get("particle_rate_manual", 0.0)
            result["particle_energy"] = 0.0
            result["director_genre"] = ""
            result["director_section"] = ""
            result["director_climax"] = 0.0
            # Still compute + smooth auto values from audio for UI display
            genre_id = features.get("genre_id", 6)
            auto_targets = self._compute_auto_values(features, genre_id)
            self._smooth_auto_state(auto_targets, delta_time)
            s = self._state
            result["_auto_values"] = {
                "brightness": s.auto_brightness,
                "bloom_intensity": s.auto_bloom_intensity,
                "bloom_tint": (s.auto_tint_r, s.auto_tint_g, s.auto_tint_b),
                "anamorphic_flare": s.auto_anamorphic_flare,
                "loop_count": int(round(s.auto_loop_count)),
                "noise_mult": s.auto_noise_mult,
                "rotation_speed": s.auto_rotation_speed,
                "depth_fog": s.auto_depth_fog,
            }
            return result

        intensity = base_settings.get("director_intensity", 0.8)
        genre_id = features.get("genre_id", 6)
        profile = _PROFILES.get(genre_id, _PROFILES[6])

        # Compute raw targets for director multipliers (trail, particles, hue, etc.)
        targets = self._compute_targets(features, profile, base_settings)

        # Compute audio-driven auto values for the 8 auto settings
        auto_targets = self._compute_auto_values(features, genre_id)

        # Smooth all state
        self._smooth_state(targets, delta_time)
        self._smooth_auto_state(auto_targets, delta_time)

        # Apply to settings
        return self._apply_to_settings(base_settings, intensity, features, profile)

    def _compute_targets(self, features: dict, profile: _GenreProfile, settings: dict) -> dict:
        """Compute raw target multipliers from musical context."""
        section_type = features.get("section_type", 1)
        climax = features.get("climax_score", 0.0)
        climax_type = features.get("climax_type", 0)
        arousal = features.get("arousal", 0.5)
        valence = features.get("valence", 0.5)
        vocal = features.get("vocal_presence", 0.0)
        la_energy = features.get("lookahead_energy_delta", 0.0)
        la_section = features.get("lookahead_section_change", 0.0)
        la_climax = features.get("lookahead_climax", 0.0)
        tempo = features.get("tempo", 120.0)
        groove = features.get("groove_factor", 0.0)
        energy_traj = features.get("energy_trajectory", 0.0)
        rhythmic_density = features.get("rhythmic_density", 0.0)
        explosion = features.get("explosion_factor", 0.0)
        bass = features.get("bass_energy", 0.0)

        section_transitions = settings.get("section_transitions", True)
        climax_reactions = settings.get("climax_reactions", True)

        # Get section visual adjustments
        sv = profile.section_visuals.get(section_type, _DEFAULT_SECTION)

        # Start from section baseline
        brightness = sv["brightness"] if section_transitions else 1.0
        energy = sv["energy"] if section_transitions else 1.0
        bloom = sv["bloom"] if section_transitions else 1.0
        noise = sv["noise"] if section_transitions else 1.0
        rotation = sv["rotation"] if section_transitions else 1.0
        anamorphic = sv["anamorphic"] if section_transitions else 1.0
        fog_offset = sv["fog_offset"] if section_transitions else 0.0
        loop_offset = sv["loop_offset"] if section_transitions else 0
        trail = sv["trail"] if section_transitions else 0.0
        particle = sv["particle"] if section_transitions else 0.0

        # New output targets
        saturation_offset = 0.0
        vignette_offset = 0.0
        chromatic_offset = 0.0
        bg_intensity_offset = 0.0
        color_temp_shift = 0.0

        # Tempo → rotation scaling (normalized around 120 BPM)
        rotation_tempo_scale = tempo / 120.0

        # Valence → saturation (happy=vivid, sad=muted)
        saturation_offset = (valence - 0.5) * 0.2

        # Valence → color temperature (happy=warm, sad=cool)
        color_temp_shift = (valence - 0.5) * 0.15

        # Valence → fog (sad = more fog, halved to avoid stacking with section fog)
        fog_offset += (1.0 - valence) * 0.05

        # Arousal → vignette (excited=tighter, calm=opener)
        vignette_offset = arousal * 0.15 - 0.05

        # Arousal → background (calm=more visible background)
        bg_intensity_offset = (1.0 - arousal) * 0.03

        # Energy trajectory → trails (falling energy = ghostly trails)
        if energy_traj < -0.2:
            trail += abs(energy_traj) * 0.3

        # Rhythmic density → active loops and noise
        loop_offset += int(rhythmic_density * 3)
        noise *= (1.0 + rhythmic_density * 0.2)

        # Vocal presence → trail boost (vocals = smoother/dreamier)
        trail += vocal * 0.15

        # Explosion + bass → chromatic aberration
        chromatic_offset += explosion * 0.005 + max(0.0, bass - 0.3) * 0.003

        # Climax boost (general)
        if climax_reactions and climax > 0.3:
            climax_factor = (climax - 0.3) / 0.7 * profile.climax_intensity
            brightness += climax_factor * 0.3
            bloom += climax_factor * 0.5
            anamorphic += climax_factor * 0.3
            particle += climax_factor * 0.3
            saturation_offset += climax_factor * 0.05

            # Climax type → distinct visual signatures
            if climax_type == 1:  # drop
                bloom *= 1.5
                anamorphic *= 1.5
                particle += 0.5
                noise *= 1.3
                fog_offset -= 0.15
                vignette_offset -= 0.2  # opens wide
                chromatic_offset += 0.003
            elif climax_type == 2:  # chorus_peak
                bloom *= 1.3
                brightness *= 1.2
                saturation_offset += 0.1
                color_temp_shift += 0.1  # warm bloom surge
            elif climax_type == 4:  # solo_peak
                brightness *= 1.3
                noise *= 0.6
                fog_offset -= 0.1
                loop_offset -= 3  # tight focus
            elif climax_type == 5:  # breakdown_return
                bloom *= 1.4
                particle += 0.4
                trail = 0.0  # snap back
                chromatic_offset += 0.002

        # Look-ahead preparation (enhanced)
        if settings.get("beat_sync", True):
            # Before climax: tension buildup + vignette tightening + bloom pre-glow
            if la_climax > 0.3:
                tension = (la_climax - 0.3) / 0.7
                energy *= (1.0 - tension * 0.1)  # slight contraction
                noise *= (1.0 + tension * 0.3)    # anxiety
                rotation *= (1.0 + tension * 0.2)
                vignette_offset += tension * 0.2   # tightening
                bloom *= (1.0 + tension * 0.2)     # pre-glow

            # Before section change: subtle fog "inhale" + brightness breath
            if la_section > 0.3:
                fog_offset += la_section * 0.1
            if la_section > 0.5:
                brightness *= (1.0 - la_section * 0.05)  # brief breath

            # Energy delta lookahead
            if la_energy > 0:
                brightness *= (1.0 + la_energy * 0.05)   # pre-brighten
            elif la_energy < 0:
                trail += abs(la_energy) * 0.15            # pre-trail

        # Arousal-driven rotation
        rotation *= (0.7 + arousal * 0.6)

        # Valence-driven hue shift
        hue_shift = (valence - 0.5) * 0.1  # warm for major, cool for minor

        # Vocal presence: reduce noise, add fog for depth
        if vocal > 0.5:
            vocal_factor = (vocal - 0.5) * 2.0
            noise *= (1.0 - vocal_factor * 0.3)
            fog_offset += vocal_factor * 0.05

        # Auto trail/particle from genre profile
        if settings.get("auto_trail", True):
            trail = max(trail, profile.trail_preference * arousal * 0.5)
        else:
            trail = settings.get("trail_decay_manual", 0.0)

        if settings.get("auto_particles", True):
            particle = max(particle, profile.particle_preference * arousal * 0.3)
        else:
            particle = settings.get("particle_rate_manual", 0.0)

        return {
            "brightness": brightness,
            "energy": energy,
            "bloom": bloom,
            "noise": noise,
            "rotation": rotation,
            "anamorphic": anamorphic,
            "fog_offset": fog_offset,
            "hue_shift": hue_shift,
            "loop_offset": loop_offset,
            "trail": trail,
            "particle": particle,
            "particle_energy": arousal * profile.beat_reactivity,
            "saturation_offset": saturation_offset,
            "vignette_offset": vignette_offset,
            "chromatic_offset": chromatic_offset,
            "bg_intensity_offset": bg_intensity_offset,
            "color_temp_shift": color_temp_shift,
            "rotation_tempo_scale": rotation_tempo_scale,
        }

    def _smooth_state(self, targets: dict, dt: float):
        """Exponential smoothing of director state (non-auto outputs)."""
        s = self._state
        s.brightness_mult = _exp_smooth(s.brightness_mult, targets["brightness"], _TAU_MEDIUM, dt)
        s.energy_mult_mult = _exp_smooth(s.energy_mult_mult, targets["energy"], _TAU_MEDIUM, dt)
        s.bloom_mult = _exp_smooth(s.bloom_mult, targets["bloom"], _TAU_MEDIUM, dt)
        s.noise_mult_mult = _exp_smooth(s.noise_mult_mult, targets["noise"], _TAU_SLOW, dt)
        s.rotation_mult = _exp_smooth(s.rotation_mult, targets["rotation"], _TAU_SLOW, dt)
        s.anamorphic_mult = _exp_smooth(s.anamorphic_mult, targets["anamorphic"], _TAU_MEDIUM, dt)
        s.depth_fog_offset = _exp_smooth(s.depth_fog_offset, targets["fog_offset"], _TAU_SLOW, dt)
        s.hue_shift = _exp_smooth(s.hue_shift, targets["hue_shift"], _TAU_SLOW, dt)
        s.loop_count_offset = _exp_smooth(s.loop_count_offset, targets["loop_offset"], _TAU_SLOW, dt)
        s.trail_decay = _exp_smooth(s.trail_decay, targets["trail"], _TAU_MEDIUM, dt)
        s.particle_spawn_rate = _exp_smooth(s.particle_spawn_rate, targets["particle"], _TAU_FAST, dt)
        s.particle_energy = _exp_smooth(s.particle_energy, targets["particle_energy"], _TAU_FAST, dt)
        s.saturation_offset = _exp_smooth(s.saturation_offset, targets["saturation_offset"], _TAU_SLOW, dt)
        s.vignette_offset = _exp_smooth(s.vignette_offset, targets["vignette_offset"], _TAU_MEDIUM, dt)
        s.chromatic_offset = _exp_smooth(s.chromatic_offset, targets["chromatic_offset"], _TAU_FAST, dt)
        s.bg_intensity_offset = _exp_smooth(s.bg_intensity_offset, targets["bg_intensity_offset"], _TAU_SLOW, dt)
        s.color_temp_shift = _exp_smooth(s.color_temp_shift, targets["color_temp_shift"], _TAU_SLOW, dt)
        s.rotation_tempo_scale = _exp_smooth(s.rotation_tempo_scale, targets["rotation_tempo_scale"], _TAU_SLOW, dt)

    def _apply_to_settings(self, base: dict, intensity: float, features: dict,
                           profile: _GenreProfile) -> dict:
        """Blend director state with user settings using director_intensity."""
        s = self._state
        result = dict(base)

        # Smoothed audio-driven auto values
        auto_brightness = s.auto_brightness
        auto_bloom = s.auto_bloom_intensity
        auto_noise = s.auto_noise_mult
        auto_rotation = s.auto_rotation_speed
        auto_anamorphic = s.auto_anamorphic_flare
        auto_fog = s.auto_depth_fog
        auto_loops = int(round(s.auto_loop_count))
        auto_tint = (s.auto_tint_r, s.auto_tint_g, s.auto_tint_b)

        # Blend multiplier for manual-mode settings (director still modulates them)
        def _blend_mult(base_val, mult):
            adjusted = base_val * (1.0 + (mult - 1.0) * intensity)
            return max(0.0, adjusted)

        # When auto ON: use audio-driven value directly (already contains all modulation).
        # When auto OFF: use manual slider value × director blend_mult (existing behavior).
        if base.get("auto_brightness", True):
            result["brightness"] = auto_brightness
        else:
            result["brightness"] = _blend_mult(base["brightness"], s.brightness_mult)

        result["energy_mult"] = _blend_mult(base["energy_mult"], s.energy_mult_mult)

        if base.get("auto_bloom_intensity", True):
            result["bloom_intensity"] = auto_bloom
        else:
            result["bloom_intensity"] = _blend_mult(base["bloom_intensity"], s.bloom_mult)

        if base.get("auto_noise_mult", True):
            result["noise_mult"] = auto_noise
        else:
            result["noise_mult"] = _blend_mult(base["noise_mult"], s.noise_mult_mult)

        if base.get("auto_rotation_speed", True):
            result["rotation_speed"] = auto_rotation
        else:
            result["rotation_speed"] = _blend_mult(base["rotation_speed"], s.rotation_mult)

        if base.get("auto_anamorphic_flare", True):
            result["anamorphic_flare"] = min(1.0, auto_anamorphic)
        else:
            result["anamorphic_flare"] = min(1.0, _blend_mult(base["anamorphic_flare"], s.anamorphic_mult))

        if base.get("auto_depth_fog", True):
            result["depth_fog"] = max(0.0, min(1.0, auto_fog))
        else:
            result["depth_fog"] = max(0.0, min(1.0, base["depth_fog"] + s.depth_fog_offset * intensity))

        if base.get("auto_loop_count", True):
            result["loop_count"] = max(5, min(25, auto_loops))
        else:
            result["loop_count"] = max(5, min(25, int(base["loop_count"] + s.loop_count_offset * intensity)))

        if base.get("auto_bloom_tint", True):
            result["bloom_tint"] = auto_tint

        # _auto_values for UI display — always shows what audio-driven auto produces
        result["_auto_values"] = {
            "brightness": auto_brightness,
            "bloom_intensity": auto_bloom,
            "bloom_tint": auto_tint,
            "anamorphic_flare": min(1.0, auto_anamorphic),
            "loop_count": max(5, min(25, auto_loops)),
            "noise_mult": auto_noise,
            "rotation_speed": auto_rotation,
            "depth_fog": max(0.0, min(1.0, auto_fog)),
        }

        # Trail and particle rates (direct from director)
        result["trail_decay"] = s.trail_decay * intensity
        result["particle_spawn_rate"] = s.particle_spawn_rate * intensity
        result["particle_energy"] = s.particle_energy * intensity

        # Hue shift (applied in render)
        result["director_hue_shift"] = s.hue_shift * intensity

        # New dynamic outputs
        result["director_saturation"] = s.saturation_offset * intensity
        result["director_vignette"] = s.vignette_offset * intensity
        result["director_chromatic"] = s.chromatic_offset * intensity
        result["director_bg_boost"] = s.bg_intensity_offset * intensity
        result["director_color_temp"] = s.color_temp_shift * intensity
        result["director_rotation_tempo"] = 1.0 + (s.rotation_tempo_scale - 1.0) * intensity

        # Status info for UI overlay
        genre_id = features.get("genre_id", 6)
        section_type = features.get("section_type", 1)
        genre_name = GENRE_NAMES[genre_id] if 0 <= genre_id < len(GENRE_NAMES) else "Unknown"
        section_name = SECTION_NAMES[section_type] if 0 <= section_type < len(SECTION_NAMES) else "Unknown"
        result["director_genre"] = genre_name
        result["director_section"] = section_name
        result["director_climax"] = features.get("climax_score", 0.0)

        return result
