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


class _DirectorState:
    """Smooth internal state maintained across frames."""
    __slots__ = (
        "brightness_mult", "energy_mult_mult", "bloom_mult",
        "noise_mult_mult", "rotation_mult", "anamorphic_mult",
        "depth_fog_offset", "hue_shift", "loop_count_offset",
        "trail_decay", "particle_spawn_rate", "particle_energy",
        "prev_section_type", "section_transition_t",
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
            return result

        intensity = base_settings.get("director_intensity", 0.8)
        genre_id = features.get("genre_id", 6)
        profile = _PROFILES.get(genre_id, _PROFILES[6])

        # Compute raw targets
        targets = self._compute_targets(features, profile, base_settings)

        # Smooth state
        self._smooth_state(targets, delta_time)

        # Apply to settings
        return self._apply_to_settings(base_settings, intensity, features, profile)

    def _compute_targets(self, features: dict, profile: _GenreProfile, settings: dict) -> dict:
        """Compute raw target multipliers from musical context."""
        section_type = features.get("section_type", 1)
        climax = features.get("climax_score", 0.0)
        arousal = features.get("arousal", 0.5)
        valence = features.get("valence", 0.5)
        vocal = features.get("vocal_presence", 0.0)
        la_energy = features.get("lookahead_energy_delta", 0.0)
        la_section = features.get("lookahead_section_change", 0.0)
        la_climax = features.get("lookahead_climax", 0.0)

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

        # Climax boost
        if climax_reactions and climax > 0.3:
            climax_factor = (climax - 0.3) / 0.7 * profile.climax_intensity
            brightness += climax_factor * 0.3
            bloom += climax_factor * 0.5
            anamorphic += climax_factor * 0.3
            particle += climax_factor * 0.3

        # Look-ahead preparation
        if settings.get("beat_sync", True):
            # Before climax: slight tension buildup
            if la_climax > 0.3:
                tension = (la_climax - 0.3) / 0.7
                energy *= (1.0 - tension * 0.1)  # slight contraction
                noise *= (1.0 + tension * 0.3)    # anxiety
                rotation *= (1.0 + tension * 0.2)

            # Before section change: subtle fog "inhale"
            if la_section > 0.3:
                fog_offset += la_section * 0.1

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
        }

    def _smooth_state(self, targets: dict, dt: float):
        """Exponential smoothing of director state."""
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

    def _apply_to_settings(self, base: dict, intensity: float, features: dict,
                           profile: _GenreProfile) -> dict:
        """Blend director state with user settings using director_intensity."""
        s = self._state
        result = dict(base)

        # Blend multipliers: lerp between 1.0 (no change) and director value
        def _blend_mult(base_val, mult):
            adjusted = base_val * (1.0 + (mult - 1.0) * intensity)
            return max(0.0, adjusted)

        result["brightness"] = _blend_mult(base["brightness"], s.brightness_mult)
        result["energy_mult"] = _blend_mult(base["energy_mult"], s.energy_mult_mult)
        result["bloom_intensity"] = _blend_mult(base["bloom_intensity"], s.bloom_mult)
        result["noise_mult"] = _blend_mult(base["noise_mult"], s.noise_mult_mult)
        result["rotation_speed"] = _blend_mult(base["rotation_speed"], s.rotation_mult)
        result["anamorphic_flare"] = min(1.0, _blend_mult(base["anamorphic_flare"], s.anamorphic_mult))

        # Additive offsets
        result["depth_fog"] = max(0.0, min(1.0, base["depth_fog"] + s.depth_fog_offset * intensity))
        result["loop_count"] = max(5, min(25, int(base["loop_count"] + s.loop_count_offset * intensity)))

        # Trail and particle rates (direct from director)
        result["trail_decay"] = s.trail_decay * intensity
        result["particle_spawn_rate"] = s.particle_spawn_rate * intensity
        result["particle_energy"] = s.particle_energy * intensity

        # Hue shift (applied in render)
        result["director_hue_shift"] = s.hue_shift * intensity

        # Status info for UI overlay
        genre_id = features.get("genre_id", 6)
        section_type = features.get("section_type", 1)
        genre_name = GENRE_NAMES[genre_id] if 0 <= genre_id < len(GENRE_NAMES) else "Unknown"
        section_name = SECTION_NAMES[section_type] if 0 <= section_type < len(SECTION_NAMES) else "Unknown"
        result["director_genre"] = genre_name
        result["director_section"] = section_name
        result["director_climax"] = features.get("climax_score", 0.0)

        return result
