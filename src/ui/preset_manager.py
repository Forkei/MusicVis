"""Preset management: built-in presets + user JSON presets."""

import json
import os

PRESETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "assets", "presets",
)

BUILTIN = {
    "Default": {
        "bloom_tint": [0.3, 0.6, 1.0],
        "brightness": 2.6,
        "energy_mult": 3.0,
        "bloom_intensity": 5.0,
        "loop_count": 15,
        "noise_mult": 0.3,
        "rotation_speed": 2.18,
        "depth_fog": 0.4,
        "show_ring": True,
        "ring_opacity": 0.5,
        "anamorphic_flare": 0.3,
        "zoom": 1.0,
    },
    "EDM Rave": {
        "bloom_tint": [0.8, 0.2, 1.0],
        "brightness": 3.0,
        "energy_mult": 3.0,
        "bloom_intensity": 5.0,
        "loop_count": 22,
        "noise_mult": 1.5,
        "rotation_speed": 2.8,
        "depth_fog": 0.15,
        "show_ring": True,
        "ring_opacity": 0.8,
        "anamorphic_flare": 0.7,
        "zoom": 1.0,
    },
    "Chill Lo-fi": {
        "bloom_tint": [1.0, 0.7, 0.3],
        "brightness": 1.8,
        "energy_mult": 1.5,
        "bloom_intensity": 3.0,
        "loop_count": 8,
        "noise_mult": 0.2,
        "rotation_speed": 0.8,
        "depth_fog": 0.7,
        "show_ring": True,
        "ring_opacity": 0.3,
        "anamorphic_flare": 0.1,
        "zoom": 1.0,
    },
    "Minimal": {
        "bloom_tint": [0.9, 0.9, 1.0],
        "brightness": 2.0,
        "energy_mult": 2.0,
        "bloom_intensity": 1.5,
        "loop_count": 5,
        "noise_mult": 0.15,
        "rotation_speed": 1.0,
        "depth_fog": 0.8,
        "show_ring": False,
        "ring_opacity": 0.0,
        "anamorphic_flare": 0.0,
        "zoom": 1.0,
    },
    "Cinematic": {
        "bloom_tint": [0.3, 0.8, 0.9],
        "brightness": 2.2,
        "energy_mult": 2.5,
        "bloom_intensity": 4.0,
        "loop_count": 15,
        "noise_mult": 0.4,
        "rotation_speed": 1.2,
        "depth_fog": 0.5,
        "show_ring": True,
        "ring_opacity": 0.4,
        "anamorphic_flare": 0.6,
        "zoom": 1.0,
    },
}


class PresetManager:
    """Manages built-in and user presets."""

    def get_all_names(self) -> list[str]:
        """Return all preset names (built-in first, then user alphabetically)."""
        names = list(BUILTIN.keys())
        user = self._list_user_presets()
        names.extend(sorted(user))
        return names

    def load(self, name: str) -> dict | None:
        """Load a preset by name. Returns settings dict or None."""
        if name in BUILTIN:
            # Return a copy with tuples for bloom_tint
            preset = dict(BUILTIN[name])
            preset["bloom_tint"] = tuple(preset["bloom_tint"])
            return preset

        path = os.path.join(PRESETS_DIR, f"{name}.json")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    preset = json.load(f)
                if "bloom_tint" in preset:
                    preset["bloom_tint"] = tuple(preset["bloom_tint"])
                return preset
            except Exception:
                return None
        return None

    def save(self, name: str, settings: dict):
        """Save settings as a user preset JSON file."""
        os.makedirs(PRESETS_DIR, exist_ok=True)
        path = os.path.join(PRESETS_DIR, f"{name}.json")
        # Convert tuples to lists for JSON serialization
        data = dict(settings)
        if "bloom_tint" in data:
            data["bloom_tint"] = list(data["bloom_tint"])
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def delete(self, name: str):
        """Delete a user preset file."""
        path = os.path.join(PRESETS_DIR, f"{name}.json")
        try:
            os.remove(path)
        except OSError:
            pass

    def _list_user_presets(self) -> list[str]:
        """List user preset names from the presets directory."""
        if not os.path.isdir(PRESETS_DIR):
            return []
        names = []
        for fname in os.listdir(PRESETS_DIR):
            if fname.endswith(".json"):
                name = os.path.splitext(fname)[0]
                if name not in BUILTIN:
                    names.append(name)
        return names
