"""Settings panel UI with imgui."""

from imgui_bundle import imgui


DEFAULT_SETTINGS = {
    "bloom_tint": (0.0, 0.0, 1.0),
    "brightness": 2.6,
    "energy_mult": 3.0,
    "bloom_intensity": 5.0,
    "loop_count": 10,
    "noise_mult": 0.3,
    "rotation_speed": 2.18,
}


class SettingsPanel:
    """Collapsible settings panel for visualization tuning."""

    def __init__(self):
        self.settings = dict(DEFAULT_SETTINGS)

    def draw(self) -> dict:
        """Draw the settings panel. Returns current settings dict."""
        viewport = imgui.get_main_viewport()
        vp_size = viewport.size

        imgui.set_next_window_pos(
            (vp_size.x - 320, 20),
            imgui.Cond_.once,
        )
        imgui.set_next_window_size((300, 0), imgui.Cond_.once)
        imgui.set_next_window_bg_alpha(0.8)

        expanded, _ = imgui.begin(
            "Settings",
            None,
            imgui.WindowFlags_.no_focus_on_appearing,
        )

        if expanded:
            # Rendering
            if imgui.collapsing_header("Rendering", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_float(
                    "Brightness", self.settings["brightness"], 0.1, 3.0
                )
                if changed:
                    self.settings["brightness"] = val

                changed, val = imgui.slider_float(
                    "Bloom", self.settings["bloom_intensity"], 0.0, 5.0
                )
                if changed:
                    self.settings["bloom_intensity"] = val

                changed, color = imgui.color_edit3(
                    "Bloom Tint", list(self.settings["bloom_tint"])
                )
                if changed:
                    self.settings["bloom_tint"] = tuple(color)

            # Shape
            if imgui.collapsing_header("Shape", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_int(
                    "Loop Count", int(self.settings["loop_count"]), 5, 25
                )
                if changed:
                    self.settings["loop_count"] = val

                changed, val = imgui.slider_float(
                    "Noise", self.settings["noise_mult"], 0.1, 3.0
                )
                if changed:
                    self.settings["noise_mult"] = val

                changed, val = imgui.slider_float(
                    "Rotation", self.settings["rotation_speed"], 0.0, 3.0
                )
                if changed:
                    self.settings["rotation_speed"] = val

                changed, val = imgui.slider_float(
                    "Energy", self.settings["energy_mult"], 0.3, 3.0
                )
                if changed:
                    self.settings["energy_mult"] = val

            # Reset button
            imgui.separator()
            if imgui.button("Reset Defaults"):
                self.settings = dict(DEFAULT_SETTINGS)

        imgui.end()
        return self.settings
