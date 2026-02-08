"""Settings panel UI with imgui."""

from imgui_bundle import imgui


DEFAULT_SETTINGS = {
    "bloom_tint": (0.5, 0.7, 1.0),
    "brightness": 1.8,
    "energy_mult": 3.0,
    "bloom_intensity": 2.5,
    "loop_count": 15,
    "noise_mult": 0.3,
    "rotation_speed": 2.18,
    "depth_fog": 0.4,
    "show_ring": True,
    "ring_opacity": 0.5,
    "anamorphic_flare": 0.3,
    "zoom": 1.0,
    # Musical Director
    "director_enabled": True,
    "director_intensity": 0.8,
    "section_transitions": True,
    "climax_reactions": True,
    "beat_sync": True,
    "auto_trail": True,
    "auto_particles": True,
    "trail_decay_manual": 0.0,
    "particle_rate_manual": 0.0,
    # Auto mode flags (all default True)
    "auto_brightness": True,
    "auto_bloom_intensity": True,
    "auto_bloom_tint": True,
    "auto_anamorphic_flare": True,
    "auto_loop_count": True,
    "auto_noise_mult": True,
    "auto_rotation_speed": True,
    "auto_depth_fog": True,
}


class SettingsPanel:
    """Collapsible settings panel for visualization tuning."""

    def __init__(self):
        self.settings = dict(DEFAULT_SETTINGS)
        self._preset_manager = None
        self._current_preset = ""
        self._save_popup_open = False
        self._save_name_buf = ""

    def set_preset_manager(self, mgr):
        """Set the preset manager for preset UI integration."""
        self._preset_manager = mgr

    def _draw_auto_float(self, label, key, min_val, max_val, auto_values):
        """Draw a float slider with an auto checkbox. Returns nothing; mutates self.settings."""
        auto_key = f"auto_{key}"
        is_auto = self.settings.get(auto_key, True)

        # Auto checkbox
        changed, val = imgui.checkbox(f"##auto_{key}", is_auto)
        if changed:
            self.settings[auto_key] = val
            if not val and auto_values and key in auto_values:
                # Snap manual slider to current auto value when disabling auto
                self.settings[key] = auto_values[key]
            self._current_preset = ""
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Auto {label}")
        imgui.same_line()

        if is_auto and auto_values and key in auto_values:
            # Show live auto value in grayed slider
            imgui.begin_disabled()
            imgui.slider_float(label, auto_values[key], min_val, max_val)
            imgui.end_disabled()
        else:
            changed, val = imgui.slider_float(label, self.settings[key], min_val, max_val)
            if changed:
                self.settings[key] = val
                self._current_preset = ""

    def _draw_auto_int(self, label, key, min_val, max_val, auto_values):
        """Draw an int slider with an auto checkbox."""
        auto_key = f"auto_{key}"
        is_auto = self.settings.get(auto_key, True)

        changed, val = imgui.checkbox(f"##auto_{key}", is_auto)
        if changed:
            self.settings[auto_key] = val
            if not val and auto_values and key in auto_values:
                self.settings[key] = auto_values[key]
            self._current_preset = ""
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Auto {label}")
        imgui.same_line()

        if is_auto and auto_values and key in auto_values:
            imgui.begin_disabled()
            imgui.slider_int(label, auto_values[key], min_val, max_val)
            imgui.end_disabled()
        else:
            changed, val = imgui.slider_int(label, int(self.settings[key]), min_val, max_val)
            if changed:
                self.settings[key] = val
                self._current_preset = ""

    def _draw_auto_color(self, label, key, auto_values):
        """Draw a color picker with an auto checkbox."""
        auto_key = f"auto_{key}"
        is_auto = self.settings.get(auto_key, True)

        changed, val = imgui.checkbox(f"##auto_{key}", is_auto)
        if changed:
            self.settings[auto_key] = val
            if not val and auto_values and key in auto_values:
                self.settings[key] = auto_values[key]
            self._current_preset = ""
        if imgui.is_item_hovered():
            imgui.set_tooltip(f"Auto {label}")
        imgui.same_line()

        if is_auto and auto_values and key in auto_values:
            imgui.begin_disabled()
            imgui.color_edit3(label, list(auto_values[key]))
            imgui.end_disabled()
        else:
            changed, color = imgui.color_edit3(label, list(self.settings[key]))
            if changed:
                self.settings[key] = tuple(color)
                self._current_preset = ""

    def draw(self, auto_values=None) -> dict:
        """Draw the settings panel. Returns current settings dict.

        Args:
            auto_values: dict of auto-computed values from the director, or None.
        """
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
            # Preset selector
            if self._preset_manager is not None:
                self._draw_preset_ui()
                imgui.separator()

            # Rendering
            if imgui.collapsing_header("Rendering", imgui.TreeNodeFlags_.default_open):
                self._draw_auto_float("Brightness", "brightness", 0.1, 3.0, auto_values)
                self._draw_auto_float("Bloom", "bloom_intensity", 0.0, 5.0, auto_values)
                self._draw_auto_color("Bloom Tint", "bloom_tint", auto_values)
                self._draw_auto_float("Anamorphic", "anamorphic_flare", 0.0, 1.0, auto_values)

            # Shape
            if imgui.collapsing_header("Shape", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_float(
                    "Zoom", self.settings["zoom"], 0.5, 3.0
                )
                if changed:
                    self.settings["zoom"] = val
                    self._current_preset = ""

                self._draw_auto_int("Loop Count", "loop_count", 5, 25, auto_values)
                self._draw_auto_float("Noise", "noise_mult", 0.1, 3.0, auto_values)
                self._draw_auto_float("Rotation", "rotation_speed", 0.0, 3.0, auto_values)

                changed, val = imgui.slider_float(
                    "Energy", self.settings["energy_mult"], 0.3, 3.0
                )
                if changed:
                    self.settings["energy_mult"] = val
                    self._current_preset = ""

            # Effects
            if imgui.collapsing_header("Effects", imgui.TreeNodeFlags_.default_open):
                self._draw_auto_float("Depth Fog", "depth_fog", 0.0, 1.0, auto_values)

                changed, val = imgui.checkbox(
                    "Show Ring", self.settings["show_ring"]
                )
                if changed:
                    self.settings["show_ring"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Ring Opacity", self.settings["ring_opacity"], 0.0, 1.0
                )
                if changed:
                    self.settings["ring_opacity"] = val
                    self._current_preset = ""

            # Musical Director
            if imgui.collapsing_header("Musical Director", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.checkbox(
                    "Enable Director", self.settings["director_enabled"]
                )
                if changed:
                    self.settings["director_enabled"] = val
                    self._current_preset = ""

                if self.settings["director_enabled"]:
                    changed, val = imgui.slider_float(
                        "Intensity", self.settings["director_intensity"], 0.0, 1.0
                    )
                    if changed:
                        self.settings["director_intensity"] = val
                        self._current_preset = ""

                    changed, val = imgui.checkbox(
                        "Section Transitions", self.settings["section_transitions"]
                    )
                    if changed:
                        self.settings["section_transitions"] = val
                        self._current_preset = ""

                    changed, val = imgui.checkbox(
                        "Climax Reactions", self.settings["climax_reactions"]
                    )
                    if changed:
                        self.settings["climax_reactions"] = val
                        self._current_preset = ""

                    changed, val = imgui.checkbox(
                        "Beat Sync", self.settings["beat_sync"]
                    )
                    if changed:
                        self.settings["beat_sync"] = val
                        self._current_preset = ""

                    imgui.separator()

                    # Auto Trail (unified pattern)
                    changed, val = imgui.checkbox("##auto_trail", self.settings["auto_trail"])
                    if changed:
                        self.settings["auto_trail"] = val
                        self._current_preset = ""
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Auto Trail")
                    imgui.same_line()
                    if self.settings["auto_trail"]:
                        imgui.begin_disabled()
                        imgui.slider_float("Trail Decay", self.settings["trail_decay_manual"], 0.0, 0.98)
                        imgui.end_disabled()
                    else:
                        changed, val = imgui.slider_float(
                            "Trail Decay", self.settings["trail_decay_manual"], 0.0, 0.98
                        )
                        if changed:
                            self.settings["trail_decay_manual"] = val
                            self._current_preset = ""

                    # Auto Particles (unified pattern)
                    changed, val = imgui.checkbox("##auto_particles", self.settings["auto_particles"])
                    if changed:
                        self.settings["auto_particles"] = val
                        self._current_preset = ""
                    if imgui.is_item_hovered():
                        imgui.set_tooltip("Auto Particles")
                    imgui.same_line()
                    if self.settings["auto_particles"]:
                        imgui.begin_disabled()
                        imgui.slider_float("Particle Rate", self.settings["particle_rate_manual"], 0.0, 1.0)
                        imgui.end_disabled()
                    else:
                        changed, val = imgui.slider_float(
                            "Particle Rate", self.settings["particle_rate_manual"], 0.0, 1.0
                        )
                        if changed:
                            self.settings["particle_rate_manual"] = val
                            self._current_preset = ""

            # Reset button
            imgui.separator()
            if imgui.button("Reset Defaults"):
                self.settings = dict(DEFAULT_SETTINGS)
                self._current_preset = ""

        imgui.end()
        return self.settings

    def draw_director_status(self, settings: dict):
        """Draw director status overlay at top-center during playback."""
        genre = settings.get("director_genre", "")
        if not genre:
            return

        section = settings.get("director_section", "")
        climax = settings.get("director_climax", 0.0)

        label = f"{genre} | {section} | Climax: {int(climax * 100)}%"

        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        text_size = imgui.calc_text_size(label)
        win_w = text_size.x + 24
        win_h = 30

        imgui.set_next_window_pos(
            ((vp_size.x - win_w) / 2, 8),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size((win_w, win_h), imgui.Cond_.always)
        imgui.set_next_window_bg_alpha(0.5)

        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
            | imgui.WindowFlags_.no_focus_on_appearing
            | imgui.WindowFlags_.no_inputs
        )
        expanded, _ = imgui.begin("##director_status", None, flags)
        if expanded:
            imgui.text(label)
        imgui.end()

    def _draw_preset_ui(self):
        """Draw preset combo and save button."""
        mgr = self._preset_manager
        names = mgr.get_all_names()

        # Find current index
        current_idx = -1
        if self._current_preset in names:
            current_idx = names.index(self._current_preset)

        # Display label
        preview = self._current_preset if self._current_preset else "(Custom)"

        if imgui.begin_combo("Preset", preview):
            for i, name in enumerate(names):
                selected = (i == current_idx)
                clicked, _ = imgui.selectable(name, selected)
                if clicked:
                    self._current_preset = name
                    loaded = mgr.load(name)
                    if loaded:
                        self.settings.update(loaded)
            imgui.end_combo()

        imgui.same_line()
        if imgui.button("Save As..."):
            self._save_popup_open = True
            self._save_name_buf = ""

        if self._save_popup_open:
            imgui.open_popup("Save Preset")

        if imgui.begin_popup_modal("Save Preset")[0]:
            changed, self._save_name_buf = imgui.input_text(
                "Name", self._save_name_buf
            )
            if imgui.button("Save", (120, 0)):
                if self._save_name_buf.strip():
                    mgr.save(self._save_name_buf.strip(), dict(self.settings))
                    self._current_preset = self._save_name_buf.strip()
                    self._save_popup_open = False
                    imgui.close_current_popup()
            imgui.same_line()
            if imgui.button("Cancel", (120, 0)):
                self._save_popup_open = False
                imgui.close_current_popup()
            imgui.end_popup()
