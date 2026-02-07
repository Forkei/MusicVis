"""Settings panel UI with imgui."""

from imgui_bundle import imgui


DEFAULT_SETTINGS = {
    "bloom_tint": (0.3, 0.6, 1.0),
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
            # Preset selector
            if self._preset_manager is not None:
                self._draw_preset_ui()
                imgui.separator()

            # Rendering
            if imgui.collapsing_header("Rendering", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_float(
                    "Brightness", self.settings["brightness"], 0.1, 3.0
                )
                if changed:
                    self.settings["brightness"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Bloom", self.settings["bloom_intensity"], 0.0, 5.0
                )
                if changed:
                    self.settings["bloom_intensity"] = val
                    self._current_preset = ""

                changed, color = imgui.color_edit3(
                    "Bloom Tint", list(self.settings["bloom_tint"])
                )
                if changed:
                    self.settings["bloom_tint"] = tuple(color)
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Anamorphic", self.settings["anamorphic_flare"], 0.0, 1.0
                )
                if changed:
                    self.settings["anamorphic_flare"] = val
                    self._current_preset = ""

            # Shape
            if imgui.collapsing_header("Shape", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_float(
                    "Zoom", self.settings["zoom"], 0.5, 3.0
                )
                if changed:
                    self.settings["zoom"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_int(
                    "Loop Count", int(self.settings["loop_count"]), 5, 25
                )
                if changed:
                    self.settings["loop_count"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Noise", self.settings["noise_mult"], 0.1, 3.0
                )
                if changed:
                    self.settings["noise_mult"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Rotation", self.settings["rotation_speed"], 0.0, 3.0
                )
                if changed:
                    self.settings["rotation_speed"] = val
                    self._current_preset = ""

                changed, val = imgui.slider_float(
                    "Energy", self.settings["energy_mult"], 0.3, 3.0
                )
                if changed:
                    self.settings["energy_mult"] = val
                    self._current_preset = ""

            # Effects
            if imgui.collapsing_header("Effects", imgui.TreeNodeFlags_.default_open):
                changed, val = imgui.slider_float(
                    "Depth Fog", self.settings["depth_fog"], 0.0, 1.0
                )
                if changed:
                    self.settings["depth_fog"] = val
                    self._current_preset = ""

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

            # Reset button
            imgui.separator()
            if imgui.button("Reset Defaults"):
                self.settings = dict(DEFAULT_SETTINGS)
                self._current_preset = ""

        imgui.end()
        return self.settings

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
                "Name", self._save_name_buf, 256
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
