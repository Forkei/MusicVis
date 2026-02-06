"""Player controls UI with imgui."""

from imgui_bundle import imgui


class PlayerControls:
    """Play/pause, seek bar, time display."""

    def __init__(self):
        self._seek_pos: float | None = None

    def draw(self, is_playing: bool, position: float, duration: float) -> dict:
        """Draw player controls overlay at bottom of screen.

        Returns dict with actions: {toggle: bool, seek: float|None}
        """
        actions = {"toggle": False, "seek": None}

        viewport = imgui.get_main_viewport()
        vp_size = viewport.size
        bar_height = 60
        padding = 10

        imgui.set_next_window_pos(
            (padding, vp_size.y - bar_height - padding),
            imgui.Cond_.always,
        )
        imgui.set_next_window_size(
            (vp_size.x - padding * 2, bar_height),
            imgui.Cond_.always,
        )

        flags = (
            imgui.WindowFlags_.no_title_bar
            | imgui.WindowFlags_.no_resize
            | imgui.WindowFlags_.no_move
            | imgui.WindowFlags_.no_scrollbar
        )

        imgui.set_next_window_bg_alpha(0.7)
        expanded, _ = imgui.begin("##player_controls", None, flags)
        if expanded:
            # Play/pause button
            label = "Pause" if is_playing else "Play"
            if imgui.button(label, (60, 0)):
                actions["toggle"] = True

            imgui.same_line()

            # Time display
            cur_min, cur_sec = divmod(int(position), 60)
            tot_min, tot_sec = divmod(int(duration), 60)
            time_str = f"{cur_min}:{cur_sec:02d} / {tot_min}:{tot_sec:02d}"
            imgui.text(time_str)

            # Seek bar
            imgui.set_next_item_width(-1)
            changed, value = imgui.slider_float(
                "##seek", position, 0.0, max(duration, 0.01), ""
            )
            if changed:
                actions["seek"] = value

        imgui.end()
        return actions
