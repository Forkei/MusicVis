"""Library panel UI: list downloaded songs with Play/Delete/Re-analyze."""

from imgui_bundle import imgui


class LibraryPanel:
    """ImGui panel listing downloaded songs."""

    def __init__(self):
        # video_id of song pending delete confirmation
        self._confirm_delete_id: str | None = None

    def draw(self, entries: list[dict]) -> dict | None:
        """Draw the library panel.

        Returns:
            {"action": "play"|"delete"|"reanalyze", "entry": ...} or None
        """
        result = None

        imgui.set_next_window_pos((20, 20), imgui.Cond_.once)
        imgui.set_next_window_size((340, 500), imgui.Cond_.once)

        expanded, _ = imgui.begin("Library", None, imgui.WindowFlags_.no_collapse)
        if expanded:
            if not entries:
                imgui.text_disabled("No downloaded songs yet")
            else:
                for entry in entries:
                    vid = entry["video_id"]
                    title = entry.get("title", "Unknown")
                    dur = entry.get("duration_str", "")
                    has_cache = entry.get("has_analysis", False)

                    imgui.push_id(vid)

                    # Title + duration
                    imgui.text(title)
                    if dur:
                        imgui.same_line()
                        imgui.text_disabled(f"({dur})")

                    # Buttons row
                    play_label = "Play" if has_cache else "Play (analyze)"
                    if imgui.small_button(play_label):
                        result = {"action": "play", "entry": entry}

                    imgui.same_line()
                    if imgui.small_button("Re-analyze"):
                        result = {"action": "reanalyze", "entry": entry}

                    imgui.same_line()

                    # Delete with confirmation
                    if self._confirm_delete_id == vid:
                        if imgui.small_button("Confirm?"):
                            result = {"action": "delete", "entry": entry}
                            self._confirm_delete_id = None
                        imgui.same_line()
                        if imgui.small_button("Cancel"):
                            self._confirm_delete_id = None
                    else:
                        if imgui.small_button("Delete"):
                            self._confirm_delete_id = vid

                    imgui.separator()
                    imgui.pop_id()

        imgui.end()
        return result
